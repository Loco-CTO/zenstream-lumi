#!/usr/bin/env python3
"""Check Lumi data-provenance joins and release-blocking use permissions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST_DEFAULT = ROOT / "provenance" / "data_sources.json"
GENERATION_MANIFEST_DEFAULT = ROOT / "provenance" / "synthetic_data.json"
SHA256_PATTERN = re.compile(r"^sha256:[A-Fa-f0-9]{64}$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
USES = {
    "tokenizer_training", "pretraining", "instruction_training",
    "zenstream_specific_training", "evaluation", "filtering", "generation", "other",
}
TRAINING_USES = {
    "tokenizer_training", "pretraining", "instruction_training", "zenstream_specific_training",
}
RECORD_SPLITS = {"unassigned", "train", "development", "public_test", "sealed_holdout", "excluded"}
CASE_SPLIT_TO_RECORD = {"development": "development", "final_holdout": "sealed_holdout"}
PERMISSIONS = {"permitted", "prohibited", "conditional", "unknown", "not_applicable"}
CASE_FINGERPRINT_FIELDS = (
    "schema_version", "case_id", "family_id", "split", "language", "categories", "turns",
    "trusted_context", "gold",
)


class ProvenanceError(ValueError):
    """Raised when a data source, sample, or evaluation record fails the provenance gate."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ProvenanceError(f"{path}: invalid JSON: {exc}") from exc


def load_json_document(path: Path) -> Any:
    """Read a UTF-8 JSON document using the provenance validator's strict parser."""
    return _read_json(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ProvenanceError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ProvenanceError(f"{path}:{line_number}: each row must be an object")
            rows.append(row)
    return rows


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _hash_string(value: Any) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _uri_string(value: Any) -> bool:
    if not _nonempty(value):
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    if not parsed.scheme:
        return False
    return parsed.scheme.lower() not in {"http", "https"} or bool(parsed.netloc)


def _date_string(value: Any) -> bool:
    if not isinstance(value, str) or DATE_PATTERN.fullmatch(value) is None:
        return False
    try:
        from datetime import date

        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def evaluation_case_sha256(case: dict[str, Any]) -> str:
    """Hash the authored evaluation content, excluding review and provenance metadata."""
    payload = {key: case.get(key) for key in CASE_FINGERPRINT_FIELDS}
    return canonical_sha256(payload)


def _manifest_sources(manifest: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(manifest, dict) or set(manifest) != {"manifest_version", "status", "sources"}:
        raise ProvenanceError("source manifest must contain only manifest_version, status, and sources")
    if manifest["manifest_version"] != 1 or isinstance(manifest["manifest_version"], bool):
        raise ProvenanceError("unsupported source manifest version")
    sources = manifest["sources"]
    if not isinstance(sources, list):
        raise ProvenanceError("source manifest sources must be a list")
    if not isinstance(manifest["status"], str) or manifest["status"] not in {"empty", "reviewing", "approved_sources_present"}:
        raise ProvenanceError("invalid source manifest status")
    if (manifest["status"] == "empty") != (len(sources) == 0):
        raise ProvenanceError("source manifest status does not match its source list")
    if manifest["status"] == "approved_sources_present" and not any(
        isinstance(source, dict) and source.get("decision") == "approved_for_use" for source in sources
    ):
        raise ProvenanceError("approved_sources_present requires at least one approved source")

    by_id: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(sources, start=1):
        prefix = f"source {index}"
        if not isinstance(source, dict):
            raise ProvenanceError(f"{prefix} must be an object")
        required_fields = {
            "source_id", "source_type", "name", "canonical_identifier", "authors_or_organization",
            "version", "revision", "release_date", "access_date", "license", "permissions",
            "lumi_uses", "decision", "direct_or_indirect", "transformations", "filters",
            "deduplication", "document_count", "example_count", "token_count", "checksum",
            "review", "generation_record_id",
        }
        if required_fields - source.keys() or source.keys() - required_fields:
            raise ProvenanceError(f"{prefix} does not match the source manifest record fields")
        source_id = source.get("source_id")
        if not _nonempty(source_id):
            raise ProvenanceError(f"{prefix} has no nonempty source_id")
        if source_id in by_id:
            raise ProvenanceError(f"duplicate source_id {source_id!r}")
        source_type = source.get("source_type")
        if not isinstance(source_type, str) or source_type not in {"external_dataset", "synthetic", "manually_authored", "public_metadata", "other"}:
            raise ProvenanceError(f"{prefix} has invalid source_type")
        decision = source.get("decision")
        if not isinstance(decision, str) or decision not in {"review_pending", "approved_for_use", "excluded"}:
            raise ProvenanceError(f"{prefix} has invalid decision")
        for field in ("name", "canonical_identifier", "authors_or_organization", "version", "revision"):
            if not _nonempty(source.get(field)):
                raise ProvenanceError(f"{prefix} needs {field}")
        if source["release_date"] is not None and not _date_string(source["release_date"]):
            raise ProvenanceError(f"{prefix} has invalid release_date")
        if not _date_string(source["access_date"]):
            raise ProvenanceError(f"{prefix} has invalid access_date")
        if not isinstance(source.get("direct_or_indirect"), str) or source["direct_or_indirect"] not in {"direct", "indirect"}:
            raise ProvenanceError(f"{prefix} has invalid direct_or_indirect")
        for field in ("transformations", "filters", "deduplication"):
            values = source.get(field)
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise ProvenanceError(f"{prefix} {field} must be a list of strings")
        for field in ("document_count", "example_count", "token_count"):
            value = source.get(field)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise ProvenanceError(f"{prefix} {field} must be a nonnegative integer or null")
        if source.get("checksum") is not None and not _hash_string(source["checksum"]):
            raise ProvenanceError(f"{prefix} checksum must be a SHA-256 or null")
        uses = source.get("lumi_uses")
        if (
            not isinstance(uses, list)
            or not uses
            or any(not isinstance(use, str) or use not in USES for use in uses)
            or len(set(uses)) != len(uses)
        ):
            raise ProvenanceError(f"{prefix} has invalid or duplicate lumi_uses")
        permissions = source.get("permissions")
        required_permissions = {
            "commercial_use", "redistribution", "modification", "derivative_works",
            "training_use", "evaluation_use",
        }
        if not isinstance(permissions, dict) or set(permissions) != required_permissions:
            raise ProvenanceError(f"{prefix} permissions do not match the source manifest fields")
        if any(not isinstance(value, str) or value not in PERMISSIONS for value in permissions.values()):
            raise ProvenanceError(f"{prefix} has an invalid permission value")
        license_record = source.get("license")
        license_fields = {"identifier", "url", "copyright_notice", "attribution_requirement", "terms_review"}
        if not isinstance(license_record, dict) or set(license_record) != license_fields:
            raise ProvenanceError(f"{prefix} license record is incomplete or has unsupported fields")
        if (
            not _nonempty(license_record.get("identifier"))
            or not _uri_string(license_record.get("url"))
            or not isinstance(license_record.get("copyright_notice"), str)
            or not isinstance(license_record.get("attribution_requirement"), str)
            or not _nonempty(license_record.get("terms_review"))
        ):
            raise ProvenanceError(f"{prefix} needs recorded license terms review")
        review = source.get("review")
        review_fields = {"reviewed_by", "reviewed_date", "rationale"}
        if (
            not isinstance(review, dict)
            or set(review) != review_fields
            or not _nonempty(review.get("reviewed_by"))
            or not _date_string(review.get("reviewed_date"))
            or not _nonempty(review.get("rationale"))
        ):
            raise ProvenanceError(f"{prefix} needs complete source review metadata")
        if source["source_type"] == "synthetic" and not _nonempty(source.get("generation_record_id")):
            raise ProvenanceError(f"{prefix} synthetic source needs a generation_record_id")
        by_id[source_id] = source
    return by_id


def _manifest_generations(manifest: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(manifest, dict) or set(manifest) != {"manifest_version", "status", "generations"}:
        raise ProvenanceError("synthetic manifest must contain only manifest_version, status, and generations")
    if manifest["manifest_version"] != 1 or isinstance(manifest["manifest_version"], bool):
        raise ProvenanceError("unsupported synthetic manifest version")
    rows = manifest["generations"]
    if not isinstance(rows, list):
        raise ProvenanceError("synthetic manifest generations must be a list")
    if not isinstance(manifest["status"], str) or manifest["status"] not in {"empty", "records_present"}:
        raise ProvenanceError("invalid synthetic manifest status")
    if (manifest["status"] == "empty") != (len(rows) == 0):
        raise ProvenanceError("synthetic manifest status does not match its generation list")
    by_id: dict[str, dict[str, Any]] = {}
    for index, generation in enumerate(rows, start=1):
        prefix = f"generation {index}"
        if not isinstance(generation, dict):
            raise ProvenanceError(f"{prefix} must be an object")
        required_fields = {
            "generation_id", "provider", "model_name", "model_version", "generation_date",
            "purpose", "prompt_family", "prompt_template_id", "prompt_template_sha256", "settings",
            "example_count", "filtered", "filter_details", "deduplicated", "deduplication_details",
            "automatically_validated", "automatic_validation_details", "manually_reviewed",
            "manual_review_details", "downstream_uses", "provider_terms_url", "output_checksum",
            "review_notes",
        }
        if required_fields - generation.keys() or generation.keys() - required_fields:
            raise ProvenanceError(f"{prefix} does not match the synthetic manifest record fields")
        generation_id = generation.get("generation_id")
        if not _nonempty(generation_id):
            raise ProvenanceError(f"{prefix} has no nonempty generation_id")
        if generation_id in by_id:
            raise ProvenanceError(f"duplicate generation_id {generation_id!r}")
        for field in ("provider", "model_name", "model_version", "purpose", "prompt_family", "prompt_template_id", "review_notes"):
            if not _nonempty(generation.get(field)):
                raise ProvenanceError(f"{prefix} needs {field}")
        if not _date_string(generation.get("generation_date")):
            raise ProvenanceError(f"{prefix} has invalid generation_date")
        if not _hash_string(generation.get("prompt_template_sha256")):
            raise ProvenanceError(f"{prefix} needs a SHA-256 for the prompt template")
        if not isinstance(generation.get("settings"), dict):
            raise ProvenanceError(f"{prefix} settings must be an object")
        count = generation.get("example_count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ProvenanceError(f"{prefix} example_count must be a nonnegative integer")
        for field in ("filtered", "deduplicated", "automatically_validated", "manually_reviewed"):
            if not isinstance(generation.get(field), bool):
                raise ProvenanceError(f"{prefix} {field} must be a boolean")
        for field in ("filter_details", "deduplication_details", "automatic_validation_details", "manual_review_details"):
            if not isinstance(generation.get(field), str):
                raise ProvenanceError(f"{prefix} {field} must be a string")
        uses = generation.get("downstream_uses")
        if (
            not isinstance(uses, list)
            or not uses
            or any(not isinstance(use, str) or use not in USES for use in uses)
            or len(set(uses)) != len(uses)
        ):
            raise ProvenanceError(f"{prefix} has invalid downstream_uses")
        if not _uri_string(generation.get("provider_terms_url")):
            raise ProvenanceError(f"{prefix} needs a provider terms URI")
        if generation.get("output_checksum") is not None and not _hash_string(generation["output_checksum"]):
            raise ProvenanceError(f"{prefix} output_checksum must be a SHA-256 or null")
        by_id[generation_id] = generation
    return by_id


def _check_record_shape(record: Any, index: int) -> dict[str, Any]:
    prefix = f"sample record {index}"
    if not isinstance(record, dict):
        raise ProvenanceError(f"{prefix} must be an object")
    required_fields = {
        "schema_version", "record_id", "sample_kind", "sample_sha256", "language_tags", "lumi_uses",
        "decision", "split", "contamination_status", "privacy_status", "source_items", "transformations",
        "parent_record_ids", "review",
    }
    optional_fields = {"generator", "author_consent_record_id"}
    if required_fields - record.keys() or record.keys() - required_fields - optional_fields:
        raise ProvenanceError(f"{prefix} does not match the sample provenance record fields")
    record_id = record.get("record_id")
    if not _nonempty(record_id):
        raise ProvenanceError(f"{prefix} needs a nonempty record_id")
    if record.get("schema_version") != 1 or isinstance(record.get("schema_version"), bool):
        raise ProvenanceError(f"{prefix} has unsupported schema_version")
    sample_kind = record.get("sample_kind")
    if not isinstance(sample_kind, str) or sample_kind not in {"external_document", "transformed_sample", "manually_authored", "synthetic"}:
        raise ProvenanceError(f"{prefix} has invalid sample_kind")
    if not _hash_string(record.get("sample_sha256")):
        raise ProvenanceError(f"{prefix} needs a sample SHA-256")
    language_tags = record.get("language_tags")
    if (
        not isinstance(language_tags, list)
        or not language_tags
        or any(not _nonempty(tag) or len(tag) < 2 for tag in language_tags)
        or len(set(language_tags)) != len(language_tags)
    ):
        raise ProvenanceError(f"{prefix} has invalid or duplicate language_tags")
    uses = record.get("lumi_uses")
    if (
        not isinstance(uses, list)
        or not uses
        or any(not isinstance(use, str) or use not in USES for use in uses)
        or len(set(uses)) != len(uses)
    ):
        raise ProvenanceError(f"{prefix} has invalid or duplicate lumi_uses")
    decision = record.get("decision")
    if not isinstance(decision, str) or decision not in {"review_pending", "approved_for_use", "excluded"}:
        raise ProvenanceError(f"{prefix} has invalid decision")
    split = record.get("split")
    if not isinstance(split, str) or split not in RECORD_SPLITS:
        raise ProvenanceError(f"{prefix} has invalid split")
    contamination_status = record.get("contamination_status")
    if not isinstance(contamination_status, str) or contamination_status not in {"not_checked", "checked_clear", "known_overlap", "excluded"}:
        raise ProvenanceError(f"{prefix} has invalid contamination_status")
    privacy_status = record.get("privacy_status")
    if not isinstance(privacy_status, str) or privacy_status not in {"not_reviewed", "cleared", "redacted", "excluded"}:
        raise ProvenanceError(f"{prefix} has invalid privacy_status")
    items = record.get("source_items")
    if not isinstance(items, list) or not items:
        raise ProvenanceError(f"{prefix} needs at least one source item")
    transformations = record.get("transformations")
    if not isinstance(transformations, list):
        raise ProvenanceError(f"{prefix} transformations must be a list")
    for transform_index, transform in enumerate(transformations, start=1):
        transform_fields = {"operation_id", "revision", "config_sha256"}
        if not isinstance(transform, dict) or set(transform) != transform_fields:
            raise ProvenanceError(f"{prefix} transformation {transform_index} is incomplete")
        if not _nonempty(transform["operation_id"]) or not _nonempty(transform["revision"]):
            raise ProvenanceError(f"{prefix} transformation {transform_index} needs operation and revision")
        if not _hash_string(transform["config_sha256"]):
            raise ProvenanceError(f"{prefix} transformation {transform_index} needs a config SHA-256")
    parents = record.get("parent_record_ids")
    if not isinstance(parents, list) or any(not _nonempty(parent) for parent in parents) or len(set(parents)) != len(parents):
        raise ProvenanceError(f"{prefix} has invalid or duplicate parent_record_ids")
    review = record.get("review")
    review_fields = {"reviewed_by", "reviewed_date", "rationale"}
    if (
        not isinstance(review, dict)
        or set(review) != review_fields
        or not _nonempty(review.get("reviewed_by"))
        or (review.get("reviewed_date") is not None and not _date_string(review["reviewed_date"]))
        or not _nonempty(review.get("rationale"))
    ):
        raise ProvenanceError(f"{prefix} needs complete review metadata")
    if record["decision"] == "approved_for_use":
        if record["contamination_status"] != "checked_clear":
            raise ProvenanceError(f"{prefix} cannot be approved before contamination review is clear")
        if record["privacy_status"] not in {"cleared", "redacted"}:
            raise ProvenanceError(f"{prefix} cannot be approved before privacy review")
        if not _nonempty(review.get("reviewed_by")) or not _date_string(review.get("reviewed_date")):
            raise ProvenanceError(f"{prefix} approval needs reviewer and date")
    training = bool(TRAINING_USES.intersection(uses))
    evaluation = "evaluation" in uses
    if training and evaluation:
        raise ProvenanceError(f"{prefix} mixes training and evaluation uses")
    if training and record["split"] != "train":
        raise ProvenanceError(f"{prefix} with training uses must be in the train split")
    if evaluation and record["split"] not in {"development", "public_test", "sealed_holdout"}:
        raise ProvenanceError(f"{prefix} evaluation use needs a development, public_test, or sealed_holdout split")
    if record["split"] == "train" and evaluation:
        raise ProvenanceError(f"{prefix} evaluation data cannot be assigned to train")
    if record["sample_kind"] == "synthetic" and not isinstance(record.get("generator"), dict):
        raise ProvenanceError(f"{prefix} synthetic record needs generator metadata")
    if "generator" in record:
        generator = record["generator"]
        generator_fields = {"model_or_tool", "version", "prompt_template_id", "prompt_template_sha256"}
        if (
            not isinstance(generator, dict)
            or set(generator) != generator_fields
            or any(not _nonempty(generator.get(field)) for field in ("model_or_tool", "version", "prompt_template_id"))
            or not _hash_string(generator.get("prompt_template_sha256"))
        ):
            raise ProvenanceError(f"{prefix} has incomplete generator metadata")
    if record["sample_kind"] == "manually_authored" and not _nonempty(record.get("author_consent_record_id")):
        raise ProvenanceError(f"{prefix} manually authored record needs a consent record")
    if "author_consent_record_id" in record and not _nonempty(record["author_consent_record_id"]):
        raise ProvenanceError(f"{prefix} author_consent_record_id must be nonempty")
    return record


def _record_permission_for_use(use: str) -> str | None:
    if use == "evaluation":
        return "evaluation_use"
    if use in TRAINING_USES:
        return "training_use"
    return None


def _check_record_sources(
    record: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    generations: dict[str, dict[str, Any]],
) -> None:
    record_id = record["record_id"]
    uses = set(record["lumi_uses"])
    is_approved = record["decision"] == "approved_for_use"
    source_types: set[str] = set()
    generation_ids: set[str] = set()
    for index, item in enumerate(record["source_items"], start=1):
        prefix = f"sample record {record_id!r} source item {index}"
        if not isinstance(item, dict):
            raise ProvenanceError(f"{prefix} must be an object")
        item_fields = {
            "source_id", "item_id", "source_uri", "source_revision", "source_sha256",
            "rights_basis", "license_identifier", "license_uri", "rights_evidence_uris",
            "rights_review_status", "rights_reviewed_date", "permissions", "attribution",
        }
        if set(item) != item_fields:
            raise ProvenanceError(f"{prefix} does not match the sample source-item fields")
        if not _nonempty(item.get("item_id")) or not _nonempty(item.get("source_revision")):
            raise ProvenanceError(f"{prefix} needs an item_id and source_revision")
        if item.get("source_uri") is not None and not _uri_string(item["source_uri"]):
            raise ProvenanceError(f"{prefix} has an invalid source_uri")
        if not _hash_string(item.get("source_sha256")):
            raise ProvenanceError(f"{prefix} needs a source SHA-256")
        rights_basis = item.get("rights_basis")
        if not isinstance(rights_basis, str) or rights_basis not in {"license", "public_domain", "direct_permission", "other", "unknown"}:
            raise ProvenanceError(f"{prefix} has an invalid rights_basis")
        if not _nonempty(item.get("license_identifier")):
            raise ProvenanceError(f"{prefix} needs a license_identifier")
        if item.get("license_uri") is not None and not _uri_string(item["license_uri"]):
            raise ProvenanceError(f"{prefix} has an invalid license_uri")
        evidence = item.get("rights_evidence_uris")
        if not isinstance(evidence, list) or not evidence or any(not _uri_string(uri) for uri in evidence):
            raise ProvenanceError(f"{prefix} needs valid rights evidence URIs")
        rights_review_status = item.get("rights_review_status")
        if not isinstance(rights_review_status, str) or rights_review_status not in {"pending", "reviewed", "excluded"}:
            raise ProvenanceError(f"{prefix} has an invalid rights_review_status")
        rights_reviewed_date = item.get("rights_reviewed_date")
        if rights_reviewed_date is not None and not _date_string(rights_reviewed_date):
            raise ProvenanceError(f"{prefix} has an invalid rights_reviewed_date")
        if not isinstance(item.get("attribution"), str):
            raise ProvenanceError(f"{prefix} attribution must be a string")
        item_permissions = item.get("permissions")
        item_permission_fields = {"training_use", "evaluation_use", "commercial_use", "modification", "redistribution"}
        if (
            not isinstance(item_permissions, dict)
            or set(item_permissions) != item_permission_fields
            or any(not isinstance(value, str) or value not in PERMISSIONS for value in item_permissions.values())
        ):
            raise ProvenanceError(f"{prefix} has incomplete or invalid item-level permissions")
        source_id = item.get("source_id")
        if not isinstance(source_id, str) or source_id not in sources:
            raise ProvenanceError(f"{prefix} references unknown source_id {source_id!r}")
        source = sources[source_id]
        source_types.add(source["source_type"])
        if item.get("source_revision") != source.get("revision"):
            raise ProvenanceError(f"{prefix} source revision differs from the manifest")
        for use in uses:
            if use not in source.get("lumi_uses", []):
                raise ProvenanceError(f"{prefix} source {source_id!r} was not reviewed for {use}")
            if is_approved and source.get("decision") != "approved_for_use":
                raise ProvenanceError(f"{prefix} source {source_id!r} is not approved")
            required_permission = _record_permission_for_use(use)
            if required_permission and is_approved:
                if item_permissions.get(required_permission) != "permitted":
                    raise ProvenanceError(f"{prefix} does not permit {use} at item level")
                source_permission = source.get("permissions", {}).get(required_permission)
                if source_permission != "permitted":
                    raise ProvenanceError(f"{prefix} source manifest does not permit {use}")
        if is_approved:
            rights_basis = item.get("rights_basis")
            if not isinstance(rights_basis, str) or rights_basis not in {"license", "public_domain", "direct_permission"}:
                raise ProvenanceError(f"{prefix} has no approved rights basis")
            if item.get("rights_review_status") != "reviewed" or not _date_string(item.get("rights_reviewed_date")):
                raise ProvenanceError(f"{prefix} needs a completed rights review")
            evidence = item.get("rights_evidence_uris")
            if not isinstance(evidence, list) or not evidence or any(not _nonempty(uri) for uri in evidence):
                raise ProvenanceError(f"{prefix} needs rights evidence references")
        if source["source_type"] == "synthetic":
            generation_id = source.get("generation_record_id")
            if generation_id not in generations:
                raise ProvenanceError(f"{prefix} references unknown generation {generation_id!r}")
            generation_ids.add(generation_id)
            if not uses.issubset(set(generations[generation_id].get("downstream_uses", []))):
                raise ProvenanceError(f"{prefix} generation record does not authorize every downstream use")
            generator = record.get("generator", {})
            expected = {
                "model_or_tool": generations[generation_id].get("model_name"),
                "version": generations[generation_id].get("model_version"),
                "prompt_template_id": generations[generation_id].get("prompt_template_id"),
                "prompt_template_sha256": generations[generation_id].get("prompt_template_sha256"),
            }
            if any(generator.get(key) != value for key, value in expected.items()):
                raise ProvenanceError(f"{prefix} generator metadata differs from the generation manifest")
    kind = record["sample_kind"]
    if kind == "synthetic" and "synthetic" not in source_types:
        raise ProvenanceError(f"sample record {record_id!r} has no synthetic source item")
    if kind == "manually_authored" and "manually_authored" not in source_types:
        raise ProvenanceError(f"sample record {record_id!r} has no manually authored source")
    if len(generation_ids) > 1:
        raise ProvenanceError(f"sample record {record_id!r} mixes unrelated synthetic generation records")


def validate_evaluation_case_provenance(
    cases: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> None:
    """Require every evaluated case to resolve to one approved, split-matched content record."""
    record_by_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ProvenanceError(f"sample record {index} must be an object")
        record_id = record.get("record_id")
        if not _nonempty(record_id):
            raise ProvenanceError(f"sample record {index} needs a nonempty record_id")
        if record_id in record_by_id:
            raise ProvenanceError(f"duplicate record_id {record_id!r}")
        record_by_id[record_id] = record
    used_record_ids: set[str] = set()
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ProvenanceError(f"evaluation case {index} must be an object")
        version = case.get("schema_version")
        if not isinstance(version, int) or isinstance(version, bool) or version != 3:
            raise ProvenanceError(f"evaluation case {case.get('case_id')!r} schema_version must be 3")
        if "trusted_context" not in case:
            raise ProvenanceError(f"evaluation case {case.get('case_id')!r} lacks trusted_context")
        record_id = case.get("provenance_record_id")
        if not _nonempty(record_id):
            raise ProvenanceError(f"evaluation case {case.get('case_id')!r} has no nonempty provenance_record_id")
        record = record_by_id.get(record_id)
        if record is None:
            raise ProvenanceError(f"evaluation case {case.get('case_id')!r} has no matching provenance record")
        if record_id in used_record_ids:
            raise ProvenanceError(f"evaluation provenance record {record_id!r} is reused by multiple cases")
        used_record_ids.add(record_id)
        if record.get("decision") != "approved_for_use":
            raise ProvenanceError(f"evaluation case {case.get('case_id')!r} provenance is not approved")
        if "evaluation" not in record.get("lumi_uses", []):
            raise ProvenanceError(f"evaluation case {case.get('case_id')!r} provenance lacks evaluation use")
        case_split = case.get("split")
        expected_split = CASE_SPLIT_TO_RECORD.get(case_split) if isinstance(case_split, str) else None
        if expected_split is None or record.get("split") != expected_split:
            raise ProvenanceError(f"evaluation case {case.get('case_id')!r} split differs from its provenance record")
        if record.get("sample_sha256") != evaluation_case_sha256(case):
            raise ProvenanceError(f"evaluation case {case.get('case_id')!r} content hash differs from its provenance record")


def validate_bundle(
    source_manifest: Any,
    generation_manifest: Any,
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]] | None = None,
) -> dict[str, int | str]:
    sources = _manifest_sources(source_manifest)
    generations = _manifest_generations(generation_manifest)
    for source in sources.values():
        if source["source_type"] != "synthetic":
            continue
        generation_id = source.get("generation_record_id")
        generation = generations.get(generation_id)
        if generation is None:
            raise ProvenanceError(f"synthetic source {source['source_id']!r} references unknown generation {generation_id!r}")
        if not set(source["lumi_uses"]).issubset(set(generation["downstream_uses"])):
            raise ProvenanceError(f"synthetic source {source['source_id']!r} exceeds its generation record's downstream uses")
    record_by_id: dict[str, dict[str, Any]] = {}
    for index, raw_record in enumerate(records, start=1):
        record = _check_record_shape(raw_record, index)
        record_id = record["record_id"]
        if record_id in record_by_id:
            raise ProvenanceError(f"duplicate record_id {record_id!r}")
        record_by_id[record_id] = record
    for record in record_by_id.values():
        _check_record_sources(record, sources, generations)
        for parent_id in record["parent_record_ids"]:
            if parent_id not in record_by_id:
                raise ProvenanceError(f"sample record {record['record_id']!r} references missing parent {parent_id!r}")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(record_id: str) -> None:
        if record_id in visiting:
            raise ProvenanceError(f"sample provenance lineage contains a cycle at {record_id!r}")
        if record_id in visited:
            return
        visiting.add(record_id)
        for parent_id in record_by_id[record_id]["parent_record_ids"]:
            visit(parent_id)
        visiting.remove(record_id)
        visited.add(record_id)

    for record_id in record_by_id:
        visit(record_id)

    lineage_splits: dict[str, set[str]] = {}
    visited_lineages: set[tuple[str, str]] = set()

    def assign_lineage_split(record_id: str, split: str) -> None:
        key = (record_id, split)
        if key in visited_lineages:
            return
        visited_lineages.add(key)
        record = record_by_id[record_id]
        if record["split"] == "excluded":
            raise ProvenanceError(f"sample record {record_id!r} is excluded but appears in an active data lineage")
        splits = lineage_splits.setdefault(record_id, set())
        splits.add(split)
        if len(splits) > 1:
            raise ProvenanceError(f"sample record {record_id!r} or its descendants cross data splits")
        for parent_id in record["parent_record_ids"]:
            assign_lineage_split(parent_id, split)

    active_splits = {"train", "development", "public_test", "sealed_holdout"}
    for record_id, record in record_by_id.items():
        if record["split"] in active_splits:
            assign_lineage_split(record_id, record["split"])
    if cases is not None:
        validate_evaluation_case_provenance(cases, records)
    return {
        "status": "valid",
        "source_count": len(sources),
        "generation_count": len(generations),
        "sample_record_count": len(record_by_id),
        "evaluation_case_count": len(cases) if cases is not None else 0,
    }


def validate_files(
    source_manifest_path: Path,
    generation_manifest_path: Path,
    records_path: Path,
    cases_path: Path | None = None,
) -> dict[str, int | str]:
    source_manifest = load_json_document(source_manifest_path)
    generation_manifest = load_json_document(generation_manifest_path)
    records = _read_jsonl(records_path)
    cases = _read_jsonl(cases_path) if cases_path is not None else None
    return validate_bundle(source_manifest, generation_manifest, records, cases)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path, help="sample provenance JSONL")
    parser.add_argument("--cases", type=Path, help="optional evaluation case JSONL to verify record links and hashes")
    parser.add_argument("--sources", type=Path, default=SOURCE_MANIFEST_DEFAULT, help="source manifest JSON")
    parser.add_argument("--generations", type=Path, default=GENERATION_MANIFEST_DEFAULT, help="synthetic generation manifest JSON")
    parser.add_argument("--output", type=Path, help="write validation report to this path; stdout if omitted")
    args = parser.parse_args(argv)
    try:
        report = validate_files(args.sources, args.generations, args.records, args.cases)
    except (OSError, ProvenanceError) as exc:
        parser.error(str(exc))
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
