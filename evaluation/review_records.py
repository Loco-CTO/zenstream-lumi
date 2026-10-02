"""Validate independent semantic, response-contract, and language-review records."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from provenance.validate import canonical_sha256, evaluation_case_sha256


SHA256_PATTERN = re.compile(r"^sha256:[A-Fa-f0-9]{64}$")
TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
RECORD_TYPES = {"independent_annotation", "language_review", "adjudication"}
GOLD_KEYS = {"decision", "action", "arguments", "requires_clarification"}
GOLD_OPTIONAL_KEYS = {"response_contract", "presentation_intent"}
PRESENTATION_INTENTS = {
    "none", "text", "media_results", "media_details", "playback_handoff", "confirmation",
}
RESPONSE_REQUIREMENTS = {"required", "optional", "forbidden"}
RESPONSE_LANGUAGES = {"same_as_case", "en", "ja", "en_ja", "any"}


class ReviewRecordError(ValueError):
    """Raised when the controlled annotation ledger cannot verify ready cases."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r} is not allowed")
        value[key] = item
    return value


def read_review_records(path: Path) -> list[dict[str, Any]]:
    """Read a strict UTF-8 JSONL ledger without accepting non-standard numbers."""
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(
                        line,
                        parse_constant=_reject_json_constant,
                        object_pairs_hook=_reject_duplicate_keys,
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ReviewRecordError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(record, dict):
                    raise ReviewRecordError(f"{path}:{line_number}: each review record must be an object")
                records.append(record)
    except UnicodeDecodeError as exc:
        raise ReviewRecordError(f"{path}: review ledger is not UTF-8: {exc}") from exc
    return records


def review_records_sha256(records: list[dict[str, Any]]) -> str:
    """Return a stable digest for callers scoring in-memory records."""
    return canonical_sha256(records)


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_semantic_output(value: Any) -> bool:
    if (
        not isinstance(value, dict)
        or not GOLD_KEYS.issubset(value)
        or set(value) - GOLD_KEYS - GOLD_OPTIONAL_KEYS
    ):
        return False
    decision = value["decision"]
    action = value["action"]
    arguments = value["arguments"]
    clarification = value["requires_clarification"]
    if not isinstance(decision, str) or decision not in {"act", "no_action", "clarify", "respond"}:
        return False
    if not isinstance(arguments, dict) or not isinstance(clarification, bool):
        return False
    if decision == "act":
        valid_core = _nonempty(action) and not clarification
    elif action is not None or arguments:
        return False
    else:
        valid_core = clarification is (decision == "clarify")
    if not valid_core:
        return False
    if "response_contract" in value:
        contract = value["response_contract"]
        if (
            not isinstance(contract, dict)
            or set(contract) != {"requirement", "language"}
            or not isinstance(contract["requirement"], str)
            or contract["requirement"] not in RESPONSE_REQUIREMENTS
            or not isinstance(contract["language"], str)
            or contract["language"] not in RESPONSE_LANGUAGES
        ):
            return False
    if "presentation_intent" in value and (
        not isinstance(value["presentation_intent"], str)
        or value["presentation_intent"] not in PRESENTATION_INTENTS
    ):
        return False
    return True


def _same_json_value(left: Any, right: Any) -> bool:
    return canonical_sha256(left) == canonical_sha256(right)


def _parse_reviewed_at(value: Any, prefix: str) -> datetime:
    if not _nonempty(value) or not TIMESTAMP_PATTERN.fullmatch(value):
        raise ReviewRecordError(f"{prefix}: reviewed_at must be a timezone-aware ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as exc:
        raise ReviewRecordError(
            f"{prefix}: reviewed_at must be a timezone-aware ISO 8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ReviewRecordError(f"{prefix}: reviewed_at must include a timezone")
    return parsed


def _validate_record(record: dict[str, Any], index: int) -> None:
    prefix = f"review record {index}"
    record_type = record.get("record_type")
    type_fields = {
        "independent_annotation": {"proposed_gold"},
        "language_review": {
            "language", "qualification", "naturalness_status", "meaning_preservation_status",
        },
        "adjudication": {"basis_record_ids", "adjudicated_gold", "rationale"},
    }
    if not isinstance(record_type, str) or record_type not in RECORD_TYPES:
        raise ReviewRecordError(f"{prefix}: invalid record_type")
    common = {
        "schema_version", "review_record_id", "case_id", "case_sha256", "reviewer_id",
        "reviewed_at", "record_type", "supersedes_record_ids",
    }
    if set(record) != common | type_fields[record_type]:
        raise ReviewRecordError(f"{prefix}: fields do not match record_type {record_type!r}")
    if (
        not isinstance(record["schema_version"], int)
        or isinstance(record["schema_version"], bool)
        or record["schema_version"] != 1
    ):
        raise ReviewRecordError(f"{prefix}: schema_version must be 1")
    for field in ("review_record_id", "case_id", "reviewer_id"):
        if not _nonempty(record[field]):
            raise ReviewRecordError(f"{prefix}: {field} must be a nonempty string")
    if not isinstance(record["case_sha256"], str) or not SHA256_PATTERN.fullmatch(record["case_sha256"]):
        raise ReviewRecordError(f"{prefix}: case_sha256 must be a SHA-256 digest")
    _parse_reviewed_at(record["reviewed_at"], prefix)
    supersedes = record["supersedes_record_ids"]
    if (
        not isinstance(supersedes, list)
        or not all(_nonempty(item) for item in supersedes)
        or len(set(supersedes)) != len(supersedes)
    ):
        raise ReviewRecordError(f"{prefix}: supersedes_record_ids must be a unique string list")

    if record_type == "independent_annotation":
        if not _valid_semantic_output(record["proposed_gold"]):
            raise ReviewRecordError(f"{prefix}: proposed_gold is not a valid semantic output")
    elif record_type == "language_review":
        if not isinstance(record["language"], str) or record["language"] not in {"ja", "en_ja"}:
            raise ReviewRecordError(f"{prefix}: language review applies only to ja or en_ja")
        if not isinstance(record["qualification"], str) or record["qualification"] not in {
            "native_japanese", "fluent_japanese", "fluent_bilingual",
        }:
            raise ReviewRecordError(f"{prefix}: invalid Japanese-language qualification")
        for field in ("naturalness_status", "meaning_preservation_status"):
            if not isinstance(record[field], str) or record[field] not in {"approved", "needs_revision"}:
                raise ReviewRecordError(f"{prefix}: {field} must be approved or needs_revision")
    else:
        basis = record["basis_record_ids"]
        if (
            not isinstance(basis, list)
            or len(basis) < 2
            or not all(_nonempty(item) for item in basis)
            or len(set(basis)) != len(basis)
        ):
            raise ReviewRecordError(f"{prefix}: basis_record_ids must contain unique annotation record IDs")
        if not _valid_semantic_output(record["adjudicated_gold"]):
            raise ReviewRecordError(f"{prefix}: adjudicated_gold is not a valid semantic output")
        if not _nonempty(record["rationale"]):
            raise ReviewRecordError(f"{prefix}: adjudication rationale must be nonempty")


def validate_case_review_records(
    cases: list[dict[str, Any]],
    review_records: list[dict[str, Any]],
) -> None:
    """Verify current review references and every ready case's independent review gate.

    A ledger passed here must cover only the case inventory being checked. Historical
    records for those same cases may remain in the ledger with older content hashes.
    This keeps development and final-holdout review files separate.
    """
    cases_by_id: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = case.get("case_id") if isinstance(case, dict) else None
        if not _nonempty(case_id):
            raise ReviewRecordError("case inventory contains a case without a case_id")
        if case_id in cases_by_id:
            raise ReviewRecordError(f"duplicate case_id {case_id!r} in review validation")
        cases_by_id[case_id] = case

    records_by_id: dict[str, dict[str, Any]] = {}
    records_by_case: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(review_records, start=1):
        if not isinstance(record, dict):
            raise ReviewRecordError(f"review record {index} must be an object")
        _validate_record(record, index)
        record_id = record["review_record_id"]
        if record_id in records_by_id:
            raise ReviewRecordError(f"duplicate review_record_id {record_id!r}")
        if record["case_id"] not in cases_by_id:
            raise ReviewRecordError(
                f"review record {record_id!r} is outside the supplied single-split case inventory"
            )
        records_by_id[record_id] = record
        records_by_case.setdefault(record["case_id"], []).append(record)

    superseded_ids: set[str] = set()
    superseded_by: dict[str, str] = {}
    for record in records_by_id.values():
        for superseded_id in record["supersedes_record_ids"]:
            previous = records_by_id.get(superseded_id)
            if previous is None:
                raise ReviewRecordError(
                    f"review record {record['review_record_id']!r} supersedes an unknown record {superseded_id!r}"
                )
            if any(
                previous[field] != record[field]
                for field in ("case_id", "case_sha256", "reviewer_id", "record_type")
            ):
                raise ReviewRecordError(
                    f"review record {record['review_record_id']!r} can supersede only the same reviewer's same-version record"
                )
            if _parse_reviewed_at(record["reviewed_at"], "review record") <= _parse_reviewed_at(
                previous["reviewed_at"], "superseded review record"
            ):
                raise ReviewRecordError("a superseding review record must have a later reviewed_at timestamp")
            if superseded_id in superseded_by:
                raise ReviewRecordError(f"review record {superseded_id!r} has multiple superseding records")
            superseded_by[superseded_id] = record["review_record_id"]
            superseded_ids.add(superseded_id)

    for case_id, case in cases_by_id.items():
        review = case.get("review")
        active_ids = review.get("review_record_ids", []) if isinstance(review, dict) else []
        if (
            not isinstance(active_ids, list)
            or not all(_nonempty(item) for item in active_ids)
            or len(set(active_ids)) != len(active_ids)
        ):
            raise ReviewRecordError(f"case {case_id!r}: review_record_ids must be a unique string list")
        current_sha256 = evaluation_case_sha256(case)
        active: list[dict[str, Any]] = []
        for record_id in active_ids:
            record = records_by_id.get(record_id)
            if record is None:
                raise ReviewRecordError(f"case {case_id!r}: unknown review_record_id {record_id!r}")
            if record["case_id"] != case_id:
                raise ReviewRecordError(f"case {case_id!r}: review record {record_id!r} belongs to another case")
            if record["case_sha256"] != current_sha256:
                raise ReviewRecordError(f"case {case_id!r}: active review record {record_id!r} has a stale case hash")
            active.append(record)

        current_record_ids = {
            record["review_record_id"]
            for record in records_by_case.get(case_id, [])
            if record["case_sha256"] == current_sha256
            and record["review_record_id"] not in superseded_ids
        }
        if set(active_ids) != current_record_ids:
            raise ReviewRecordError(
                f"case {case_id!r}: review_record_ids must include every current unsuperseded ledger record"
            )

        if case.get("review_status") != "ready":
            continue

        annotations = [record for record in active if record["record_type"] == "independent_annotation"]
        if len(annotations) < 2:
            raise ReviewRecordError(f"ready case {case_id!r} needs at least two independent annotations")
        annotation_reviewers = [record["reviewer_id"] for record in annotations]
        if len(set(annotation_reviewers)) < 2 or len(set(annotation_reviewers)) != len(annotation_reviewers):
            raise ReviewRecordError(
                f"ready case {case_id!r} needs one independent annotation from each distinct reviewer"
            )

        agreed_with_gold = all(
            _same_json_value(record["proposed_gold"], case["gold"])
            for record in annotations
        )
        adjudications = [record for record in active if record["record_type"] == "adjudication"]
        if agreed_with_gold:
            if adjudications:
                raise ReviewRecordError(f"case {case_id!r}: unnecessary adjudication record")
        else:
            if len(adjudications) != 1:
                raise ReviewRecordError(
                    f"case {case_id!r}: disagreement or changed gold requires one adjudication record"
                )
            adjudication = adjudications[0]
            annotation_ids = {record["review_record_id"] for record in annotations}
            if set(adjudication["basis_record_ids"]) != annotation_ids:
                raise ReviewRecordError(
                    f"case {case_id!r}: adjudication must reference every active independent annotation"
                )
            adjudication_time = _parse_reviewed_at(adjudication["reviewed_at"], "adjudication record")
            if any(
                adjudication_time <= _parse_reviewed_at(record["reviewed_at"], "annotation record")
                for record in annotations
            ):
                raise ReviewRecordError(
                    f"case {case_id!r}: adjudication must follow every independent annotation"
                )
            if not _same_json_value(adjudication["adjudicated_gold"], case["gold"]):
                raise ReviewRecordError(f"case {case_id!r}: adjudicated_gold does not match case gold")
            if adjudication["reviewer_id"] in set(annotation_reviewers):
                raise ReviewRecordError(f"case {case_id!r}: adjudicator must be independent of annotators")

        language_reviews = [record for record in active if record["record_type"] == "language_review"]
        language = case["language"]
        language_reviewers: set[str] = set()
        if language in {"ja", "en_ja"}:
            if len(language_reviews) != 1:
                raise ReviewRecordError(f"ready {language} case {case_id!r} needs one language review record")
            language_review = language_reviews[0]
            if language_review["language"] != language:
                raise ReviewRecordError(f"case {case_id!r}: language review does not match case language")
            if (
                language_review["naturalness_status"] != "approved"
                or language_review["meaning_preservation_status"] != "approved"
            ):
                raise ReviewRecordError(
                    f"ready {language} case {case_id!r} needs approved naturalness and meaning-preservation review"
                )
            valid_qualifications = (
                {"native_japanese", "fluent_japanese"}
                if language == "ja"
                else {"fluent_bilingual"}
            )
            if language_review["qualification"] not in valid_qualifications:
                raise ReviewRecordError(f"case {case_id!r}: language reviewer qualification is insufficient")
            language_reviewers.add(language_review["reviewer_id"])
            if language_reviewers & (set(annotation_reviewers) | {
                record["reviewer_id"] for record in adjudications
            }):
                raise ReviewRecordError(f"case {case_id!r}: language reviewer must be independent")
        elif language_reviews:
            raise ReviewRecordError(f"English case {case_id!r} must not reference a Japanese-language review")

        expected_reviewers = {record["reviewer_id"] for record in active}
        declared_reviewers = set(case["review"].get("reviewer_ids", []))
        if declared_reviewers != expected_reviewers:
            raise ReviewRecordError(
                f"case {case_id!r}: reviewer_ids must match the active review records"
            )
