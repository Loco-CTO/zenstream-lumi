#!/usr/bin/env python3
"""Train a bounded random-init intent/slot component on paired MASSIVE rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import random_init_conversation_response_smoke as engine
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "massive-1.1-intent-slot-pilot"
SOURCE_REVISION = "1.1-f966f21846043aabef9b0f974fa7970027f43738"
REPOSITORY_REVISION = "f966f21846043aabef9b0f974fa7970027f43738"
TRAIN_FAMILY_COUNT = 256
DEV_FAMILY_COUNT = 64
LOCALES = ("en-US", "ja-JP")
ARCHIVE_SHA256 = "sha256:4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"
MEMBER_SHA256 = {
    "1.1/data/en-US.jsonl": "sha256:c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e",
    "1.1/data/ja-JP.jsonl": "sha256:c22df382db6aa4a23dd1e7f62a2ac8f01c6158865771ad25201696be7201ab79",
}
SOURCE_MANIFEST = ROOT / "provenance" / "data_sources.json"
SLOT_PATTERN = re.compile(r"\[([^:\[\]]+)\s*:\s*([^\[\]]*?)\]")
EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?\d[\d().\s-]{6,}\d)(?!\d)")
TRANSFORM_CONFIG = {
    "source_fields": ["locale", "id", "partition", "intent", "utt", "annot_utt"],
    "task_prompt": "Locale: {locale}\nUtterance: {utterance}\nReturn the intent and slots as JSON.\n",
    "slot_pattern": SLOT_PATTERN.pattern,
    "slot_target": "intent string and sorted slot-name to ordered-value-list object",
    "email_pattern": EMAIL_PATTERN.pattern,
    "phone_pattern": PHONE_PATTERN.pattern,
    "redaction_token": "[REDACTED]",
    "maximum_target_bytes": 512,
}
RIGHTS_EVIDENCE_URIS = [
    f"https://github.com/alexa/massive/blob/{REPOSITORY_REVISION}/README.md",
    f"https://github.com/alexa/massive/blob/{REPOSITORY_REVISION}/NOTICE.md",
    f"https://github.com/alexa/massive/blob/{REPOSITORY_REVISION}/1.1/LICENSE",
    "https://aclanthology.org/2023.acl-long.235/",
    "https://creativecommons.org/licenses/by/4.0/",
]


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value) + b"\n")


def _inside_git_worktree(path: Path) -> bool:
    return any((parent / ".git").exists() for parent in (path, *path.parents))


def _read_approved_source() -> tuple[dict[str, Any], str]:
    manifest_bytes = SOURCE_MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    source = next(
        (row for row in manifest.get("sources", []) if row.get("source_id") == SOURCE_ID),
        None,
    )
    if not isinstance(source, dict) or source.get("decision") != "approved_for_use":
        raise ValueError("the scoped MASSIVE source is not approved")
    if source.get("revision") != SOURCE_REVISION or source.get("checksum") != ARCHIVE_SHA256:
        raise ValueError("the MASSIVE source manifest differs from the runner's fixed pin")
    if not {"instruction_training", "evaluation"}.issubset(set(source.get("lumi_uses", []))):
        raise ValueError("the MASSIVE source does not approve both pilot uses")
    return source, _sha256_bytes(manifest_bytes)


def _read_index(bundle: tarfile.TarFile, member_name: str, expected_locale: str) -> dict[str, dict[str, str]]:
    member = bundle.extractfile(member_name)
    if member is None:
        raise ValueError(f"missing archive member: {member_name}")
    digest = hashlib.sha256()
    index: dict[str, dict[str, str]] = {}
    for line_number, raw_line in enumerate(member, start=1):
        digest.update(raw_line)
        try:
            row = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid JSON row in {member_name}:{line_number}") from exc
        if row.get("locale") != expected_locale:
            raise ValueError(f"unexpected locale in {member_name}:{line_number}")
        item_id = row.get("id")
        split = row.get("partition")
        intent = row.get("intent")
        if not all(isinstance(value, str) and value for value in (item_id, split, intent)):
            raise ValueError(f"missing ID, split, or intent in {member_name}:{line_number}")
        if split not in {"train", "dev", "test"} or item_id in index:
            raise ValueError(f"invalid split or duplicate ID in {member_name}:{line_number}")
        index[item_id] = {"partition": split, "intent": intent}
    if "sha256:" + digest.hexdigest() != MEMBER_SHA256[member_name]:
        raise ValueError(f"archive member hash does not match the pinned file: {member_name}")
    if len(index) != 16521:
        raise ValueError(f"unexpected row count in {member_name}: {len(index)}")
    return index


def _select_families(
    english: dict[str, dict[str, str]], japanese: dict[str, dict[str, str]]
) -> tuple[list[str], list[str]]:
    if set(english) != set(japanese):
        raise ValueError("English and Japanese source IDs do not match")
    for item_id in english:
        if english[item_id] != japanese[item_id]:
            raise ValueError(f"aligned locale labels or split differ for source ID {item_id!r}")
    train_ids = [item_id for item_id, row in english.items() if row["partition"] == "train"]
    dev_ids = [item_id for item_id, row in english.items() if row["partition"] == "dev"]
    if len(train_ids) != 11514 or len(dev_ids) != 2033:
        raise ValueError("official train/dev split sizes differ from the pinned audit")

    def ranked(split: str, ids: list[str], count: int) -> list[str]:
        return sorted(
            ids,
            key=lambda item_id: hashlib.sha256(
                f"{SOURCE_ID}\0{split}\0{item_id}".encode("utf-8")
            ).digest(),
        )[:count]

    train = ranked("train", train_ids, TRAIN_FAMILY_COUNT)
    dev = ranked("dev", dev_ids, DEV_FAMILY_COUNT)
    if set(train) & set(dev):
        raise ValueError("selected training and development families overlap")
    return train, dev


def _read_selected_rows(
    bundle: tarfile.TarFile,
    locale: str,
    selected_ids: set[str],
    expected_split: str,
) -> dict[str, dict[str, Any]]:
    member_name = f"1.1/data/{locale}.jsonl"
    member = bundle.extractfile(member_name)
    if member is None:
        raise ValueError(f"missing archive member: {member_name}")
    selected: dict[str, dict[str, Any]] = {}
    for line_number, raw_line in enumerate(member, start=1):
        try:
            row = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid JSON row in {member_name}:{line_number}") from exc
        if row.get("id") not in selected_ids:
            continue
        if row.get("partition") != expected_split or row.get("locale") != locale:
            raise ValueError(f"selected family has the wrong split or locale: {locale}:{row.get('id')}")
        if not isinstance(row.get("utt"), str) or not isinstance(row.get("annot_utt"), str):
            raise ValueError(f"selected item lacks utterance annotations: {locale}:{row.get('id')}")
        selected[row["id"]] = {
            "row": row,
            "source_sha256": _sha256_bytes(raw_line.rstrip(b"\r\n")),
        }
    if set(selected) != selected_ids:
        raise ValueError(f"not all selected rows were found for {locale}")
    return selected


def _slot_values(annotated_utterance: str) -> dict[str, list[str]]:
    matches = SLOT_PATTERN.findall(annotated_utterance)
    if annotated_utterance.count("[") != len(matches) or annotated_utterance.count("]") != len(matches):
        raise ValueError("annotated utterance contains an unparsed slot marker")
    slots: dict[str, list[str]] = {}
    for raw_name, raw_value in matches:
        name = raw_name.strip()
        value = raw_value.strip()
        if not name or not value:
            raise ValueError("annotated slot has an empty name or value")
        slots.setdefault(name, []).append(value)
    return {name: values for name, values in sorted(slots.items())}


def _redact_sensitive(text: str) -> tuple[str, int]:
    count = 0
    for pattern in (EMAIL_PATTERN, PHONE_PATTERN):
        text, hits = pattern.subn("[REDACTED]", text)
        count += hits
    return text, count


def _build_example(
    locale: str, item_id: str, row: dict[str, Any], source_sha256: str, split: str
) -> tuple[dict[str, str], dict[str, Any]]:
    utterance, redactions = _redact_sensitive(row["utt"])
    raw_slots = _slot_values(row["annot_utt"])
    slots: dict[str, list[str]] = {}
    for name, values in raw_slots.items():
        clean_values: list[str] = []
        for value in values:
            clean, hits = _redact_sensitive(value)
            redactions += hits
            clean_values.append(clean)
        slots[name] = clean_values
    prompt = TRANSFORM_CONFIG["task_prompt"].format(locale=locale, utterance=utterance)
    target = _json_bytes({"intent": row["intent"], "slots": slots}).decode("utf-8")
    if len(target.encode("utf-8")) > 512:
        raise ValueError(f"selected target exceeds the registered 512-byte limit: {locale}:{item_id}")
    row_record = {
        "item_id": item_id,
        "locale": locale,
        "family_id": item_id,
        "source_split": split,
        "source_sha256": source_sha256,
        "processed_sample_sha256": _sha256_bytes(_json_bytes({"input": prompt, "target": target})),
        "redacted_sensitive_spans": redactions,
    }
    example = {
        "id": f"{locale}:{item_id}",
        "family_id": item_id,
        "locale": locale,
        "text": prompt,
        "response": target,
    }
    return example, row_record


def _build_records(
    train_rows: list[dict[str, str]],
    dev_rows: list[dict[str, str]],
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    source: dict[str, Any],
) -> list[dict[str, Any]]:
    source_permissions = source["permissions"]
    item_permissions = {
        key: source_permissions[key]
        for key in ("training_use", "evaluation_use", "commercial_use", "modification", "redistribution")
    }
    transform_sha256 = _sha256_bytes(_json_bytes(TRANSFORM_CONFIG))
    attribution = (
        "MASSIVE 1.1 by Amazon Science and its cited authors, with English source text from SLURP; "
        "CC BY 4.0. Locale, ID, utterance, intent, and annotated slot values were selected and "
        "transformed; worker_id and other source metadata were not used."
    )
    records: list[dict[str, Any]] = []
    for rows, provenance_records, split, use, record_split in (
        (train_rows, train_records, "train", "instruction_training", "train"),
        (dev_rows, dev_records, "dev", "evaluation", "development"),
    ):
        for row, record in zip(rows, provenance_records, strict=True):
            locale = row["locale"]
            item_id = row["id"].split(":", 1)[1]
            records.append(
                {
                    "schema_version": 1,
                    "record_id": f"{SOURCE_ID}:{split}:{locale}:{item_id}",
                    "sample_kind": "transformed_sample",
                    "sample_sha256": record["processed_sample_sha256"],
                    "language_tags": [locale],
                    "lumi_uses": [use],
                    "decision": "approved_for_use",
                    "split": record_split,
                    "contamination_status": "checked_clear",
                    "privacy_status": "redacted" if record["redacted_sensitive_spans"] else "cleared",
                    "source_items": [
                        {
                            "source_id": SOURCE_ID,
                            "item_id": f"{locale}:{item_id}",
                            "source_uri": source["canonical_identifier"],
                            "source_revision": SOURCE_REVISION,
                            "source_sha256": record["source_sha256"],
                            "rights_basis": "license",
                            "license_identifier": source["license"]["identifier"],
                            "license_uri": source["license"]["url"],
                            "rights_evidence_uris": RIGHTS_EVIDENCE_URIS,
                            "rights_review_status": "reviewed",
                            "rights_reviewed_date": "2026-10-03",
                            "permissions": item_permissions,
                            "attribution": attribution,
                        }
                    ],
                    "transformations": [
                        {
                            "operation_id": "massive-intent-slot-pilot-normalization",
                            "revision": "1",
                            "config_sha256": transform_sha256,
                        }
                    ],
                    "parent_record_ids": [],
                    "review": {
                        "reviewed_by": "Codex-assisted pinned-archive and item-scope review",
                        "reviewed_date": "2026-10-03",
                        "rationale": (
                            "The exact source row, paired-locale family, official split, license evidence, and "
                            "transformed content hash are recorded. The selected source text and slot values "
                            "were scanned for email and phone patterns. This is a provenance and rights review, "
                            "not a qualified Japanese naturalness or Lumi semantic review. The excluded 101-case "
                            "Codex draft and all derivatives were neither loaded nor compared."
                        ),
                    },
                }
            )
    return records


def _write_and_validate_records(
    output_dir: Path, records: list[dict[str, Any]]
) -> tuple[Path, dict[str, Any]]:
    records_path = output_dir / "document-records.jsonl"
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(_json_bytes(record).decode("utf-8") + "\n")
    result = subprocess.run(
        [sys.executable, str(ROOT / "provenance" / "validate.py"), "--records", str(records_path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    validation = json.loads(result.stdout)
    _write_json(output_dir / "provenance-validation.json", validation)
    return records_path, validation


def _prepare(
    archive_path: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    archive = archive_path.expanduser().resolve()
    if _inside_git_worktree(archive) or not archive.is_file():
        raise ValueError("the pinned MASSIVE archive must exist outside every Git worktree")
    if _hash_file(archive) != ARCHIVE_SHA256:
        raise ValueError("MASSIVE archive hash differs from the fixed 1.1 pin")
    with tarfile.open(archive, mode="r:gz") as bundle:
        english_index = _read_index(bundle, "1.1/data/en-US.jsonl", "en-US")
        japanese_index = _read_index(bundle, "1.1/data/ja-JP.jsonl", "ja-JP")
        train_ids, dev_ids = _select_families(english_index, japanese_index)
        train_set, dev_set = set(train_ids), set(dev_ids)
        loaded: dict[str, dict[str, dict[str, Any]]] = {}
        for locale in LOCALES:
            loaded[locale] = {
                **_read_selected_rows(bundle, locale, train_set, "train"),
                **_read_selected_rows(bundle, locale, dev_set, "dev"),
            }

    train_rows: list[dict[str, str]] = []
    dev_rows: list[dict[str, str]] = []
    train_item_records: list[dict[str, Any]] = []
    dev_item_records: list[dict[str, Any]] = []
    for split, family_ids in (("train", train_ids), ("dev", dev_ids)):
        for item_id in family_ids:
            paired_intents: set[str] = set()
            for locale in LOCALES:
                source_row = loaded[locale][item_id]
                raw = source_row["row"]
                paired_intents.add(raw["intent"])
                example, item_record = _build_example(
                    locale, item_id, raw, source_row["source_sha256"], split
                )
                if split == "train":
                    train_rows.append(example)
                    train_item_records.append(item_record)
                else:
                    dev_rows.append(example)
                    dev_item_records.append(item_record)
            if len(paired_intents) != 1:
                raise ValueError(f"paired locale intent labels differ for source ID {item_id!r}")

    if {row["family_id"] for row in train_rows} & {row["family_id"] for row in dev_rows}:
        raise ValueError("training and development ID families overlap")
    selection = {
        "selection_version": 1,
        "source_id": SOURCE_ID,
        "source_revision": SOURCE_REVISION,
        "archive_sha256": ARCHIVE_SHA256,
        "member_sha256": MEMBER_SHA256,
        "sampling": "lowest SHA-256 rank of source_id, official split, and family ID",
        "selected_train_families": train_ids,
        "selected_development_families": dev_ids,
        "training_items": train_item_records,
        "development_items": dev_item_records,
        "locale_pairing": "same MASSIVE ID in en-US and ja-JP is one family; both locale rows are used",
        "intent_labels_match_within_locale_pairs": True,
        "official_test_split_used": False,
        "selected_email_or_phone_pattern_hits": sum(
            item["redacted_sensitive_spans"]
            for item in [*train_item_records, *dev_item_records]
        ),
        "worker_id_used": False,
        "raw_source_text_committed": False,
    }
    preparation = {
        "selection": selection,
        "train_rows": train_rows,
        "dev_rows": dev_rows,
        "train_records": train_item_records,
        "dev_records": dev_item_records,
        "train_bytes": sum(len(row["text"].encode("utf-8")) + len(row["response"].encode("utf-8")) for row in train_rows),
        "dev_bytes": sum(len(row["text"].encode("utf-8")) + len(row["response"].encode("utf-8")) for row in dev_rows),
        "families": {"train": len(train_ids), "development": len(dev_ids)},
    }
    return train_rows, dev_rows, preparation


def _empty_output_dir(path: Path) -> Path:
    output = path.expanduser().resolve()
    if _inside_git_worktree(output):
        raise ValueError("run artifacts must stay outside every Git worktree")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _slot_set(value: Any) -> set[tuple[str, str]]:
    if not isinstance(value, dict):
        return set()
    pairs: set[tuple[str, str]] = set()
    for name, values in value.items():
        if isinstance(values, list):
            pairs.update((str(name), str(item)) for item in values)
        else:
            pairs.add((str(name), str(values)))
    return pairs


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    language_results: dict[str, list[dict[str, Any]]] = {locale: [] for locale in LOCALES}
    for row in rows:
        language_results[row["locale"]].append(row)

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        true_positive = false_positive = false_negative = 0
        for row in group:
            expected = _slot_set(row["reference"].get("slots"))
            predicted = _slot_set(row["parsed"].get("slots")) if row["valid_output_shape"] else set()
            true_positive += len(expected & predicted)
            false_positive += len(predicted - expected)
            false_negative += len(expected - predicted)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
        f1 = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0
        return {
            "case_count": len(group),
            "valid_utf8_count": sum(bool(row["valid_utf8"]) for row in group),
            "ended_with_eos_count": sum(bool(row["ended_with_eos"]) for row in group),
            "valid_json_count": sum(bool(row["valid_json"]) for row in group),
            "valid_output_shape_count": sum(bool(row["valid_output_shape"]) for row in group),
            "exact_intent_match_count": sum(bool(row["exact_intent_match"]) for row in group),
            "exact_slot_object_match_count": sum(bool(row["exact_slot_object_match"]) for row in group),
            "slot_value_micro_true_positive": true_positive,
            "slot_value_micro_false_positive": false_positive,
            "slot_value_micro_false_negative": false_negative,
            "slot_value_micro_precision": precision,
            "slot_value_micro_recall": recall,
            "slot_value_micro_f1": f1,
        }

    overall = summarize(rows)
    overall["by_locale"] = {locale: summarize(group) for locale, group in language_results.items()}
    return overall


def run(args: argparse.Namespace) -> dict[str, Any]:
    source, source_manifest_sha256 = _read_approved_source()
    train_rows, dev_rows, preparation = _prepare(args.archive)
    output_dir = _empty_output_dir(args.output_dir)
    selection_path = output_dir / "selection-manifest.json"
    _write_json(selection_path, preparation["selection"])
    records_path, provenance_validation = _write_and_validate_records(
        output_dir,
        _build_records(
            train_rows,
            dev_rows,
            preparation["train_records"],
            preparation["dev_records"],
            source,
        ),
    )
    transform_config_sha256 = _sha256_bytes(_json_bytes(TRANSFORM_CONFIG))

    started = time.perf_counter()
    parameters, training = engine.fit(train_rows, args.seed, args.epochs, args.learning_rate)
    training_seconds = time.perf_counter() - started
    model_path = output_dir / "lumi-random-init-massive-intent-slots.npz"
    np.savez_compressed(model_path, **parameters)
    code_hashes = {
        "pilot_script_sha256": _hash_file(Path(__file__).resolve()),
        "shared_engine_sha256": _hash_file(Path(engine.__file__).resolve()),
    }
    runtime_environment = {
        "os": platform.platform(),
        "cpu_architecture": platform.machine(),
        "processor_identifier": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "cpu_thread_limit": 1,
    }
    training_config = {
        "objective": "per-example mean byte-level next-token cross-entropy, then example mean",
        "optimizer": "Adam",
        "optimizer_beta1": 0.9,
        "optimizer_beta2": 0.999,
        "optimizer_epsilon": 1e-8,
        "global_gradient_norm_clip": 1.0,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "shuffle_seed": args.seed + 1,
    }
    generation_config = {
        "decoding": "greedy argmax",
        "max_output_bytes": args.max_output_bytes,
        "tokenizer": "none; UTF-8 byte IDs with reserved BOS/EOS",
    }
    parameter_count = sum(int(value.size) for value in parameters.values())
    metadata = {
        "model_id": "lumi-random-init-massive-intent-slot-smoke-v1",
        "architecture": "single-layer tanh recurrent byte-level encoder-decoder",
        "initialization": "Gaussian random initialization; no external weights or embeddings",
        "parameter_count": parameter_count,
        "seed": args.seed,
        "source_id": SOURCE_ID,
        "source_revision": SOURCE_REVISION,
        "source_manifest_sha256": source_manifest_sha256,
        "document_records_sha256": _hash_file(records_path),
        "transform_config_sha256": transform_config_sha256,
        "selection_manifest_sha256": _hash_file(selection_path),
        "provenance_validation": provenance_validation,
        "code_hashes": code_hashes,
        "training_config": training_config,
        "generation_config": generation_config,
        "runtime_environment": runtime_environment,
    }
    _write_json(model_path.with_suffix(".json"), metadata)

    load_started = time.perf_counter()
    with np.load(model_path, allow_pickle=False) as archive:
        reloaded = {name: archive[name].copy() for name in engine.PARAMETER_NAMES}
    load_ms = (time.perf_counter() - load_started) * 1000.0
    output_rows: list[dict[str, Any]] = []
    warm_times: list[float] = []
    for row in dev_rows:
        inference_started = time.perf_counter()
        generated, ended_with_eos, valid_utf8 = engine.generate(
            reloaded, row["text"], args.max_output_bytes
        )
        warm_times.append((time.perf_counter() - inference_started) * 1000.0)
        try:
            parsed = json.loads(generated)
            valid_json = True
            valid_output_shape = (
                isinstance(parsed, dict)
                and set(parsed) == {"intent", "slots"}
                and isinstance(parsed.get("intent"), str)
                and isinstance(parsed.get("slots"), dict)
            )
        except (json.JSONDecodeError, TypeError):
            parsed = None
            valid_json = False
            valid_output_shape = False
        reference = json.loads(row["response"])
        output_rows.append(
            {
                "id": row["id"],
                "family_id": row["family_id"],
                "locale": row["locale"],
                "text": row["text"],
                "reference": reference,
                "generated_output": generated,
                "parsed": parsed,
                "valid_utf8": valid_utf8,
                "ended_with_eos": ended_with_eos,
                "valid_json": valid_json,
                "valid_output_shape": valid_output_shape,
                "exact_intent_match": bool(
                    valid_output_shape and parsed.get("intent") == reference.get("intent")
                ),
                "exact_slot_object_match": bool(
                    valid_output_shape and parsed.get("slots") == reference.get("slots")
                ),
            }
        )

    predictions_path = output_dir / "development-predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    ordered_times = sorted(warm_times)
    p95_index = max(0, math.ceil(0.95 * len(ordered_times)) - 1)
    report = {
        "evidence_level": "exploratory",
        "component_scope": "structured intent and slot extraction only; no natural-language response output",
        "model_id": metadata["model_id"],
        "random_initialization": True,
        "pretrained_artifacts_used": False,
        "source_id": SOURCE_ID,
        "source_revision": SOURCE_REVISION,
        "archive_sha256": ARCHIVE_SHA256,
        "member_sha256": MEMBER_SHA256,
        "source_manifest_sha256": source_manifest_sha256,
        "document_records_sha256": metadata["document_records_sha256"],
        "transform_config_sha256": transform_config_sha256,
        "selection_manifest_sha256": metadata["selection_manifest_sha256"],
        "provenance_validation": provenance_validation,
        "code_hashes": code_hashes,
        "selected_training_families": preparation["families"]["train"],
        "selected_development_families": preparation["families"]["development"],
        "selected_training_rows": len(train_rows),
        "selected_development_rows": len(dev_rows),
        "training_and_development_family_overlap": False,
        "training_utf8_bytes": preparation["train_bytes"],
        "development_utf8_bytes": preparation["dev_bytes"],
        "redacted_sensitive_spans": preparation["selection"]["selected_email_or_phone_pattern_hits"],
        "seed": args.seed,
        "parameter_count": parameter_count,
        "parameter_state_sha256": engine._parameter_state_sha256(parameters),
        "training_config": training_config,
        "training": {key: value for key, value in training.items() if key != "loss_by_epoch"},
        "training_seconds": training_seconds,
        "runtime_environment": runtime_environment,
        "development_structured_metrics": _metrics(output_rows),
        "cpu_inference": {
            "numpy_cpu_only": True,
            "gpu_used": False,
            "load_ms": load_ms,
            "warm_latency_p50_ms": statistics.median(warm_times),
            "warm_latency_p95_ms": ordered_times[p95_index],
            "warm_latency_samples": len(warm_times),
        },
        "memory": {
            "peak_working_set_bytes": engine._working_set_bytes(peak=True),
            "scope_note": "Windows process working set includes Python and NumPy runtime; not a standalone model measurement.",
        },
        "model_artifact": {
            "path": str(model_path),
            "bytes": model_path.stat().st_size,
            "sha256": _hash_file(model_path),
        },
        "development_predictions": {
            "path": str(predictions_path),
            "sha256": _hash_file(predictions_path),
        },
        "limitations": [
            "MASSIVE is professionally localized English virtual-assistant data, not spontaneous Japanese, natural EN/JA code-switching, or ZenStream media dialogue.",
            "The experiment measures only structured intent and slot output; it does not generate or evaluate user-facing conversation.",
            "Its public dev split is a small exploratory diagnostic, not a Lumi-qualified development set or final holdout.",
            "No false-action, clarification, unsupported-request, trusted-state grounding, or qualified Japanese naturalness claim is supported.",
            "No tokenizer was learned, no public test item was used, and the excluded 101-case Codex draft and every derivative were not loaded or compared.",
            "Weights remain local pending a separate review of the exact artifact, lineage, attribution, and public distribution implications.",
        ],
    }
    _write_json(output_dir / "training-curve.json", {"seed": args.seed, "loss_by_epoch": training["loss_by_epoch"]})
    _write_json(output_dir / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="pinned MASSIVE archive outside Git")
    parser.add_argument("--output-dir", required=True, type=Path, help="empty controlled output directory outside Git")
    parser.add_argument("--seed", default=314159, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--learning-rate", default=0.01, type=float)
    parser.add_argument("--max-output-bytes", default=512, type=int)
    args = parser.parse_args(argv)
    if args.max_output_bytes < 512:
        parser.error("max-output-bytes must be at least the registered 512-byte target cap")
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
