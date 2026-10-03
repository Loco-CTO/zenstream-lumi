#!/usr/bin/env python3
"""Run a bounded random-init Japanese state-and-response pilot on JMultiWOZ."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import random_init_conversation_response_smoke as engine
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "jmultiwoz-v1-training-pilot"
TRAIN_DIALOGUE_COUNT = 64
DEV_DIALOGUE_COUNT = 24
UPSTREAM_REVISION = "a78dd334b907e12318b76bb23d0a9ed1498b8d15"
ARCHIVE_SHA256 = "sha256:283dea36f5abeea8f016fea2bd4df4dfba67a40f89c4600e47ed701c57ea2bc2"
MEMBER_SHA256 = {
    "JMultiWOZ_1.0/dialogues.json": "sha256:0f716c603b954f716e26865b7bf096546cb3a719202da55b09e31265a97c392c",
    "JMultiWOZ_1.0/split_list.json": "sha256:9b7c5d326d1669322000750cba12ba2eac30b8bbf32f7696701ebb0440eb1bd6",
}
SOURCE_MANIFEST = ROOT / "provenance" / "data_sources.json"
PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?81[-\s]?)?0\d{1,4}(?:[-\s]?\d{1,4}){1,2}(?!\d)"
)
EXCLUDED_SLOTS = {"phone", "address", "reference"}
TRANSFORM_CONFIG = {
    "input_context_utterances": 4,
    "input_speakers": ["USER", "SYSTEM"],
    "last_annotated_system_turn": True,
    "phone_pattern": PHONE_PATTERN.pattern,
    "phone_replacement": "[PHONE]",
    "state_fields": ["general.city", "active_domain.non_null_slots"],
    "excluded_slots": sorted(EXCLUDED_SLOTS),
    "output_format": "sorted compact UTF-8 JSON with arguments, domain, response",
    "maximum_target_bytes": 512,
}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _inside_git_worktree(path: Path) -> bool:
    return any((parent / ".git").exists() for parent in (path, *path.parents))


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value) + b"\n")


def _read_approved_source() -> tuple[dict[str, Any], str]:
    manifest_bytes = SOURCE_MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    source = next(
        (item for item in manifest.get("sources", []) if item.get("source_id") == SOURCE_ID),
        None,
    )
    if not isinstance(source, dict):
        raise ValueError(f"{SOURCE_ID!r} is missing from the tracked source manifest")
    if source.get("decision") != "approved_for_use":
        raise ValueError("JMultiWOZ is not approved for the declared pilot uses")
    if not {"instruction_training", "evaluation"}.issubset(set(source.get("lumi_uses", []))):
        raise ValueError("JMultiWOZ source manifest does not approve both pilot uses")
    if source.get("revision") != UPSTREAM_REVISION or source.get("checksum") != ARCHIVE_SHA256:
        raise ValueError("tracked source pin does not match the pilot's fixed revision and archive")
    return source, _sha256_bytes(manifest_bytes)


def _select_ids(ids: list[str], split: str, count: int) -> list[str]:
    ranked = sorted(
        ids,
        key=lambda item_id: hashlib.sha256(
            f"{SOURCE_ID}\0{split}\0{item_id}".encode("utf-8")
        ).digest(),
    )
    return ranked[:count]


def _hash_zip_member(bundle: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with bundle.open(name) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _iter_top_level_json_items(stream: io.TextIOBase):
    """Parse one top-level JSON object value at a time from the large dialogue member."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0

    def fill() -> bool:
        nonlocal buffer
        chunk = stream.read(1024 * 1024)
        if not chunk:
            return False
        buffer += chunk
        return True

    while True:
        while position < len(buffer) and buffer[position].isspace():
            position += 1
        if position < len(buffer):
            break
        buffer = ""
        position = 0
        if not fill():
            raise ValueError("dialogues member is empty")
    if position >= len(buffer) or buffer[position] != "{":
        raise ValueError("dialogues member is not a top-level JSON object")
    position += 1

    while True:
        while True:
            if position >= len(buffer):
                buffer = ""
                position = 0
                if not fill():
                    raise ValueError("dialogues member ended before its object close")
            if buffer[position].isspace() or buffer[position] == ",":
                position += 1
                continue
            break
        if buffer[position] == "}":
            return

        while True:
            try:
                key, key_end = decoder.raw_decode(buffer, position)
                break
            except json.JSONDecodeError:
                if not fill():
                    raise ValueError("dialogues member contains an incomplete key")
        if not isinstance(key, str):
            raise ValueError("dialogues member contains a non-string key")
        position = key_end
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                break
            buffer = ""
            position = 0
            if not fill():
                raise ValueError("dialogues member ended before a key separator")
        if buffer[position] != ":":
            raise ValueError("dialogues member contains a key without a value separator")
        position += 1
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                break
            buffer = ""
            position = 0
            if not fill():
                raise ValueError("dialogues member ended before a value")
        while True:
            try:
                value, value_end = decoder.raw_decode(buffer, position)
                break
            except json.JSONDecodeError:
                if not fill():
                    raise ValueError("dialogues member contains an incomplete value")
        buffer = buffer[value_end:]
        position = 0
        yield key, value


def _compact_state(state: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    belief = state.get("belief_state")
    if not isinstance(belief, dict):
        return None, {}
    general = belief.get("general")
    domain = general.get("active_domain") if isinstance(general, dict) else None
    if not isinstance(domain, str) or not domain:
        return None, {}

    arguments: dict[str, Any] = {}
    city = general.get("city") if isinstance(general, dict) else None
    if city is not None:
        arguments["city"] = city
    domain_state = belief.get(domain)
    if isinstance(domain_state, dict):
        for slot, value in domain_state.items():
            if value is not None and slot not in EXCLUDED_SLOTS:
                arguments[slot] = value
    return domain, arguments


def _redact_phone_like_text(text: str) -> tuple[str, int]:
    count = 0

    def replace(_: Any) -> str:
        nonlocal count
        count += 1
        return "[PHONE]"

    return PHONE_PATTERN.sub(replace, text), count


def _build_examples(
    dialogues: dict[str, Any], ids: list[str], split: str
) -> tuple[list[dict[str, str]], list[dict[str, Any]], int]:
    rows: list[dict[str, str]] = []
    item_records: list[dict[str, Any]] = []
    redaction_count = 0
    for dialogue_id in ids:
        dialogue_redactions = 0
        dialogue = dialogues.get(dialogue_id)
        if not isinstance(dialogue, dict):
            raise ValueError(f"split {split!r} points to missing dialogue {dialogue_id!r}")
        turns = dialogue.get("turns")
        if not isinstance(turns, list):
            raise ValueError(f"dialogue {dialogue_id!r} has no turn list")
        system_turns = [
            (index, turn)
            for index, turn in enumerate(turns)
            if isinstance(turn, dict)
            and turn.get("speaker") == "SYSTEM"
            and isinstance(turn.get("dialogue_state"), dict)
        ]
        if not system_turns:
            raise ValueError(f"dialogue {dialogue_id!r} has no annotated system turn")
        target_index, target_turn = system_turns[-1]
        domain, arguments = _compact_state(target_turn["dialogue_state"])
        if domain is None:
            raise ValueError(f"dialogue {dialogue_id!r} has no active belief-state domain")

        preceding_utterances: list[dict[str, Any]] = []
        for turn in reversed(turns[:target_index]):
            if (
                not isinstance(turn, dict)
                or turn.get("speaker") not in {"USER", "SYSTEM"}
                or not isinstance(turn.get("utterance"), str)
            ):
                continue
            preceding_utterances.append(turn)
            if len(preceding_utterances) == 4:
                break
        history: list[str] = []
        for turn in reversed(preceding_utterances):
            utterance, redactions = _redact_phone_like_text(turn["utterance"])
            dialogue_redactions += redactions
            speaker = turn["speaker"]
            history.append(f"{speaker}: {utterance}")
        if not history:
            raise ValueError(f"dialogue {dialogue_id!r} has no usable preceding context")
        prompt = "Context:\n" + "\n".join(history) + "\nTask: state and reply\n"

        response, redactions = _redact_phone_like_text(target_turn.get("utterance", ""))
        dialogue_redactions += redactions
        redaction_count += dialogue_redactions
        target = _json_bytes(
            {"arguments": arguments, "domain": domain, "response": response}
        ).decode("utf-8")
        if len(target.encode("utf-8")) > 512:
            raise ValueError(
                f"selected target {dialogue_id!r} exceeds the 512-byte registered output cap"
            )
        family_id = dialogue.get("dialogue_name", dialogue_id)
        processed_sample_sha256 = _sha256_bytes(
            _json_bytes({"input": prompt, "target": target})
        )
        rows.append(
            {
                "id": f"{dialogue_id}:last-system-turn",
                "family_id": family_id,
                "text": prompt,
                "response": target,
            }
        )
        item_records.append(
            {
                "dialogue_id": dialogue_id,
                "family_id": family_id,
                "source_split": split,
                "canonical_dialogue_sha256": _sha256_bytes(_json_bytes(dialogue)),
                "processed_sample_sha256": processed_sample_sha256,
                "redacted_phone_like_spans": dialogue_redactions,
            }
        )
    return rows, item_records, redaction_count


def _load_data(
    archive_path: Path, train_count: int, dev_count: int
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    if train_count != TRAIN_DIALOGUE_COUNT or dev_count != DEV_DIALOGUE_COUNT:
        raise ValueError("the admitted source scope is locked to 64 train and 24 dev dialogues")
    archive = archive_path.expanduser().resolve()
    if _inside_git_worktree(archive):
        raise ValueError("the source archive must remain outside every Git worktree")
    if not archive.is_file():
        raise ValueError(f"source archive does not exist: {archive}")
    archive_sha256 = engine._sha256(archive)
    if archive_sha256 != ARCHIVE_SHA256:
        raise ValueError("JMultiWOZ archive hash does not match the pinned v1.0 release")

    with zipfile.ZipFile(archive) as bundle:
        split_name = "JMultiWOZ_1.0/split_list.json"
        split_bytes = bundle.read(split_name)
        if _sha256_bytes(split_bytes) != MEMBER_SHA256[split_name]:
            raise ValueError(f"pinned source member hash mismatch: {split_name}")
        dialogue_name = "JMultiWOZ_1.0/dialogues.json"
        if _hash_zip_member(bundle, dialogue_name) != MEMBER_SHA256[dialogue_name]:
            raise ValueError(f"pinned source member hash mismatch: {dialogue_name}")
        splits = json.loads(split_bytes)
        if {name: len(ids) for name, ids in splits.items()} != {
        "train": 3646,
        "dev": 300,
        "test": 300,
        }:
            raise ValueError("source official split counts changed")

        if train_count < 1 or train_count > len(splits["train"]):
            raise ValueError("train-dialogues is outside the pinned training split")
        if dev_count < 1 or dev_count > len(splits["dev"]):
            raise ValueError("dev-dialogues is outside the pinned development split")
        train_ids = _select_ids(splits["train"], "train", train_count)
        dev_ids = _select_ids(splits["dev"], "dev", dev_count)
        if set(train_ids) & set(dev_ids):
            raise ValueError("training and development dialogue IDs overlap")
        selected_ids = set(train_ids) | set(dev_ids)
        selected_dialogues: dict[str, Any] = {}
        total_dialogues = 0
        with bundle.open(dialogue_name) as raw_handle:
            with io.TextIOWrapper(raw_handle, encoding="utf-8") as text_handle:
                for dialogue_id, dialogue in _iter_top_level_json_items(text_handle):
                    total_dialogues += 1
                    if dialogue_id in selected_ids:
                        selected_dialogues[dialogue_id] = dialogue
        if total_dialogues != 4246 or set(selected_dialogues) != selected_ids:
            raise ValueError("pinned dialogue count or selected IDs changed")

    train_rows, train_records, train_redactions = _build_examples(
        selected_dialogues, train_ids, "train"
    )
    dev_rows, dev_records, dev_redactions = _build_examples(
        selected_dialogues, dev_ids, "dev"
    )
    train_hashes = {item["canonical_dialogue_sha256"] for item in train_records}
    dev_hashes = {item["canonical_dialogue_sha256"] for item in dev_records}
    exact_cross_split_duplicates = train_hashes & dev_hashes
    if exact_cross_split_duplicates:
        raise ValueError("selected training and development dialogues contain exact duplicates")
    selection = {
        "selection_version": 1,
        "source_id": SOURCE_ID,
        "upstream_revision": UPSTREAM_REVISION,
        "archive_sha256": archive_sha256,
        "dialogue_member_sha256": MEMBER_SHA256["JMultiWOZ_1.0/dialogues.json"],
        "split_member_sha256": MEMBER_SHA256["JMultiWOZ_1.0/split_list.json"],
        "sampling": "lowest SHA-256 rank of source_id, official split, and dialogue ID",
        "input_transform": "last four preceding utterances; phone-like sequences replaced with [PHONE]",
        "target_transform": (
            "compact JSON of active domain, active-domain arguments plus general city, and the final system response; "
            "phone, address, and reference slots omitted; phone-like sequences replaced with [PHONE]"
        ),
        "training_items": train_records,
        "development_items": dev_records,
        "official_test_split_used_for_training_or_development": False,
        "exact_train_dev_dialogue_duplicates": 0,
        "raw_dialogue_text_committed": False,
    }
    preparation = {
        "archive_sha256": archive_sha256,
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "train_input_target_utf8_bytes": sum(
            len(row["text"].encode("utf-8")) + len(row["response"].encode("utf-8"))
            for row in train_rows
        ),
        "dev_input_target_utf8_bytes": sum(
            len(row["text"].encode("utf-8")) + len(row["response"].encode("utf-8"))
            for row in dev_rows
        ),
        "phone_like_spans_redacted": train_redactions + dev_redactions,
        "training_families": sorted({row["family_id"] for row in train_rows}),
        "development_families": sorted({row["family_id"] for row in dev_rows}),
        "selection": selection,
    }
    return train_rows, dev_rows, preparation


def _document_records(
    train_rows: list[dict[str, str]],
    dev_rows: list[dict[str, str]],
    preparation: dict[str, Any],
) -> list[dict[str, Any]]:
    manifest = json.loads(SOURCE_MANIFEST.read_bytes())
    source = next(
        item for item in manifest["sources"] if item.get("source_id") == SOURCE_ID
    )
    rows_by_id = {
        row["id"].split(":", 1)[0]: row for row in [*train_rows, *dev_rows]
    }
    transform_hash = _sha256_bytes(_json_bytes(TRANSFORM_CONFIG))
    evidence_uris = [
        f"https://github.com/nu-dialogue/jmultiwoz/blob/{UPSTREAM_REVISION}/LICENSE",
        f"https://github.com/nu-dialogue/jmultiwoz/blob/{UPSTREAM_REVISION}/README.md",
        f"https://github.com/nu-dialogue/jmultiwoz/blob/{UPSTREAM_REVISION}/dataset/README.md",
        "https://aclanthology.org/2024.lrec-main.835/",
    ]
    attribution = (
        "JMultiWOZ 1.0 by Ohashi et al.; licensed CC BY-SA 4.0. "
        "This item was filtered to the final annotated SYSTEM turn, transformed to "
        "a four-utterance context and compact state/response JSON, and had phone-like "
        "spans redacted and phone, address, and reference slots omitted."
    )
    records: list[dict[str, Any]] = []
    for split, items, lumi_use, record_split in (
        ("train", preparation["selection"]["training_items"], "instruction_training", "train"),
        ("dev", preparation["selection"]["development_items"], "evaluation", "development"),
    ):
        for item in items:
            dialogue_id = item["dialogue_id"]
            row = rows_by_id[dialogue_id]
            records.append(
                {
                    "schema_version": 1,
                    "record_id": f"{SOURCE_ID}:{split}:{dialogue_id}",
                    "sample_kind": "transformed_sample",
                    "sample_sha256": item["processed_sample_sha256"],
                    "language_tags": ["ja"],
                    "lumi_uses": [lumi_use],
                    "decision": "approved_for_use",
                    "split": record_split,
                    "contamination_status": "checked_clear",
                    "privacy_status": (
                        "redacted" if item["redacted_phone_like_spans"] else "cleared"
                    ),
                    "source_items": [
                        {
                            "source_id": SOURCE_ID,
                            "item_id": dialogue_id,
                            "source_uri": source["canonical_identifier"],
                            "source_revision": UPSTREAM_REVISION,
                            "source_sha256": item["canonical_dialogue_sha256"],
                            "rights_basis": "license",
                            "license_identifier": source["license"]["identifier"],
                            "license_uri": source["license"]["url"],
                            "rights_evidence_uris": evidence_uris,
                            "rights_review_status": "reviewed",
                            "rights_reviewed_date": "2026-10-03",
                            "permissions": {
                                key: source["permissions"][key]
                                for key in (
                                    "training_use",
                                    "evaluation_use",
                                    "commercial_use",
                                    "modification",
                                    "redistribution",
                                )
                            },
                            "attribution": attribution,
                        }
                    ],
                    "transformations": [
                        {
                            "operation_id": "jmultiwoz-pilot-normalization",
                            "revision": "1",
                            "config_sha256": transform_hash,
                        }
                    ],
                    "parent_record_ids": [],
                    "review": {
                        "reviewed_by": "Codex-assisted pinned-source rights audit",
                        "reviewed_date": "2026-10-03",
                        "rationale": (
                            "Pinned source rights and intended train/evaluation use were reviewed; "
                            "the item ID, canonical dialogue hash, split membership, and deterministic "
                            "transformation are recorded. This is a provenance and rights review, not "
                            "a qualified Japanese response-quality review. The excluded 101-case Codex "
                            "draft and all derivatives were not loaded or compared."
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
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "provenance" / "validate.py"),
            "--records",
            str(records_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    validation = json.loads(completed.stdout)
    _write_json(output_dir / "provenance-validation.json", validation)
    return records_path, validation


def _empty_output_dir(path: Path) -> Path:
    output = path.expanduser().resolve()
    if _inside_git_worktree(output):
        raise ValueError("model and report artifacts must stay outside every Git worktree")
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise ValueError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    source, source_manifest_sha256 = _read_approved_source()
    train_rows, dev_rows, preparation = _load_data(
        args.archive, TRAIN_DIALOGUE_COUNT, DEV_DIALOGUE_COUNT
    )
    if set(preparation["training_families"]) & set(preparation["development_families"]):
        raise ValueError("train and development dialogue families overlap")
    output_dir = _empty_output_dir(args.output_dir)
    selection_path = output_dir / "selection-manifest.json"
    _write_json(selection_path, preparation["selection"])

    records_path, provenance_validation = _write_and_validate_records(
        output_dir,
        _document_records(train_rows, dev_rows, preparation),
    )
    transform_config_sha256 = _sha256_bytes(_json_bytes(TRANSFORM_CONFIG))

    training_started = time.perf_counter()
    parameters, training = engine.fit(
        train_rows, args.seed, args.epochs, args.learning_rate
    )
    training_seconds = time.perf_counter() - training_started
    artifact_path = output_dir / "lumi-random-init-jmultiwoz.npz"
    np.savez_compressed(artifact_path, **parameters)
    code_hashes = {
        "pilot_script_sha256": engine._sha256(Path(__file__).resolve()),
        "response_engine_sha256": engine._sha256(Path(engine.__file__).resolve()),
    }
    metadata = {
        "model_id": "lumi-random-init-jmultiwoz-state-response-smoke-v1",
        "architecture": (
            "single-layer tanh recurrent byte-level encoder-decoder with shared input "
            "embedding and autoregressive byte softmax"
        ),
        "initialization": "Gaussian random initialization; no external weights or embeddings",
        "seed": args.seed,
        "embedding_dimension": engine.EMBED_DIM,
        "hidden_dimension": engine.HIDDEN_DIM,
        "parameter_count": sum(int(value.size) for value in parameters.values()),
        "source_id": SOURCE_ID,
        "source_revision": UPSTREAM_REVISION,
        "source_manifest_sha256": source_manifest_sha256,
        "document_records_sha256": engine._sha256(records_path),
        "transform_config_sha256": transform_config_sha256,
        "provenance_validation": provenance_validation,
        "code_hashes": code_hashes,
        "training_config": {
            "objective": "per-example mean byte-level next-token cross-entropy, then example mean",
            "optimizer": "Adam",
            "optimizer_beta1": 0.9,
            "optimizer_beta2": 0.999,
            "optimizer_epsilon": 1e-8,
            "global_gradient_norm_clip": 1.0,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "shuffle_seed": args.seed + 1,
        },
        "generation_config": {
            "decoding": "greedy argmax",
            "max_output_bytes": args.max_output_bytes,
            "tokenizer": "none; UTF-8 byte IDs with reserved BOS/EOS",
        },
        "runtime_environment": {
            "os": platform.platform(),
            "cpu_architecture": platform.machine(),
            "processor_identifier": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "cpu_thread_limit": 1,
    }
    _write_json(artifact_path.with_suffix(".json"), metadata)

    load_started = time.perf_counter()
    with np.load(artifact_path, allow_pickle=False) as archive:
        reloaded = {name: archive[name].copy() for name in engine.PARAMETER_NAMES}
    load_ms = (time.perf_counter() - load_started) * 1000.0

    prediction_rows: list[dict[str, Any]] = []
    warm_times: list[float] = []
    for row in dev_rows:
        call_started = time.perf_counter()
        generated, ended_with_eos, valid_utf8 = engine.generate(
            reloaded, row["text"], args.max_output_bytes
        )
        warm_times.append((time.perf_counter() - call_started) * 1000.0)
        try:
            parsed = json.loads(generated)
            valid_json = True
            valid_output_shape = (
                isinstance(parsed, dict)
                and set(parsed) == {"arguments", "domain", "response"}
                and isinstance(parsed.get("arguments"), dict)
                and isinstance(parsed.get("domain"), str)
                and isinstance(parsed.get("response"), str)
            )
        except (json.JSONDecodeError, TypeError):
            parsed = None
            valid_json = False
            valid_output_shape = False
        try:
            reference = json.loads(row["response"])
        except json.JSONDecodeError:
            reference = {}
        prediction_rows.append(
            {
                "id": row["id"],
                "family_id": row["family_id"],
                "text": row["text"],
                "reference_output": row["response"],
                "generated_output": generated,
                "valid_utf8": valid_utf8,
                "ended_with_eos": ended_with_eos,
                "valid_json": valid_json,
                "valid_output_shape": valid_output_shape,
                "exact_domain_match": bool(
                    valid_output_shape and parsed.get("domain") == reference.get("domain")
                ),
                "exact_arguments_match": bool(
                    valid_output_shape
                    and parsed.get("arguments") == reference.get("arguments")
                ),
                "human_response_quality_reviewed": False,
                "exact_response_string_match_used_as_quality_score": False,
            }
        )

    ordered_times = sorted(warm_times)
    p95_index = max(0, math.ceil(0.95 * len(ordered_times)) - 1)
    prediction_path = output_dir / "development-predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in prediction_rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                + "\n"
            )

    selection_sha256 = engine._sha256(selection_path)
    report = {
        "evidence_level": "exploratory",
        "model_id": metadata["model_id"],
        "random_initialization": True,
        "pretrained_artifacts_used": False,
        "source_id": SOURCE_ID,
        "source_revision": UPSTREAM_REVISION,
        "source_manifest_sha256": source_manifest_sha256,
        "document_records_sha256": engine._sha256(records_path),
        "transform_config_sha256": transform_config_sha256,
        "provenance_validation": provenance_validation,
        "source_manifest_decision": source["decision"],
        "code_hashes": code_hashes,
        "selection_manifest_sha256": selection_sha256,
        "archive_sha256": preparation["archive_sha256"],
        "selected_train_dialogues": len(train_rows),
        "selected_dev_dialogues": len(dev_rows),
        "train_dev_family_overlap": False,
        "training_utf8_byte_count": preparation["train_input_target_utf8_bytes"],
        "development_utf8_byte_count": preparation["dev_input_target_utf8_bytes"],
        "phone_like_spans_redacted": preparation["phone_like_spans_redacted"],
        "seed": args.seed,
        "training_config": metadata["training_config"],
        "generation_config": metadata["generation_config"],
        "runtime_environment": metadata["runtime_environment"],
        "parameter_count": metadata["parameter_count"],
        "parameter_state_sha256": engine._parameter_state_sha256(parameters),
        "training": {key: value for key, value in training.items() if key != "loss_by_epoch"},
        "training_seconds": training_seconds,
        "development_generation": {
            "case_count": len(prediction_rows),
            "valid_utf8_count": sum(bool(row["valid_utf8"]) for row in prediction_rows),
            "ended_with_eos_count": sum(bool(row["ended_with_eos"]) for row in prediction_rows),
            "valid_json_count": sum(bool(row["valid_json"]) for row in prediction_rows),
            "valid_output_shape_count": sum(
                bool(row["valid_output_shape"]) for row in prediction_rows
            ),
            "exact_domain_match_count": sum(
                bool(row["exact_domain_match"]) for row in prediction_rows
            ),
            "exact_arguments_match_count": sum(
                bool(row["exact_arguments_match"]) for row in prediction_rows
            ),
            "human_response_quality_reviewed": False,
            "exact_response_string_match_used_as_quality_score": False,
        },
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
            "scope_note": "Windows process working set includes Python and NumPy runtime.",
        },
        "artifact": {
            "model_path": str(artifact_path),
            "metadata_path": str(artifact_path.with_suffix(".json")),
            "model_bytes": artifact_path.stat().st_size,
            "model_sha256": engine._sha256(artifact_path),
        },
        "prediction_artifact": {
            "path": str(prediction_path),
            "sha256": engine._sha256(prediction_path),
        },
        "limitations": [
            "This is a small Japanese travel-domain state-and-response smoke, not a Lumi-quality result.",
            "Only a deterministic 64-dialogue training subset and 24-dialogue public development subset are used by default.",
            "The model has 18,321 trainable parameters and a byte-level vocabulary; it is a toy candidate, not a selected architecture.",
            "The public source is Japanese travel dialogue; English, natural EN/JA code-switching, media behavior, unsupported requests, and false-action safety are not exercised.",
            "Development references are public dataset labels, not the qualified human-reviewed Lumi evaluation set or final holdout.",
            "Generated Japanese responses were not human-reviewed; JSON validity and exact slot matches are engineering diagnostics only.",
            "Phone-like spans are redacted and phone, address, and reference slots are omitted; no raw source text or model artifact is committed.",
            "No artifact from the separately excluded 101-case Codex draft or any derivative was loaded or used.",
            "The run does not measure ZenStream integration, grounding, service lifecycle, cold start, idle use, unload, GPU, or other CPU classes.",
        ],
    }
    _write_json(output_dir / "report.json", report)
    _write_json(
        output_dir / "training-curve.json",
        {
            "seed": args.seed,
            "initial_training_loss": training["initial_loss"],
            "loss_by_epoch": training["loss_by_epoch"],
        },
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="pinned archive outside every Git worktree")
    parser.add_argument("--output-dir", type=Path, required=True, help="empty controlled output directory outside every Git worktree")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-output-bytes", type=int, default=512)
    args = parser.parse_args(argv)
    if args.max_output_bytes < 512:
        parser.error("max-output-bytes must be at least 512 for the registered targets")
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
