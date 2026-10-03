#!/usr/bin/env python3
"""Score existing JECS byte-LM checkpoints on one pinned public CSR-L test slice.

This script never trains, tunes, or writes model weights. Raw queries, item-level
provenance, and output files must stay outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

from jecs_byte_lm_pilot import (
    BATCH_SIZE,
    EMAIL,
    HIDDEN_SIZE,
    LANGUAGE_CODES,
    PHONE,
    URL,
    VOCAB_SIZE,
    evaluate,
    file_sha256,
    inside_git_worktree,
    loss_grad,
    make_sequences,
    working_set_bytes,
)


SOURCE_ID = "csr-l-humaneval-ja-en-public-test-transfer-probe"
DATASET_REVISION = "d7634f8e08cf9249cdb2169fb153b0de85d705c8"
PARQUET_SHA256 = "e1dc7284c97c0d4e6dfd7eae7ab2dd1f5af223523b9dad23f0b41c40e2f6aba0"
PARQUET_BYTES = 26901
NORMALIZED_JSONL_SHA256 = "e1e5d2098393141c63b77d3019b6c2b47ccd60798b293ae36b0b339cdef1060c"
EXPECTED_PHONE_PATTERN_FALSE_POSITIVES = {
    "00119": "7ac6f058394b77ea1607badbee0ef021e1d471d1ba11088a15ccbf688b5c0356",
    "00062": "56c8e63b21b0206f7a35304e6069aa31ebf4c13e006ab7843b47214d32ba1af6",
}
WEIGHT_NAMES = {"embedding", "recurrent", "hidden_bias", "output", "output_bias"}
TRANSFORM_CONFIG = {
    "operation": "csr-l-ja-en-query-byte-input-v1",
    "revision": DATASET_REVISION,
    "input_text_normalization": "none; preserve the published query exactly",
    "encoding": "UTF-8 bytes",
    "sequence": ["BOS", "code-switch-language-tag", "UTF-8 query bytes", "EOS"],
    "uses": ["evaluation"],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def require_external(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if inside_git_worktree(resolved):
        raise ValueError(f"{label} must remain outside every Git worktree")
    return resolved


def query_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_records(
    rows: list[dict[str, Any]],
    *,
    source_uri: str,
    script_sha256: str,
    jecs_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != 158:
        raise ValueError(f"expected 158 pinned queries, received {len(rows)}")
    if any(set(row) != {"id", "text"} for row in rows):
        raise ValueError("each normalized query row must contain only id and text")
    ids = [row["id"] for row in rows]
    if any(not isinstance(item_id, str) or not item_id.strip() for item_id in ids):
        raise ValueError("query IDs must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate source query IDs")

    hashes = [query_hash(row["text"]) for row in rows if isinstance(row["text"], str)]
    if len(hashes) != len(rows):
        raise ValueError("all query texts must be strings")
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate UTF-8 query texts")
    if any(not row["text"].strip() for row in rows):
        raise ValueError("empty query text")
    if any(
        unicodedata.category(char) == "Cc" and char not in "\t\n\r"
        for row in rows
        for char in row["text"]
    ):
        raise ValueError("unexpected control character in query text")

    email_ids = [row["id"] for row in rows if EMAIL.search(row["text"])]
    url_ids = [row["id"] for row in rows if URL.search(row["text"])]
    if email_ids or url_ids:
        raise ValueError("email or URL privacy-pattern match in query slice")
    phone_hashes = {
        row["id"]: query_hash(row["text"])
        for row in rows
        if PHONE.search(row["text"])
    }
    if phone_hashes != EXPECTED_PHONE_PATTERN_FALSE_POSITIVES:
        raise ValueError("phone-like pattern results differ from the manually reviewed code/date examples")

    if len(jecs_records) != 2977:
        raise ValueError("expected all 2,977 eligible JECS sample records for overlap checking")
    jecs_source_ids = {
        item.get("source_id")
        for record in jecs_records
        for item in record.get("source_items", [])
    }
    if jecs_source_ids != {"jecs-v1-neutral-text-only"}:
        raise ValueError("overlap input must contain only the eligible JECS source records")
    eligible_hashes = {
        record.get("sample_sha256", "").removeprefix("sha256:")
        for record in jecs_records
    }
    overlaps = sorted(set(hashes) & eligible_hashes)
    if overlaps:
        raise ValueError(f"exact UTF-8 hash overlap with JECS samples: {len(overlaps)}")

    reviewed_date = "2026-10-03"
    transform_hash = "sha256:" + hashlib.sha256(
        json.dumps(TRANSFORM_CONFIG, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    rights_evidence = [
        f"https://huggingface.co/datasets/UTokyo-Yokoya-Lab/HumanEvalRetrieval-CSR-L/resolve/{DATASET_REVISION}/README.md",
        "https://aclanthology.org/2026.findings-acl.636.pdf",
        "https://huggingface.co/datasets/mteb/HumanEvalRetrieval",
    ]
    records = []
    for row, digest in zip(rows, hashes, strict=True):
        records.append({
            "schema_version": 1,
            "record_id": f"csr-l-humaneval-ja-en-{row['id']}",
            "sample_kind": "external_document",
            "sample_sha256": f"sha256:{digest}",
            "language_tags": ["ja", "en"],
            "lumi_uses": ["evaluation"],
            "decision": "approved_for_use",
            "split": "public_test",
            "contamination_status": "checked_clear",
            "privacy_status": "cleared",
            "source_items": [{
                "source_id": SOURCE_ID,
                "item_id": row["id"],
                "source_uri": source_uri,
                "source_revision": DATASET_REVISION,
                "source_sha256": f"sha256:{digest}",
                "rights_basis": "license",
                "license_identifier": "MIT",
                "license_uri": "https://opensource.org/license/mit/",
                "rights_evidence_uris": rights_evidence,
                "rights_review_status": "reviewed",
                "rights_reviewed_date": reviewed_date,
                "permissions": {
                    "training_use": "permitted",
                    "evaluation_use": "permitted",
                    "commercial_use": "permitted",
                    "modification": "permitted",
                    "redistribution": "permitted",
                },
                "attribution": "Cite the CSR-L authors and dataset, plus the underlying HumanEval source; retain MIT notices.",
            }],
            "transformations": [{
                "operation_id": TRANSFORM_CONFIG["operation"],
                "revision": f"script-sha256:{script_sha256}",
                "config_sha256": transform_hash,
            }],
            "parent_record_ids": [],
            "review": {
                "reviewed_by": "Codex-assisted pinned-source, rights, privacy, and overlap review",
                "reviewed_date": reviewed_date,
                "rationale": (
                    "Exact pinned public-test query; source SHA recorded per item. "
                    "The only phone-like matches are the manually inspected code/date examples "
                    "in these exact item IDs: " + ",".join(sorted(phone_hashes))
                    + ". Exact UTF-8 hashes were checked against all 2,977 eligible JECS records. "
                    "Evaluation only; not a sealed holdout or conversational-quality sample."
                ),
            },
        })
    checks = {
        "row_count": len(rows),
        "unique_id_count": len(set(ids)),
        "unique_text_count": len(set(hashes)),
        "email_pattern_matches": 0,
        "url_pattern_matches": 0,
        "phone_pattern_match_count": sum(len(PHONE.findall(row["text"])) for row in rows),
        "manually_reviewed_code_or_date_false_positive_ids": sorted(phone_hashes),
        "eligible_jecs_record_count": len(jecs_records),
        "exact_utf8_hash_overlap_count": 0,
    }
    return records, checks


def load_weights(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != WEIGHT_NAMES:
            raise ValueError(f"unexpected weight keys in {path.name}")
        params = {name: archive[name].copy() for name in archive.files}
    expected_shapes = {
        "embedding": (VOCAB_SIZE, HIDDEN_SIZE),
        "recurrent": (HIDDEN_SIZE, HIDDEN_SIZE),
        "hidden_bias": (HIDDEN_SIZE,),
        "output": (HIDDEN_SIZE, VOCAB_SIZE),
        "output_bias": (VOCAB_SIZE,),
    }
    if any(params[name].shape != shape for name, shape in expected_shapes.items()):
        raise ValueError(f"unexpected model shape in {path.name}")
    if any(not np.isfinite(array).all() for array in params.values()):
        raise ValueError(f"non-finite model weight in {path.name}")
    return params


def score_rows(params: dict[str, np.ndarray], sequences: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = evaluate(params, sequences)["cs"]
    item_bpc = []
    for row in sequences:
        _, _, byte_nats, byte_count, _ = loss_grad(params, [row], gradients=False)
        if byte_count != row["byte_count"]:
            raise ValueError("scored byte count differs from source query bytes")
        item_bpc.append(byte_nats / math.log(2))
    return {"aggregate": metrics, "per_query_bits_per_byte": np.asarray(item_bpc, dtype=np.float64)}


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, type=Path, help="normalized pinned query JSONL outside Git")
    parser.add_argument("--parquet", required=True, type=Path, help="pinned Parquet source file outside Git")
    parser.add_argument("--jecs-records", required=True, type=Path, help="eligible JECS item records outside Git")
    parser.add_argument("--pilot-report", required=True, type=Path, help="JECS experiment report outside Git")
    parser.add_argument("--baseline-weights", required=True, type=Path, help="existing bilingual baseline NPZ outside Git")
    parser.add_argument("--codeswitch-weights", required=True, type=Path, help="existing code-switch augmented NPZ outside Git")
    parser.add_argument("--output-dir", required=True, type=Path, help="local output directory outside Git")
    parser.add_argument("--source-manifest", type=Path, default=root / "provenance" / "data_sources.json")
    args = parser.parse_args(argv)

    external = {
        name: require_external(getattr(args, name), name)
        for name in (
            "queries", "parquet", "jecs_records", "pilot_report",
            "baseline_weights", "codeswitch_weights", "output_dir",
        )
    }
    if not external["output_dir"].is_dir():
        external["output_dir"].mkdir(parents=True)
    source_manifest = args.source_manifest.expanduser().resolve()
    try:
        source_manifest.relative_to(root)
    except ValueError as exc:
        raise ValueError("the reviewed source manifest must be in the task worktree") from exc
    if not source_manifest.is_file():
        raise ValueError("the reviewed source manifest must be in the task worktree")
    if external["parquet"].stat().st_size != PARQUET_BYTES or file_sha256(external["parquet"]) != PARQUET_SHA256:
        raise ValueError("pinned CSR-L Parquet size or SHA-256 mismatch")
    if file_sha256(external["queries"]) != NORMALIZED_JSONL_SHA256:
        raise ValueError("normalized query JSONL does not match the pinned 158-row acquisition")

    rows = read_jsonl(external["queries"])
    jecs_records = read_jsonl(external["jecs_records"])
    script_sha256 = file_sha256(Path(__file__))
    parquet_uri = (
        "https://huggingface.co/datasets/UTokyo-Yokoya-Lab/HumanEvalRetrieval-CSR-L/"
        f"resolve/{DATASET_REVISION}/queries_ja_en/test-00000-of-00001.parquet"
    )
    sample_records, data_checks = build_records(
        rows,
        source_uri=parquet_uri,
        script_sha256=script_sha256,
        jecs_records=jecs_records,
    )
    records_path = external["output_dir"] / "csr_l_sample_records.jsonl"
    validation_path = external["output_dir"] / "csr_l_provenance_validation.json"
    write_jsonl(records_path, sample_records)

    validation = subprocess.run(
        [
            sys.executable,
            str(root / "provenance" / "validate.py"),
            "--records", str(records_path),
            "--sources", str(source_manifest),
            "--output", str(validation_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if validation.returncode != 0:
        raise ValueError("provenance gate failed: " + validation.stderr[-1200:])
    validation_report = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation_report.get("status") != "valid" or validation_report.get("sample_record_count") != 158:
        raise ValueError("provenance validator did not approve all 158 exact query records")

    pilot = json.loads(external["pilot_report"].read_text(encoding="utf-8"))
    candidates = pilot["candidates"]
    candidate_specs = (
        ("bilingual_baseline", external["baseline_weights"]),
        ("code_switch_augmented", external["codeswitch_weights"]),
    )
    parameters = {}
    candidate_metadata = {}
    for name, path in candidate_specs:
        metadata = candidates[name]
        digest = file_sha256(path)
        if not metadata.get("random_initialization") or metadata.get("pretrained_weights_or_embeddings_used"):
            raise ValueError(f"{name} was not verified as randomly initialized")
        if metadata["weights_sha256"].removeprefix("sha256:") != digest:
            raise ValueError(f"{name} weight hash differs from the original pilot report")
        if metadata["seed"] != 314159 or metadata["parameter_count"] != 18086:
            raise ValueError(f"{name} model lineage differs from the pinned JECS pilot")
        if name == "code_switch_augmented" and metadata["training_unique_code_switch_rows"] <= 0:
            raise ValueError("the augmented checkpoint has no code-switch training rows")
        parameters[name] = load_weights(path)
        candidate_metadata[name] = {
            "weights_sha256": f"sha256:{digest}",
            "random_initialization": metadata["random_initialization"],
            "pretrained_weights_or_embeddings_used": metadata["pretrained_weights_or_embeddings_used"],
            "seed": metadata["seed"],
            "epochs": metadata["epochs"],
            "training_optimizer_updates": metadata["training_optimizer_updates"],
            "training_sample_presentations": metadata["training_sample_presentations"],
            "training_unique_code_switch_rows": metadata["training_unique_code_switch_rows"],
        }
    if candidates["bilingual_baseline"]["architecture"] != candidates["code_switch_augmented"]["architecture"]:
        raise ValueError("candidate architectures differ")

    sequence_rows = [
        {"language": "cs", "text": row["text"], "family_id": row["id"]}
        for row in rows
    ]
    sequences = make_sequences(sequence_rows)
    rss_start = working_set_bytes(peak=True)
    start = time.perf_counter()
    baseline = score_rows(parameters["bilingual_baseline"], sequences)
    augmented = score_rows(parameters["code_switch_augmented"], sequences)
    total_scoring_seconds = time.perf_counter() - start
    baseline_items = baseline.pop("per_query_bits_per_byte")
    augmented_items = augmented.pop("per_query_bits_per_byte")
    paired_delta = augmented_items - baseline_items
    bootstrap_rng = np.random.default_rng(20261003)
    bootstrap = bootstrap_rng.choice(paired_delta, size=(3000, len(paired_delta)), replace=True).mean(axis=1)
    rss_end = working_set_bytes(peak=True)

    output = {
        "schema_version": 1,
        "experiment_id": "jecs-csr-l-ja-en-byte-transfer-v1",
        "created_date": "2026-10-03",
        "evidence_level": "bounded_external_transfer_diagnostic",
        "random_initialization": True,
        "pretrained_weights_or_embeddings_used": False,
        "training_performed_by_this_probe": False,
        "source": {
            "dataset": "CSR-L HumanEval Japanese-English queries",
            "dataset_revision": DATASET_REVISION,
            "split": "public_test",
            "query_count": len(rows),
            "parquet_sha256": f"sha256:{PARQUET_SHA256}",
            "normalized_jsonl_sha256": f"sha256:{NORMALIZED_JSONL_SHA256}",
            "source_manifest_sha256": file_sha256(source_manifest),
            "sample_records_sha256": file_sha256(records_path),
            "provenance_validation": validation_report,
            "rights_scope": "MIT-declared dataset; evaluation only; no raw text committed",
        },
        "training_data": {
            "source": "existing JECS text-only pilot",
            "pilot_report_sha256": f"sha256:{file_sha256(external['pilot_report'])}",
            "original_training_source_manifest_sha256": pilot["source_manifest_sha256"],
            "existing_checkpoint_candidates": candidate_metadata,
        },
        "input": {
            "language_tag": LANGUAGE_CODES["cs"],
            "encoding": "UTF-8 bytes",
            "normalization": "none",
            "target_bytes": sum(row["byte_count"] for row in sequences),
            "data_checks": data_checks,
        },
        "metrics": {
            "metric": "next-byte negative log likelihood in bits per UTF-8 byte",
            "lower_is_better": True,
            "bilingual_baseline": baseline["aggregate"],
            "code_switch_augmented": augmented["aggregate"],
            "aggregate_delta_augmented_minus_baseline": (
                augmented["aggregate"]["bits_per_utf8_byte"]
                - baseline["aggregate"]["bits_per_utf8_byte"]
            ),
            "paired_per_query_delta_bits_per_byte": {
                "mean": float(paired_delta.mean()),
                "median": float(np.median(paired_delta)),
                "bootstrap_95_percentile_interval": [
                    float(np.percentile(bootstrap, 2.5)),
                    float(np.percentile(bootstrap, 97.5)),
                ],
                "bootstrap_seed": 20261003,
                "bootstrap_resamples": 3000,
            },
            "total_scoring_seconds": total_scoring_seconds,
        },
        "runtime": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "peak_working_set_bytes_at_start": rss_start,
            "peak_working_set_bytes_at_end": rss_end,
            "batch_size": BATCH_SIZE,
        },
        "scope_limits": [
            "This is next-byte prediction on 158 public Japanese-English technical retrieval queries, not generated response quality or retrieval effectiveness.",
            "The source is human-rewritten and human-validated, but it is not spontaneous conversation or ZenStream intent data.",
            "The public test split is not a sealed holdout. Do not use it for training, tokenizer fitting, filtering, prompt construction, tuning, or release evidence.",
            "The two existing one-seed checkpoints come from a small JECS mixture ablation whose per-language exposure was not controlled.",
            "The probe does not support any claim about natural Japanese dialogue, action safety, grounding, user value, or public model distribution.",
        ],
    }
    result_path = external["output_dir"] / "jecs_csr_l_transfer_probe.json"
    write_json(result_path, output)
    print(json.dumps({
        "report_path": str(result_path),
        "report_sha256": f"sha256:{file_sha256(result_path)}",
        "source_records_path": str(records_path),
        "provenance_status": validation_report["status"],
        "rows": len(rows),
        "bilingual_baseline_bits_per_utf8_byte": baseline["aggregate"]["bits_per_utf8_byte"],
        "code_switch_augmented_bits_per_utf8_byte": augmented["aggregate"]["bits_per_utf8_byte"],
        "augmented_minus_baseline": output["metrics"]["aggregate_delta_augmented_minus_baseline"],
        "scoring_seconds": total_scoring_seconds,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
