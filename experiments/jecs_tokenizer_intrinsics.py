#!/usr/bin/env python3
"""Measure small JECS-only tokenizer intrinsic and CPU characteristics."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import distribution, version
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import sentencepiece as spm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments import jecs_byte_lm_pilot as jecs  # noqa: E402

EXPECTED_SENTENCEPIECE = "0.2.2"
VOCAB_SIZES = (2048, 4096)
SEED = 314159
SENTENCEPIECE_DEFAULT_SEED = 4294967295
TOKENIZER_TRANSFORM = {
    "operation_id": "jecs-tokenizer-train-corpus-v1",
    "revision": "1",
    "config": {
        "split": "existing JECS training families only",
        "input": "parsed transcript text; remove source IDs and code-switch asterisk delimiters",
        "sample_order": "source file order: ja, en, cs; no shuffle",
        "normalization": "identity; preserve spaces",
        "sentencepiece_version": EXPECTED_SENTENCEPIECE,
        "candidate_models": ["bpe", "unigram"],
        "requested_vocab_sizes": list(VOCAB_SIZES),
        "byte_fallback": True,
        "family_split_seed": SEED,
        "sentencepiece_seed_policy": "pinned library default; explicit random_seed is rejected by the installed wheel",
        "sentencepiece_default_seed": SENTENCEPIECE_DEFAULT_SEED,
        "training_threads": 1,
    },
}
TOKENIZER_TRANSFORM["config_sha256"] = "sha256:" + hashlib.sha256(
    json.dumps(TOKENIZER_TRANSFORM["config"], sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sentencepiece_native_sha256() -> str:
    package = distribution("sentencepiece")
    candidates = [
        path for path in (package.files or [])
        if Path(str(path)).name.startswith("_sentencepiece")
        and Path(str(path)).suffix.lower() in {".pyd", ".so", ".dylib"}
    ]
    if len(candidates) != 1:
        raise ValueError("expected exactly one installed SentencePiece native module")
    return sha256_file(Path(package.locate_file(candidates[0])))


def inside_git_worktree(path: Path) -> bool:
    resolved = path.resolve()
    return any((parent / ".git").exists() for parent in (resolved, *resolved.parents))


def require_source_scope() -> tuple[dict[str, Any], str]:
    manifest_path = REPO_ROOT / "provenance" / "data_sources.json"
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    matches = [
        source for source in manifest["sources"]
        if source.get("source_id") == jecs.SOURCE_ID
    ]
    required_uses = {"tokenizer_training", "evaluation", "filtering"}
    if len(matches) != 1:
        raise ValueError("exactly one JECS source record is required")
    source = matches[0]
    if source.get("decision") != "approved_for_use":
        raise ValueError("JECS is not approved for this exploratory tokenizer scope")
    if not required_uses.issubset(set(source.get("lumi_uses", []))):
        raise ValueError("JECS manifest does not explicitly admit tokenizer_training/evaluation/filtering")
    if source.get("permissions", {}).get("training_use") != "permitted":
        raise ValueError("JECS source record does not permit local training use")
    return source, sha256_bytes(raw)


def attach_switch_spans(data_dir: Path, rows: list[dict[str, Any]]) -> None:
    by_family = {row["family_id"]: row for row in rows if row["language"] == "cs"}
    found: set[str] = set()
    for line_number, line in enumerate(
        (data_dir / "transcripts_cs.txt").read_text(encoding="utf-8", errors="strict").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        item_id, separator, marked_text = line.partition(": ")
        if not separator:
            raise ValueError(f"transcripts_cs.txt:{line_number}: invalid row")
        item_match = __import__("re").fullmatch(r"JECS(\d{4})_CS", item_id)
        if item_match is None:
            raise ValueError(f"transcripts_cs.txt:{line_number}: invalid item ID")
        family = f"JECS{int(item_match.group(1)):04d}"
        if family not in by_family:
            raise ValueError(f"transcripts_cs.txt:{line_number}: family absent from parsed rows")
        clean = marked_text.strip()
        if clean.count("*") != 2:
            raise ValueError(f"transcripts_cs.txt:{line_number}: expected one marked segment")
        start = clean.index("*")
        end = clean.rindex("*")
        plain = clean[:start] + clean[start + 1:end] + clean[end + 1:]
        row = by_family[family]
        if plain != row["text"]:
            raise ValueError(f"transcripts_cs.txt:{line_number}: parsed switch text differs")
        row["switch_boundaries_chars"] = [start, start + (end - start - 1)]
        found.add(family)
    if found != set(by_family):
        raise ValueError("not every parsed code-switch row has a source-marked segment")


def add_pilot_scope(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["uses"] = (
            ["tokenizer_training", "filtering"]
            if row["split"] == "train"
            else ["evaluation", "filtering"]
        )
        if row["split"] == "train":
            row["transformations"].append({
                "operation_id": TOKENIZER_TRANSFORM["operation_id"],
                "revision": TOKENIZER_TRANSFORM["revision"],
                "config_sha256": TOKENIZER_TRANSFORM["config_sha256"],
            })


def quantile(values: list[int | float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return float(ordered[index])


def percentiles(values: list[int | float]) -> dict[str, float | None]:
    return {
        "median": quantile(values, 0.50),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
    }


def byte_offsets_for_char_offsets(text: str) -> list[int]:
    offsets = [0]
    total = 0
    for char in text:
        total += len(char.encode("utf-8"))
        offsets.append(total)
    return offsets


def measure_candidate(
    name: str,
    encoder: Any,
    model_path: Path | None,
    dev_rows: list[dict[str, Any]],
    tokenizer_dir: Path,
) -> dict[str, Any]:
    token_counts: dict[str, list[int]] = {"en": [], "ja": [], "cs": []}
    scalar_counts: dict[str, list[int]] = {"en": [], "ja": [], "cs": []}
    byte_counts: dict[str, list[int]] = {"en": [], "ja": [], "cs": []}
    english_words: list[int] = []
    roundtrip_failures = 0
    switch_boundary_count = 0
    switch_boundary_inside_token = 0
    latencies_ms: list[float] = []
    measured_bytes = 0

    for row in dev_rows:
        text = row["text"]
        language = row["language"]
        utf8 = text.encode("utf-8", errors="strict")
        measured_bytes += len(utf8)
        start = time.perf_counter_ns()
        if name == "utf8_bytes":
            ids = list(utf8)
        else:
            ids = encoder.encode(text, return_type=int)
        latencies_ms.append((time.perf_counter_ns() - start) / 1_000_000)
        token_counts[language].append(len(ids))
        scalar_counts[language].append(len(text))
        byte_counts[language].append(len(utf8))
        if language == "en":
            english_words.append(len(text.split()))
        if name == "utf8_bytes":
            reconstructed = bytes(ids).decode("utf-8", errors="strict")
        else:
            reconstructed = encoder.decode(ids)
        roundtrip_failures += int(reconstructed != text)

        if language == "cs":
            char_to_byte = byte_offsets_for_char_offsets(text)
            if name == "utf8_bytes":
                pieces = [(offset, offset + 1) for offset in range(len(utf8))]
                boundaries = [char_to_byte[index] for index in row["switch_boundaries_chars"]]
            else:
                offset_result = encoder.encode(text, return_type="offset_mapping")
                pieces = offset_result["offsets"]
                boundaries = row["switch_boundaries_chars"]
            for boundary in boundaries:
                switch_boundary_count += 1
                if any(begin < boundary < end for begin, end in pieces):
                    switch_boundary_inside_token += 1

    language_metrics: dict[str, Any] = {}
    for language in ("en", "ja", "cs"):
        counts = token_counts[language]
        scalars = scalar_counts[language]
        byte_values = byte_counts[language]
        total_tokens = sum(counts)
        total_scalars = sum(scalars)
        total_bytes = sum(byte_values)
        metrics: dict[str, Any] = {
            "rows": len(counts),
            "utf8_bytes": total_bytes,
            "unicode_scalars": total_scalars,
            "token_count_distribution": percentiles(counts),
            "tokens_per_unicode_scalar": (total_tokens / total_scalars) if total_scalars else None,
            "utf8_bytes_per_token": (total_bytes / total_tokens) if total_tokens else None,
        }
        if language == "en":
            words = sum(english_words)
            metrics["whitespace_word_fertility"] = (total_tokens / words) if words else None
            metrics["whitespace_words"] = words
        language_metrics[language] = metrics

    total_tokens = sum(sum(values) for values in token_counts.values())
    warm_seconds = sum(latencies_ms) / 1000.0
    result: dict[str, Any] = {
        "vocabulary_entries": 256 if name == "utf8_bytes" else int(encoder.get_piece_size()),
        "development_rows": len(dev_rows),
        "development_utf8_bytes": measured_bytes,
        "roundtrip_failures": roundtrip_failures,
        "languages": language_metrics,
        "switch_boundaries": {
            "source_marked_boundary_count": switch_boundary_count,
            "inside_token_count": switch_boundary_inside_token,
            "inside_token_rate": (
                switch_boundary_inside_token / switch_boundary_count
                if switch_boundary_count else None
            ),
            "matrix_direction": "not labeled by the source rows; not inferred from script",
        },
        "warm_per_row_latency_ms": percentiles(latencies_ms),
        "warm_rows_per_second": (len(dev_rows) / warm_seconds) if warm_seconds else None,
        "total_tokens": total_tokens,
    }
    if name == "utf8_bytes":
        result["model_file_bytes"] = 0
        result["vocabulary_file_bytes"] = 0
        result["token_embedding_parameters_at_width_32"] = 256 * 32
        result["token_embedding_bytes_fp32_at_width_32"] = 256 * 32 * 4
        result["tokenizer_model_sha256"] = None
        result["tokenizer_vocab_sha256"] = None
    else:
        vocab_path = model_path.with_suffix(".vocab")
        model_bytes = model_path.stat().st_size
        vocab_bytes = vocab_path.stat().st_size
        result["model_file_bytes"] = model_bytes
        result["vocabulary_file_bytes"] = vocab_bytes
        result["token_embedding_parameters_at_width_32"] = encoder.get_piece_size() * 32
        result["token_embedding_bytes_fp32_at_width_32"] = encoder.get_piece_size() * 32 * 4
        result["tokenizer_model_sha256"] = sha256_file(model_path)
        result["tokenizer_vocab_sha256"] = sha256_file(vocab_path)
        result["model_file"] = model_path.name
        result["vocabulary_file"] = vocab_path.name
    return result


def train_sentencepiece(
    model_type: str,
    vocab_size: int,
    corpus_path: Path,
    artifact_dir: Path,
    output_dir: Path,
    dev_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    prefix = artifact_dir / "tokenizers" / f"jecs-{model_type}-{vocab_size}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    spm.SentencePieceTrainer.train(
        input=str(corpus_path), model_prefix=str(prefix), model_type=model_type,
        vocab_size=vocab_size, character_coverage=1.0, hard_vocab_limit=False,
        byte_fallback=True, unk_id=0, bos_id=-1, eos_id=-1, pad_id=-1,
        normalization_rule_name="identity", add_dummy_prefix=False,
        remove_extra_whitespaces=False, escape_whitespaces=True,
        split_by_unicode_script=False, split_by_whitespace=True,
        input_sentence_size=0, shuffle_input_sentence=False,
        num_threads=1, max_sentence_length=8192,
        max_sentencepiece_length=16, minloglevel=2,
    )
    training_seconds = time.perf_counter() - started
    cold_started = time.perf_counter()
    processor = spm.SentencePieceProcessor(model_file=str(prefix.with_suffix(".model")))
    load_seconds = time.perf_counter() - cold_started
    measured = measure_candidate(model_type, processor, prefix.with_suffix(".model"), dev_rows, prefix.parent)
    measured.update({
        "model_type": model_type,
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": processor.get_piece_size(),
        "training_seconds": training_seconds,
        "cold_model_load_seconds": load_seconds,
        "sentencepiece_seed_policy": "library default 4294967295; the pinned wheel rejected explicit random_seed",
        "training_threads": 1,
        "training_corpus_sha256": sha256_file(corpus_path),
    })
    return measured


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def run(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    installed_sentencepiece = version("sentencepiece")
    if installed_sentencepiece != EXPECTED_SENTENCEPIECE:
        raise ValueError(f"requires sentencepiece=={EXPECTED_SENTENCEPIECE}; found {installed_sentencepiece}")
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    if data_dir == output_dir or data_dir in output_dir.parents or output_dir in data_dir.parents:
        raise ValueError("input and output directories must not overlap")
    if inside_git_worktree(output_dir):
        raise ValueError("all text, tokenizer models, and reports must remain outside Git worktrees")
    source, source_manifest_hash = require_source_scope()
    rows, data_summary = jecs.read_rows(data_dir)
    attach_switch_spans(data_dir, rows)
    add_pilot_scope(rows)
    train_rows = [row for row in rows if row["split"] == "train"]
    dev_rows = [row for row in rows if row["split"] == "development"]
    if not train_rows or not dev_rows:
        raise ValueError("both train and separate public-development rows are required")

    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_hashes = jecs.write_provenance(
        output_dir, rows, data_summary, source_manifest_hash
    )
    records_path = output_dir / "document_records.jsonl"
    validator = REPO_ROOT / "provenance" / "validate.py"
    validation = os.spawnv(
        os.P_WAIT, sys.executable,
        [sys.executable, str(validator), "--records", str(records_path)],
    )
    if validation != 0:
        raise ValueError(f"provenance validator failed with exit code {validation}")

    artifact_dir = output_dir.parent / "lumi-jecs-tokenizer-intrinsics-v1"
    if inside_git_worktree(artifact_dir):
        raise ValueError("all text and tokenizer models must remain outside Git worktrees")
    corpus = artifact_dir / "tokenizer-training-text.txt"
    corpus_bytes = ("".join(f"{row['text']}\n" for row in train_rows)).encode("utf-8", errors="strict")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    corpus_created = False
    try:
        with corpus.open("xb") as stream:
            stream.write(corpus_bytes)
        corpus_created = True
    except FileExistsError:
        if sha256_file(corpus) != sha256_bytes(corpus_bytes):
            raise ValueError(f"refusing to overwrite unrelated training text at {corpus}")
    candidates: dict[str, Any] = {
        "utf8_bytes": measure_candidate("utf8_bytes", None, None, dev_rows, output_dir)
    }
    for model_type in ("bpe", "unigram"):
        for vocab_size in VOCAB_SIZES:
            key = f"{model_type}_{vocab_size}"
            candidates[key] = train_sentencepiece(
                model_type, vocab_size, corpus, artifact_dir, output_dir, dev_rows
            )

    report = {
        "experiment_id": "jecs-neutral-text-tokenizer-intrinsics-level1-v1",
        "evidence_level": "1 exploratory intrinsic/system measurements only",
        "decision": "no tokenizer-quality selection; no product or response-quality claim",
        "created_date": "2026-10-03",
        "source": source,
        "source_manifest_sha256": f"sha256:{source_manifest_hash}",
        "source_file_sha256": data_summary["source_file_sha256"],
        "document_records_sha256": provenance_hashes["document_records_sha256"],
        "selection_manifest_sha256": provenance_hashes["selection_manifest_sha256"],
        "runner_sha256": sha256_file(Path(__file__)),
        "requirements_sha256": sha256_file(REPO_ROOT / "experiments" / "requirements-tokenizer-pilot.txt"),
        "tokenizer_training_text_sha256": sha256_bytes(corpus_bytes),
        "data": data_summary,
        "training": {
            "families": data_summary["training_family_count"],
            "rows": len(train_rows),
            "utf8_bytes": len(corpus_bytes),
            "vocabulary_sizes": list(VOCAB_SIZES),
            "sentencepiece_version": installed_sentencepiece,
            "sentencepiece_distribution": "Apache-2.0",
            "normalization": "identity; exact Unicode content and whitespace preserved",
            "byte_fallback": True,
            "family_split_seed": SEED,
            "sentencepiece_seed_policy": "library default 4294967295; the pinned wheel rejected explicit random_seed",
            "threads": 1,
            "pretrained_weights_or_embeddings_used": False,
        },
        "development": {
            "families": data_summary["development_family_count"],
            "rows": len(dev_rows),
            "used_only_for_intrinsic_measurement": True,
            "used_to_fit_vocabularies_or_filter_training_items": False,
            "source_matrix_direction_labels": "unavailable; not inferred from Unicode script",
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "execution": "CPU only, one tokenizer-training thread",
            "sentencepiece_version": installed_sentencepiece,
            "sentencepiece_native_module_sha256": sentencepiece_native_sha256(),
        },
        "candidates": candidates,
        "scope_limits": [
            "Only pinned JECS neutral transcript text and the existing grouped train/development split were used; audio and other archive members were not acquired or opened.",
            "The 101-case Codex draft and every derivative remained excluded.",
            "This small acted/read corpus does not establish natural Japanese, natural code-switching, Lumi behavior, or product quality.",
            "The corpus does not label the code-switch matrix direction; the report does not infer it from character script.",
            "Development text was not used to fit, tune, or filter any tokenizer candidate.",
            "All raw text and tokenizer artifacts remain local and are not approved for public distribution.",
            "Tokenizer candidate metrics do not select an architecture, runtime, or final vocabulary.",
        ],
    }
    write_json(output_dir / "report.json", report)
    if corpus_created:
        corpus.unlink(missing_ok=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = run(args.data_dir, args.output_dir)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "experiment_id": report["experiment_id"],
        "train_families": report["training"]["families"],
        "development_families": report["development"]["families"],
        "candidates": {
            name: {
                "vocabulary_entries": value["vocabulary_entries"],
                "roundtrip_failures": value["roundtrip_failures"],
                "languages": {
                    language: {
                        "tokens_per_unicode_scalar": metrics["tokens_per_unicode_scalar"],
                        "utf8_bytes_per_token": metrics["utf8_bytes_per_token"],
                        "token_count_distribution": metrics["token_count_distribution"],
                    }
                    for language, metrics in value["languages"].items()
                },
                "switch_boundary_inside_token_rate": value["switch_boundaries"]["inside_token_rate"],
                "warm_rows_per_second": value["warm_rows_per_second"],
                "model_file_bytes": value["model_file_bytes"],
                "tokenizer_model_sha256": value["tokenizer_model_sha256"],
            }
            for name, value in report["candidates"].items()
        },
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
