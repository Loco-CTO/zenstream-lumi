#!/usr/bin/env python3
"""Train one tiny random-init model on synthetic combined Lumi outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GOAL_OBJECTIVE_SHA256 = "sha256:fc1bc61d354a9534a7a6692646c52d42c2cb2c5a5413025f0b48fbe461e5232c"
BASE_COMMIT = "039b54310921a7aa32cdacd505f8fa7f1391db2e"
RESPONSE_ENGINE_SHA256 = "sha256:6cb2a3c8d1f56e7c304194db40132eef3b34f39d505f77e53b4385ffba1cf780"
DATASET_ID = "lumi-unified-synthetic-smoke-v1"
HISTORICAL_TEMPLATE_BUILDER_SHA256 = "sha256:3bbfb41e068e75ed57c450dc371c94598e33020d70ba673d110f4f71b78f3ef3"
TRAIN_INPUT_SHA256 = "sha256:1a28bb3abc9be316d46f6e3e0c7cced4e55a16d7f9fe3512818a725626d15247"
DEV_INPUT_SHA256 = "sha256:88c7cd5614f589543838ccb15b545821b320de9397e33eaaabe4ffec4bc306fe"
ALL_INPUTS_SHA256 = "sha256:02790c733a85a76bb4be63a3f23413b1bbe13e6bdd06272ef77ded0e2e5bdaca"
SEED = 1729
MAX_OUTPUT_BYTES = 512
ALLOWED_DECISIONS = {"act", "no_action", "clarify", "respond"}
ALLOWED_PRESENTATION = {"none", "text", "media_results", "media_details", "playback_handoff", "confirmation"}


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(b"".join(_json_bytes(row) + b"\n" for row in rows))


def _inside_git_worktree(path: Path) -> bool:
    return any((parent / ".git").exists() for parent in (path, *path.parents))


def _output_path(path: Path) -> Path:
    result = path.expanduser().resolve()
    if _inside_git_worktree(result):
        raise ValueError("controlled data, model, and report must remain outside every Git worktree")
    if ".lumi-data" not in {part.casefold() for part in result.parts}:
        raise ValueError("output directory must be below a controlled .lumi-data directory")
    if result.exists():
        if not result.is_dir() or any(result.iterdir()):
            raise ValueError(f"output directory already exists and is not empty: {result}")
    else:
        result.mkdir(parents=True, exist_ok=False)
    return result


def _output(
    decision: str,
    action: str | None,
    arguments: dict[str, Any],
    requires_clarification: bool,
    message: str,
    presentation_intent: str = "text",
) -> dict[str, Any]:
    return {
        "decision": decision,
        "action": action,
        "arguments": arguments,
        "requires_clarification": requires_clarification,
        "message": message,
        "presentation_intent": presentation_intent,
    }


def _read_frozen_jsonl(path: Path, expected_sha256: str, split: str) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if _inside_git_worktree(resolved):
        raise ValueError("dataset text must remain outside every Git worktree")
    if ".lumi-data" not in {part.casefold() for part in resolved.parts}:
        raise ValueError("dataset files must be below a controlled .lumi-data directory")
    if not resolved.is_file():
        raise ValueError(f"required frozen dataset file is missing: {resolved.name}")
    if _sha256_file(resolved) != expected_sha256:
        raise ValueError(f"{resolved.name} does not match the pinned {split} SHA-256")
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{resolved.name}:{line_number}: blank JSONL row")
            value = _parse_json(line)
            if not isinstance(value, dict):
                raise ValueError(f"{resolved.name}:{line_number}: expected one JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{resolved.name}: dataset split cannot be empty")
    return rows


def load_dataset(dataset_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load only the frozen exploratory packet; sample text is not embedded in this runner."""
    root = dataset_dir.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("dataset directory must exist in controlled local storage")
    if _inside_git_worktree(root):
        raise ValueError("dataset text must remain outside every Git worktree")
    if ".lumi-data" not in {part.casefold() for part in root.parts}:
        raise ValueError("dataset directory must be below a controlled .lumi-data directory")
    train = _read_frozen_jsonl(root / "train.jsonl", TRAIN_INPUT_SHA256, "training")
    dev = _read_frozen_jsonl(root / "exploratory-dev.jsonl", DEV_INPUT_SHA256, "exploratory development")
    combined = b"".join(_json_bytes(row) + b"\n" for row in train + dev)
    if _sha256_bytes(combined) != ALL_INPUTS_SHA256:
        raise ValueError("combined frozen packet does not match its pinned SHA-256")
    return train, dev


def validate_dataset(train: list[dict[str, Any]], dev: list[dict[str, Any]]) -> None:
    if not train or not dev:
        raise ValueError("both train and exploratory development examples are required")
    train_families = {row["family_id"] for row in train}
    dev_families = {row["family_id"] for row in dev}
    if train_families & dev_families:
        raise ValueError("train and exploratory development family IDs overlap")
    ids: set[str] = set()
    texts: set[str] = set()
    for split, rows in (("train", train), ("exploratory_dev", dev)):
        for row in rows:
            if row["split"] != split:
                raise ValueError(f"{row['id']}: unexpected split")
            if row["id"] in ids or row["text"] in texts:
                raise ValueError(f"{row['id']}: duplicate id or input text")
            ids.add(row["id"])
            texts.add(row["text"])
            if not _valid_output(row["gold"]):
                raise ValueError(f"{row['id']}: authored gold output fails the candidate shape contract")
            if row["response"] != _json_bytes(row["gold"]).decode("utf-8"):
                raise ValueError(f"{row['id']}: target serialization is not canonical")


def _valid_output(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    required = {"decision", "action", "arguments", "requires_clarification", "message", "presentation_intent"}
    if set(value) != required:
        return False
    if not isinstance(value["decision"], str) or value["decision"] not in ALLOWED_DECISIONS:
        return False
    if not isinstance(value["presentation_intent"], str) or value["presentation_intent"] not in ALLOWED_PRESENTATION:
        return False
    if not isinstance(value["arguments"], dict) or not isinstance(value["message"], str) or not value["message"].strip():
        return False
    if not isinstance(value["requires_clarification"], bool):
        return False
    if value["decision"] == "act":
        return isinstance(value["action"], str) and bool(value["action"]) and not value["requires_clarification"]
    if value["action"] is not None:
        return False
    if value["decision"] == "clarify":
        return value["requires_clarification"] and not value["arguments"]
    return not value["requires_clarification"] and not value["arguments"]


def _parse_json(text: str) -> Any | None:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=reject_duplicates, parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (json.JSONDecodeError, ValueError):
        return None
    return value


def _load_model(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        parameters = {name: archive[name].copy() for name in _ENGINE.PARAMETER_NAMES}
    return parameters


def _parameter_count(parameters: dict[str, np.ndarray]) -> int:
    return sum(int(value.size) for value in parameters.values())


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _evaluate_one(row: dict[str, Any], generated: str) -> dict[str, Any]:
    parsed = _parse_json(generated)
    syntactically_valid = parsed is not None
    schema_valid = _valid_output(parsed)
    prediction = parsed if schema_valid else None
    gold = row["gold"]
    if prediction is None:
        return {
            "id": row["id"],
            "family_id": row["family_id"],
            "language": row["language"],
            "categories": row["categories"],
            "syntactically_valid_json": syntactically_valid,
            "schema_valid_output": False,
            "exact_output": False,
            "decision_exact": False,
            "action_exact": False,
            "arguments_exact": False,
            "clarification_exact": False,
            "message_exact": False,
            "false_action": None,
            "generated_utf8_bytes": len(generated.encode("utf-8")),
            "target_utf8_bytes": len(row["response"].encode("utf-8")),
            "generated_text": generated,
            "parsed_output": parsed,
        }
    return {
        "id": row["id"],
        "family_id": row["family_id"],
        "language": row["language"],
        "categories": row["categories"],
        "syntactically_valid_json": True,
        "schema_valid_output": True,
        "exact_output": prediction == gold,
        "decision_exact": prediction["decision"] == gold["decision"],
        "action_exact": prediction["action"] == gold["action"],
        "arguments_exact": prediction["arguments"] == gold["arguments"],
        "clarification_exact": prediction["requires_clarification"] == gold["requires_clarification"],
        "message_exact": prediction["message"] == gold["message"],
        "false_action": prediction["decision"] == "act" and gold["decision"] != "act",
        "generated_utf8_bytes": len(generated.encode("utf-8")),
        "target_utf8_bytes": len(row["response"].encode("utf-8")),
        "generated_text": generated,
        "parsed_output": prediction,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    metrics = (
        "syntactically_valid_json", "schema_valid_output", "exact_output", "decision_exact", "action_exact",
        "arguments_exact", "clarification_exact", "message_exact", "false_action",
    )
    result: dict[str, int | float] = {"count": count}
    for metric in metrics:
        scored = [row[metric] for row in rows if row[metric] is not None]
        hits = sum(bool(value) for value in scored)
        result[f"{metric}_scored_count"] = len(scored)
        result[f"{metric}_count"] = hits
        result[f"{metric}_rate"] = (hits / len(scored)) if scored else None
    return result


def _metric_slices(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"language:{row['language']}"].append(row)
        for category in row["categories"]:
            groups[f"category:{category}"].append(row)
    return {name: _summarize(group) for name, group in sorted(groups.items())}


def run(args: argparse.Namespace) -> dict[str, Any]:
    global _ENGINE
    import random_init_conversation_response_smoke as _response_engine

    _ENGINE = _response_engine
    engine_path = ROOT / "experiments" / "random_init_conversation_response_smoke.py"
    engine_sha = _sha256_file(engine_path)
    if engine_sha != RESPONSE_ENGINE_SHA256:
        raise ValueError("the pinned random-init byte-GRU implementation has changed")
    if args.seed != SEED:
        raise ValueError(f"this recorded experiment pins seed {SEED}; use a new experiment ID for another seed")
    if args.epochs < 1 or args.epochs > 120:
        raise ValueError("epochs must be between 1 and 120")
    train_rows, dev_rows = load_dataset(args.dataset_dir)
    validate_dataset(train_rows, dev_rows)
    output_dir = _output_path(args.output_dir)

    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "exploratory-dev.jsonl"
    combined_path = output_dir / "all-examples.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(dev_path, dev_rows)
    _write_jsonl(combined_path, train_rows + dev_rows)

    generation_families = {
        "train": sorted({row["family_id"] for row in train_rows}),
        "exploratory_dev": sorted({row["family_id"] for row in dev_rows}),
    }
    source_manifest = {
        "dataset_id": DATASET_ID,
        "evidence_level": 1,
        "source": "fixed, hand-authored synthetic JSONL retained in controlled local storage",
        "goal_objective_sha256": GOAL_OBJECTIVE_SHA256,
        "generator_path": "historical template-builder source not included in the repository",
        "generator_sha256": HISTORICAL_TEMPLATE_BUILDER_SHA256,
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "local_dataset_directory": args.dataset_dir.name,
        "generation_method": "No samples are generated during this run; the frozen local JSONL is verified against pinned SHA-256 values before use.",
        "seed": args.seed,
        "train_example_count": len(train_rows),
        "exploratory_dev_example_count": len(dev_rows),
        "family_count": {"train": len(generation_families["train"]), "exploratory_dev": len(generation_families["exploratory_dev"])},
        "families": generation_families,
        "train_sha256": TRAIN_INPUT_SHA256,
        "exploratory_dev_sha256": DEV_INPUT_SHA256,
        "all_examples_sha256": ALL_INPUTS_SHA256,
        "data_use": "local exploratory training and one exploratory development measurement only; not formal evaluation, tokenizer fitting, filtering, or release evidence",
        "distribution_approval": False,
        "independent_review": "none; examples and Japanese/code-switch gold wording are not human-reviewed",
        "external_datasets": [],
        "pretrained_models_or_weights": [],
        "excluded_data_assertions": {
            "101_case_codex_draft_opened": False,
            "101_case_draft_or_derivatives_used": False,
            "jecs_used": False,
            "jmultiwoz_used": False,
            "massive_used": False,
            "llm_teacher_used": False,
        },
    }
    _write_json(output_dir / "source-manifest.json", source_manifest)

    training_started = time.perf_counter()
    parameters, training = _ENGINE.fit(
        train_rows,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=0.01,
    )
    training_seconds = time.perf_counter() - training_started
    artifact_path = output_dir / "lumi-unified-synthetic-smoke.npz"
    np.savez_compressed(artifact_path, **parameters)
    parameter_state_sha = _ENGINE._parameter_state_sha256(parameters)
    if parameter_state_sha != _ENGINE._parameter_state_sha256(_load_model(artifact_path)):
        raise RuntimeError("saved artifact parameters did not reload exactly")

    rss_before_load = _ENGINE._working_set_bytes()
    load_started = time.perf_counter()
    reloaded = _load_model(artifact_path)
    load_ms = (time.perf_counter() - load_started) * 1000.0
    rss_after_load = _ENGINE._working_set_bytes()

    generation_rows: list[dict[str, Any]] = []
    warm_latencies: list[float] = []
    first_inference_ms: float | None = None
    inference_start = time.perf_counter()
    for index, row in enumerate(dev_rows):
        started = time.perf_counter()
        raw, ended_with_eos, valid_utf8 = _ENGINE.generate(reloaded, row["text"], MAX_OUTPUT_BYTES)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if index == 0:
            first_inference_ms = latency_ms
        else:
            warm_latencies.append(latency_ms)
        evaluation = _evaluate_one(row, raw)
        evaluation["ended_with_eos"] = ended_with_eos
        evaluation["valid_utf8"] = valid_utf8
        evaluation["latency_ms"] = latency_ms
        generation_rows.append(evaluation)
    total_inference_seconds = time.perf_counter() - inference_start

    train_generations: list[dict[str, Any]] = []
    for row in train_rows:
        raw, ended_with_eos, valid_utf8 = _ENGINE.generate(reloaded, row["text"], MAX_OUTPUT_BYTES)
        result = _evaluate_one(row, raw)
        result["ended_with_eos"] = ended_with_eos
        result["valid_utf8"] = valid_utf8
        train_generations.append(result)

    dev_output_counts = Counter(row["generated_text"] for row in generation_rows)
    dominant_dev_output, dominant_dev_output_count = dev_output_counts.most_common(1)[0]

    _write_jsonl(output_dir / "exploratory-dev-predictions.jsonl", generation_rows)
    _write_jsonl(output_dir / "train-replay-predictions.jsonl", train_generations)
    report = {
        "experiment_id": DATASET_ID,
        "evidence_level": 1,
        "base_commit": BASE_COMMIT,
        "generator_sha256": source_manifest["generator_sha256"],
        "goal_objective_sha256": GOAL_OBJECTIVE_SHA256,
        "response_engine_path": "experiments/random_init_conversation_response_smoke.py",
        "response_engine_sha256": engine_sha,
        "source_manifest_sha256": _sha256_file(output_dir / "source-manifest.json"),
        "input_hashes": {
            "train": source_manifest["train_sha256"],
            "exploratory_dev": source_manifest["exploratory_dev_sha256"],
            "all_examples": source_manifest["all_examples_sha256"],
        },
        "model": {
            "architecture": "single-layer tanh recurrent byte-level encoder-decoder; shared learned UTF-8 byte embedding; model generates one canonical JSON object containing decision, action, arguments, clarification flag, presentation intent, and user-facing message",
            "initialization": "NumPy seeded Gaussian random initialization; no pretrained weights, checkpoint, adapter, external embedding, or teacher output",
            "seed": args.seed,
            "epochs": args.epochs,
            "optimizer": "Adam, learning rate 0.01, one shuffled update per training example per epoch, global gradient norm clipped to 1.0",
            "embedding_dimension": _ENGINE.EMBED_DIM,
            "hidden_dimension": _ENGINE.HIDDEN_DIM,
            "input_vocabulary_size": _ENGINE.INPUT_VOCAB_SIZE,
            "output_vocabulary_size": _ENGINE.OUTPUT_VOCAB_SIZE,
            "maximum_generated_bytes": MAX_OUTPUT_BYTES,
            "parameter_count": _parameter_count(parameters),
            "cpu_threads": 1,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or None,
            "gpu_used": False,
        },
        "data": {
            "train_examples": len(train_rows),
            "dataset_text_storage": "controlled external .lumi-data; no input text is embedded in tracked code",
            "exploratory_dev_examples": len(dev_rows),
            "train_families": len(generation_families["train"]),
            "exploratory_dev_families": len(generation_families["exploratory_dev"]),
            "family_ids_disjoint": not (set(generation_families["train"]) & set(generation_families["exploratory_dev"])),
            "train_languages": dict(sorted(Counter(row["language"] for row in train_rows).items())),
            "exploratory_dev_languages": dict(sorted(Counter(row["language"] for row in dev_rows).items())),
            "train_rows_by_category": dict(sorted(Counter(category for row in train_rows for category in row["categories"]).items())),
            "exploratory_dev_rows_by_category": dict(sorted(Counter(category for row in dev_rows for category in row["categories"]).items())),
        },
        "training": {
            "initial_mean_byte_cross_entropy": training["initial_loss"],
            "final_mean_byte_cross_entropy": training["final_loss"],
            "loss_delta": training["initial_loss"] - training["final_loss"],
            "loss_by_epoch": training["loss_by_epoch"],
            "updates": training["updates"],
            "wall_seconds": training_seconds,
        },
        "artifact": {
            "path": artifact_path.name,
            "sha256": _sha256_file(artifact_path),
            "compressed_bytes": artifact_path.stat().st_size,
            "parameter_state_sha256": parameter_state_sha,
            "reload_parameter_state_matches": True,
            "reload_ms": load_ms,
        },
        "inference": {
            "device": "cpu",
            "first_exploratory_dev_generation_ms": first_inference_ms,
            "warm_exploratory_dev_generation_p50_ms": statistics.median(warm_latencies) if warm_latencies else None,
            "warm_exploratory_dev_generation_p95_ms": _percentile(warm_latencies, 95),
            "all_exploratory_dev_generation_seconds": total_inference_seconds,
            "maximum_output_bytes": MAX_OUTPUT_BYTES,
            "process_working_set_before_reload_bytes": rss_before_load,
            "process_working_set_after_reload_bytes": rss_after_load,
            "process_peak_working_set_bytes": _ENGINE._working_set_bytes(peak=True),
        },
        "exploratory_dev": {
            "summary": _summarize(generation_rows),
            "slices": _metric_slices(generation_rows),
            "valid_utf8_count": sum(bool(row["valid_utf8"]) for row in generation_rows),
            "eos_count": sum(bool(row["ended_with_eos"]) for row in generation_rows),
        },
        "train_replay": {
            "summary": _summarize(train_generations),
            "valid_utf8_count": sum(bool(row["valid_utf8"]) for row in train_generations),
            "eos_count": sum(bool(row["ended_with_eos"]) for row in train_generations),
        },
        "free_run_diagnostics": {
            "train_replay_unique_outputs": len({row["generated_text"] for row in train_generations}),
            "exploratory_dev_unique_outputs": len(dev_output_counts),
            "modal_exploratory_dev_output": dominant_dev_output,
            "modal_exploratory_dev_output_count": dominant_dev_output_count,
            "mean_exploratory_dev_generated_utf8_bytes": statistics.mean(row["generated_utf8_bytes"] for row in generation_rows),
            "mean_exploratory_dev_target_utf8_bytes": statistics.mean(row["target_utf8_bytes"] for row in generation_rows),
            "premature_eos_count_before_maximum_output_bytes": sum(bool(row["ended_with_eos"]) for row in generation_rows),
        },
        "limitations": [
            "All 84 examples are hand-authored synthetic templates from one current user goal and remain outside Git; they are not independent evidence of natural language coverage.",
            "The exploratory development examples are family-disjoint but authored by the same generator and not human reviewed; they are not formal evaluation cases.",
            "Japanese and code-switch wording has no qualified bilingual review, so no naturalness or language-quality claim is supported.",
            "A 32-unit recurrent byte model and tiny dataset are an engineering smoke only; no quality threshold, useful generalization, grounding, multi-turn behavior, or low false-action target is established.",
            "Action proposals are inert predictions; the run does not call ZenStream, access trusted library state, or mutate media state.",
            "Only one CPU and one seed were measured; cold process start, idle/unload behavior, other CPU classes, production traffic, GPU, and artifact distribution rights remain unmeasured or unresolved.",
            "The excluded 101-case Codex draft and every derivative were not opened, loaded, tokenized, derived from, or evaluated on; JECS, JMultiWOZ, and MASSIVE were not used.",
        ],
    }
    _write_json(output_dir / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="controlled .lumi-data directory with the exact frozen train/dev JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="new controlled .lumi-data output directory")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "training_loss": [
            report["training"]["initial_mean_byte_cross_entropy"],
            report["training"]["final_mean_byte_cross_entropy"],
        ],
        "parameters": report["model"]["parameter_count"],
        "artifact_bytes": report["artifact"]["compressed_bytes"],
        "exploratory_dev": report["exploratory_dev"]["summary"],
        "report": str((args.output_dir.resolve() / "report.json")),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
