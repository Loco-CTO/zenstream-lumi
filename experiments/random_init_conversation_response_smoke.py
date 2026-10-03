#!/usr/bin/env python3
"""Train a tiny random-init byte-level conversation responder as an engineering smoke."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BYTE_COUNT = 256
EOS = 256
BOS = 257
INPUT_VOCAB_SIZE = 258
OUTPUT_VOCAB_SIZE = 257
EMBED_DIM = 24
HIDDEN_DIM = 32
PARAMETER_NAMES = (
    "embedding",
    "encoder_input",
    "encoder_recurrent",
    "encoder_bias",
    "decoder_input",
    "decoder_recurrent",
    "decoder_bias",
    "output_weight",
    "output_bias",
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _parameter_state_sha256(parameters: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in PARAMETER_NAMES:
        digest.update(name.encode("ascii") + b"\0")
        digest.update(np.ascontiguousarray(parameters[name]).tobytes())
    return "sha256:" + digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is not allowed")
        result[key] = value
    return result


def _inside_git_worktree(path: Path) -> bool:
    return any((parent / ".git").exists() for parent in (path, *path.parents))


def read_dataset(path: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    data_path = path.expanduser().resolve()
    manifest_file = manifest_path.expanduser().resolve()
    if _inside_git_worktree(data_path) or _inside_git_worktree(manifest_file):
        raise ValueError("controlled examples and their manifest must remain outside every Git worktree")
    if data_path == manifest_file:
        raise ValueError("the examples and source manifest must be separate files")

    try:
        manifest = json.loads(
            manifest_file.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"could not read a strict source manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("source manifest must be a JSON object")
    if manifest.get("dataset_id") != "lumi-user-brief-conversation-response-smoke-v1":
        raise ValueError("source manifest has an unexpected dataset_id")
    if manifest.get("permitted_scope") != (
        "local exploratory training and development-only smoke evaluation for this user-requested project"
    ):
        raise ValueError("source manifest does not declare the scoped exploratory use")
    if manifest.get("release_or_distribution_approval") is not False:
        raise ValueError("this smoke must not treat the local source as approved for distribution")
    if manifest.get("data_sha256") != _sha256(data_path):
        raise ValueError("source manifest data_sha256 does not match the examples file")

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    try:
        with data_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(
                        line,
                        parse_constant=_reject_constant,
                        object_pairs_hook=_reject_duplicate_keys,
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"{data_path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{data_path}:{line_number}: each row must be an object")
                if set(row) != {"id", "family_id", "split", "language", "text", "response"}:
                    raise ValueError(f"{data_path}:{line_number}: unexpected row fields")
                for field in ("id", "family_id", "text", "response"):
                    if not isinstance(row[field], str) or not row[field].strip():
                        raise ValueError(f"{data_path}:{line_number}: {field} must be nonempty text")
                if row["split"] not in {"train", "dev"}:
                    raise ValueError(f"{data_path}:{line_number}: split must be train or dev")
                if row["language"] != "en":
                    raise ValueError(
                        f"{data_path}:{line_number}: this English-only smoke cannot claim Japanese coverage"
                    )
                if row["id"] in seen_ids:
                    raise ValueError(f"{data_path}:{line_number}: duplicate id {row['id']!r}")
                seen_ids.add(row["id"])
                rows.append(row)
    except UnicodeDecodeError as exc:
        raise ValueError(f"{data_path}: examples must be UTF-8") from exc

    train_rows = [row for row in rows if row["split"] == "train"]
    dev_rows = [row for row in rows if row["split"] == "dev"]
    if not train_rows or not dev_rows:
        raise ValueError("both train and dev examples are required")
    train_families = {row["family_id"] for row in train_rows}
    dev_families = {row["family_id"] for row in dev_rows}
    if train_families & dev_families:
        raise ValueError("train and dev semantic family IDs must be disjoint")
    if manifest.get("training_examples") != len(train_rows):
        raise ValueError("source manifest training_examples does not match the dataset")
    if manifest.get("development_examples") != len(dev_rows):
        raise ValueError("source manifest development_examples does not match the dataset")
    return train_rows, dev_rows, manifest


def _initialize(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "embedding": rng.normal(0.0, 0.05, (INPUT_VOCAB_SIZE, EMBED_DIM)).astype(np.float32),
        "encoder_input": rng.normal(
            0.0, math.sqrt(2.0 / (EMBED_DIM + HIDDEN_DIM)), (EMBED_DIM, HIDDEN_DIM)
        ).astype(np.float32),
        "encoder_recurrent": rng.normal(
            0.0, math.sqrt(1.0 / HIDDEN_DIM), (HIDDEN_DIM, HIDDEN_DIM)
        ).astype(np.float32),
        "encoder_bias": np.zeros(HIDDEN_DIM, dtype=np.float32),
        "decoder_input": rng.normal(
            0.0, math.sqrt(2.0 / (EMBED_DIM + HIDDEN_DIM)), (EMBED_DIM, HIDDEN_DIM)
        ).astype(np.float32),
        "decoder_recurrent": rng.normal(
            0.0, math.sqrt(1.0 / HIDDEN_DIM), (HIDDEN_DIM, HIDDEN_DIM)
        ).astype(np.float32),
        "decoder_bias": np.zeros(HIDDEN_DIM, dtype=np.float32),
        "output_weight": rng.normal(
            0.0, math.sqrt(1.0 / HIDDEN_DIM), (HIDDEN_DIM, OUTPUT_VOCAB_SIZE)
        ).astype(np.float32),
        "output_bias": np.zeros(OUTPUT_VOCAB_SIZE, dtype=np.float32),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _encoded_prompt(parameters: dict[str, np.ndarray], text: str) -> tuple[np.ndarray, list[Any]]:
    tokens = [BOS, *text.encode("utf-8"), EOS]
    hidden = np.zeros(HIDDEN_DIM, dtype=np.float32)
    cache: list[Any] = []
    for token in tokens:
        previous = hidden
        preactivation = (
            parameters["embedding"][token] @ parameters["encoder_input"]
            + previous @ parameters["encoder_recurrent"]
            + parameters["encoder_bias"]
        )
        hidden = np.tanh(preactivation).astype(np.float32)
        cache.append((token, previous, hidden))
    return hidden, cache


def _example_loss_and_gradients(
    parameters: dict[str, np.ndarray], row: dict[str, Any], *, gradients: bool
) -> tuple[float, dict[str, np.ndarray] | None]:
    encoded, encoder_cache = _encoded_prompt(parameters, row["text"])
    target = [*row["response"].encode("utf-8"), EOS]
    hidden = encoded
    decoder_cache: list[Any] = []
    loss = 0.0

    for position, target_token in enumerate(target):
        input_token = BOS if position == 0 else target[position - 1]
        previous = hidden
        preactivation = (
            parameters["embedding"][input_token] @ parameters["decoder_input"]
            + previous @ parameters["decoder_recurrent"]
            + parameters["decoder_bias"]
        )
        hidden = np.tanh(preactivation).astype(np.float32)
        probabilities = _softmax(
            hidden @ parameters["output_weight"] + parameters["output_bias"]
        )
        loss -= math.log(max(float(probabilities[target_token]), 1e-12))
        if gradients:
            decoder_cache.append((input_token, previous, hidden, probabilities, target_token))

    loss /= len(target)
    if not gradients:
        return loss, None

    grads = {name: np.zeros_like(parameters[name]) for name in PARAMETER_NAMES}
    decoder_hidden_gradient = np.zeros(HIDDEN_DIM, dtype=np.float32)
    inverse_target_length = 1.0 / len(target)
    for input_token, previous, current, probabilities, target_token in reversed(decoder_cache):
        output_gradient = probabilities.copy()
        output_gradient[target_token] -= 1.0
        output_gradient *= inverse_target_length
        grads["output_weight"] += np.outer(current, output_gradient)
        grads["output_bias"] += output_gradient

        hidden_gradient = (
            output_gradient @ parameters["output_weight"].T + decoder_hidden_gradient
        )
        preactivation_gradient = hidden_gradient * (1.0 - current * current)
        grads["embedding"][input_token] += (
            preactivation_gradient @ parameters["decoder_input"].T
        )
        grads["decoder_input"] += np.outer(
            parameters["embedding"][input_token], preactivation_gradient
        )
        grads["decoder_recurrent"] += np.outer(previous, preactivation_gradient)
        grads["decoder_bias"] += preactivation_gradient
        decoder_hidden_gradient = (
            preactivation_gradient @ parameters["decoder_recurrent"].T
        )

    encoder_hidden_gradient = decoder_hidden_gradient
    for token, previous, current in reversed(encoder_cache):
        preactivation_gradient = encoder_hidden_gradient * (1.0 - current * current)
        grads["embedding"][token] += preactivation_gradient @ parameters["encoder_input"].T
        grads["encoder_input"] += np.outer(
            parameters["embedding"][token], preactivation_gradient
        )
        grads["encoder_recurrent"] += np.outer(previous, preactivation_gradient)
        grads["encoder_bias"] += preactivation_gradient
        encoder_hidden_gradient = (
            preactivation_gradient @ parameters["encoder_recurrent"].T
        )

    return loss, grads


def _dataset_loss(parameters: dict[str, np.ndarray], rows: list[dict[str, Any]]) -> float:
    losses = [
        _example_loss_and_gradients(parameters, row, gradients=False)[0]
        for row in rows
    ]
    return float(np.mean(losses))


class Adam:
    def __init__(self, parameters: dict[str, np.ndarray]) -> None:
        self.first = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.second = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.step_number = 0

    def step(
        self,
        parameters: dict[str, np.ndarray],
        gradients: dict[str, np.ndarray],
        learning_rate: float,
    ) -> None:
        self.step_number += 1
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        correction1 = 1.0 - beta1**self.step_number
        correction2 = 1.0 - beta2**self.step_number
        for name in PARAMETER_NAMES:
            self.first[name] *= beta1
            self.first[name] += (1.0 - beta1) * gradients[name]
            self.second[name] *= beta2
            self.second[name] += (1.0 - beta2) * gradients[name] ** 2
            first_hat = self.first[name] / correction1
            second_hat = self.second[name] / correction2
            parameters[name] -= learning_rate * first_hat / (
                np.sqrt(second_hat) + epsilon
            )


def fit(
    rows: list[dict[str, Any]], seed: int, epochs: int, learning_rate: float
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    parameters = _initialize(seed)
    initial_loss = _dataset_loss(parameters, rows)
    optimizer = Adam(parameters)
    rng = np.random.default_rng(seed + 1)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(rows))
        for index in order:
            _, gradients = _example_loss_and_gradients(
                parameters, rows[int(index)], gradients=True
            )
            assert gradients is not None
            norm = math.sqrt(
                sum(float(np.sum(value * value)) for value in gradients.values())
            )
            if norm > 1.0:
                scale = 1.0 / norm
                for name in PARAMETER_NAMES:
                    gradients[name] *= scale
            optimizer.step(parameters, gradients, learning_rate)
        history.append({"epoch": epoch, "training_loss": _dataset_loss(parameters, rows)})

    final_loss = _dataset_loss(parameters, rows)
    return parameters, {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "epochs": epochs,
        "updates": optimizer.step_number,
        "loss_by_epoch": history,
    }


def generate(
    parameters: dict[str, np.ndarray], prompt: str, max_output_bytes: int
) -> tuple[str, bool, bool]:
    hidden, _ = _encoded_prompt(parameters, prompt)
    input_token = BOS
    output = bytearray()
    ended_with_eos = False
    for _ in range(max_output_bytes):
        hidden = np.tanh(
            parameters["embedding"][input_token] @ parameters["decoder_input"]
            + hidden @ parameters["decoder_recurrent"]
            + parameters["decoder_bias"]
        ).astype(np.float32)
        probabilities = _softmax(
            hidden @ parameters["output_weight"] + parameters["output_bias"]
        )
        token = int(np.argmax(probabilities))
        if token == EOS:
            ended_with_eos = True
            break
        output.append(token)
        input_token = token
    try:
        decoded = bytes(output).decode("utf-8")
        valid_utf8 = True
    except UnicodeDecodeError:
        decoded = bytes(output).decode("utf-8", errors="replace")
        valid_utf8 = False
    return decoded, ended_with_eos, valid_utf8


def _working_set_bytes(peak: bool = False) -> int | None:
    if sys.platform != "win32":
        return None
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    process = kernel32.GetCurrentProcess()
    success = psapi.GetProcessMemoryInfo(
        process, ctypes.byref(counters), wintypes.DWORD(counters.cb)
    )
    if not success:
        return None
    return int(counters.PeakWorkingSetSize if peak else counters.WorkingSetSize)


def _resolve_output_path(path: Path, data_path: Path, manifest_path: Path) -> Path:
    output = path.expanduser().resolve()
    if _inside_git_worktree(output):
        raise ValueError("output artifacts must stay outside every Git worktree")
    if output in {data_path.resolve(), manifest_path.resolve()}:
        raise ValueError("output directory cannot replace a controlled input file")
    if output.exists():
        if not output.is_dir():
            raise ValueError("output path must be a directory")
        if any(output.iterdir()):
            raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value) + b"\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_path = args.data.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    train_rows, dev_rows, source_manifest = read_dataset(data_path, manifest_path)
    output_dir = _resolve_output_path(args.output_dir, data_path, manifest_path)

    started = time.perf_counter()
    parameters, training = fit(
        train_rows, args.seed, args.epochs, args.learning_rate
    )
    training_seconds = time.perf_counter() - started
    artifact_path = output_dir / "lumi-random-init-conversation.npz"
    np.savez_compressed(artifact_path, **parameters)
    metadata = {
        "model_id": "lumi-random-init-conversation-response-smoke-v1",
        "architecture": (
            "single-layer tanh recurrent byte-level encoder-decoder with shared input "
            "embedding and autoregressive byte softmax"
        ),
        "initialization": "Gaussian random initialization; no external weights or embeddings",
        "seed": args.seed,
        "embedding_dimension": EMBED_DIM,
        "hidden_dimension": HIDDEN_DIM,
        "input_vocabulary_size": INPUT_VOCAB_SIZE,
        "output_vocabulary_size": OUTPUT_VOCAB_SIZE,
        "special_tokens": {"eos": EOS, "bos": BOS},
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "cpu_thread_limit": 1,
    }
    _write_json(artifact_path.with_suffix(".json"), metadata)

    memory_before_reload = _working_set_bytes()
    load_started = time.perf_counter()
    with np.load(artifact_path, allow_pickle=False) as archive:
        reloaded = {name: archive[name].copy() for name in PARAMETER_NAMES}
    load_ms = (time.perf_counter() - load_started) * 1000.0
    memory_after_reload = _working_set_bytes()

    first_started = time.perf_counter()
    first_generation = generate(
        reloaded, dev_rows[0]["text"], args.max_output_bytes
    )
    first_generation_ms = (time.perf_counter() - first_started) * 1000.0

    generated_rows: list[dict[str, Any]] = []
    for row in dev_rows:
        response, ended_with_eos, valid_utf8 = generate(
            reloaded, row["text"], args.max_output_bytes
        )
        generated_rows.append(
            {
                "id": row["id"],
                "family_id": row["family_id"],
                "split": row["split"],
                "language": row["language"],
                "text": row["text"],
                "reference_response": row["response"],
                "generated_response": response,
                "nonempty_response": bool(response.strip()),
                "ended_with_eos": ended_with_eos,
                "valid_utf8": valid_utf8,
            }
        )

    warm_times: list[float] = []
    for _ in range(args.latency_repeats):
        for row in dev_rows:
            call_started = time.perf_counter()
            generate(reloaded, row["text"], args.max_output_bytes)
            warm_times.append((time.perf_counter() - call_started) * 1000.0)
    ordered_times = sorted(warm_times)
    p95_index = max(0, math.ceil(0.95 * len(ordered_times)) - 1)
    parameter_count = sum(int(value.size) for value in parameters.values())
    report = {
        "evidence_level": "exploratory",
        "model_id": metadata["model_id"],
        "random_initialization": True,
        "pretrained_artifacts_used": False,
        "source_dataset_id": source_manifest["dataset_id"],
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "training_data_path": str(data_path),
        "training_data_sha256": _sha256(data_path),
        "training_example_count": len(train_rows),
        "development_example_count": len(dev_rows),
        "training_families": sorted({row["family_id"] for row in train_rows}),
        "development_families": sorted({row["family_id"] for row in dev_rows}),
        "training_utf8_byte_count": sum(
            len(row["text"].encode("utf-8")) + len(row["response"].encode("utf-8"))
            for row in train_rows
        ),
        "seed": args.seed,
        "parameter_count": parameter_count,
        "parameter_state_sha256": _parameter_state_sha256(parameters),
        "training": {
            key: value for key, value in training.items() if key != "loss_by_epoch"
        },
        "training_seconds": training_seconds,
        "development_generation": {
            "case_count": len(generated_rows),
            "nonempty_response_count": sum(
                row["nonempty_response"] for row in generated_rows
            ),
            "valid_utf8_count": sum(row["valid_utf8"] for row in generated_rows),
            "ended_with_eos_count": sum(row["ended_with_eos"] for row in generated_rows),
            "human_quality_reviewed": False,
            "exact_string_match_used_as_quality_score": False,
            "predictions": generated_rows,
        },
        "cpu_inference": {
            "numpy_cpu_only": True,
            "gpu_used": False,
            "load_ms": load_ms,
            "first_development_generation_ms": first_generation_ms,
            "first_reloaded_generation_nonempty": bool(first_generation[0].strip()),
            "warm_latency_p50_ms": statistics.median(warm_times),
            "warm_latency_p95_ms": ordered_times[p95_index],
            "warm_latency_samples": len(warm_times),
        },
        "memory": {
            "working_set_before_reload_bytes": memory_before_reload,
            "working_set_after_reload_bytes": memory_after_reload,
            "peak_working_set_bytes": _working_set_bytes(peak=True),
            "scope_note": (
                "Windows process working set includes the Python and NumPy runtime; it is "
                "not a standalone model-memory estimate."
            ),
        },
        "artifact": {
            "model_path": str(artifact_path),
            "metadata_path": str(artifact_path.with_suffix(".json")),
            "model_bytes": artifact_path.stat().st_size,
            "model_sha256": _sha256(artifact_path),
        },
        "limitations": [
            "This is an English-only sequence-generation smoke, not a selected Lumi architecture.",
            "The four training and two development examples are illustrative user-provided examples, not a benchmark or product training corpus.",
            "Development generations were not human-reviewed; nonempty text and UTF-8 validity are not measures of conversational quality.",
            "The exact illustrative wording must not be treated as a fixed response template or memorized product solution.",
            "Japanese, code-switching, multi-turn context, structured intent, tool integration, and grounding were not exercised by this component.",
            "The candidate is not integrated with the separate intent smoke or ZenStream and cannot execute actions.",
            "The process was measured after training; service cold start, idle use, unload, wake, and other CPU classes were not measured.",
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
    with (output_dir / "development-generations.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in generated_rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                + "\n"
            )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="controlled UTF-8 JSONL outside all Git worktrees")
    parser.add_argument("--manifest", type=Path, required=True, help="local source and use-scope manifest")
    parser.add_argument("--output-dir", type=Path, required=True, help="empty output directory outside all Git worktrees")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--latency-repeats", type=int, default=20)
    parser.add_argument("--max-output-bytes", type=int, default=240)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.latency_repeats < 1 or args.max_output_bytes < 1:
        parser.error("epochs, latency-repeats, and max-output-bytes must be positive")
    try:
        report = run(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": "complete",
                "evidence_level": report["evidence_level"],
                "training_examples": report["training_example_count"],
                "development_examples": report["development_example_count"],
                "training_loss": [
                    report["training"]["initial_loss"],
                    report["training"]["final_loss"],
                ],
                "development_nonempty": [
                    report["development_generation"]["nonempty_response_count"],
                    report["development_generation"]["case_count"],
                ],
                "model_parameters": report["parameter_count"],
                "model_bytes": report["artifact"]["model_bytes"],
                "report_path": str(args.output_dir.resolve() / "report.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
