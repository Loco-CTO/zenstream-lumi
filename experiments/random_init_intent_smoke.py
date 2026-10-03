#!/usr/bin/env python3
"""Train and measure a tiny random-initialized structured-intent baseline."""

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
FEATURE_DIM = 2048
INTENT_HIDDEN = 48
SLOT_HIDDEN = 32
SLOT_LABELS = ("O", "B-TITLE", "I-TITLE")
SCHEMA_REQUIRED = {"decision", "action", "arguments", "requires_clarification"}
SCHEMA_OPTIONAL = {"message", "presentation_intent"}
TITLE_CONFIDENCE = 0.45


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


def _hash_feature(value: str) -> tuple[int, float]:
    digest = hashlib.blake2b(
        value.encode("utf-8"), digest_size=8, person=b"lumi-feature-v1"
    ).digest()
    index = int.from_bytes(digest[:4], "little") % FEATURE_DIM
    sign = 1.0 if digest[4] & 1 else -1.0
    return index, sign


def _add_feature(vector: np.ndarray, feature: str, weight: float = 1.0) -> None:
    index, sign = _hash_feature(feature)
    vector[index] += sign * weight


def _unit_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm:
        vector /= norm
    return vector


def intent_features(text: str) -> np.ndarray:
    normalized = text.casefold()
    bounded = f"^{normalized}$"
    vector = np.zeros(FEATURE_DIM, dtype=np.float32)
    for width in (1, 2, 3, 4):
        for start in range(max(0, len(bounded) - width + 1)):
            _add_feature(vector, f"intent:{width}:{bounded[start:start + width]}")
    return _unit_normalize(vector)


def slot_features(text: str) -> np.ndarray:
    normalized = text.casefold()
    padded = "\u0002\u0002\u0002" + normalized + "\u0003\u0003\u0003"
    rows = np.zeros((len(text), FEATURE_DIM), dtype=np.float32)
    for position in range(len(text)):
        center = position + 3
        for offset in range(-3, 4):
            _add_feature(rows[position], f"slot-char:{offset}:{padded[center + offset]}")
        for start_offset in (-2, -1, 0):
            start = center + start_offset
            for width in (2, 3):
                _add_feature(
                    rows[position],
                    f"slot-ngram:{start_offset}:{width}:{padded[start:start + width]}",
                )
    for row in rows:
        _unit_normalize(row)
    return rows


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _cross_entropy(
    probabilities: np.ndarray, labels: np.ndarray, sample_weights: np.ndarray | None
) -> float:
    selected = np.clip(probabilities[np.arange(len(labels)), labels], 1e-9, 1.0)
    losses = -np.log(selected)
    if sample_weights is None:
        return float(np.mean(losses))
    return float(np.sum(losses * sample_weights) / np.sum(sample_weights))


def _initial_parameters(
    input_dim: int, hidden_dim: int, output_dim: int, seed: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.normal(
            0.0, math.sqrt(2.0 / input_dim), (input_dim, hidden_dim)
        ).astype(np.float32),
        "b1": np.zeros(hidden_dim, dtype=np.float32),
        "w2": rng.normal(
            0.0, math.sqrt(1.0 / hidden_dim), (hidden_dim, output_dim)
        ).astype(np.float32),
        "b2": np.zeros(output_dim, dtype=np.float32),
    }


def _forward(
    parameters: dict[str, np.ndarray], features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preactivation = features @ parameters["w1"] + parameters["b1"]
    hidden = np.maximum(preactivation, 0.0)
    logits = hidden @ parameters["w2"] + parameters["b2"]
    return preactivation, hidden, _softmax(logits)


def _loss(
    parameters: dict[str, np.ndarray],
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray | None,
) -> float:
    return _cross_entropy(_forward(parameters, features)[2], labels, sample_weights)


def fit_mlp(
    features: np.ndarray,
    labels: np.ndarray,
    output_dim: int,
    seed: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    class_weights: tuple[float, ...] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if len(features) == 0 or len(labels) != len(features):
        raise ValueError("training features and labels must have matching nonzero rows")
    parameters = _initial_parameters(features.shape[1], hidden_dim, output_dim, seed)
    sample_weights = (
        np.asarray(class_weights, dtype=np.float32)[labels]
        if class_weights is not None
        else None
    )
    initial_loss = _loss(parameters, features, labels, sample_weights)
    first_moment = {key: np.zeros_like(value) for key, value in parameters.items()}
    second_moment = {key: np.zeros_like(value) for key, value in parameters.items()}
    rng = np.random.default_rng(seed + 1)
    history: list[float] = []
    step = 0
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8

    for _epoch in range(epochs):
        order = rng.permutation(len(labels))
        for start in range(0, len(order), batch_size):
            batch_indices = order[start:start + batch_size]
            batch_x = features[batch_indices]
            batch_y = labels[batch_indices]
            batch_weights = (
                sample_weights[batch_indices] if sample_weights is not None else None
            )
            preactivation, hidden, probabilities = _forward(parameters, batch_x)
            delta = probabilities
            delta[np.arange(len(batch_y)), batch_y] -= 1.0
            if batch_weights is None:
                delta /= len(batch_y)
            else:
                delta *= batch_weights[:, None] / float(np.sum(batch_weights))
            grad_w2 = hidden.T @ delta
            grad_b2 = np.sum(delta, axis=0)
            grad_hidden = delta @ parameters["w2"].T
            grad_preactivation = grad_hidden * (preactivation > 0.0)
            grad_w1 = batch_x.T @ grad_preactivation
            grad_b1 = np.sum(grad_preactivation, axis=0)
            gradients = {
                "w1": grad_w1,
                "b1": grad_b1,
                "w2": grad_w2,
                "b2": grad_b2,
            }
            step += 1
            for key, gradient in gradients.items():
                first_moment[key] = beta1 * first_moment[key] + (1.0 - beta1) * gradient
                second_moment[key] = (
                    beta2 * second_moment[key] + (1.0 - beta2) * gradient * gradient
                )
                corrected_first = first_moment[key] / (1.0 - beta1**step)
                corrected_second = second_moment[key] / (1.0 - beta2**step)
                parameters[key] -= learning_rate * corrected_first / (
                    np.sqrt(corrected_second) + epsilon
                )
        history.append(_loss(parameters, features, labels, sample_weights))

    return parameters, {
        "initial_loss": initial_loss,
        "final_loss": history[-1],
        "epochs": epochs,
        "loss_by_epoch": history,
        "training_accuracy": float(
            np.mean(np.argmax(_forward(parameters, features)[2], axis=1) == labels)
        ),
    }


def _label_for_gold(gold: dict[str, Any]) -> str:
    if gold["decision"] == "act":
        return "act:" + gold["action"]
    return gold["decision"]


def _validate_gold_shape(gold: Any, row_id: str) -> dict[str, Any]:
    if not isinstance(gold, dict) or set(gold) - (SCHEMA_REQUIRED | SCHEMA_OPTIONAL):
        raise ValueError(f"{row_id}: gold output is not a supported structured record")
    if SCHEMA_REQUIRED - gold.keys():
        raise ValueError(f"{row_id}: gold output is missing required fields")
    if gold["decision"] not in {"act", "no_action", "clarify", "respond"}:
        raise ValueError(f"{row_id}: unsupported decision")
    if not isinstance(gold["arguments"], dict):
        raise ValueError(f"{row_id}: arguments must be an object")
    if not isinstance(gold["requires_clarification"], bool):
        raise ValueError(f"{row_id}: requires_clarification must be boolean")
    if gold["decision"] == "act":
        if not isinstance(gold["action"], str) or not gold["action"]:
            raise ValueError(f"{row_id}: act needs a nonempty action")
        if gold["requires_clarification"]:
            raise ValueError(f"{row_id}: act cannot require clarification")
    else:
        if gold["action"] is not None:
            raise ValueError(f"{row_id}: non-act decision must have a null action")
        if gold["decision"] == "clarify":
            if not gold["requires_clarification"] or gold["arguments"]:
                raise ValueError(f"{row_id}: clarify needs empty arguments and a true flag")
        elif gold["requires_clarification"] or gold["arguments"]:
            raise ValueError(f"{row_id}: non-action decision needs empty arguments")
    return gold


def read_dataset(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id or row_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: id must be nonempty and unique")
            if row.get("split") not in {"train", "dev"}:
                raise ValueError(f"{row_id}: split must be train or dev")
            if not isinstance(row.get("family_id"), str) or not row["family_id"]:
                raise ValueError(f"{row_id}: family_id is required")
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"{row_id}: nonempty text is required")
            row["gold"] = _validate_gold_shape(row.get("gold"), row_id)
            seen_ids.add(row_id)
            rows.append(row)
    train = [row for row in rows if row["split"] == "train"]
    dev = [row for row in rows if row["split"] == "dev"]
    if not train or not dev:
        raise ValueError("dataset must contain separate nonempty train and dev splits")
    train_families = {row["family_id"] for row in train}
    dev_families = {row["family_id"] for row in dev}
    overlap = train_families & dev_families
    if overlap:
        raise ValueError("semantic families cross train/dev: " + ", ".join(sorted(overlap)))
    return train, dev


def _intent_dataset(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels = sorted({_label_for_gold(row["gold"]) for row in rows})
    label_index = {label: index for index, label in enumerate(labels)}
    features = np.stack([intent_features(row["text"]) for row in rows])
    targets = np.asarray(
        [label_index[_label_for_gold(row["gold"])] for row in rows], dtype=np.int64
    )
    return features, targets, labels


def _slot_dataset(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    matrices: list[np.ndarray] = []
    targets: list[int] = []
    for row in rows:
        text = row["text"]
        slot = row["gold"]["arguments"].get("target_text")
        if slot is None:
            span = None
        else:
            if not isinstance(slot, str) or not slot:
                raise ValueError(f"{row['id']}: target_text must be a nonempty string")
            start = text.casefold().find(slot.casefold())
            if start < 0:
                raise ValueError(f"{row['id']}: target_text is not present in source text")
            span = (start, start + len(slot))
        rows_x = slot_features(text)
        for position in range(len(text)):
            if span is None or position < span[0] or position >= span[1]:
                tag = 0
            elif position == span[0]:
                tag = 1
            else:
                tag = 2
            matrices.append(rows_x[position])
            targets.append(tag)
    return np.stack(matrices), np.asarray(targets, dtype=np.int64)


def _predict_slot(
    text: str, parameters: dict[str, np.ndarray], threshold: float = TITLE_CONFIDENCE
) -> tuple[str | None, float]:
    probabilities = _forward(parameters, slot_features(text))[2]
    beginnings = probabilities[:, 1]
    if len(beginnings) == 0:
        return None, 0.0
    start = int(np.argmax(beginnings))
    confidence = float(beginnings[start])
    if confidence < threshold:
        return None, confidence
    end = start + 1
    while end < len(text) and int(np.argmax(probabilities[end])) == 2:
        end += 1
    value = text[start:end].strip(" \t\r\n.,!?\"'“”‘’()[]{}")
    return (value if value else None), confidence


def predict_output(
    text: str,
    labels: list[str],
    intent_parameters: dict[str, np.ndarray],
    slot_parameters: dict[str, np.ndarray],
) -> dict[str, Any]:
    probabilities = _forward(intent_parameters, intent_features(text)[None, :])[2][0]
    label = labels[int(np.argmax(probabilities))]
    if label == "no_action":
        return {
            "decision": "no_action",
            "action": None,
            "arguments": {},
            "requires_clarification": False,
        }
    if label == "respond":
        return {
            "decision": "respond",
            "action": None,
            "arguments": {},
            "requires_clarification": False,
            "presentation_intent": "text",
        }
    if label.startswith("act:"):
        action = label[4:]
        if action == "playback.start":
            target, _confidence = _predict_slot(text, slot_parameters)
            if target is None:
                return {
                    "decision": "clarify",
                    "action": None,
                    "arguments": {},
                    "requires_clarification": True,
                }
            return {
                "decision": "act",
                "action": action,
                "arguments": {"target_text": target},
                "requires_clarification": False,
                "presentation_intent": "playback_handoff",
            }
        if action == "catalog.search":
            return {
                "decision": "act",
                "action": action,
                "arguments": {"query": text[:200]},
                "requires_clarification": False,
                "presentation_intent": "media_results",
            }
        if action == "catalog.home.read":
            return {
                "decision": "act",
                "action": action,
                "arguments": {"section": "continueWatching"},
                "requires_clarification": False,
                "presentation_intent": "media_results",
            }
    return {
        "decision": "clarify",
        "action": None,
        "arguments": {},
        "requires_clarification": True,
    }


def output_shape_is_valid(output: dict[str, Any]) -> bool:
    if not isinstance(output, dict) or set(output) - (SCHEMA_REQUIRED | SCHEMA_OPTIONAL):
        return False
    if not SCHEMA_REQUIRED.issubset(output):
        return False
    if output["decision"] not in {"act", "no_action", "clarify", "respond"}:
        return False
    if not isinstance(output["arguments"], dict):
        return False
    if not isinstance(output["requires_clarification"], bool):
        return False
    if output["decision"] == "act":
        return isinstance(output["action"], str) and bool(output["action"]) and not output["requires_clarification"]
    if output["action"] is not None:
        return False
    if output["decision"] == "clarify":
        return output["requires_clarification"] and not output["arguments"]
    return not output["requires_clarification"] and not output["arguments"]


def _working_set_bytes(peak: bool = False) -> int | None:
    if sys.platform != "win32":
        return None
    try:
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
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
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
        if not ok:
            return None
        return int(counters.PeakWorkingSetSize if peak else counters.WorkingSetSize)
    except (AttributeError, OSError):
        return None


def _resolve_output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    for parent in (resolved, *resolved.parents):
        if (parent / ".git").exists():
            raise ValueError("model artifacts must be written outside every Git worktree")
    return resolved


def _load_model(path: Path) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as archive:
        intent_parameters = {
            key.removeprefix("intent_"): archive[key].copy()
            for key in ("intent_w1", "intent_b1", "intent_w2", "intent_b2")
        }
        slot_parameters = {
            key.removeprefix("slot_"): archive[key].copy()
            for key in ("slot_w1", "slot_b1", "slot_w2", "slot_b2")
        }
    return metadata["intent_labels"], intent_parameters, slot_parameters


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_path = args.data.expanduser().resolve()
    output_dir = _resolve_output_path(args.output_dir)
    if not data_path.is_file():
        raise ValueError(f"training data does not exist: {data_path}")
    if any((parent / ".git").exists() for parent in (data_path, *data_path.parents)):
        raise ValueError("training data must remain outside every Git worktree")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows, dev_rows = read_dataset(data_path)
    intent_x, intent_y, intent_labels = _intent_dataset(train_rows)
    slot_x, slot_y = _slot_dataset(train_rows)
    training_start = time.perf_counter()
    intent_parameters, intent_training = fit_mlp(
        intent_x,
        intent_y,
        len(intent_labels),
        args.seed,
        INTENT_HIDDEN,
        args.epochs,
        args.batch_size,
        args.learning_rate,
    )
    slot_parameters, slot_training = fit_mlp(
        slot_x,
        slot_y,
        len(SLOT_LABELS),
        args.seed + 17,
        SLOT_HIDDEN,
        args.epochs,
        args.batch_size * 4,
        args.learning_rate,
        class_weights=(1.0, 6.0, 3.0),
    )
    training_seconds = time.perf_counter() - training_start

    artifact_path = output_dir / "lumi-random-init-intent.npz"
    np.savez_compressed(
        artifact_path,
        **{f"intent_{key}": value for key, value in intent_parameters.items()},
        **{f"slot_{key}": value for key, value in slot_parameters.items()},
    )
    metadata = {
        "model_id": "lumi-random-init-intent-smoke-v1",
        "architecture": "two independent one-hidden-layer MLP heads over fixed hashed Unicode character n-gram features",
        "initialization": "NumPy Gaussian random initialization; no external weights or embeddings",
        "seed": args.seed,
        "feature_dimension": FEATURE_DIM,
        "intent_hidden_dimension": INTENT_HIDDEN,
        "slot_hidden_dimension": SLOT_HIDDEN,
        "intent_labels": intent_labels,
        "slot_labels": list(SLOT_LABELS),
        "slot_confidence_threshold": TITLE_CONFIDENCE,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "cpu_thread_limit": 1,
    }
    _write_json(artifact_path.with_suffix(".json"), metadata)

    rss_before_load = _working_set_bytes()
    load_started = time.perf_counter()
    reloaded_labels, reloaded_intent, reloaded_slot = _load_model(artifact_path)
    model_load_ms = (time.perf_counter() - load_started) * 1000.0
    rss_after_load = _working_set_bytes()
    first_started = time.perf_counter()
    first_prediction = predict_output(
        dev_rows[0]["text"], reloaded_labels, reloaded_intent, reloaded_slot
    )
    first_inference_ms = (time.perf_counter() - first_started) * 1000.0

    prediction_rows: list[dict[str, Any]] = []
    valid_outputs = 0
    exact_matches = 0
    false_actions = 0
    required_title_slots = 0
    exact_title_slots = 0
    for row in dev_rows:
        prediction = predict_output(
            row["text"], reloaded_labels, reloaded_intent, reloaded_slot
        )
        valid = output_shape_is_valid(prediction)
        valid_outputs += int(valid)
        gold = row["gold"]
        exact = all(
            prediction[key] == gold[key]
            for key in ("decision", "action", "arguments", "requires_clarification")
        )
        exact_matches += int(exact)
        false_actions += int(prediction["decision"] == "act" and gold["decision"] != "act")
        gold_title = gold["arguments"].get("target_text")
        if gold_title is not None:
            required_title_slots += 1
            exact_title_slots += int(prediction["arguments"].get("target_text") == gold_title)
        prediction_rows.append(
            {
                "id": row["id"],
                "gold": gold,
                "prediction": prediction,
                "structured_output_valid": valid,
                "semantic_exact_match": exact,
            }
        )

    warm_times: list[float] = []
    for _ in range(args.latency_repeats):
        for row in dev_rows:
            started = time.perf_counter()
            predict_output(row["text"], reloaded_labels, reloaded_intent, reloaded_slot)
            warm_times.append((time.perf_counter() - started) * 1000.0)
    warm_sorted = sorted(warm_times)
    p95_index = max(0, math.ceil(0.95 * len(warm_sorted)) - 1)
    parameter_count = sum(value.size for value in intent_parameters.values()) + sum(
        value.size for value in slot_parameters.values()
    )
    dev_report = {
        "case_count": len(dev_rows),
        "semantic_exact_count": exact_matches,
        "semantic_exact_rate": exact_matches / len(dev_rows),
        "structured_valid_count": valid_outputs,
        "structured_valid_rate": valid_outputs / len(dev_rows),
        "false_action_count": false_actions,
        "title_slot_exact_count": exact_title_slots,
        "title_slot_denominator": required_title_slots,
        "title_slot_exact_rate": (
            exact_title_slots / required_title_slots if required_title_slots else None
        ),
        "predictions": prediction_rows,
    }
    report = {
        "evidence_level": "exploratory",
        "model_id": metadata["model_id"],
        "random_initialization": True,
        "pretrained_artifacts_used": False,
        "training_input_source": args.source_description,
        "training_data_path": str(data_path),
        "training_data_sha256": _sha256(data_path),
        "training_example_count": len(train_rows),
        "development_example_count": len(dev_rows),
        "training_families": len({row["family_id"] for row in train_rows}),
        "development_families": len({row["family_id"] for row in dev_rows}),
        "training_unicode_scalar_count": sum(len(row["text"]) for row in train_rows),
        "training_seed": args.seed,
        "parameter_count": int(parameter_count),
        "intent_training": {
            key: value for key, value in intent_training.items() if key != "loss_by_epoch"
        },
        "slot_training": {
            key: value for key, value in slot_training.items() if key != "loss_by_epoch"
        },
        "training_seconds": training_seconds,
        "development": dev_report,
        "cpu_inference": {
            "numpy_cpu_only": True,
            "gpu_used": False,
            "load_ms": model_load_ms,
            "first_inference_ms": first_inference_ms,
            "warm_latency_p50_ms": statistics.median(warm_times),
            "warm_latency_p95_ms": warm_sorted[p95_index],
            "warm_latency_samples": len(warm_times),
            "first_reloaded_inference_valid": output_shape_is_valid(first_prediction),
        },
        "memory": {
            "working_set_before_reload_bytes": rss_before_load,
            "working_set_after_reload_bytes": rss_after_load,
            "peak_working_set_bytes": _working_set_bytes(peak=True),
            "scope_note": "Windows process working set; includes the Python and NumPy runtime and is not a standalone model-memory estimate.",
        },
        "artifact": {
            "model_path": str(artifact_path),
            "metadata_path": str(artifact_path.with_suffix(".json")),
            "model_bytes": artifact_path.stat().st_size,
            "model_sha256": _sha256(artifact_path),
        },
        "limitations": [
            "This is a tiny task-focused structured predictor, not a generative language model.",
            "The data is a small exploratory split from examples supplied in the user task brief; it is not a benchmark or a product-quality evaluation.",
            "Japanese naturalness has not received qualified human review.",
            "Search arguments copy the original query; genre, year, runtime, and watched-state constraints are not extracted.",
            "The candidate emits proposals only and has no ZenStream service or playback execution path.",
            "The model is loaded by a short-lived command process; idle-service and unload behavior are not measured.",
        ],
    }
    _write_json(output_dir / "report.json", report)
    _write_json(
        output_dir / "training-loss.json",
        {
            "seed": args.seed,
            "intent_initial_loss": intent_training["initial_loss"],
            "intent_loss_by_epoch": intent_training["loss_by_epoch"],
            "slot_initial_loss": slot_training["initial_loss"],
            "slot_loss_by_epoch": slot_training["loss_by_epoch"],
        },
    )
    with (output_dir / "development-predictions.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="controlled local JSONL; never stored in Git")
    parser.add_argument("--output-dir", type=Path, required=True, help="empty output directory outside every Git worktree")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--latency-repeats", type=int, default=25)
    parser.add_argument(
        "--source-description",
        default="User-supplied representative examples in the Lumi task brief; exploratory internal use only.",
    )
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.latency_repeats < 1:
        parser.error("epochs, batch-size, and latency-repeats must be positive")
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
                "intent_loss": [
                    report["intent_training"]["initial_loss"],
                    report["intent_training"]["final_loss"],
                ],
                "slot_loss": [
                    report["slot_training"]["initial_loss"],
                    report["slot_training"]["final_loss"],
                ],
                "development_exact": [
                    report["development"]["semantic_exact_count"],
                    report["development"]["case_count"],
                ],
                "false_actions": report["development"]["false_action_count"],
                "model_bytes": report["artifact"]["model_bytes"],
                "report_path": str(args.output_dir.resolve() / "report.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
