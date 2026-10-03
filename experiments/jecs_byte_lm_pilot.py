#!/usr/bin/env python3
"""Run a small random-init byte-level LM ablation on pinned text-only JECS rows."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_ID = "jecs-v1-neutral-text-only"
SOURCE_REVISION = (
    "jecs_ver1.zip; Google Drive file id "
    "1fkqANFinsVJcD4GXmQXgent9ImyiWUZB"
)
EXPECTED_FILES = {
    "README.md": "4d68ea0502e58698e245f104098f4dd9679e07398a4d44bc95b586732536f10d",
    "transcripts_cs.txt": "7c26eea872f3099e205035973c0784fcaa473c89f26018a5b4a1a6d72a2e468c",
    "transcripts_en.txt": "a4d2e8980d42e8747e2e670c9d112953bd8298da292ba43c1bd2ffa6f8ffc083",
    "transcripts_ja.txt": "20551876ae990563b2fda3b107fb3c5249e485d5b01847452f26a2f7956d2b5e",
}
LANGUAGE_CODES = {"ja": 259, "en": 260, "cs": 261}
PAD, BOS, EOS, VOCAB_SIZE = 256, 257, 258, 262
HIDDEN_SIZE, BATCH_SIZE, EPOCHS = 32, 32, 8
SEED, DEV_MODULUS, LEARNING_RATE = 314159, 5, 0.003
AUTHOR_PAGE = "https://sites.google.com/site/shinnnosuketakamichi/research-topics/jecs_corpus"
JEC_RESOURCE = "https://nlp.ist.i.kyoto-u.ac.jp/EN/?JEC+Basic+Sentence+Data="
JEC_CATALOG = "https://www.jaist.ac.jp/project/NLP_Portal/doc/LR/lr-cat-e.html"
CC_BY_3 = "https://creativecommons.org/licenses/by/3.0/"
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
PHONE = re.compile(r"(?<!\d)\+?\d[\d\s().-]{8,}\d(?!\d)")

TRANSFORMS = {
    "jecs-row-parser-v1": {"format": "JECSNNNN_LANG: UTF-8 utterance", "trim": True},
    "jecs-switch-markup-v1": {"remove_paired_asterisks": True},
    "jecs-family-split-v1": {"seed": SEED, "dev_modulus": DEV_MODULUS, "dev_remainder": 0},
    "jecs-privacy-scan-v1": {"patterns": ["email", "url", "phone-like digit sequence"]},
}
TRANSFORM_HASHES = {
    key: "sha256:"
    + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for key, value in TRANSFORMS.items()
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inside_git_worktree(path: Path) -> bool:
    resolved = path.resolve()
    return any((parent / ".git").exists() for parent in (resolved, *resolved.parents))


def working_set_bytes(peak: bool = False) -> int | None:
    if sys.platform != "win32":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
            ("peak_working_set", ctypes.c_size_t), ("working_set", ctypes.c_size_t),
            ("peak_paged_pool", ctypes.c_size_t), ("paged_pool", ctypes.c_size_t),
            ("peak_nonpaged_pool", ctypes.c_size_t), ("nonpaged_pool", ctypes.c_size_t),
            ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    ok = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    )
    if not ok:
        return None
    return int(counters.peak_working_set if peak else counters.working_set)


def read_rows(data_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_names = set(EXPECTED_FILES)
    if not data_dir.is_dir():
        raise ValueError("input directory must contain only the four pinned JECS text/evidence files")
    children = list(data_dir.iterdir())
    if len(children) != len(expected_names) or any(not path.is_file() for path in children) or {
        path.name for path in children
    } != expected_names:
        raise ValueError("input directory must contain only the four pinned JECS text/evidence files")
    file_hashes = {}
    for name, expected in EXPECTED_FILES.items():
        path = data_dir / name
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"pinned JECS hash mismatch: {name}")
        file_hashes[name] = actual

    rows: list[dict[str, Any]] = []
    ids_by_language: dict[str, set[str]] = {}
    for language in ("ja", "en", "cs"):
        member = f"transcripts_{language}.txt"
        lines = (data_dir / member).read_text(encoding="utf-8", errors="strict").splitlines()
        seen_ids: set[str] = set()
        seen_text: set[str] = set()
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            item_id, separator, text = line.partition(": ")
            match = re.fullmatch(r"JECS(\d{4})_([A-Z]{2})", item_id)
            if not separator or match is None or match.group(2).lower() != language:
                raise ValueError(f"{member}:{line_number}: invalid pinned row format")
            family = f"JECS{int(match.group(1)):04d}"
            if family in seen_ids:
                raise ValueError(f"{member}:{line_number}: duplicate family ID")
            seen_ids.add(family)
            text = text.strip()
            if language == "cs":
                if text.count("*") != 2 or text.index("*") >= text.rindex("*"):
                    raise ValueError(f"{member}:{line_number}: expected one paired switch marker")
                text = text.replace("*", "")
            if not text or text in seen_text:
                raise ValueError(f"{member}:{line_number}: empty or duplicate text")
            seen_text.add(text)
            rows.append({
                "language": language, "family_id": family, "item_id": item_id,
                "member": f"jecs_ver1/neutral/{member}", "text": text,
                "line_sha256": sha256(line.encode("utf-8")),
                "sample_sha256": sha256(text.encode("utf-8")),
            })
        ids_by_language[language] = seen_ids

    counts = {lang: len(ids) for lang, ids in ids_by_language.items()}
    if counts != {"ja": 1219, "en": 1219, "cs": 539}:
        raise ValueError(f"pinned row counts changed: {counts}")
    if ids_by_language["ja"] != ids_by_language["en"]:
        raise ValueError("Japanese and English rows are not aligned")
    if not ids_by_language["cs"].issubset(ids_by_language["ja"]):
        raise ValueError("a code-switch row has no Japanese/English family")

    for row in rows:
        if any(pattern.search(row["text"]) for pattern in (EMAIL, URL, PHONE)):
            raise ValueError(f"contact-detail pattern scan hit {row['item_id']}; inspect locally")
        value = int.from_bytes(
            hashlib.sha256(f"jecs-v1|{SEED}|{row['family_id']}".encode("ascii")).digest()[:4],
            "big",
        )
        row["split"] = "development" if value % DEV_MODULUS == 0 else "train"
        row["uses"] = (
            ["pretraining", "filtering"]
            if row["split"] == "train"
            else ["evaluation", "filtering"]
        )
        row["transformations"] = [
            {"operation_id": "jecs-row-parser-v1", "revision": "1",
             "config_sha256": TRANSFORM_HASHES["jecs-row-parser-v1"]},
            {"operation_id": "jecs-family-split-v1", "revision": "1",
             "config_sha256": TRANSFORM_HASHES["jecs-family-split-v1"]},
            {"operation_id": "jecs-privacy-scan-v1", "revision": "1",
             "config_sha256": TRANSFORM_HASHES["jecs-privacy-scan-v1"]},
        ]
        if row["language"] == "cs":
            row["transformations"].append(
                {"operation_id": "jecs-switch-markup-v1", "revision": "1",
                 "config_sha256": TRANSFORM_HASHES["jecs-switch-markup-v1"]}
            )

    splits_by_sample: dict[str, set[str]] = {}
    languages_by_sample: dict[str, set[str]] = {}
    for row in rows:
        splits_by_sample.setdefault(row["sample_sha256"], set()).add(row["split"])
        languages_by_sample.setdefault(row["sample_sha256"], set()).add(row["language"])
    if any(len(splits) > 1 for splits in splits_by_sample.values()):
        raise ValueError("an exact normalized text appears in both training and development")
    cross_language_duplicates = sum(
        len(languages) > 1 for languages in languages_by_sample.values()
    )

    dev_families = {r["family_id"] for r in rows if r["split"] == "development"}
    train_families = {r["family_id"] for r in rows if r["split"] == "train"}
    if train_families & dev_families or train_families | dev_families != ids_by_language["ja"]:
        raise ValueError("family split integrity check failed")
    summary = {
        "source_id": SOURCE_ID, "source_revision": SOURCE_REVISION,
        "source_file_sha256": file_hashes, "source_counts": counts,
        "training_family_count": len(train_families),
        "development_family_count": len(dev_families),
        "training_rows_by_language": {
            lang: sum(r["language"] == lang and r["split"] == "train" for r in rows)
            for lang in ("ja", "en", "cs")
        },
        "development_rows_by_language": {
            lang: sum(r["language"] == lang and r["split"] == "development" for r in rows)
            for lang in ("ja", "en", "cs")
        },
        "privacy_pattern_hits": 0, "duplicate_texts_within_language": 0,
        "cross_split_duplicate_hash_groups": 0,
        "cross_language_duplicate_hash_groups": cross_language_duplicates,
    }
    return rows, summary


def source_manifest(repo_root: Path) -> tuple[dict[str, Any], str]:
    path = repo_root / "provenance" / "data_sources.json"
    raw = path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    matches = [s for s in manifest["sources"] if s.get("source_id") == SOURCE_ID]
    if len(matches) != 1 or matches[0].get("decision") != "approved_for_use":
        raise ValueError("one approved JECS source record is required")
    if not {"pretraining", "evaluation", "filtering"}.issubset(matches[0].get("lumi_uses", [])):
        raise ValueError("JECS manifest does not declare this pilot's uses")
    return matches[0], sha256(raw)


def make_record(row: dict[str, Any]) -> dict[str, Any]:
    attribution = (
        "JECS: Japanese-English code-switching speech corpus, Shinnosuke Takamichi, "
        "Yoshifumi Nakano, Ai Morimatsu, Takaaki Saeki, and Hiroshi Saruwatari; "
        "text includes material based on JEC Basic Sentence Data by Kurohashi-Kawahara "
        "Lab., Kyoto University, and the NICT MASTAR Project, Multilingual Translation "
        "Lab. The utterance-id prefix and code-switch asterisks were removed. No "
        "endorsement is implied."
    )
    item = {
        "source_id": SOURCE_ID, "item_id": row["item_id"],
        "source_uri": "https://drive.google.com/file/d/1fkqANFinsVJcD4GXmQXgent9ImyiWUZB/view",
        "source_revision": SOURCE_REVISION,
        "source_sha256": f"sha256:{row['line_sha256']}",
        "rights_basis": "license", "license_identifier": "CC-BY-3.0",
        "license_uri": CC_BY_3,
        "rights_evidence_uris": [AUTHOR_PAGE, CC_BY_3, JEC_RESOURCE, JEC_CATALOG],
        "rights_review_status": "reviewed", "rights_reviewed_date": "2026-10-03",
        "permissions": {k: "permitted" for k in (
            "training_use", "evaluation_use", "commercial_use", "modification", "redistribution"
        )},
        "attribution": attribution,
    }
    return {
        "schema_version": 1,
        "record_id": f"jecs-v1-{row['language']}-{row['family_id']}",
        "sample_kind": "transformed_sample",
        "sample_sha256": f"sha256:{row['sample_sha256']}",
        "language_tags": [row["language"]], "lumi_uses": row["uses"],
        "decision": "approved_for_use", "split": row["split"],
        "contamination_status": "checked_clear", "privacy_status": "cleared",
        "source_items": [item], "transformations": row["transformations"],
        "parent_record_ids": [],
        "review": {
            "reviewed_by": "Codex-assisted primary-source, rights, and structural audit",
            "reviewed_date": "2026-10-03",
            "rationale": (
                "Pinned text-only members and line hashes passed; IDs are unique, JA/EN "
                "families align, train/development families are disjoint, and email/URL/"
                "phone-like pattern scans found no matches. This is not a semantic or "
                "native-speaker quality review."
            ),
        },
    }


def write_provenance(
    output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any], source_manifest_hash: str
) -> dict[str, str]:
    records_path = output_dir / "document_records.jsonl"
    selection_path = output_dir / "selection_manifest.json"
    records = [
        make_record(row)
        for row in rows
    ]
    records_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records),
        encoding="utf-8",
    )
    selection = {
        "manifest_version": 1, "source_id": SOURCE_ID, "source_revision": SOURCE_REVISION,
        "source_manifest_sha256": f"sha256:{source_manifest_hash}",
        "source_file_sha256": summary["source_file_sha256"],
        "split": {
            "algorithm": "sha256(f'jecs-v1|314159|{family_id}')[:4] mod 5 == 0",
            "seed": SEED, "development_modulus": DEV_MODULUS, "development_remainder": 0,
            "family_grouping": "same numeric JECS utterance id across all language rows",
        },
        "row_count": len(rows), "raw_text_in_manifest": False,
        "rows": [
            {"item_id": r["item_id"], "language": r["language"], "family_id": r["family_id"],
             "split": r["split"], "sample_sha256": f"sha256:{r['sample_sha256']}",
             "source_line_sha256": f"sha256:{r['line_sha256']}"}
            for r in rows
        ],
    }
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "document_records_sha256": file_sha256(records_path),
        "selection_manifest_sha256": file_sha256(selection_path),
    }


def make_sequences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        data = row["text"].encode("utf-8")
        inputs = np.asarray([BOS, LANGUAGE_CODES[row["language"]], *data], dtype=np.int32)
        targets = np.asarray([LANGUAGE_CODES[row["language"]], *data, EOS], dtype=np.int32)
        result.append(row | {"inputs": inputs, "targets": targets, "byte_count": len(data)})
    return result


def initialise(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "embedding": rng.normal(0, 0.04, (VOCAB_SIZE, HIDDEN_SIZE)).astype(np.float32),
        "recurrent": rng.normal(0, math.sqrt(1 / HIDDEN_SIZE), (HIDDEN_SIZE, HIDDEN_SIZE)).astype(np.float32),
        "hidden_bias": np.zeros(HIDDEN_SIZE, dtype=np.float32),
        "output": rng.normal(0, math.sqrt(1 / HIDDEN_SIZE), (HIDDEN_SIZE, VOCAB_SIZE)).astype(np.float32),
        "output_bias": np.zeros(VOCAB_SIZE, dtype=np.float32),
    }


def batch_arrays(batch: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = max(len(row["inputs"]) for row in batch)
    x = np.full((steps, len(batch)), PAD, dtype=np.int32)
    y = np.full_like(x, PAD)
    mask = np.zeros_like(x, dtype=np.float32)
    for col, row in enumerate(batch):
        n = len(row["inputs"])
        x[:n, col], y[:n, col], mask[:n, col] = row["inputs"], row["targets"], 1
    return x, y, mask


def loss_grad(
    p: dict[str, np.ndarray], batch: list[dict[str, Any]], gradients: bool
) -> tuple[float, int, float, int, dict[str, np.ndarray] | None]:
    x, y, mask = batch_arrays(batch)
    steps, size = x.shape
    token_count = int(mask.sum())
    byte_count = int(((y < 256) * mask).sum())
    hprev = np.zeros((size, HIDDEN_SIZE), dtype=np.float32)
    cache = []
    total, bytes_loss = 0.0, 0.0
    indices = np.arange(size)
    for t in range(steps):
        h = np.tanh(p["embedding"][x[t]] + hprev @ p["recurrent"] + p["hidden_bias"]).astype(np.float32)
        logits = h @ p["output"] + p["output_bias"]
        maximum = logits.max(axis=1)
        exp = np.exp(logits - maximum[:, None])
        probs = exp / exp.sum(axis=1)[:, None]
        ce = np.log(exp.sum(axis=1)) + maximum - logits[indices, y[t]]
        total += float((ce * mask[t]).sum())
        bytes_loss += float((ce * mask[t] * (y[t] < 256)).sum())
        if gradients:
            cache.append((x[t], hprev, h, probs, y[t]))
        hprev = h
    total, bytes_loss = total / token_count, bytes_loss / byte_count
    if not gradients:
        return total, token_count, bytes_loss, byte_count, None
    g = {name: np.zeros_like(value) for name, value in p.items()}
    dhnext = np.zeros((size, HIDDEN_SIZE), dtype=np.float32)
    for t in range(steps - 1, -1, -1):
        token, previous, h, probs, target = cache[t]
        dlogits = probs.copy()
        dlogits[indices, target] -= 1
        dlogits *= mask[t, :, None] / token_count
        g["output"] += h.T @ dlogits
        g["output_bias"] += dlogits.sum(axis=0)
        dh = dlogits @ p["output"].T + dhnext
        dz = dh * (1 - h * h)
        g["hidden_bias"] += dz.sum(axis=0)
        g["recurrent"] += previous.T @ dz
        np.add.at(g["embedding"], token, dz)
        dhnext = dz @ p["recurrent"].T
    return total, token_count, bytes_loss, byte_count, g


def evaluate(p: dict[str, np.ndarray], rows: list[dict[str, Any]]) -> dict[str, Any]:
    results = {}
    for lang in ("ja", "en", "cs"):
        selected = [r for r in rows if r["language"] == lang]
        if not selected:
            results[lang] = None
            continue
        nll, count, byte_nll, byte_count = 0.0, 0, 0.0, 0
        started = time.perf_counter()
        for i in range(0, len(selected), BATCH_SIZE):
            loss, n, bits_loss, nbytes, _ = loss_grad(
                p, selected[i:i + BATCH_SIZE], gradients=False
            )
            nll += loss * n
            count += n
            byte_nll += bits_loss * nbytes
            byte_count += nbytes
        elapsed = time.perf_counter() - started
        results[lang] = {
            "rows": len(selected), "target_tokens": count, "utf8_bytes": byte_count,
            "nats_per_target_token": nll / count,
            "bits_per_utf8_byte": (byte_nll / byte_count) / math.log(2),
            "scoring_seconds": elapsed,
            "scored_utf8_bytes_per_second": byte_count / elapsed if elapsed else None,
        }
    return results


def train(
    train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]], *,
    include_cs: bool, presentations_per_epoch: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    selected = [r for r in train_rows if include_cs or r["language"] != "cs"]
    p = initialise(SEED)
    m, v = ({name: np.zeros_like(a) for name, a in p.items()} for _ in range(2))
    initial_dev = evaluate(p, dev_rows)
    updates, presented, target_tokens, bytes_seen = 0, 0, 0, 0
    updates_per_epoch = math.ceil(presentations_per_epoch / BATCH_SIZE)
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        rng = np.random.default_rng(SEED + epoch)
        if include_cs:
            order = rng.permutation(len(selected))
            epoch_rows = [selected[int(i)] for i in order]
        else:
            picks = rng.integers(0, len(selected), size=presentations_per_epoch)
            epoch_rows = [selected[int(i)] for i in picks]
        if len(epoch_rows) != presentations_per_epoch:
            raise ValueError("candidate sample-presentation budget mismatch")
        for start in range(0, len(epoch_rows), BATCH_SIZE):
            batch = epoch_rows[start:start + BATCH_SIZE]
            loss, n, _, _, g = loss_grad(p, batch, gradients=True)
            if g is None or not math.isfinite(loss):
                raise ValueError("non-finite loss or missing gradients")
            norm = math.sqrt(sum(float((value * value).sum()) for value in g.values()))
            if norm > 1:
                for value in g.values():
                    value *= 1 / norm
            updates += 1
            for name in p:
                m[name] = 0.9 * m[name] + 0.1 * g[name]
                v[name] = 0.999 * v[name] + 0.001 * g[name] ** 2
                p[name] -= LEARNING_RATE * (m[name] / (1 - 0.9**updates)) / (
                    np.sqrt(v[name] / (1 - 0.999**updates)) + 1e-8
                )
            presented += len(batch)
            target_tokens += n
            bytes_seen += sum(r["byte_count"] for r in batch)
        if math.ceil(len(epoch_rows) / BATCH_SIZE) != updates_per_epoch:
            raise ValueError("optimizer update budget mismatch")
    seconds = time.perf_counter() - started
    used_train = [r for r in train_rows if include_cs or r["language"] != "cs"]
    result = {
        "random_initialization": True, "pretrained_weights_or_embeddings_used": False,
        "architecture": "one-layer tanh recurrent UTF-8 byte LM with BOS/EOS and language tags",
        "seed": SEED, "hidden_size": HIDDEN_SIZE, "vocabulary_size": VOCAB_SIZE,
        "parameter_count": sum(a.size for a in p.values()), "epochs": EPOCHS,
        "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "optimizer": "Adam; global gradient norm clipped to 1.0",
        "training_unique_rows": len(selected),
        "training_unique_code_switch_rows": sum(r["language"] == "cs" for r in selected),
        "training_sample_presentations": presented, "training_optimizer_updates": updates,
        "training_target_tokens_seen": target_tokens, "training_utf8_bytes_seen": bytes_seen,
        "training_seconds": seconds, "initial_development_metrics": initial_dev,
        "final_training_metrics": evaluate(p, used_train),
        "final_development_metrics": evaluate(p, dev_rows),
    }
    return p, result


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def run(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    if data_dir == output_dir or data_dir in output_dir.parents or output_dir in data_dir.parents:
        raise ValueError("input and output directories must not overlap")
    if inside_git_worktree(output_dir):
        raise ValueError("outputs must be outside every Git worktree")
    source, manifest_hash = source_manifest(repo)
    rows, summary = read_rows(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_hashes = write_provenance(output_dir, rows, summary, manifest_hash)

    seqs = make_sequences(rows)
    train_rows = [r for r, s in zip(rows, seqs, strict=True) if r["split"] == "train"]
    dev_rows = [r for r in rows if r["split"] == "development"]
    train_rows = [
        r | {"inputs": s["inputs"], "targets": s["targets"],
             "byte_count": len(r["text"].encode("utf-8"))}
        for r, s in zip(train_rows, [s for s in seqs if s["split"] == "train"], strict=True)
    ]
    dev_rows = [
        r | {"inputs": s["inputs"], "targets": s["targets"],
             "byte_count": len(r["text"].encode("utf-8"))}
        for r, s in zip(dev_rows, [s for s in seqs if s["split"] == "development"], strict=True)
    ]
    if not train_rows or not dev_rows:
        raise ValueError("train and development rows are required")

    working_before, peak_before = working_set_bytes(), working_set_bytes(peak=True)
    presentations = len(train_rows)
    candidates = {}
    for name, include_cs in (("bilingual_baseline", False), ("code_switch_augmented", True)):
        params, metrics = train(
            train_rows, dev_rows, include_cs=include_cs,
            presentations_per_epoch=presentations,
        )
        artifact = output_dir / f"lumi-jecs-{name}.npz"
        np.savez_compressed(artifact, **params)
        reload_started = time.perf_counter()
        with np.load(artifact, allow_pickle=False) as archive:
            {key: archive[key].copy() for key in params}
        metrics["model_file_bytes"] = artifact.stat().st_size
        metrics["model_reload_seconds"] = time.perf_counter() - reload_started
        metrics["weights_sha256"] = file_sha256(artifact)
        candidates[name] = metrics

    report = {
        "experiment_id": "jecs-neutral-text-random-init-byte-lm-ablation-v1",
        "evidence_level": "exploratory",
        "conclusion_status": "no product or conversational-quality claim",
        "created_date": "2026-10-03",
        "source": source, "source_manifest_sha256": f"sha256:{manifest_hash}",
        "provenance_artifacts": provenance_hashes, "data": summary,
        "text_written_to_outputs": False,
        "candidate_comparison": {
            "shared_seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "updates_each": EPOCHS * math.ceil(presentations / BATCH_SIZE),
            "budget_note": "The bilingual baseline resamples JA/EN rows to match sample presentations; UTF-8 byte counts are reported per candidate.",
        },
        "candidates": candidates,
        "runtime": {
            "python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform(), "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(), "execution": "CPU only", "gpu_used": False,
            "working_set_bytes_before": working_before, "working_set_peak_bytes_before": peak_before,
            "working_set_bytes_after": working_set_bytes(),
            "working_set_peak_bytes_after": working_set_bytes(peak=True),
        },
        "scope_limits": [
            "Only the exact neutral JECS transcript text members were used; audio was not downloaded or opened.",
            "This short read-speech corpus is not a natural conversational ZenStream set or final holdout.",
            "No tokenizer fitting, instruction tuning, response generation, or public model release was performed.",
            "Output weights are local exploratory artifacts and are not release candidates.",
        ],
    }
    write_json(output_dir / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run(args.data_dir, args.output_dir)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    summary = {
        "experiment_id": result["experiment_id"],
        "data": result["data"],
        "candidates": {
            name: {
                "parameter_count": metrics["parameter_count"],
                "training_optimizer_updates": metrics["training_optimizer_updates"],
                "training_seconds": metrics["training_seconds"],
                "development_bits_per_utf8_byte": {
                    lang: values["bits_per_utf8_byte"] if values else None
                    for lang, values in metrics["final_development_metrics"].items()
                },
                "model_file_bytes": metrics["model_file_bytes"],
                "weights_sha256": metrics["weights_sha256"],
            }
            for name, metrics in result["candidates"].items()
        },
        "runtime": result["runtime"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
