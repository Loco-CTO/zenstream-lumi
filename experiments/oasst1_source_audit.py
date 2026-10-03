#!/usr/bin/env python3
"""Audit pinned OASST1 parquet rows without printing or exporting message text."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet
import pyarrow


DATASET_REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
SHARDS = {
    "train": {
        "name": "data-train-00000-of-00001-b42a775f407cee45.parquet",
        "sha256": "bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e",
        "size_bytes": 39_516_251,
        "rows": 84_437,
    },
    "validation": {
        "name": "data-validation-00000-of-00001-134b8fd0c89408b6.parquet",
        "sha256": "24002597bb13a7edd42d92f773762f25e285f72c31a70449393d0ded1dc7b416",
        "size_bytes": 2_080_179,
        "rows": 4_401,
    },
}
REQUIRED_COLUMNS = {
    "message_id",
    "parent_id",
    "text",
    "role",
    "lang",
    "review_count",
    "review_result",
    "deleted",
    "synthetic",
    "model_name",
    "message_tree_id",
    "tree_state",
    "labels",
}
DISALLOWED_LABELS = {
    "spam",
    "lang_mismatch",
    "pii",
    "not_appropriate",
    "hate_speech",
    "sexual_content",
    "violence",
    "threat",
    "identity_attack",
    "sexual_explicit",
}
EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_LIKE_PATTERN = re.compile(r"(?<!\d)(?:\+?\d[\d().\s-]{6,}\d)(?!\d)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside_git_worktree(path: Path) -> bool:
    return any((parent / ".git").exists() for parent in (path, *path.parents))


def normalized_text_hash(text: str) -> str | None:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def label_values(value: Any) -> dict[str, float]:
    """Normalize either the card's map example or its parquet list-of-structs form."""
    result: dict[str, float] = {}
    if isinstance(value, dict):
        entries = value.items()
        for name, item in entries:
            raw = item.get("value") if isinstance(item, dict) else item
            try:
                result[str(name)] = float(raw)
            except (TypeError, ValueError):
                continue
        return result
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            try:
                result[str(item["name"])] = float(item.get("value"))
            except (TypeError, ValueError):
                continue
    return result


def _load_shard(path: Path, split: str) -> list[dict[str, Any]]:
    expected = SHARDS[split]
    if path.name != expected["name"]:
        raise ValueError(f"{split} input filename does not match the pinned OASST1 shard")
    if path.stat().st_size != expected["size_bytes"]:
        raise ValueError(f"{split} input has an unexpected byte length")
    if sha256_file(path) != expected["sha256"]:
        raise ValueError(f"{split} input SHA-256 does not match the pinned OASST1 shard")
    table = parquet.read_table(path)
    missing = REQUIRED_COLUMNS.difference(table.column_names)
    if missing:
        raise ValueError(f"{split} input is missing required columns: {', '.join(sorted(missing))}")
    if table.num_rows != expected["rows"]:
        raise ValueError(f"{split} input row count does not match the pinned card")
    return table.select(sorted(REQUIRED_COLUMNS)).to_pylist()


def _row_exclusion_reasons(row: dict[str, Any]) -> set[str]:
    reasons: set[str] = set()
    if row.get("synthetic") is not False:
        reasons.add("synthetic_true_or_unknown")
    if str(row.get("model_name") or "").strip():
        reasons.add("generator_name_present")
    if row.get("deleted") is not False:
        reasons.add("deleted_true_or_unknown")
    if row.get("review_result") is not True:
        reasons.add("review_not_positive_or_unknown")
    if row.get("tree_state") != "ready_for_export":
        reasons.add("tree_not_ready_or_unknown")
    if row.get("role") not in {"prompter", "assistant"}:
        reasons.add("role_unknown")
    if row.get("lang") not in {"en", "ja"}:
        reasons.add("language_unsupported_or_unknown")
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        reasons.add("text_empty_or_unknown")
    else:
        if EMAIL_PATTERN.search(text):
            reasons.add("email_like_text")
        if PHONE_LIKE_PATTERN.search(text):
            reasons.add("phone_like_text")
    flags = label_values(row.get("labels"))
    if any(flags.get(name, 0.0) > 0.0 for name in DISALLOWED_LABELS):
        reasons.add("disallowed_review_label")
    return reasons


def audit(train_path: Path, validation_path: Path) -> dict[str, Any]:
    split_rows = {
        "train": _load_shard(train_path, "train"),
        "validation": _load_shard(validation_path, "validation"),
    }
    all_rows = [row for rows in split_rows.values() for row in rows]
    row_ids = [row.get("message_id") for row in all_rows]
    nonempty_ids = [value for value in row_ids if isinstance(value, str) and value]
    id_counts = Counter(nonempty_ids)
    duplicate_ids = {value for value, count in id_counts.items() if count > 1}
    row_by_id = {
        row["message_id"]: row
        for row in all_rows
        if isinstance(row.get("message_id"), str)
        and row.get("message_id")
        and id_counts.get(row["message_id"]) == 1
    }
    rows_by_tree: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_without_tree_id = 0
    tree_splits: dict[str, set[str]] = defaultdict(set)
    structure_reasons_by_tree: dict[str, set[str]] = defaultdict(set)
    missing_parent_reference_rows = 0
    parent_tree_mismatch_rows = 0
    same_role_parent_child_rows = 0
    cyclic_parent_chain_rows = 0
    language_counts: dict[str, Counter[str]] = {
        split: Counter(str(row.get("lang") or "unknown") for row in rows)
        for split, rows in split_rows.items()
    }
    labels_by_name: Counter[str] = Counter()
    duplicate_text_trees: dict[str, set[str]] = defaultdict(set)
    duplicate_text_rows: Counter[str] = Counter()

    for split, rows in split_rows.items():
        for row in rows:
            tree_id = row.get("message_tree_id")
            if isinstance(tree_id, str) and tree_id:
                rows_by_tree[tree_id].append(row)
                tree_splits[tree_id].add(split)
            else:
                rows_without_tree_id += 1
            parent_id = row.get("parent_id")
            if parent_id not in (None, ""):
                parent = row_by_id.get(parent_id)
                if parent is None:
                    missing_parent_reference_rows += 1
                    if isinstance(tree_id, str) and tree_id:
                        structure_reasons_by_tree[tree_id].add("parent_reference_missing_or_ambiguous")
                else:
                    parent_tree_id = parent.get("message_tree_id")
                    if parent_tree_id != tree_id:
                        parent_tree_mismatch_rows += 1
                        if isinstance(tree_id, str) and tree_id:
                            structure_reasons_by_tree[tree_id].add("parent_tree_mismatch")
                        if isinstance(parent_tree_id, str) and parent_tree_id:
                            structure_reasons_by_tree[parent_tree_id].add("parent_tree_mismatch")
                    if parent.get("role") == row.get("role"):
                        same_role_parent_child_rows += 1
                        if isinstance(tree_id, str) and tree_id:
                            structure_reasons_by_tree[tree_id].add("parent_role_did_not_alternate")
            for name, value in label_values(row.get("labels")).items():
                if name in DISALLOWED_LABELS and value > 0.0:
                    labels_by_name[name] += 1
            text = row.get("text")
            if isinstance(text, str):
                digest = normalized_text_hash(text)
                if digest:
                    duplicate_text_rows[digest] += 1
                    if isinstance(tree_id, str) and tree_id:
                        duplicate_text_trees[digest].add(tree_id)

    for row in all_rows:
        seen: set[str] = set()
        current = row
        while True:
            current_id = current.get("message_id")
            if not isinstance(current_id, str) or not current_id:
                break
            if current_id in seen:
                cyclic_parent_chain_rows += 1
                tree_id = row.get("message_tree_id")
                if isinstance(tree_id, str) and tree_id:
                    structure_reasons_by_tree[tree_id].add("parent_cycle")
                break
            seen.add(current_id)
            parent_id = current.get("parent_id")
            if parent_id in (None, ""):
                break
            current = row_by_id.get(parent_id)
            if current is None:
                break

    reason_tree_counts: Counter[str] = Counter()
    eligible_rows_by_language: Counter[str] = Counter()
    eligible_trees_by_language: Counter[str] = Counter()
    eligible_rows_by_split_and_language: dict[str, Counter[str]] = {
        split: Counter() for split in split_rows
    }
    eligible_trees_by_split_and_language: dict[str, Counter[str]] = {
        split: Counter() for split in split_rows
    }
    synthetic_true_rows = 0
    synthetic_unknown_rows = 0
    generator_named_rows = 0
    for rows in all_rows:
        if rows.get("synthetic") is True:
            synthetic_true_rows += 1
        elif rows.get("synthetic") is not False:
            synthetic_unknown_rows += 1
        if str(rows.get("model_name") or "").strip():
            generator_named_rows += 1

    for tree_id, family_rows in rows_by_tree.items():
        reasons: set[str] = set()
        reasons.update(structure_reasons_by_tree.get(tree_id, set()))
        langs = {row.get("lang") for row in family_rows}
        if len(langs) != 1:
            reasons.add("mixed_language_tree")
        family_splits = tree_splits[tree_id]
        if len(family_splits) != 1:
            reasons.add("family_spans_published_splits")
        for row in family_rows:
            reasons.update(_row_exclusion_reasons(row))
            row_id = row.get("message_id")
            if not isinstance(row_id, str) or id_counts.get(row_id, 0) != 1:
                reasons.add("missing_or_duplicate_message_id")
        if reasons:
            reason_tree_counts.update(reasons)
            continue
        language = next(iter(langs))
        split = next(iter(family_splits))
        eligible_trees_by_language[str(language)] += 1
        eligible_rows_by_language[str(language)] += len(family_rows)
        eligible_trees_by_split_and_language[split][str(language)] += 1
        eligible_rows_by_split_and_language[split][str(language)] += len(family_rows)

    cross_split_families = sum(1 for splits in tree_splits.values() if len(splits) > 1)
    duplicate_text_groups = [
        (digest, count)
        for digest, count in duplicate_text_rows.items()
        if count > 1
    ]
    cross_family_duplicate_groups = sum(
        1
        for digest, _ in duplicate_text_groups
        if len(duplicate_text_trees[digest]) > 1
    )
    duplicate_rows_beyond_first = sum(count - 1 for _, count in duplicate_text_groups)

    return {
        "report_version": 1,
        "dataset": "OpenAssistant/oasst1",
        "dataset_revision": DATASET_REVISION,
        "source_audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pyarrow_version": pyarrow.__version__,
        "training_or_evaluation_admitted": False,
        "source_files": {
            split: {
                "name": SHARDS[split]["name"],
                "sha256": SHARDS[split]["sha256"],
                "size_bytes": SHARDS[split]["size_bytes"],
                "rows": len(split_rows[split]),
            }
            for split in split_rows
        },
        "inventory": {
            "total_rows": len(all_rows),
            "unique_message_ids": len(id_counts),
            "duplicate_message_id_count": len(duplicate_ids),
            "rows_without_tree_id": rows_without_tree_id,
            "rows_with_missing_or_ambiguous_parent_reference": missing_parent_reference_rows,
            "rows_with_parent_tree_mismatch": parent_tree_mismatch_rows,
            "rows_with_non_alternating_parent_role": same_role_parent_child_rows,
            "starting_rows_with_cyclic_parent_chain": cyclic_parent_chain_rows,
            "unique_tree_families": len(rows_by_tree),
            "families_spanning_published_splits": cross_split_families,
            "rows_by_published_split_and_language": {
                split: dict(sorted(counts.items()))
                for split, counts in language_counts.items()
            },
            "synthetic_true_rows": synthetic_true_rows,
            "synthetic_unknown_rows": synthetic_unknown_rows,
            "rows_with_generator_name": generator_named_rows,
            "positive_disallowed_label_rows": dict(sorted(labels_by_name.items())),
            "normalized_duplicate_text_groups": len(duplicate_text_groups),
            "normalized_duplicate_rows_beyond_first": duplicate_rows_beyond_first,
            "duplicate_text_groups_spanning_tree_families": cross_family_duplicate_groups,
        },
        "automatic_screen": {
            "family_exclusion_counts_by_reason": dict(sorted(reason_tree_counts.items())),
            "candidate_rows_by_language": dict(sorted(eligible_rows_by_language.items())),
            "candidate_tree_families_by_language": dict(sorted(eligible_trees_by_language.items())),
            "candidate_rows_by_published_split_and_language": {
                split: dict(sorted(counts.items()))
                for split, counts in eligible_rows_by_split_and_language.items()
            },
            "candidate_tree_families_by_published_split_and_language": {
                split: dict(sorted(counts.items()))
                for split, counts in eligible_trees_by_split_and_language.items()
            },
            "candidate_status": "requires_manual_privacy_and_language_review",
            "required_before_training": [
                "review candidate Japanese samples with a qualified Japanese reviewer",
                "manually inspect privacy-screen hits and a controlled sample of no-hit records",
                "check overlap with approved Lumi training inputs and the frozen evaluation inventory",
                "create item-level provenance records and explicitly approve the scoped training use",
                "re-review public model-artifact rights and exact training lineage",
            ],
        },
        "privacy_note": "Counts only; no message text, participant identifiers, message IDs, or tree IDs are written to this report.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if inside_git_worktree(output):
        parser.error("audit output must be stored outside every Git worktree")
    if output.exists() and not args.force:
        parser.error("output exists; pass --force only when replacing a controlled audit report")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = audit(args.train.resolve(), args.validation.resolve())
    except (OSError, ValueError) as error:
        print(f"OASST1 audit refused: {error}", file=sys.stderr)
        return 2
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    output.write_bytes(encoded)
    print(json.dumps({
        "report_path": str(output),
        "report_sha256": hashlib.sha256(encoded).hexdigest(),
        "training_or_evaluation_admitted": False,
        "candidate_rows_by_language": report["automatic_screen"]["candidate_rows_by_language"],
        "candidate_tree_families_by_language": report["automatic_screen"]["candidate_tree_families_by_language"],
        "candidate_rows_by_published_split_and_language": report["automatic_screen"]["candidate_rows_by_published_split_and_language"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
