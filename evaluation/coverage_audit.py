"""Audit reviewed-case presence across Lumi's required evaluation slices."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scorer import (  # noqa: E402
    EvaluationInputError,
    LANGUAGES,
    SPLITS,
    _validate_case,
)


REQUIRED_SLICES_PATH = Path(__file__).resolve().with_name("required_slices.json")


class CoverageInputError(ValueError):
    """Raised when a coverage taxonomy or case inventory is malformed."""


def _load_required_slices() -> list[dict[str, Any]]:
    document = json.loads(REQUIRED_SLICES_PATH.read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or set(document) != {"schema_version", "slices"}
        or not isinstance(document["schema_version"], int)
        or isinstance(document["schema_version"], bool)
        or document["schema_version"] != 1
        or not isinstance(document["slices"], list)
        or not document["slices"]
    ):
        raise CoverageInputError("required slice taxonomy has an invalid shape")

    seen: set[str] = set()
    for index, item in enumerate(document["slices"], start=1):
        if not isinstance(item, dict) or set(item) != {"id", "title", "languages"}:
            raise CoverageInputError(f"required slice {index} has an invalid shape")
        if (
            not isinstance(item["id"], str)
            or not item["id"].strip()
            or not isinstance(item["title"], str)
            or not item["title"].strip()
        ):
            raise CoverageInputError(f"required slice {index} needs an id and title")
        if item["id"] in seen:
            raise CoverageInputError(f"duplicate required slice id {item['id']!r}")
        seen.add(item["id"])
        languages = item["languages"]
        if (
            not isinstance(languages, list)
            or not languages
            or not all(isinstance(language, str) for language in languages)
            or len(set(languages)) != len(languages)
            or not set(languages).issubset(LANGUAGES)
        ):
            raise CoverageInputError(f"required slice {item['id']!r} has invalid languages")
    return document["slices"]


def _read_cases(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line, parse_constant=_reject_json_constant))
                except (json.JSONDecodeError, ValueError) as exc:
                    raise CoverageInputError(
                        f"invalid case JSON on line {line_number}: {exc}"
                    ) from exc
    except UnicodeDecodeError as exc:
        raise CoverageInputError(f"case file is not UTF-8: {exc}") from exc
    return records


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _validate_inventory(cases: list[dict[str, Any]], split: str) -> None:
    case_ids: set[str] = set()
    for index, case in enumerate(cases, start=1):
        try:
            _validate_case(case, index)
        except EvaluationInputError as exc:
            raise CoverageInputError(str(exc)) from exc
        if case["split"] != split:
            raise CoverageInputError(
                "case inventory must contain only the requested split; provide separate development and final-holdout files"
            )
        case_id = case["case_id"]
        if case_id in case_ids:
            raise CoverageInputError(f"duplicate case_id {case_id!r}")
        case_ids.add(case_id)


def _case_counts(cases: list[dict[str, Any]]) -> dict[str, Any]:
    ready = [case for case in cases if case["review_status"] == "ready"]
    draft = [case for case in cases if case["review_status"] == "draft"]
    excluded = [case for case in cases if case["review_status"] == "excluded"]
    if ready:
        presence_status = "ready_cases_present"
    elif draft:
        presence_status = "draft_only"
    elif excluded:
        presence_status = "excluded_only"
    else:
        presence_status = "missing"
    return {
        "case_count": len(cases),
        "ready_case_count": len(ready),
        "draft_case_count": len(draft),
        "excluded_case_count": len(excluded),
        "family_count": len({case["family_id"] for case in cases}),
        "ready_family_count": len({case["family_id"] for case in ready}),
        "presence_status": presence_status,
    }


def audit_cases(
    cases: list[dict[str, Any]],
    *,
    split: str = "development",
    allow_final_holdout: bool = False,
) -> dict[str, Any]:
    """Report case presence only; it makes no quality or statistical sufficiency claim."""
    if split not in SPLITS:
        raise CoverageInputError("split must be development or final_holdout")
    if split == "final_holdout" and not allow_final_holdout:
        raise CoverageInputError(
            "final holdout requires explicit --final-audit authorization or allow_final_holdout=True"
        )
    if allow_final_holdout and split != "final_holdout":
        raise CoverageInputError("final-audit authorization is valid only for final_holdout")

    _validate_inventory(cases, split)
    selected = cases
    required_slices = _load_required_slices()

    matrix: list[dict[str, Any]] = []
    all_statuses: list[str] = []
    required_category_ids = {item["id"] for item in required_slices}
    for item in required_slices:
        by_language: dict[str, Any] = {}
        for language in item["languages"]:
            matched = [
                case for case in selected
                if case["language"] == language and item["id"] in case["categories"]
            ]
            counts = _case_counts(matched)
            by_language[language] = counts
            all_statuses.append(counts["presence_status"])
        matrix.append({"id": item["id"], "title": item["title"], "by_language": by_language})

    ready_presence_count = sum(status == "ready_cases_present" for status in all_statuses)
    draft_only_count = sum(status == "draft_only" for status in all_statuses)
    excluded_only_count = sum(status == "excluded_only" for status in all_statuses)
    missing_count = sum(status == "missing" for status in all_statuses)
    all_required_slices_present = (
        bool(all_statuses)
        and missing_count == 0
        and draft_only_count == 0
        and excluded_only_count == 0
    )
    category_tags = {tag for case in selected for tag in case["categories"]}
    return {
        "report_schema_version": 1,
        "split": split,
        "final_audit": split == "final_holdout",
        "case_inventory": _case_counts(selected),
        "language_inventory": {
            language: _case_counts([case for case in selected if case["language"] == language])
            for language in sorted(LANGUAGES)
        },
        "slice_presence": {
            "ready_slice_language_count": ready_presence_count,
            "draft_only_slice_language_count": draft_only_count,
            "excluded_only_slice_language_count": excluded_only_count,
            "missing_slice_language_count": missing_count,
            "all_required_slices_have_ready_cases": all_required_slices_present,
            "interpretation": "Presence requires at least one ready case and does not establish adequate sample size, quality, or independence.",
        },
        "required_slices": matrix,
        "unmapped_category_tags": sorted(category_tags - required_category_ids),
        "non_case_dimensions": [
            "structured_response_validity",
            "unsupported_factual_claims",
            "disallowed_output_content",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path, help="JSONL case inventory")
    parser.add_argument("--split", choices=sorted(SPLITS), default="development")
    parser.add_argument(
        "--final-audit",
        action="store_true",
        help="explicitly authorize reporting the sealed final-holdout inventory",
    )
    parser.add_argument("--output", type=Path, help="write JSON report here; stdout if omitted")
    args = parser.parse_args(argv)
    if args.final_audit and args.split != "final_holdout":
        parser.error("--final-audit is valid only with --split final_holdout")
    if args.split == "final_holdout" and not args.final_audit:
        parser.error("final holdout requires --split final_holdout and --final-audit")
    if args.output:
        output_path = args.output.resolve()
        protected_paths = {args.cases.resolve(), REQUIRED_SLICES_PATH.resolve()}
        if output_path in protected_paths:
            parser.error("--output must not overwrite the case inventory or required-slice taxonomy")
    try:
        report = audit_cases(
            _read_cases(args.cases),
            split=args.split,
            allow_final_holdout=args.final_audit,
        )
        report["case_manifest_sha256"] = _sha256_file(args.cases)
        report["required_slices_sha256"] = _sha256_file(REQUIRED_SLICES_PATH)
    except (OSError, CoverageInputError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
