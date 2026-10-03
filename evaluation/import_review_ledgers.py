"""Combine controlled reviewer ledgers without approving evaluation cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.review_records import (  # noqa: E402
    ReviewRecordError,
    _validate_record,
    read_review_records,
    review_records_sha256,
    validate_case_review_records,
)
from evaluation.scorer import EvaluationInputError, _validate_case  # noqa: E402
from provenance.validate import evaluation_case_sha256  # noqa: E402


class LedgerImportError(ValueError):
    """Raised when case and reviewer files cannot be safely combined."""


def _validate_records(records: list[dict[str, Any]]) -> None:
    record_ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        try:
            _validate_record(record, index)
        except ReviewRecordError as exc:
            raise LedgerImportError(f"review record {index}: {exc}") from exc
        record_id = record["review_record_id"]
        if record_id in record_ids:
            raise LedgerImportError(f"duplicate review_record_id {record_id!r}")
        record_ids.add(record_id)


def _validate_existing_case_references(
    cases: list[dict[str, Any]], records: list[dict[str, Any]]
) -> None:
    records_by_id = {record["review_record_id"]: record for record in records}
    superseded_ids = {
        record_id for record in records for record_id in record["supersedes_record_ids"]
    }
    for case in cases:
        current_hash = evaluation_case_sha256(case)
        declared_ids = case["review"]["review_record_ids"]
        declared_records: list[dict[str, Any]] = []
        for record_id in declared_ids:
            record = records_by_id.get(record_id)
            if (
                record is None
                or record_id in superseded_ids
                or record["case_id"] != case["case_id"]
                or record["case_sha256"] != current_hash
            ):
                raise LedgerImportError(
                    f"case {case['case_id']!r} contains a review reference that is absent, stale, or superseded"
                )
            declared_records.append(record)
        declared_reviewers = set(case["review"]["reviewer_ids"])
        referenced_reviewers = {record["reviewer_id"] for record in declared_records}
        if declared_reviewers != referenced_reviewers:
            raise LedgerImportError(
                f"case {case['case_id']!r} reviewer_ids do not match its supplied review references"
            )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is not allowed")
        result[key] = value
    return result


def _read_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    case = json.loads(
                        line,
                        parse_constant=_reject_json_constant,
                        object_pairs_hook=_reject_duplicate_keys,
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    raise LedgerImportError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(case, dict):
                    raise LedgerImportError(f"{path}:{line_number}: each case must be an object")
                try:
                    _validate_case(case, line_number)
                except EvaluationInputError as exc:
                    raise LedgerImportError(f"{path}: {exc}") from exc
                if case["split"] != "development" or case["review_status"] != "draft":
                    raise LedgerImportError(
                        "ledger import accepts development drafts only; ready, excluded, and holdout rows are refused"
                    )
                expected_language_status = "not_required" if case["language"] == "en" else "pending"
                if (
                    case["review"]["annotation_status"] != "pending"
                    or case["review"]["language_review_status"] != expected_language_status
                ):
                    raise LedgerImportError(
                        f"case {case['case_id']!r} has non-pending review state; import will not alter it"
                    )
                if case["case_id"] in seen:
                    raise LedgerImportError(f"duplicate case_id {case['case_id']!r}")
                seen.add(case["case_id"])
                cases.append(case)
    except UnicodeDecodeError as exc:
        raise LedgerImportError(f"{path}: case inventory is not UTF-8") from exc
    if not cases:
        raise LedgerImportError("the development case inventory is empty")
    return cases


def _read_reviewer_ledgers(paths: list[Path]) -> list[dict[str, Any]]:
    if not paths:
        raise LedgerImportError("provide at least one controlled reviewer ledger")
    normalized_paths = [path.expanduser().resolve() for path in paths]
    if len(set(normalized_paths)) != len(normalized_paths):
        raise LedgerImportError("the same reviewer ledger was supplied more than once")

    records: list[dict[str, Any]] = []
    for path in normalized_paths:
        try:
            ledger_records = read_review_records(path)
        except (OSError, ReviewRecordError) as exc:
            raise LedgerImportError(f"{path}: could not read reviewer ledger: {exc}") from exc
        reviewer_ids = {
            record.get("reviewer_id")
            for record in ledger_records
            if isinstance(record.get("reviewer_id"), str)
        }
        if len(reviewer_ids) > 1:
            raise LedgerImportError(
                f"{path}: reviewer ledgers must contain records from one pseudonymous reviewer only"
            )
        records.extend(ledger_records)
    return records


def _active_current_records(
    case: dict[str, Any], records: list[dict[str, Any]], superseded_ids: set[str]
) -> list[dict[str, Any]]:
    current_hash = evaluation_case_sha256(case)
    return [
        record
        for record in records
        if record["case_id"] == case["case_id"]
        and record["case_sha256"] == current_hash
        and record["review_record_id"] not in superseded_ids
    ]


def _link_current_records(
    cases: list[dict[str, Any]], records: list[dict[str, Any]]
) -> dict[str, int]:
    _validate_records(records)
    for index, record in enumerate(records, start=1):
        if (
            not isinstance(record.get("review_record_id"), str)
            or not isinstance(record.get("case_id"), str)
            or not isinstance(record.get("case_sha256"), str)
            or not isinstance(record.get("reviewer_id"), str)
            or not isinstance(record.get("supersedes_record_ids"), list)
            or not all(isinstance(item, str) for item in record["supersedes_record_ids"])
        ):
            raise LedgerImportError(f"review record {index}: malformed identity or supersession fields")

    superseded_ids = {
        record_id for record in records for record_id in record["supersedes_record_ids"]
    }
    semantic_count = 0
    language_count = 0
    for case in cases:
        active = _active_current_records(case, records, superseded_ids)
        semantic_reviewers = [
            record["reviewer_id"]
            for record in active
            if record["record_type"] == "independent_annotation"
        ]
        language_reviewers = [
            record["reviewer_id"] for record in active if record["record_type"] == "language_review"
        ]
        if len(semantic_reviewers) != len(set(semantic_reviewers)):
            raise LedgerImportError(
                f"case {case['case_id']!r} has more than one active semantic annotation from the same reviewer"
            )
        if len(language_reviewers) > 1:
            raise LedgerImportError(f"case {case['case_id']!r} has more than one active language review")
        if set(semantic_reviewers) & set(language_reviewers):
            raise LedgerImportError(
                f"case {case['case_id']!r} uses one reviewer for both semantic and language review"
            )

        case["review"]["review_record_ids"] = sorted(
            record["review_record_id"] for record in active
        )
        case["review"]["reviewer_ids"] = sorted(
            {record["reviewer_id"] for record in active}
        )
        semantic_count += len(semantic_reviewers)
        language_count += len(language_reviewers)

    try:
        validate_case_review_records(cases, records)
    except ReviewRecordError as exc:
        raise LedgerImportError(f"combined review ledger is invalid: {exc}") from exc

    return {
        "semantic_annotation_count": semantic_count,
        "language_review_count": language_count,
        "active_review_record_count": sum(
            len(case["review"]["review_record_ids"]) for case in cases
        ),
    }


def _safe_output_path(path: Path, input_paths: set[Path]) -> Path:
    output = path.expanduser().resolve()
    if any((parent / ".git").exists() for parent in (output, *output.parents)):
        raise LedgerImportError("review output files must be stored outside the Git repository")
    if output in input_paths:
        raise LedgerImportError("review output must not overwrite an input case or ledger file")
    if output.suffix.lower() != ".jsonl":
        raise LedgerImportError("both output paths must end in .jsonl")
    if output.exists() and not output.is_file():
        raise LedgerImportError(f"output path is not a regular file: {output}")
    return output


def _serialize_jsonl(records: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for record in records
    ).encode("utf-8")


def _write_atomic(path: Path, payload: bytes, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise LedgerImportError(f"output already exists; use --force to replace it: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not force:
            raise LedgerImportError(f"output appeared during import; refusing to replace it: {path}")
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def import_ledgers(
    cases_path: Path,
    reviewer_ledger_paths: list[Path],
    output_cases_path: Path,
    output_ledger_path: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    cases_input = cases_path.expanduser().resolve()
    ledger_inputs = [path.expanduser().resolve() for path in reviewer_ledger_paths]
    input_paths = {cases_input, *ledger_inputs}
    output_cases = _safe_output_path(output_cases_path, input_paths)
    output_ledger = _safe_output_path(output_ledger_path, input_paths)
    if output_cases == output_ledger:
        raise LedgerImportError("case output and combined-ledger output must be different files")

    cases = _read_cases(cases_input)
    records = _read_reviewer_ledgers(ledger_inputs)
    _validate_records(records)
    _validate_existing_case_references(cases, records)
    counts = _link_current_records(cases, records)

    # Recheck both destinations before writing either output to avoid replacing prior work.
    if not force and (output_cases.exists() or output_ledger.exists()):
        existing = output_cases if output_cases.exists() else output_ledger
        raise LedgerImportError(f"output already exists; use --force to replace it: {existing}")

    ledger_bytes = _serialize_jsonl(records)
    cases_bytes = _serialize_jsonl(cases)
    _write_atomic(output_ledger, ledger_bytes, force=force)
    _write_atomic(output_cases, cases_bytes, force=force)

    return {
        "case_count": len(cases),
        "draft_case_count": sum(case["review_status"] == "draft" for case in cases),
        "ready_case_count": sum(case["review_status"] == "ready" for case in cases),
        "review_record_count": len(records),
        "review_records_sha256": review_records_sha256(records),
        "case_file_sha256": "sha256:" + hashlib.sha256(cases_bytes).hexdigest(),
        **counts,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine controlled reviewer ledgers and link them to development drafts without promoting cases."
    )
    parser.add_argument("--cases", type=Path, required=True, help="Controlled development draft JSONL.")
    parser.add_argument(
        "--review-ledger",
        type=Path,
        action="append",
        required=True,
        help="One pseudonymous reviewer's controlled JSONL ledger; repeat once per reviewer.",
    )
    parser.add_argument("--output-cases", type=Path, required=True, help="Combined development cases JSONL.")
    parser.add_argument("--output-ledger", type=Path, required=True, help="Combined review ledger JSONL.")
    parser.add_argument(
        "--force", action="store_true", help="Replace existing output files after validation succeeds."
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = import_ledgers(
        args.cases,
        args.review_ledger,
        args.output_cases,
        args.output_ledger,
        force=args.force,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LedgerImportError, ReviewRecordError, EvaluationInputError, OSError) as exc:
        print(f"review ledger import: {exc}", file=sys.stderr)
        raise SystemExit(2)
