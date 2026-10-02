#!/usr/bin/env python3
"""Score Lumi's canonical intent evaluation format using only the Python stdlib.

The canonical format is an evaluation adapter, not the future ZenStream wire
protocol. It deliberately separates structural validity from semantic accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from provenance.validate import (  # noqa: E402
    ProvenanceError,
    load_json_document,
    validate_bundle as validate_provenance_bundle,
    validate_files as validate_provenance_files,
)


THRESHOLDS_PATH = Path(__file__).with_name("thresholds.json")
DECISIONS = {"act", "no_action", "clarify", "respond"}
OUTPUT_KEYS = {"decision", "action", "arguments", "requires_clarification"}
OPTIONAL_OUTPUT_KEYS = {"message"}
LANGUAGES = {"en", "ja", "en_ja"}
SPLITS = {"development", "final_holdout"}
_Z_95 = 1.959963984540054


class EvaluationInputError(ValueError):
    """Raised when benchmark or prediction records are incomplete or unsafe to score."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise EvaluationInputError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise EvaluationInputError(f"{path}:{line_number}: each JSONL row must be an object")
            records.append(record)
    return records


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_semantic_output(value: Any, *, allow_message: bool) -> str | None:
    if not isinstance(value, dict):
        return "output must be a JSON object"
    allowed = OUTPUT_KEYS | (OPTIONAL_OUTPUT_KEYS if allow_message else set())
    if not OUTPUT_KEYS.issubset(value):
        missing = sorted(OUTPUT_KEYS - value.keys())
        return f"output is missing required field(s): {', '.join(missing)}"
    extra = sorted(value.keys() - allowed)
    if extra:
        return f"output contains unsupported field(s): {', '.join(extra)}"
    if not isinstance(value["decision"], str) or value["decision"] not in DECISIONS:
        return "decision must be one of act, no_action, clarify, or respond"
    if value["action"] is not None and not isinstance(value["action"], str):
        return "action must be a string or null"
    if not isinstance(value["arguments"], dict):
        return "arguments must be an object"
    if not isinstance(value["requires_clarification"], bool):
        return "requires_clarification must be a boolean"
    if "message" in value and not isinstance(value["message"], str):
        return "message must be a string when present"

    decision = value["decision"]
    if decision == "act":
        if not _is_nonempty_string(value["action"]):
            return "act decisions require a nonempty action"
        if value["requires_clarification"]:
            return "act decisions cannot also require clarification"
    else:
        if value["action"] is not None:
            return f"{decision} decisions require action=null"
        if value["arguments"]:
            return f"{decision} decisions require empty arguments"
        if decision == "clarify" and not value["requires_clarification"]:
            return "clarify decisions require requires_clarification=true"
        if decision != "clarify" and value["requires_clarification"]:
            return f"{decision} decisions require requires_clarification=false"
    return None


def _validate_case(record: dict[str, Any], line_number: int) -> None:
    if not isinstance(record, dict):
        raise EvaluationInputError(f"case row {line_number} must be an object")
    required = {
        "schema_version", "case_id", "family_id", "split", "review_status",
        "language", "categories", "turns", "gold", "provenance_record_id", "review",
    }
    missing = sorted(required - record.keys())
    extra = sorted(record.keys() - (required | {"context_id"}))
    prefix = f"case row {line_number}"
    if missing:
        raise EvaluationInputError(f"{prefix}: missing field(s): {', '.join(missing)}")
    if extra:
        raise EvaluationInputError(f"{prefix}: unsupported field(s): {', '.join(extra)}")
    if (
        not isinstance(record["schema_version"], int)
        or isinstance(record["schema_version"], bool)
        or record["schema_version"] != 1
    ):
        raise EvaluationInputError(f"{prefix}: schema_version must be 1")
    for field in ("case_id", "family_id", "provenance_record_id"):
        if not _is_nonempty_string(record[field]):
            raise EvaluationInputError(f"{prefix}: {field} must be a nonempty string")
    if not isinstance(record["split"], str) or record["split"] not in SPLITS:
        raise EvaluationInputError(f"{prefix}: split must be development or final_holdout")
    if (
        not isinstance(record["review_status"], str)
        or record["review_status"] not in {"draft", "ready", "excluded"}
    ):
        raise EvaluationInputError(f"{prefix}: invalid review_status")
    if not isinstance(record["language"], str) or record["language"] not in LANGUAGES:
        raise EvaluationInputError(f"{prefix}: language must be en, ja, or en_ja")
    if not isinstance(record["categories"], list) or not record["categories"]:
        raise EvaluationInputError(f"{prefix}: categories must be a nonempty list")
    if not all(_is_nonempty_string(tag) for tag in record["categories"]):
        raise EvaluationInputError(f"{prefix}: every category must be a nonempty string")
    if len(set(record["categories"])) != len(record["categories"]):
        raise EvaluationInputError(f"{prefix}: categories must be unique")
    turns = record["turns"]
    if not isinstance(turns, list) or not turns:
        raise EvaluationInputError(f"{prefix}: turns must be a nonempty list")
    for turn in turns:
        if not isinstance(turn, dict) or set(turn) != {"role", "text"}:
            raise EvaluationInputError(f"{prefix}: each turn must contain only role and text")
        if (
            not isinstance(turn["role"], str)
            or turn["role"] not in {"user", "assistant"}
            or not _is_nonempty_string(turn["text"])
        ):
            raise EvaluationInputError(f"{prefix}: invalid turn role or empty text")
    if "context_id" in record and record["context_id"] is not None and not isinstance(record["context_id"], str):
        raise EvaluationInputError(f"{prefix}: context_id must be a string or null")
    error = _validate_semantic_output(record["gold"], allow_message=False)
    if error:
        raise EvaluationInputError(f"{prefix}: invalid gold output: {error}")

    review = record["review"]
    if not isinstance(review, dict) or not {"annotation_status", "language_review_status", "reviewer_ids"}.issubset(review):
        raise EvaluationInputError(f"{prefix}: review metadata is incomplete")
    if set(review) - {"annotation_status", "language_review_status", "reviewer_ids", "adjudication_note"}:
        raise EvaluationInputError(f"{prefix}: unsupported review metadata")
    if "adjudication_note" in review and not isinstance(review["adjudication_note"], str):
        raise EvaluationInputError(f"{prefix}: adjudication_note must be a string")
    if not isinstance(review["annotation_status"], str) or review["annotation_status"] not in {"pending", "approved"}:
        raise EvaluationInputError(f"{prefix}: invalid annotation_status")
    if not isinstance(review["language_review_status"], str) or review["language_review_status"] not in {"not_required", "pending", "approved"}:
        raise EvaluationInputError(f"{prefix}: invalid language_review_status")
    reviewers = review["reviewer_ids"]
    if not isinstance(reviewers, list) or not all(_is_nonempty_string(item) for item in reviewers):
        raise EvaluationInputError(f"{prefix}: reviewer_ids must be a list of nonempty strings")
    if len(set(reviewers)) != len(reviewers):
        raise EvaluationInputError(f"{prefix}: reviewer_ids must be unique")
    if record["review_status"] == "ready":
        if review["annotation_status"] != "approved" or not reviewers:
            raise EvaluationInputError(f"{prefix}: ready cases need approved annotation and at least one reviewer")
        if record["language"] in {"ja", "en_ja"} and review["language_review_status"] != "approved":
            raise EvaluationInputError(f"{prefix}: Japanese and code-switch cases need language review approval")
        if record["language"] == "en" and review["language_review_status"] == "pending":
            raise EvaluationInputError(f"{prefix}: English cases cannot be ready with language review pending")


def _validate_prediction_wrapper(record: dict[str, Any], line_number: int) -> None:
    if not isinstance(record, dict):
        raise EvaluationInputError(f"prediction row {line_number} must be an object")
    allowed = {"schema_version", "case_id", "raw_output", "grounding_review"}
    if set(record) - allowed:
        raise EvaluationInputError(f"prediction row {line_number}: unsupported field(s)")
    version = record.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version != 1:
        raise EvaluationInputError(f"prediction row {line_number}: schema_version must be 1")
    if not _is_nonempty_string(record.get("case_id")):
        raise EvaluationInputError(f"prediction row {line_number}: case_id must be a nonempty string")
    if not isinstance(record.get("raw_output"), str):
        raise EvaluationInputError(f"prediction row {line_number}: raw_output must be a string")
    if "grounding_review" not in record:
        return
    review = record["grounding_review"]
    if not isinstance(review, dict) or set(review) - {"status", "claim_count", "unsupported_claim_count", "reviewer_id"}:
        raise EvaluationInputError(f"prediction row {line_number}: invalid grounding_review")
    if not isinstance(review.get("status"), str) or review["status"] not in {"pending", "reviewed"}:
        raise EvaluationInputError(f"prediction row {line_number}: invalid grounding review status")
    if review["status"] == "reviewed":
        if not isinstance(review.get("claim_count"), int) or isinstance(review.get("claim_count"), bool) or review["claim_count"] < 0:
            raise EvaluationInputError(f"prediction row {line_number}: reviewed claim_count must be a nonnegative integer")
        if not isinstance(review.get("unsupported_claim_count"), int) or isinstance(review.get("unsupported_claim_count"), bool) or review["unsupported_claim_count"] < 0:
            raise EvaluationInputError(f"prediction row {line_number}: unsupported_claim_count must be a nonnegative integer")
        if review["unsupported_claim_count"] > review["claim_count"]:
            raise EvaluationInputError(f"prediction row {line_number}: unsupported claims cannot exceed reviewed claims")
        if not _is_nonempty_string(review.get("reviewer_id")):
            raise EvaluationInputError(f"prediction row {line_number}: reviewed grounding records need reviewer_id")


def _normalise(value: Any) -> Any:
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFC", value)
        return " ".join(normalized.split()).casefold()
    if isinstance(value, list):
        return [_normalise(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalise(value[key]) for key in sorted(value)}
    return value


def _semantically_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    fields = ("decision", "action", "arguments", "requires_clarification")
    return all(_normalise(left[field]) == _normalise(right[field]) for field in fields)


def _action_selection_equal(prediction: dict[str, Any], gold: dict[str, Any]) -> bool:
    return prediction["decision"] == gold["decision"] and _normalise(prediction["action"]) == _normalise(gold["action"])


def _argument_equal(prediction: dict[str, Any], gold: dict[str, Any]) -> bool:
    return _normalise(prediction["arguments"]) == _normalise(gold["arguments"])


def _wilson_interval(successes: int, total: int, confidence: float = 0.95) -> list[float] | None:
    if total == 0:
        return None
    # The evaluator currently uses the standard two-sided 95% Wilson interval.
    # Keep this fixed and explicit until a confidence-method comparison is needed.
    if confidence != 0.95:
        raise ValueError("only 95% confidence intervals are currently supported")
    p = successes / total
    z2 = _Z_95 * _Z_95
    denominator = 1.0 + z2 / total
    centre = (p + z2 / (2.0 * total)) / denominator
    margin = (_Z_95 * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def _binomial_cdf(errors: int, total: int, probability: float) -> float:
    if errors >= total:
        return 1.0
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    logs = [
        math.lgamma(total + 1) - math.lgamma(k + 1) - math.lgamma(total - k + 1)
        + k * log_p + (total - k) * log_q
        for k in range(errors + 1)
    ]
    peak = max(logs)
    return math.exp(peak) * sum(math.exp(value - peak) for value in logs)


def _one_sided_error_upper(errors: int, total: int, confidence: float = 0.95) -> float | None:
    """Exact one-sided Clopper-Pearson upper bound for a binomial error rate."""
    if total == 0:
        return None
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if errors >= total:
        return 1.0
    alpha = 1.0 - confidence
    low = errors / total
    high = 1.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if _binomial_cdf(errors, total, middle) > alpha:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _rate(successes: int, total: int, confidence: float = 0.95) -> dict[str, Any]:
    errors = total - successes
    return {
        "successes": successes,
        "errors": errors,
        "total": total,
        "rate": successes / total if total else None,
        "wilson_95": _wilson_interval(successes, total, confidence),
        "one_sided_95_error_upper": _one_sided_error_upper(errors, total, confidence),
    }


def _event_rate(events: int, total: int, confidence: float = 0.95) -> dict[str, Any]:
    """Report an undesirable event rate and its one-sided exact upper bound."""
    return {
        "events": events,
        "total": total,
        "rate": events / total if total else None,
        "wilson_95": _wilson_interval(events, total, confidence),
        "one_sided_95_event_upper": _one_sided_error_upper(events, total, confidence),
    }


def _select_cases(cases: list[dict[str, Any]], split: str, allow_final_holdout: bool) -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise EvaluationInputError("split must be development or final_holdout")
    if split == "final_holdout" and not allow_final_holdout:
        raise EvaluationInputError("final holdout scoring requires the explicit --final-audit flag")
    case_ids: set[str] = set()
    family_splits: dict[str, str] = {}
    for index, case in enumerate(cases, start=1):
        _validate_case(case, index)
        case_id = case["case_id"]
        if case_id in case_ids:
            raise EvaluationInputError(f"duplicate case_id {case_id!r}")
        case_ids.add(case_id)
        family_id = case["family_id"]
        previous = family_splits.setdefault(family_id, case["split"])
        if previous != case["split"]:
            raise EvaluationInputError(f"case family {family_id!r} crosses evaluation splits")
    selected = [case for case in cases if case["split"] == split]
    if not selected:
        raise EvaluationInputError(f"no cases found for split {split!r}")
    not_ready = [case["case_id"] for case in selected if case["review_status"] != "ready"]
    if not_ready:
        preview = ", ".join(not_ready[:5])
        raise EvaluationInputError(f"selected split contains unreviewed or excluded cases: {preview}")
    return selected


def _evaluate_prediction(raw_output: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(raw_output, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        return None, f"invalid JSON: {exc}"
    error = _validate_semantic_output(parsed, allow_message=True)
    if error:
        return None, error
    return parsed, None


def score_records(
    cases: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
    provenance_records: list[dict[str, Any]],
    source_manifest: dict[str, Any],
    generation_manifest: dict[str, Any],
    *,
    split: str = "development",
    allow_final_holdout: bool = False,
    candidate_version: str | None = None,
    confidence: float = 0.95,
) -> dict[str, Any]:
    selected = _select_cases(cases, split, allow_final_holdout)
    try:
        validate_provenance_bundle(source_manifest, generation_manifest, provenance_records, selected)
    except ProvenanceError as exc:
        raise EvaluationInputError(str(exc)) from exc
    available = {case["case_id"]: case for case in selected}
    predictions: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(prediction_records, start=1):
        _validate_prediction_wrapper(record, index)
        case_id = record["case_id"]
        if case_id not in available:
            raise EvaluationInputError(f"prediction references unknown or out-of-split case_id {case_id!r}")
        if case_id in predictions:
            raise EvaluationInputError(f"duplicate prediction case_id {case_id!r}")
        predictions[case_id] = record

    missing_count = len(set(available) - set(predictions))
    decoded: dict[str, dict[str, Any] | None] = {}
    output_errors: dict[str, str | None] = {}
    for case_id in available:
        record = predictions.get(case_id)
        if record is None:
            decoded[case_id], output_errors[case_id] = None, "missing prediction"
        else:
            decoded[case_id], output_errors[case_id] = _evaluate_prediction(record["raw_output"])

    def rate_for(subset: Iterable[dict[str, Any]], predicate: Callable[[dict[str, Any], dict[str, Any]], bool]) -> dict[str, Any]:
        selected_subset = list(subset)
        successes = 0
        for case in selected_subset:
            prediction = decoded[case["case_id"]]
            if prediction is not None and predicate(prediction, case["gold"]):
                successes += 1
        return _rate(successes, len(selected_subset), confidence)

    all_cases = list(available.values())
    exact = lambda prediction, gold: _semantically_equal(prediction, gold)
    metrics: dict[str, Any] = {
        "overall_semantic_exact_match": rate_for(all_cases, exact),
        "structured_response_validity": _rate(sum(value is not None for value in decoded.values()), len(all_cases), confidence),
        "simple_action_selection": rate_for(
            (case for case in all_cases if "simple_action" in case["categories"]),
            _action_selection_equal,
        ),
        "argument_extraction": rate_for(
            (case for case in all_cases if case["gold"]["decision"] == "act"),
            _argument_equal,
        ),
        "negation_no_action_correctness": rate_for(
            (case for case in all_cases if {"negation", "no_action"} & set(case["categories"])),
            exact,
        ),
        "english_zenstream_tasks": rate_for((case for case in all_cases if case["language"] == "en"), exact),
        "japanese_zenstream_tasks": rate_for((case for case in all_cases if case["language"] == "ja"), exact),
        "code_switching": rate_for((case for case in all_cases if case["language"] == "en_ja"), exact),
        "multi_turn_reference_resolution": rate_for(
            (case for case in all_cases if "multi_turn_reference_resolution" in case["categories"]),
            exact,
        ),
        "difficult_ambiguous_requests": rate_for(
            (case for case in all_cases if "difficult_ambiguous" in case["categories"]),
            exact,
        ),
    }

    non_action_gold_cases = [case for case in all_cases if case["gold"]["decision"] != "act"]
    false_actions = sum(
        1 for case in non_action_gold_cases
        if (prediction := decoded[case["case_id"]]) is not None
        and prediction["decision"] == "act"
    )
    metrics["false_action_rate"] = _event_rate(false_actions, len(non_action_gold_cases), confidence)

    reviewed_predictions = [
        predictions[case_id]["grounding_review"]
        for case_id in available
        if case_id in predictions
        and predictions[case_id].get("grounding_review", {}).get("status") == "reviewed"
    ]
    audited_claims = sum(item["claim_count"] for item in reviewed_predictions)
    unsupported_claims = sum(item["unsupported_claim_count"] for item in reviewed_predictions)
    metrics["unsupported_factual_claim_rate"] = _event_rate(unsupported_claims, audited_claims, confidence)
    metrics["grounding_review_coverage"] = _rate(len(reviewed_predictions), len(all_cases), confidence)
    metrics["grounding_review_counts"] = {
        "reviewed_responses": len(reviewed_predictions),
        "responses": len(all_cases),
        "claims": audited_claims,
        "unsupported_claims": unsupported_claims,
    }

    category_metrics: dict[str, Any] = {}
    for category in sorted({tag for case in all_cases for tag in case["categories"]}):
        category_metrics[category] = rate_for(
            (case for case in all_cases if category in case["categories"]),
            exact,
        )

    thresholds_doc = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    target_assessment: dict[str, Any] = {}
    for name, spec in thresholds_doc["targets"].items():
        result = metrics[name]
        rate = result["rate"]
        minimum = spec["minimum"]
        if rate is None:
            status = "not_measured"
        elif rate < minimum:
            status = "below_target"
        elif spec["evidence"] == "one_sided_exact_error_bound":
            error_bound = result["one_sided_95_error_upper"]
            status = "meets_target_with_95pct_bound" if error_bound <= 1.0 - minimum else "inconclusive_sample_evidence"
        else:
            wilson = result["wilson_95"]
            status = "meets_point_target" if wilson and wilson[0] >= minimum else "point_target_met_interval_crosses_target"
        target_assessment[name] = {
            "minimum": minimum,
            "status": status,
            "evidence_method": spec["evidence"],
        }

    return {
        "report_schema_version": 1,
        "candidate_version": candidate_version,
        "split": split,
        "final_audit": split == "final_holdout",
        "case_count": len(all_cases),
        "prediction_count": len(prediction_records),
        "missing_prediction_count": missing_count,
        "invalid_response_count": sum(value is None for value in decoded.values()),
        "invalid_response_examples": {
            case_id: error for case_id, error in output_errors.items() if error is not None
        },
        "metrics": metrics,
        "metrics_by_category": category_metrics,
        "target_assessment": target_assessment,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def score_files(
    cases_path: Path,
    predictions_path: Path,
    provenance_records_path: Path,
    *,
    split: str,
    allow_final_holdout: bool,
    candidate_version: str | None = None,
    source_manifest_path: Path = PROJECT_ROOT / "provenance" / "data_sources.json",
    generation_manifest_path: Path = PROJECT_ROOT / "provenance" / "synthetic_data.json",
) -> dict[str, Any]:
    cases = _read_jsonl(cases_path)
    predictions = _read_jsonl(predictions_path)
    provenance_records = _read_jsonl(provenance_records_path)
    validate_provenance_files(
        source_manifest_path,
        generation_manifest_path,
        provenance_records_path,
        cases_path,
    )
    source_manifest = load_json_document(source_manifest_path)
    generation_manifest = load_json_document(generation_manifest_path)
    report = score_records(
        cases,
        predictions,
        provenance_records,
        source_manifest,
        generation_manifest,
        split=split,
        allow_final_holdout=allow_final_holdout,
        candidate_version=candidate_version,
    )
    report["case_manifest_sha256"] = _sha256(cases_path)
    report["prediction_file_sha256"] = _sha256(predictions_path)
    report["provenance_records_sha256"] = _sha256(provenance_records_path)
    report["source_manifest_sha256"] = _sha256(source_manifest_path)
    report["synthetic_manifest_sha256"] = _sha256(generation_manifest_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path, help="JSONL evaluation cases")
    parser.add_argument("--predictions", required=True, type=Path, help="JSONL raw prediction wrappers")
    parser.add_argument("--provenance-records", required=True, type=Path, help="sample-level provenance JSONL for the evaluation cases")
    parser.add_argument("--sources", type=Path, default=PROJECT_ROOT / "provenance" / "data_sources.json", help="data-source manifest JSON")
    parser.add_argument("--generations", type=Path, default=PROJECT_ROOT / "provenance" / "synthetic_data.json", help="synthetic-generation manifest JSON")
    parser.add_argument("--split", choices=sorted(SPLITS), default="development")
    parser.add_argument("--final-audit", action="store_true", help="explicitly authorize scoring the sealed final holdout")
    parser.add_argument("--candidate-version", help="candidate identifier written into the report")
    parser.add_argument("--output", type=Path, help="write JSON report to this path; stdout if omitted")
    args = parser.parse_args(argv)
    if args.final_audit and args.split != "final_holdout":
        parser.error("--final-audit is valid only with --split final_holdout")
    try:
        report = score_files(
            args.cases,
            args.predictions,
            args.provenance_records,
            split=args.split,
            allow_final_holdout=args.final_audit,
            candidate_version=args.candidate_version,
            source_manifest_path=args.sources,
            generation_manifest_path=args.generations,
        )
    except (OSError, EvaluationInputError, ProvenanceError, json.JSONDecodeError) as exc:
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
