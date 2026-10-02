"""Run a synchronous candidate callback over provenance-approved cases."""

from __future__ import annotations

from typing import Any, Protocol

from evaluation.input_adapter import CandidateInput, build_candidate_input
from evaluation.scorer import EvaluationInputError, select_evaluation_cases
from provenance.validate import ProvenanceError, validate_bundle


PREDICTION_SCHEMA_VERSION = 1


class Candidate(Protocol):
    """A model- or architecture-specific implementation of Lumi prediction."""

    def predict(self, candidate_input: CandidateInput, /) -> str:
        """Return the raw model output string without parsing or repairing it."""


class CandidateOutputError(TypeError):
    """Raised when candidate output cannot be retained as a raw string."""


def run_candidate_cases(
    cases: list[dict[str, Any]],
    candidate: Candidate,
    provenance_records: list[dict[str, Any]],
    source_manifest: dict[str, Any],
    generation_manifest: dict[str, Any],
    *,
    split: str = "development",
    allow_final_holdout: bool = False,
) -> list[dict[str, Any]]:
    """Validate and run ready cases, then wrap raw output for the existing scorer.

    Case structure/readiness, selected-split membership, and source/generation/sample
    provenance are checked before invoking candidate code. Final holdout runs require
    explicit opt-in. This is a synchronous callback runner; it does not load a model,
    execute tools, or simulate changes to application state.
    """
    selected = select_evaluation_cases(cases, split, allow_final_holdout)
    try:
        validate_bundle(source_manifest, generation_manifest, provenance_records, selected)
    except ProvenanceError as exc:
        raise EvaluationInputError(str(exc)) from exc

    predictions: list[dict[str, Any]] = []
    for case in selected:
        raw_output = candidate.predict(build_candidate_input(case))
        if not isinstance(raw_output, str):
            raise CandidateOutputError("candidate must return its raw output as a string")
        predictions.append({
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "case_id": case["case_id"],
            "raw_output": raw_output,
        })
    return predictions
