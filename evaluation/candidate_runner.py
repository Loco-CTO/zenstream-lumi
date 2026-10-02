"""Run a synchronous candidate callback over provenance-approved cases."""

from __future__ import annotations

from typing import Any, Protocol

from evaluation.input_adapter import CandidateInput, build_candidate_input
from evaluation.scorer import EvaluationInputError, select_evaluation_cases
from evaluation.tool_trajectory import FixtureOnlyToolSimulator, parse_tool_call
from provenance.validate import ProvenanceError, validate_bundle


PREDICTION_SCHEMA_VERSION = 3
TOOL_TRAJECTORY_PREDICTION_VERSION = 1


class Candidate(Protocol):
    """A model- or architecture-specific implementation of Lumi prediction."""

    def predict(self, candidate_input: CandidateInput, /) -> str:
        """Return a raw final output or tool-call envelope without repairing it.

        For cases with a tool scenario, the runner calls predict repeatedly with
        accumulated fixture observations until the candidate returns a final output
        or reaches a scenario limit.
        """


class CandidateOutputError(TypeError):
    """Raised when candidate output cannot be retained as a raw string."""


def _run_tool_scenario(case: dict[str, Any], candidate: Candidate) -> dict[str, Any]:
    scenario = case["tool_scenario"]
    simulator = FixtureOnlyToolSimulator(scenario)
    history: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    tool_call_count = 0
    stop_reason = "step_limit"
    raw_output = ""

    for _ in range(scenario["max_model_steps"]):
        raw_output = candidate.predict(build_candidate_input(case, tool_history=history))
        if not isinstance(raw_output, str):
            raise CandidateOutputError("candidate must return its raw output as a string")
        step_kind, tool_call = parse_tool_call(raw_output)
        if step_kind == "invalid_tool_call":
            steps.append({"raw_output": raw_output, "tool_call": None, "observation": None})
            stop_reason = "invalid_tool_call"
            break
        if step_kind == "final":
            steps.append({"raw_output": raw_output, "tool_call": None, "observation": None})
            stop_reason = "final"
            break

        assert tool_call is not None
        tool_call_count += 1
        effect_by_name = {tool["name"]: tool["effect"] for tool in scenario["tools"]}
        if tool_call_count > scenario["max_tool_calls"]:
            observation = {
                "status": "call_limit",
                "payload": {},
                "simulated": True,
                "effect": effect_by_name.get(tool_call["tool"], "unknown"),
            }
            steps.append({
                "raw_output": raw_output,
                "tool_call": tool_call,
                "observation": observation,
            })
            stop_reason = "call_limit"
            break

        observation = simulator.invoke(tool_call["tool"], tool_call["arguments"])
        steps.append({
            "raw_output": raw_output,
            "tool_call": tool_call,
            "observation": observation,
        })
        history.append({
            "tool": tool_call["tool"],
            "arguments": tool_call["arguments"],
            "observation": observation,
        })
    else:
        stop_reason = "step_limit"

    return {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "case_id": case["case_id"],
        "raw_output": raw_output,
        "trajectory": {
            "schema_version": TOOL_TRAJECTORY_PREDICTION_VERSION,
            "stop_reason": stop_reason,
            "steps": steps,
        },
    }


def run_candidate_cases(
    cases: list[dict[str, Any]],
    candidate: Candidate,
    provenance_records: list[dict[str, Any]],
    source_manifest: dict[str, Any],
    generation_manifest: dict[str, Any],
    *,
    review_records: list[dict[str, Any]],
    split: str = "development",
    allow_final_holdout: bool = False,
) -> list[dict[str, Any]]:
    """Validate and run ready cases, then wrap raw output for the existing scorer.

    Case structure/readiness, the current independent review ledger, selected-split
    membership, and source/generation/sample provenance are checked before invoking
    candidate code. Final holdout runs require explicit opt-in. This is a synchronous
    callback runner; it does not load a model, execute tools, or simulate changes to
    application state.
    """
    selected = select_evaluation_cases(
        cases,
        review_records,
        split=split,
        allow_final_holdout=allow_final_holdout,
    )
    try:
        validate_bundle(source_manifest, generation_manifest, provenance_records, selected)
    except ProvenanceError as exc:
        raise EvaluationInputError(str(exc)) from exc

    predictions: list[dict[str, Any]] = []
    for case in selected:
        if case["tool_scenario"] is not None:
            predictions.append(_run_tool_scenario(case, candidate))
            continue
        raw_output = candidate.predict(build_candidate_input(case))
        if not isinstance(raw_output, str):
            raise CandidateOutputError("candidate must return its raw output as a string")
        predictions.append({
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "case_id": case["case_id"],
            "raw_output": raw_output,
        })
    return predictions
