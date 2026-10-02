"""Project a validated evaluation case into candidate-visible input."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, TypedDict


CANDIDATE_INPUT_SCHEMA_VERSION = 1
INTERACTIVE_CANDIDATE_INPUT_SCHEMA_VERSION = 2


class CandidateInput(TypedDict):
    """Case content exposed to a candidate without evaluation annotations."""

    schema_version: int
    turns: list[dict[str, str]]
    trusted_context: dict[str, Any] | None


class InteractiveCandidateInput(CandidateInput):
    """Candidate view for one step in an interactive fixture-only scenario."""

    tools: list[dict[str, Any]]
    tool_history: list[dict[str, Any]]


def build_candidate_input(
    case: Mapping[str, Any],
    *,
    tool_history: list[dict[str, Any]] | None = None,
) -> CandidateInput | InteractiveCandidateInput:
    """Project conversation, context, and only the currently available tool observations.

    Callers must validate the case and its provenance before using this projection.
    The returned values are deep copies so candidate code cannot mutate the authored
    case. Gold labels, fixture outcomes not yet returned, split/category metadata,
    identifiers, and review data are not exposed to the candidate.
    """
    projected: CandidateInput = {
        "schema_version": CANDIDATE_INPUT_SCHEMA_VERSION,
        "turns": deepcopy(case["turns"]),
        "trusted_context": deepcopy(case["trusted_context"]),
    }
    scenario = case.get("tool_scenario")
    if scenario is None:
        if tool_history:
            raise ValueError("tool history requires a non-null tool_scenario")
        return projected
    return {
        "schema_version": INTERACTIVE_CANDIDATE_INPUT_SCHEMA_VERSION,
        "turns": projected["turns"],
        "trusted_context": projected["trusted_context"],
        "tools": deepcopy(scenario["tools"]),
        "tool_history": deepcopy(tool_history or []),
    }
