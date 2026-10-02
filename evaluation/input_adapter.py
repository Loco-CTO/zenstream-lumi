"""Project a validated evaluation case into candidate-visible input."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, TypedDict


CANDIDATE_INPUT_SCHEMA_VERSION = 1


class CandidateInput(TypedDict):
    """Case content exposed to a candidate without evaluation annotations."""

    schema_version: int
    turns: list[dict[str, str]]
    trusted_context: dict[str, Any] | None


def build_candidate_input(case: Mapping[str, Any]) -> CandidateInput:
    """Return only conversation turns and static trusted context from a v2 case.

    Callers must validate the case and its provenance before using this projection.
    The returned values are deep copies so candidate code cannot mutate the authored
    case. Gold labels, split/category metadata, identifiers, and review data are not
    exposed to the candidate.
    """
    return {
        "schema_version": CANDIDATE_INPUT_SCHEMA_VERSION,
        "turns": deepcopy(case["turns"]),
        "trusted_context": deepcopy(case["trusted_context"]),
    }
