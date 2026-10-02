"""Validate and replay fixture-only interactive tool trajectories."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any


TOOL_SCENARIO_SCHEMA_VERSION = 1
TOOL_TRAJECTORY_SCHEMA_VERSION = 1
MAX_MODEL_STEPS = 32
MAX_TOOL_CALLS = 16
TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*$")
TOOL_RESULT_STATUSES = {
    "success", "empty", "not_found", "unavailable", "permission_denied", "timeout", "error",
}
OBSERVATION_STATUSES = TOOL_RESULT_STATUSES | {"unknown_tool", "fixture_mismatch", "call_limit"}


class ToolTrajectoryError(ValueError):
    """Raised when an authored tool scenario or trajectory record is malformed."""


def _valid_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _same_json_value(left: Any, right: Any) -> bool:
    try:
        return _canonical_json(left) == _canonical_json(right)
    except (TypeError, ValueError):
        return False


def _validate_tool_call(value: Any, prefix: str) -> None:
    if not isinstance(value, dict) or set(value) != {"tool", "arguments"}:
        raise ToolTrajectoryError(f"{prefix} must contain only tool and arguments")
    if not _valid_nonempty_string(value["tool"]) or not TOOL_NAME_PATTERN.fullmatch(value["tool"]):
        raise ToolTrajectoryError(f"{prefix}.tool is not a valid tool name")
    if not isinstance(value["arguments"], dict):
        raise ToolTrajectoryError(f"{prefix}.arguments must be an object")
    try:
        _canonical_json(value["arguments"])
    except (TypeError, ValueError) as exc:
        raise ToolTrajectoryError(f"{prefix}.arguments must contain finite JSON values") from exc


def validate_tool_scenario(scenario: Any) -> None:
    """Check the static fixture schema without loading code or contacting a service."""
    if scenario is None:
        return
    required = {"schema_version", "tools", "fixtures", "max_model_steps", "max_tool_calls"}
    if not isinstance(scenario, dict) or set(scenario) != required:
        raise ToolTrajectoryError("tool_scenario must contain only its version, tools, fixtures, and limits")
    version = scenario["schema_version"]
    if not isinstance(version, int) or isinstance(version, bool) or version != TOOL_SCENARIO_SCHEMA_VERSION:
        raise ToolTrajectoryError("tool_scenario schema_version must be 1")
    max_steps = scenario["max_model_steps"]
    max_calls = scenario["max_tool_calls"]
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or not 1 <= max_steps <= MAX_MODEL_STEPS:
        raise ToolTrajectoryError(f"tool_scenario max_model_steps must be from 1 to {MAX_MODEL_STEPS}")
    if not isinstance(max_calls, int) or isinstance(max_calls, bool) or not 0 <= max_calls <= MAX_TOOL_CALLS:
        raise ToolTrajectoryError(f"tool_scenario max_tool_calls must be from 0 to {MAX_TOOL_CALLS}")
    if max_calls >= max_steps:
        raise ToolTrajectoryError("tool_scenario must leave at least one model step for a final response")

    tools = scenario["tools"]
    if not isinstance(tools, list) or not tools:
        raise ToolTrajectoryError("tool_scenario tools must be a nonempty list")
    tools_by_name: dict[str, dict[str, Any]] = {}
    for index, tool in enumerate(tools, start=1):
        prefix = f"tool_scenario tool {index}"
        if not isinstance(tool, dict) or set(tool) != {"name", "description", "effect", "arguments_schema"}:
            raise ToolTrajectoryError(f"{prefix} has an invalid shape")
        name = tool["name"]
        if not _valid_nonempty_string(name) or not TOOL_NAME_PATTERN.fullmatch(name):
            raise ToolTrajectoryError(f"{prefix} has an invalid name")
        if name in tools_by_name:
            raise ToolTrajectoryError(f"tool_scenario has duplicate tool name {name!r}")
        if not _valid_nonempty_string(tool["description"]):
            raise ToolTrajectoryError(f"{prefix} needs a description")
        if not isinstance(tool["effect"], str) or tool["effect"] not in {"read_only", "state_changing"}:
            raise ToolTrajectoryError(f"{prefix} has an invalid effect")
        argument_schema = tool["arguments_schema"]
        if not isinstance(argument_schema, dict) or argument_schema.get("type") != "object":
            raise ToolTrajectoryError(f"{prefix}.arguments_schema must describe an object")
        try:
            _canonical_json(argument_schema)
        except (TypeError, ValueError) as exc:
            raise ToolTrajectoryError(f"{prefix}.arguments_schema must contain finite JSON values") from exc
        tools_by_name[name] = tool

    fixtures = scenario["fixtures"]
    if not isinstance(fixtures, list) or len(fixtures) > MAX_TOOL_CALLS:
        raise ToolTrajectoryError(f"tool_scenario fixtures must be a list of at most {MAX_TOOL_CALLS} items")
    if len(fixtures) > max_calls:
        raise ToolTrajectoryError("tool_scenario fixtures exceed max_tool_calls")
    if len(fixtures) + 1 > max_steps:
        raise ToolTrajectoryError("tool_scenario max_model_steps must allow the fixture calls and a final response")
    for index, fixture in enumerate(fixtures, start=1):
        prefix = f"tool_scenario fixture {index}"
        if not isinstance(fixture, dict) or set(fixture) != {"tool", "arguments", "result"}:
            raise ToolTrajectoryError(f"{prefix} has an invalid shape")
        _validate_tool_call({"tool": fixture["tool"], "arguments": fixture["arguments"]}, prefix)
        if fixture["tool"] not in tools_by_name:
            raise ToolTrajectoryError(f"{prefix} references an undeclared tool")
        result = fixture["result"]
        if (
            not isinstance(result, dict)
            or set(result) != {"status", "payload"}
            or not isinstance(result["status"], str)
            or result["status"] not in TOOL_RESULT_STATUSES
            or not isinstance(result["payload"], dict)
        ):
            raise ToolTrajectoryError(f"{prefix}.result must contain a known status and object payload")
        try:
            _canonical_json(result["payload"])
        except (TypeError, ValueError) as exc:
            raise ToolTrajectoryError(f"{prefix}.result.payload must contain finite JSON values") from exc


def validate_tool_trajectory_gold(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"calls", "completion"}:
        raise ToolTrajectoryError("gold.tool_trajectory must contain only calls and completion")
    if value["completion"] != "final":
        raise ToolTrajectoryError("gold.tool_trajectory.completion must be final")
    calls = value["calls"]
    if not isinstance(calls, list) or len(calls) > MAX_TOOL_CALLS:
        raise ToolTrajectoryError(f"gold.tool_trajectory.calls must be a list of at most {MAX_TOOL_CALLS} items")
    for index, call in enumerate(calls, start=1):
        _validate_tool_call(call, f"gold.tool_trajectory call {index}")


def validate_case_tool_trajectory(scenario: Any, gold_trajectory: Any) -> None:
    """Bind one reviewed expected ordered call path to its fixture outcomes."""
    validate_tool_scenario(scenario)
    if scenario is None:
        if gold_trajectory is not None:
            raise ToolTrajectoryError("gold.tool_trajectory requires a non-null tool_scenario")
        return
    if gold_trajectory is None:
        raise ToolTrajectoryError("non-null tool_scenario requires gold.tool_trajectory")
    validate_tool_trajectory_gold(gold_trajectory)
    fixture_calls = [
        {"tool": fixture["tool"], "arguments": fixture["arguments"]}
        for fixture in scenario["fixtures"]
    ]
    if not _same_json_value(gold_trajectory["calls"], fixture_calls):
        raise ToolTrajectoryError("gold.tool_trajectory.calls must match the ordered tool_scenario fixtures")


def validate_prediction_trajectory(
    value: Any,
    raw_output: str,
    scenario: Any = None,
) -> None:
    """Check trace structure and bind every recorded call to its unmodified raw step."""
    if not isinstance(value, dict) or set(value) != {"schema_version", "stop_reason", "steps"}:
        raise ToolTrajectoryError("trajectory prediction must contain only schema_version, stop_reason, and steps")
    version = value["schema_version"]
    if not isinstance(version, int) or isinstance(version, bool) or version != TOOL_TRAJECTORY_SCHEMA_VERSION:
        raise ToolTrajectoryError("trajectory prediction schema_version must be 1")
    stop_reason = value["stop_reason"]
    if not isinstance(stop_reason, str) or stop_reason not in {"final", "invalid_tool_call", "call_limit", "step_limit"}:
        raise ToolTrajectoryError("trajectory prediction has an invalid stop_reason")
    steps = value["steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_MODEL_STEPS:
        raise ToolTrajectoryError(f"trajectory prediction steps must contain 1 to {MAX_MODEL_STEPS} rows")

    for index, step in enumerate(steps, start=1):
        prefix = f"trajectory prediction step {index}"
        if not isinstance(step, dict) or set(step) != {"raw_output", "tool_call", "observation"}:
            raise ToolTrajectoryError(f"{prefix} has an invalid shape")
        if not isinstance(step["raw_output"], str):
            raise ToolTrajectoryError(f"{prefix}.raw_output must be a string")
        call = step["tool_call"]
        observation = step["observation"]
        if call is None:
            if observation is not None:
                raise ToolTrajectoryError(f"{prefix} cannot have an observation without a tool call")
        else:
            _validate_tool_call(call, f"{prefix}.tool_call")
            kind, parsed_call = parse_tool_call(step["raw_output"])
            if kind != "tool_call" or not _same_json_value(parsed_call, call):
                raise ToolTrajectoryError(f"{prefix}.tool_call does not match its raw candidate output")
            if (
                not isinstance(observation, dict)
                or set(observation) != {"status", "payload", "simulated", "effect"}
                or not isinstance(observation.get("status"), str)
                or observation["status"] not in OBSERVATION_STATUSES
                or not isinstance(observation.get("payload"), dict)
                or observation.get("simulated") is not True
                or not isinstance(observation.get("effect"), str)
                or observation["effect"] not in {"read_only", "state_changing", "unknown"}
            ):
                raise ToolTrajectoryError(f"{prefix}.observation is not a simulated tool result")

    last = steps[-1]
    if raw_output != last["raw_output"]:
        raise ToolTrajectoryError("prediction raw_output must match the terminal trajectory step")
    if any(step["tool_call"] is None for step in steps[:-1]):
        raise ToolTrajectoryError("only the terminal trajectory step may finish without a tool call")
    if stop_reason == "final" and last["tool_call"] is not None:
        raise ToolTrajectoryError("final trajectory stop requires a terminal candidate output")
    if stop_reason == "invalid_tool_call":
        kind, _ = parse_tool_call(last["raw_output"])
        if last["tool_call"] is not None or kind != "invalid_tool_call":
            raise ToolTrajectoryError("invalid_tool_call stop requires a malformed tool-call envelope")
    if stop_reason == "call_limit":
        if last["tool_call"] is None or last["observation"]["status"] != "call_limit":
            raise ToolTrajectoryError("call_limit stop requires a simulated call_limit observation")
    if stop_reason == "step_limit" and last["tool_call"] is None:
        raise ToolTrajectoryError("step_limit stop requires a terminal tool call")

    if scenario is not None:
        validate_tool_scenario(scenario)
        if len(steps) > scenario["max_model_steps"]:
            raise ToolTrajectoryError("trajectory exceeds tool_scenario max_model_steps")
        simulator = FixtureOnlyToolSimulator(scenario)
        call_count = 0
        tools_by_name = {tool["name"]: tool for tool in scenario["tools"]}
        for index, step in enumerate(steps):
            call = step["tool_call"]
            if call is None:
                continue
            call_count += 1
            if call_count > scenario["max_tool_calls"]:
                if index != len(steps) - 1 or stop_reason != "call_limit":
                    raise ToolTrajectoryError("trajectory exceeded max_tool_calls without stopping immediately")
                expected_observation = {
                    "status": "call_limit",
                    "payload": {},
                    "simulated": True,
                    "effect": tools_by_name.get(call["tool"], {}).get("effect", "unknown"),
                }
            else:
                expected_observation = simulator.invoke(call["tool"], call["arguments"])
            if not _same_json_value(step["observation"], expected_observation):
                raise ToolTrajectoryError(
                    f"trajectory step {index + 1} observation differs from its fixture-only simulation"
                )
        if stop_reason == "call_limit" and call_count != scenario["max_tool_calls"] + 1:
            raise ToolTrajectoryError("call_limit stop must contain exactly one over-limit tool request")
        if stop_reason == "step_limit":
            if len(steps) != scenario["max_model_steps"] or call_count > scenario["max_tool_calls"]:
                raise ToolTrajectoryError("step_limit stop does not match the configured model-step limit")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r} is not allowed")
        result[key] = value
    return result


def parse_tool_call(raw_output: str) -> tuple[str, dict[str, Any] | None]:
    """Recognize only the exact tool-call envelope; all other strings remain raw finals."""
    try:
        value = json.loads(
            raw_output,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError):
        return "final", None
    if not isinstance(value, dict) or value.get("type") != "tool_call":
        return "final", None
    if set(value) != {"type", "tool", "arguments"}:
        return "invalid_tool_call", None
    try:
        _validate_tool_call({"tool": value["tool"], "arguments": value["arguments"]}, "tool call")
    except (KeyError, ToolTrajectoryError):
        return "invalid_tool_call", None
    return "tool_call", {"tool": value["tool"], "arguments": deepcopy(value["arguments"])}


class FixtureOnlyToolSimulator:
    """Resolve calls against ordered inline outcomes; it has no external execution path."""

    def __init__(self, scenario: dict[str, Any]):
        validate_tool_scenario(scenario)
        self._tools = {tool["name"]: tool for tool in scenario["tools"]}
        self._fixtures = scenario["fixtures"]
        self._fixture_index = 0

    def invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        effect = tool["effect"] if tool is not None else "unknown"
        if tool is None:
            status, payload = "unknown_tool", {}
        elif self._fixture_index >= len(self._fixtures):
            status, payload = "fixture_mismatch", {}
        else:
            fixture = self._fixtures[self._fixture_index]
            if fixture["tool"] != name or not _same_json_value(fixture["arguments"], arguments):
                status, payload = "fixture_mismatch", {}
            else:
                self._fixture_index += 1
                status, payload = fixture["result"]["status"], deepcopy(fixture["result"]["payload"])
        return {
            "status": status,
            "payload": payload,
            "simulated": True,
            "effect": effect,
        }
