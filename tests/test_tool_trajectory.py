import json
import unittest

from evaluation.tool_trajectory import (
    FixtureOnlyToolSimulator,
    ToolTrajectoryError,
    parse_tool_call,
    validate_case_tool_trajectory,
    validate_prediction_trajectory,
)


def scenario(effect="state_changing"):
    expected_call = {"tool": "playback.start", "arguments": {"entity_id": "item-1"}}
    return {
        "schema_version": 1,
        "tools": [{
            "name": "playback.start",
            "description": "Simulated playback request.",
            "effect": effect,
            "arguments_schema": {"type": "object", "required": ["entity_id"]},
        }],
        "fixtures": [{
            **expected_call,
            "result": {"status": "success", "payload": {"accepted": True}},
        }],
        "max_model_steps": 3,
        "max_tool_calls": 2,
    }


class ToolTrajectoryTests(unittest.TestCase):
    def test_simulator_returns_only_the_next_exact_fixture_and_marks_no_effect(self):
        simulator = FixtureOnlyToolSimulator(scenario())

        mismatch = simulator.invoke("playback.start", {"entity_id": "different-item"})
        matched = simulator.invoke("playback.start", {"entity_id": "item-1"})
        exhausted = simulator.invoke("playback.start", {"entity_id": "item-1"})

        self.assertEqual(mismatch["status"], "fixture_mismatch")
        self.assertEqual(mismatch["payload"], {})
        self.assertEqual(matched, {
            "status": "success",
            "payload": {"accepted": True},
            "simulated": True,
            "effect": "state_changing",
        })
        self.assertEqual(exhausted["status"], "fixture_mismatch")
        self.assertTrue(matched["simulated"])

    def test_case_gold_path_must_match_fixture_order(self):
        fixture_scenario = scenario()
        gold = {
            "calls": [{"tool": "playback.start", "arguments": {"entity_id": "other"}}],
            "completion": "final",
        }

        with self.assertRaisesRegex(ToolTrajectoryError, "match the ordered"):
            validate_case_tool_trajectory(fixture_scenario, gold)

    def test_tool_call_parser_accepts_only_the_exact_call_envelope(self):
        valid = json.dumps({
            "type": "tool_call",
            "tool": "playback.start",
            "arguments": {"entity_id": "item-1"},
        })
        malformed = json.dumps({
            "type": "tool_call",
            "tool": "playback.start",
            "arguments": {"entity_id": "item-1"},
            "url": "https://example.invalid/",
        })

        self.assertEqual(parse_tool_call(valid), (
            "tool_call", {"tool": "playback.start", "arguments": {"entity_id": "item-1"}},
        ))
        self.assertEqual(parse_tool_call(malformed), ("invalid_tool_call", None))
        self.assertEqual(parse_tool_call('{"decision":"respond"}'), ("final", None))

    def test_prediction_trace_binds_raw_calls_observations_and_terminal_output(self):
        call_raw = '{"type":"tool_call","tool":"playback.start","arguments":{"entity_id":"item-1"}}'
        final_raw = '{"decision":"respond"}'
        trace = {
            "schema_version": 1,
            "stop_reason": "final",
            "steps": [
                {
                    "raw_output": call_raw,
                    "tool_call": {"tool": "playback.start", "arguments": {"entity_id": "item-1"}},
                    "observation": {
                        "status": "success",
                        "payload": {"accepted": True},
                        "simulated": True,
                        "effect": "state_changing",
                    },
                },
                {"raw_output": final_raw, "tool_call": None, "observation": None},
            ],
        }
        validate_prediction_trajectory(trace, final_raw)

        trace["steps"][0]["observation"]["simulated"] = False
        with self.assertRaisesRegex(ToolTrajectoryError, "simulated tool result"):
            validate_prediction_trajectory(trace, final_raw)

    def test_terminal_output_must_match_the_last_preserved_step(self):
        raw = "malformed"
        trace = {
            "schema_version": 1,
            "stop_reason": "final",
            "steps": [{"raw_output": raw, "tool_call": None, "observation": None}],
        }

        with self.assertRaisesRegex(ToolTrajectoryError, "must match the terminal"):
            validate_prediction_trajectory(trace, "different")


if __name__ == "__main__":
    unittest.main()
