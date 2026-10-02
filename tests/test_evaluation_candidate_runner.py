import unittest
import json

from evaluation.candidate_runner import CandidateOutputError, run_candidate_cases
from evaluation.scorer import EvaluationInputError
from test_provenance_validator import (
    make_case,
    make_generation_manifest,
    make_record,
    make_source_manifest,
)
from review_fixtures import make_review_records


class RecordingCandidate:
    def __init__(self, raw_output):
        self.raw_output = raw_output
        self.inputs = []

    def predict(self, candidate_input):
        self.inputs.append(candidate_input)
        return self.raw_output


class SequenceCandidate:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.inputs = []

    def predict(self, candidate_input):
        self.inputs.append(candidate_input)
        return self.outputs.pop(0)


def make_tool_case():
    case = make_case()
    expected_call = {"tool": "catalog.search", "arguments": {"query": "Example"}}
    case["tool_scenario"] = {
        "schema_version": 1,
        "tools": [{
            "name": "catalog.search",
            "description": "Search the static catalog fixture.",
            "effect": "read_only",
            "arguments_schema": {"type": "object", "required": ["query"]},
        }],
        "fixtures": [{
            **expected_call,
            "result": {"status": "success", "payload": {"items": [{"id": "item-1", "title": "Example"}]}},
        }],
        "max_model_steps": 4,
        "max_tool_calls": 3,
    }
    case["gold"]["tool_trajectory"] = {"calls": [expected_call], "completion": "final"}
    return case


def tool_call_output(tool="catalog.search", arguments=None, **extra):
    value = {"type": "tool_call", "tool": tool, "arguments": arguments or {"query": "Example"}}
    value.update(extra)
    return json.dumps(value, ensure_ascii=False)


def run_one(case, candidate, record=None, review_records=None, **kwargs):
    ledger = make_review_records(case) if review_records is None else review_records
    return run_candidate_cases(
        [case],
        candidate,
        [record or make_record(case)],
        make_source_manifest(),
        make_generation_manifest(),
        review_records=ledger,
        **kwargs,
    )


class EvaluationCandidateRunnerTests(unittest.TestCase):
    def test_validates_provenance_and_preserves_raw_output(self):
        case = make_case()
        case["trusted_context"] = {
            "schema_version": 1,
            "items": [{
                "kind": "playback_state",
                "source": "playback.current_item",
                "status": "success",
                "payload": {"title": "Example", "position_ms": 42000},
            }],
        }
        candidate = RecordingCandidate("{ malformed output")

        predictions = run_one(case, candidate)

        self.assertEqual(predictions, [{
            "schema_version": 3,
            "case_id": case["case_id"],
            "raw_output": "{ malformed output",
        }])
        self.assertEqual(
            set(candidate.inputs[0]),
            {"schema_version", "turns", "trusted_context"},
        )
        self.assertNotIn("case_id", candidate.inputs[0])
        self.assertNotIn("gold", candidate.inputs[0])

    def test_provenance_failure_prevents_candidate_execution(self):
        case = make_case()
        record = make_record(case)
        case["trusted_context"] = {
            "schema_version": 1,
            "items": [{
                "kind": "catalog",
                "source": "catalog.search",
                "status": "success",
                "payload": {"title": "Changed after review"},
            }],
        }
        candidate = RecordingCandidate('{"decision":"respond"}')

        with self.assertRaisesRegex(EvaluationInputError, "content hash differs"):
            run_one(case, candidate, record=record)

        self.assertEqual(candidate.inputs, [])

    def test_draft_case_prevents_candidate_execution(self):
        case = make_case()
        case["review_status"] = "draft"
        record = make_record(case)
        candidate = RecordingCandidate('{"decision":"respond"}')

        with self.assertRaisesRegex(EvaluationInputError, "unreviewed or excluded"):
            run_one(case, candidate, record=record)

        self.assertEqual(candidate.inputs, [])

    def test_stale_review_ledger_prevents_candidate_execution(self):
        case = make_case()
        records = make_review_records(case)
        case["turns"][0]["text"] = "Changed after annotation"
        candidate = RecordingCandidate('{"decision":"respond"}')

        with self.assertRaisesRegex(EvaluationInputError, "stale case hash"):
            run_one(case, candidate, review_records=records)

        self.assertEqual(candidate.inputs, [])

    def test_final_holdout_requires_explicit_opt_in(self):
        case = make_case()
        case["split"] = "final_holdout"
        record = make_record(case)
        candidate = RecordingCandidate('{"decision":"respond"}')

        with self.assertRaisesRegex(EvaluationInputError, "final holdout"):
            run_one(case, candidate, record=record, split="final_holdout")

        self.assertEqual(candidate.inputs, [])

    def test_final_holdout_runs_only_with_explicit_opt_in(self):
        case = make_case()
        case["split"] = "final_holdout"
        candidate = RecordingCandidate('{"decision":"respond"}')

        predictions = run_one(
            case,
            candidate,
            split="final_holdout",
            allow_final_holdout=True,
        )

        self.assertEqual(len(predictions), 1)
        self.assertEqual(candidate.inputs[0]["turns"], case["turns"])

    def test_empty_raw_output_is_preserved(self):
        case = make_case()
        candidate = RecordingCandidate("")

        predictions = run_one(case, candidate)

        self.assertEqual(predictions[0]["raw_output"], "")

    def test_candidate_must_return_a_string(self):
        case = make_case()
        candidate = RecordingCandidate(None)

        with self.assertRaisesRegex(CandidateOutputError, "raw output as a string"):
            run_one(case, candidate)

    def test_tool_trajectory_exposes_only_declared_tools_and_prior_observations(self):
        case = make_tool_case()
        final = '{"decision":"act","action":"play","arguments":{"title":"Example"},"requires_clarification":false}'
        candidate = SequenceCandidate([tool_call_output(), final])

        prediction = run_one(case, candidate)[0]

        self.assertEqual(prediction["schema_version"], 3)
        self.assertEqual(prediction["raw_output"], final)
        self.assertEqual(prediction["trajectory"]["stop_reason"], "final")
        self.assertEqual(len(prediction["trajectory"]["steps"]), 2)
        self.assertEqual(set(candidate.inputs[0]), {
            "schema_version", "turns", "trusted_context", "tools", "tool_history",
        })
        self.assertEqual(candidate.inputs[0]["schema_version"], 2)
        self.assertNotIn("fixtures", candidate.inputs[0])
        self.assertNotIn("gold", candidate.inputs[0])
        self.assertEqual(candidate.inputs[0]["tool_history"], [])
        self.assertEqual(candidate.inputs[1]["tool_history"][0]["observation"], {
            "status": "success",
            "payload": {"items": [{"id": "item-1", "title": "Example"}]},
            "simulated": True,
            "effect": "read_only",
        })

    def test_wrong_and_unknown_calls_return_safe_fixture_errors_without_consuming_expected_result(self):
        case = make_tool_case()
        final = '{"decision":"act","action":"play","arguments":{"title":"Example"},"requires_clarification":false}'
        candidate = SequenceCandidate([
            tool_call_output("unknown.lookup"),
            tool_call_output(arguments={"query": "Other"}),
            tool_call_output(),
            final,
        ])

        prediction = run_one(case, candidate)[0]

        steps = prediction["trajectory"]["steps"]
        self.assertEqual([step["observation"]["status"] for step in steps[:3]], [
            "unknown_tool", "fixture_mismatch", "success",
        ])
        self.assertTrue(all(step["observation"]["simulated"] for step in steps[:3]))
        self.assertEqual(prediction["trajectory"]["stop_reason"], "final")

    def test_tool_call_limit_stops_before_an_additional_fixture_lookup(self):
        case = make_tool_case()
        case["tool_scenario"]["max_tool_calls"] = 1
        candidate = SequenceCandidate([tool_call_output(), tool_call_output()])

        prediction = run_one(case, candidate)[0]

        self.assertEqual(prediction["trajectory"]["stop_reason"], "call_limit")
        self.assertEqual(len(candidate.inputs), 2)
        self.assertEqual(prediction["trajectory"]["steps"][-1]["observation"]["status"], "call_limit")
        self.assertTrue(all(
            step["observation"]["simulated"]
            for step in prediction["trajectory"]["steps"]
        ))

    def test_malformed_tool_call_is_retained_and_stops_the_simulator(self):
        case = make_tool_case()
        raw = tool_call_output(extra="not-permitted")
        candidate = SequenceCandidate([raw])

        prediction = run_one(case, candidate)[0]

        self.assertEqual(prediction["raw_output"], raw)
        self.assertEqual(prediction["trajectory"]["stop_reason"], "invalid_tool_call")
        self.assertEqual(prediction["trajectory"]["steps"][0]["tool_call"], None)
        self.assertEqual(len(candidate.inputs), 1)

    def test_inconsistent_gold_and_fixture_path_blocks_candidate_execution(self):
        case = make_tool_case()
        case["tool_scenario"]["fixtures"][0]["arguments"]["query"] = "Different"
        case["gold"]["tool_trajectory"]["calls"][0]["arguments"] = {"query": "Example"}
        candidate = SequenceCandidate([tool_call_output()])

        with self.assertRaisesRegex(EvaluationInputError, "must match the ordered tool_scenario fixtures"):
            run_one(case, candidate)

        self.assertEqual(candidate.inputs, [])


if __name__ == "__main__":
    unittest.main()
