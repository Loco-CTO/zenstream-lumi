import unittest

from evaluation.candidate_runner import CandidateOutputError, run_candidate_cases
from evaluation.scorer import EvaluationInputError
from test_provenance_validator import (
    make_case,
    make_generation_manifest,
    make_record,
    make_source_manifest,
)


class RecordingCandidate:
    def __init__(self, raw_output):
        self.raw_output = raw_output
        self.inputs = []

    def predict(self, candidate_input):
        self.inputs.append(candidate_input)
        return self.raw_output


def run_one(case, candidate, record=None, **kwargs):
    return run_candidate_cases(
        [case],
        candidate,
        [record or make_record(case)],
        make_source_manifest(),
        make_generation_manifest(),
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
            "schema_version": 1,
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


if __name__ == "__main__":
    unittest.main()
