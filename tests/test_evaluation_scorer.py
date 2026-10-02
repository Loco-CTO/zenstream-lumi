import unittest

from evaluation.scorer import EvaluationInputError, _one_sided_error_upper, score_records as _score_records
from test_provenance_validator import make_generation_manifest, make_record, make_source_manifest


def make_case(case_id, *, split="development", language="en", categories=None, gold=None, family_id=None, review=None):
    return {
        "schema_version": 2,
        "case_id": case_id,
        "family_id": family_id or case_id,
        "split": split,
        "review_status": "ready",
        "language": language,
        "categories": categories or ["simple_action"],
        "turns": [{"role": "user", "text": "play something"}],
        "trusted_context": None,
        "gold": gold or {
            "decision": "act",
            "action": "play",
            "arguments": {"title": "Example"},
            "requires_clarification": False,
        },
        "provenance_record_id": f"prov-{case_id}",
        "review": review or {
            "annotation_status": "approved",
            "language_review_status": "not_required",
            "reviewer_ids": ["reviewer-1"],
        },
    }


def make_prediction(case_id, output, grounding_review=None):
    record = {"schema_version": 1, "case_id": case_id, "raw_output": output}
    if grounding_review is not None:
        record["grounding_review"] = grounding_review
    return record


def score_records(cases, predictions, **kwargs):
    provenance = [make_record(case) for case in cases]
    return _score_records(
        cases,
        predictions,
        provenance,
        make_source_manifest(),
        make_generation_manifest(),
        **kwargs,
    )


class EvaluationScorerTests(unittest.TestCase):
    def test_direct_scoring_enforces_source_manifest_permissions(self):
        case = make_case("unpermitted-source")
        provenance = make_record(case)
        sources = make_source_manifest()
        sources["sources"][0]["permissions"]["evaluation_use"] = "conditional"

        with self.assertRaisesRegex(EvaluationInputError, "source manifest does not permit evaluation"):
            _score_records(
                [case],
                [make_prediction(case["case_id"], "{}")],
                [provenance],
                sources,
                make_generation_manifest(),
            )

    def test_reports_semantics_and_false_actions_separately_from_validity(self):
        cases = [
            make_case("positive", gold={
                "decision": "act", "action": "play", "arguments": {"title": "Example"},
                "requires_clarification": False,
            }),
            make_case("negative", categories=["negation", "no_action"], gold={
                "decision": "no_action", "action": None, "arguments": {},
                "requires_clarification": False,
            }),
        ]
        predictions = [
            make_prediction("positive", '{"decision":"act","action":"play","arguments":{"title":" example "},"requires_clarification":false}'),
            make_prediction("negative", '{"decision":"act","action":"play","arguments":{},"requires_clarification":false}'),
        ]

        report = score_records(cases, predictions)
        metrics = report["metrics"]

        self.assertEqual(metrics["structured_response_validity"]["rate"], 1.0)
        self.assertEqual(metrics["overall_semantic_exact_match"]["successes"], 1)
        self.assertEqual(metrics["false_action_rate"]["rate"], 1.0)
        self.assertEqual(metrics["negation_no_action_correctness"]["rate"], 0.0)

    def test_accepts_a_versioned_trusted_context_fixture(self):
        case = make_case("context")
        case["trusted_context"] = {
            "schema_version": 1,
            "items": [{
                "kind": "playback_state",
                "source": "playback.current_item",
                "status": "success",
                "payload": {"title": "Example", "position_ms": 42000},
            }],
        }
        prediction = make_prediction(
            "context",
            '{"decision":"act","action":"play","arguments":{"title":"Example"},"requires_clarification":false}',
        )

        report = score_records([case], [prediction])

        self.assertEqual(report["metrics"]["overall_semantic_exact_match"]["rate"], 1.0)

    def test_rejects_invalid_trusted_context_status(self):
        case = make_case("bad-context")
        case["trusted_context"] = {
            "schema_version": 1,
            "items": [{
                "kind": "tool_result",
                "source": "catalog.lookup",
                "status": "unrecognized",
                "payload": {},
            }],
        }
        prediction = make_prediction(
            "bad-context",
            '{"decision":"act","action":"play","arguments":{"title":"Example"},"requires_clarification":false}',
        )

        with self.assertRaisesRegex(EvaluationInputError, "invalid status"):
            score_records([case], [prediction])

    def test_false_action_rate_covers_all_non_action_gold_decisions(self):
        cases = [
            make_case("no-action", categories=["no_action"], gold={
                "decision": "no_action", "action": None, "arguments": {},
                "requires_clarification": False,
            }),
            make_case("respond", categories=["capability_question"], gold={
                "decision": "respond", "action": None, "arguments": {},
                "requires_clarification": False,
            }),
            make_case("clarify", categories=["difficult_ambiguous"], gold={
                "decision": "clarify", "action": None, "arguments": {},
                "requires_clarification": True,
            }),
        ]
        act = '{"decision":"act","action":"play","arguments":{},"requires_clarification":false}'
        predictions = [
            make_prediction("no-action", act),
            make_prediction("respond", act),
            make_prediction(
                "clarify",
                '{"decision":"clarify","action":null,"arguments":{},"requires_clarification":true}',
            ),
        ]

        metrics = score_records(cases, predictions)["metrics"]

        self.assertEqual(metrics["false_action_rate"]["events"], 2)
        self.assertEqual(metrics["false_action_rate"]["total"], 3)
        self.assertAlmostEqual(metrics["false_action_rate"]["rate"], 2 / 3)
        self.assertNotIn("false_state_changing_action_rate", metrics)

    def test_invalid_json_counts_against_validity_and_semantics(self):
        cases = [make_case("invalid")]
        predictions = [make_prediction("invalid", "{not-json")]

        report = score_records(cases, predictions)

        self.assertEqual(report["invalid_response_count"], 1)
        self.assertEqual(report["metrics"]["structured_response_validity"]["rate"], 0.0)
        self.assertEqual(report["metrics"]["overall_semantic_exact_match"]["rate"], 0.0)

    def test_invalid_decision_type_counts_as_invalid_instead_of_raising(self):
        case = make_case("invalid-decision")
        prediction = make_prediction(
            "invalid-decision",
            '{"decision":[],"action":null,"arguments":{},"requires_clarification":false}',
        )

        report = score_records([case], [prediction])

        self.assertEqual(report["invalid_response_count"], 1)
        self.assertEqual(report["metrics"]["structured_response_validity"]["rate"], 0.0)

    def test_ready_japanese_case_requires_language_review(self):
        case = make_case(
            "japanese",
            language="ja",
            review={
                "annotation_status": "approved",
                "language_review_status": "pending",
                "reviewer_ids": ["reviewer-1"],
            },
        )
        prediction = make_prediction(
            "japanese",
            '{"decision":"act","action":"play","arguments":{"title":"Example"},"requires_clarification":false}',
        )

        with self.assertRaisesRegex(EvaluationInputError, "language review approval"):
            score_records([case], [prediction])

    def test_draft_cases_cannot_be_scored(self):
        case = make_case("draft")
        case["review_status"] = "draft"
        prediction = make_prediction(
            "draft",
            '{"decision":"act","action":"play","arguments":{"title":"Example"},"requires_clarification":false}',
        )
        with self.assertRaisesRegex(EvaluationInputError, "unreviewed or excluded"):
            score_records([case], [prediction])

    def test_final_holdout_requires_explicit_authorization_and_is_reported(self):
        case = make_case("holdout", split="final_holdout")
        prediction = make_prediction(
            "holdout",
            '{"decision":"act","action":"play","arguments":{"title":"Example"},"requires_clarification":false}',
        )
        with self.assertRaisesRegex(EvaluationInputError, "explicit --final-audit"):
            score_records([case], [prediction], split="final_holdout")

        report = score_records([case], [prediction], split="final_holdout", allow_final_holdout=True)
        self.assertTrue(report["final_audit"])

    def test_case_families_cannot_cross_splits(self):
        development = make_case("dev", family_id="same-family")
        holdout = make_case("holdout", split="final_holdout", family_id="same-family")
        prediction = make_prediction(
            "dev",
            '{"decision":"act","action":"play","arguments":{"title":"Example"},"requires_clarification":false}',
        )
        with self.assertRaisesRegex(EvaluationInputError, "crosses evaluation splits"):
            score_records([development, holdout], [prediction])

    def test_unsupported_claim_metric_reports_unsupported_event_rate(self):
        case = make_case("grounded")
        prediction = make_prediction(
            "grounded",
            '{"decision":"act","action":"play","arguments":{"title":"Example"},"requires_clarification":false}',
            grounding_review={
                "status": "reviewed",
                "claim_count": 2,
                "unsupported_claim_count": 1,
                "reviewer_id": "grounding-reviewer",
            },
        )

        report = score_records([case], [prediction])
        metric = report["metrics"]["unsupported_factual_claim_rate"]

        self.assertEqual(metric["events"], 1)
        self.assertEqual(metric["total"], 2)
        self.assertEqual(metric["rate"], 0.5)
        self.assertIsNotNone(metric["one_sided_95_event_upper"])

    def test_zero_errors_in_2995_trials_supports_point_one_percent_error_bound(self):
        upper = _one_sided_error_upper(0, 2995)
        self.assertIsNotNone(upper)
        self.assertLessEqual(upper, 0.001)


if __name__ == "__main__":
    unittest.main()
