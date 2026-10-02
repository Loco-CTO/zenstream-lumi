import hashlib
import json
import unittest
from pathlib import Path

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


def semantic_output(decision, action, arguments, requires_clarification=False):
    return json.dumps({
        "decision": decision,
        "action": action,
        "arguments": arguments,
        "requires_clarification": requires_clarification,
    }, ensure_ascii=False, separators=(",", ":"))


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


def approved_capability_registry():
    path = Path(__file__).resolve().parents[1] / "evaluation" / "capabilities.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    registry["review_status"] = "approved"
    for capability in registry["capabilities"]:
        capability["review_status"] = "approved"
    return registry


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
        registry_path = (
            Path(__file__).resolve().parents[1]
            / "evaluation"
            / "capabilities.json"
        )
        registry_sha256 = hashlib.sha256(registry_path.read_bytes()).hexdigest()
        self.assertEqual(
            report["capability_registry"]["sha256"], "sha256:" + registry_sha256
        )

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
        state_changing_metric = metrics["false_state_changing_action_rate"]
        self.assertEqual(state_changing_metric["status"], "not_measured")
        self.assertEqual(state_changing_metric["registry_review_status"], "draft")
        self.assertIsNone(state_changing_metric["rate"])

    def test_approved_registry_measures_false_mutations_and_unknown_actions(self):
        state_change_gold = {
            "decision": "act", "action": "playback.start", "arguments": {},
            "requires_clarification": False,
        }
        read_only_gold = {
            "decision": "act", "action": "catalog.search", "arguments": {},
            "requires_clarification": False,
        }
        no_action_gold = {
            "decision": "no_action", "action": None, "arguments": {},
            "requires_clarification": False,
        }
        cases = [
            make_case("false-on-no-action", categories=["no_action"], gold=no_action_gold),
            make_case("false-on-read-only", gold=read_only_gold),
            make_case("correct-mutation", gold=state_change_gold),
            make_case("unknown-action", gold=no_action_gold),
        ]
        predictions = [
            make_prediction("false-on-no-action", '{"decision":"act","action":"playback.start","arguments":{},"requires_clarification":false}'),
            make_prediction("false-on-read-only", '{"decision":"act","action":"playback.start","arguments":{},"requires_clarification":false}'),
            make_prediction("correct-mutation", '{"decision":"act","action":"playback.start","arguments":{},"requires_clarification":false}'),
            make_prediction("unknown-action", '{"decision":"act","action":"unknown.mutate","arguments":{},"requires_clarification":false}'),
        ]

        report = score_records(
            cases,
            predictions,
            capability_registry=approved_capability_registry(),
        )
        metrics = report["metrics"]

        false_mutations = metrics["false_state_changing_action_rate"]
        self.assertEqual(false_mutations["status"], "measured")
        self.assertEqual(false_mutations["events"], 2)
        self.assertEqual(false_mutations["total"], 4)
        self.assertEqual(false_mutations["rate"], 0.5)
        self.assertEqual(false_mutations["denominator"], "all selected cases")

        unknown_actions = metrics["unclassified_action_rate"]
        self.assertEqual(unknown_actions["status"], "measured")
        self.assertEqual(unknown_actions["events"], 1)
        self.assertEqual(unknown_actions["total"], 4)
        self.assertEqual(unknown_actions["rate"], 0.25)
        self.assertEqual(report["capability_registry"]["review_status"], "approved")
        self.assertRegex(
            report["capability_registry"]["sha256"], r"^sha256:[0-9a-f]{64}$"
        )

    def test_invalid_capability_registry_is_rejected(self):
        registry = approved_capability_registry()
        registry["capabilities"][0]["effect"] = "unknown"
        case = make_case("invalid-registry")
        prediction = make_prediction("invalid-registry", "{}")

        with self.assertRaisesRegex(EvaluationInputError, "invalid capability registry"):
            score_records([case], [prediction], capability_registry=registry)

    def test_approved_read_only_registry_still_measures_unclassified_actions(self):
        registry = approved_capability_registry()
        for capability in registry["capabilities"]:
            capability["effect"] = "read_only"
        case = make_case("unclassified", gold={
            "decision": "no_action", "action": None, "arguments": {},
            "requires_clarification": False,
        })
        prediction = make_prediction(
            "unclassified",
            '{"decision":"act","action":"unknown.action","arguments":{},"requires_clarification":false}',
        )

        metrics = score_records(
            [case], [prediction], capability_registry=registry
        )["metrics"]

        self.assertEqual(metrics["false_state_changing_action_rate"]["status"], "not_measured")
        self.assertEqual(metrics["unclassified_action_rate"]["events"], 1)

    def test_invalid_json_counts_against_validity_and_semantics(self):
        cases = [make_case("invalid")]
        predictions = [make_prediction("invalid", "{not-json")]

        report = score_records(cases, predictions)

        self.assertEqual(report["invalid_response_count"], 1)
        self.assertEqual(report["metrics"]["structured_response_validity"]["rate"], 0.0)
        self.assertEqual(report["metrics"]["overall_semantic_exact_match"]["rate"], 0.0)

    def test_clarification_precision_and_recall_count_false_and_missed_clarifications(self):
        clarify_gold = {
            "decision": "clarify", "action": None, "arguments": {},
            "requires_clarification": True,
        }
        respond_gold = {
            "decision": "respond", "action": None, "arguments": {},
            "requires_clarification": False,
        }
        action_gold = {
            "decision": "act", "action": "catalog.search", "arguments": {"query": "Jazz"},
            "requires_clarification": False,
        }
        cases = [
            make_case("clarify-hit", gold=clarify_gold),
            make_case("clarify-miss", gold=clarify_gold),
            make_case("clarify-false-positive", gold=action_gold),
            make_case("respond-correct", gold=respond_gold),
        ]
        predictions = [
            make_prediction("clarify-hit", semantic_output("clarify", None, {}, True)),
            make_prediction("clarify-miss", semantic_output("respond", None, {})),
            make_prediction("clarify-false-positive", semantic_output("clarify", None, {}, True)),
            make_prediction("respond-correct", semantic_output("respond", None, {})),
        ]

        metrics = score_records(cases, predictions)["metrics"]

        self.assertEqual(metrics["clarification_precision"]["successes"], 1)
        self.assertEqual(metrics["clarification_precision"]["total"], 2)
        self.assertEqual(metrics["clarification_precision"]["rate"], 0.5)
        self.assertEqual(metrics["clarification_recall"]["successes"], 1)
        self.assertEqual(metrics["clarification_recall"]["total"], 2)
        self.assertEqual(metrics["clarification_recall"]["rate"], 0.5)

    def test_argument_slot_precision_recall_reports_each_action_slot(self):
        cases = [
            make_case("search-slots", gold={
                "decision": "act", "action": "catalog.search",
                "arguments": {"query": "Show", "type": "movie"},
                "requires_clarification": False,
            }),
            make_case("playback-slot", gold={
                "decision": "act", "action": "playback.start",
                "arguments": {"target_text": "play the new album"},
                "requires_clarification": False,
            }),
        ]
        predictions = [
            make_prediction("search-slots", semantic_output("act", "catalog.search", {
                "query": " show ", "type": "series", "unexpected": True,
            })),
            make_prediction("playback-slot", semantic_output("no_action", None, {})),
        ]

        metrics = score_records(cases, predictions)["metrics"]

        self.assertEqual(metrics["argument_slot_precision"]["successes"], 1)
        self.assertEqual(metrics["argument_slot_precision"]["total"], 3)
        self.assertAlmostEqual(metrics["argument_slot_precision"]["rate"], 1 / 3)
        self.assertEqual(metrics["argument_slot_recall"]["successes"], 1)
        self.assertEqual(metrics["argument_slot_recall"]["total"], 3)
        self.assertAlmostEqual(metrics["argument_slot_recall"]["rate"], 1 / 3)

        slots = metrics["argument_slots_by_action"]
        self.assertEqual(
            slots["catalog.search"]["query"]["precision"]["rate"], 1.0
        )
        self.assertEqual(
            slots["catalog.search"]["type"]["recall"]["rate"], 0.0
        )
        self.assertEqual(
            slots["catalog.search"]["unexpected"]["precision"]["rate"], 0.0
        )
        self.assertEqual(
            slots["playback.start"]["target_text"]["recall"]["rate"], 0.0
        )

    def test_required_slice_tags_feed_existing_behavioral_target_metrics(self):
        cases = [
            make_case("negation-tag", categories=["no_action_negation"], gold={
                "decision": "no_action", "action": None, "arguments": {},
                "requires_clarification": False,
            }),
            make_case("capability-question-tag", categories=["no_action_capability_question"], gold={
                "decision": "respond", "action": None, "arguments": {},
                "requires_clarification": False,
            }),
            make_case("multi-turn-tag", categories=["multi_turn_reference_resolution"]),
            make_case("ambiguity-tag", categories=["difficult_ambiguous"], gold={
                "decision": "clarify", "action": None, "arguments": {},
                "requires_clarification": True,
            }),
        ]
        predictions = [
            make_prediction("negation-tag", semantic_output("no_action", None, {})),
            make_prediction("capability-question-tag", semantic_output("respond", None, {})),
            make_prediction("multi-turn-tag", semantic_output("act", "play", {"title": "Example"})),
            make_prediction("ambiguity-tag", semantic_output("clarify", None, {}, True)),
        ]

        metrics = score_records(cases, predictions)["metrics"]

        for metric_name in (
            "negation_no_action_correctness",
            "multi_turn_reference_resolution",
            "difficult_ambiguous_requests",
        ):
            self.assertEqual(metrics[metric_name]["successes"], 2 if metric_name == "negation_no_action_correctness" else 1)
            self.assertEqual(metrics[metric_name]["total"], 2 if metric_name == "negation_no_action_correctness" else 1)

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
