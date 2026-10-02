import unittest

from evaluation.input_adapter import build_candidate_input


class EvaluationInputAdapterTests(unittest.TestCase):
    def test_exposes_only_turns_and_trusted_context(self):
        context = {
            "schema_version": 1,
            "items": [{
                "kind": "playback_state",
                "source": "playback.current_item",
                "status": "success",
                "payload": {"title": "Example", "position_ms": 42000},
            }],
        }
        case = {
            "schema_version": 4,
            "case_id": "secret-case-id",
            "family_id": "secret-family-id",
            "split": "final_holdout",
            "language": "en_ja",
            "categories": ["reference_resolution"],
            "turns": [
                {"role": "user", "text": "What was I watching?"},
                {"role": "assistant", "text": "Let me check."},
                {"role": "user", "text": "続き再生して"},
            ],
            "trusted_context": context,
            "tool_scenario": None,
            "gold": {"decision": "act", "action": "play", "arguments": {}, "requires_clarification": False},
            "provenance_record_id": "private-record-id",
            "review_status": "ready",
            "review": {
                "annotation_status": "approved",
                "reviewer_ids": ["annotator-1", "annotator-2", "language-reviewer"],
                "review_record_ids": ["review-1", "review-2", "review-language"],
                "language_review_status": "approved",
            },
        }

        candidate_input = build_candidate_input(case)

        self.assertEqual(
            candidate_input,
            {
                "schema_version": 1,
                "turns": case["turns"],
                "trusted_context": context,
            },
        )
        self.assertEqual(set(candidate_input), {"schema_version", "turns", "trusted_context"})

    def test_candidate_mutation_does_not_change_case_data(self):
        case = {
            "turns": [{"role": "user", "text": "play Example"}],
            "trusted_context": {
                "schema_version": 1,
                "items": [{"kind": "catalog", "source": "catalog.search", "status": "success", "payload": {"title": "Example"}}],
            },
        }

        candidate_input = build_candidate_input(case)
        candidate_input["turns"][0]["text"] = "changed"
        candidate_input["trusted_context"]["items"][0]["payload"]["title"] = "Changed"

        self.assertEqual(case["turns"][0]["text"], "play Example")
        self.assertEqual(case["trusted_context"]["items"][0]["payload"]["title"], "Example")

    def test_null_trusted_context_stays_null(self):
        candidate_input = build_candidate_input({
            "turns": [{"role": "user", "text": "play Example"}],
            "trusted_context": None,
        })

        self.assertIsNone(candidate_input["trusted_context"])

    def test_interactive_projection_exposes_tools_and_returned_observations_only(self):
        tool = {
            "name": "catalog.search",
            "description": "Search the fixture catalog.",
            "effect": "read_only",
            "arguments_schema": {"type": "object", "required": ["query"]},
        }
        scenario = {
            "schema_version": 1,
            "tools": [tool],
            "fixtures": [{
                "tool": "catalog.search",
                "arguments": {"query": "secret expected query"},
                "result": {"status": "success", "payload": {"items": []}},
            }],
            "max_model_steps": 3,
            "max_tool_calls": 2,
        }
        history = [{
            "tool": "catalog.search",
            "arguments": {"query": "actual query"},
            "observation": {"status": "empty", "payload": {}, "simulated": True, "effect": "read_only"},
        }]
        case = {
            "turns": [{"role": "user", "text": "Find an album."}],
            "trusted_context": None,
            "tool_scenario": scenario,
        }

        candidate_input = build_candidate_input(case, tool_history=history)

        self.assertEqual(candidate_input["schema_version"], 2)
        self.assertEqual(candidate_input["tools"], [tool])
        self.assertEqual(candidate_input["tool_history"], history)
        self.assertNotIn("fixtures", candidate_input)
        candidate_input["tools"][0]["description"] = "changed"
        candidate_input["tool_history"][0]["observation"]["payload"]["leak"] = True
        self.assertEqual(scenario["tools"][0]["description"], "Search the fixture catalog.")
        self.assertNotIn("leak", history[0]["observation"]["payload"])


if __name__ == "__main__":
    unittest.main()
