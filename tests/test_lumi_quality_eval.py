from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import lumi.model_installation as model_installation
from scripts import lumi_quality_eval as evaluation


class LumiQualityEvaluationTests(unittest.IsolatedAsyncioTestCase):
    def test_suite_has_thirty_to_fifty_multilingual_turns_and_followups(self) -> None:
        self.assertGreaterEqual(len(evaluation.SCENARIOS), 30)
        self.assertLessEqual(len(evaluation.SCENARIOS), 50)
        languages = {turn["id"].split("-", 1)[0] for turn in evaluation.SCENARIOS}
        self.assertEqual(languages, {"EN", "JA", "ZH"})
        language_counts = Counter(
            turn["id"].split("-", 1)[0] for turn in evaluation.SCENARIOS
        )
        self.assertGreaterEqual(language_counts["JA"], 6)
        self.assertGreaterEqual(language_counts["ZH"], 8)
        counts = Counter(turn["conversation"] for turn in evaluation.SCENARIOS)
        self.assertGreaterEqual(sum(count > 1 for count in counts.values()), 10)

    def test_scenarios_are_rubric_cases_not_expected_answers(self) -> None:
        self.assertEqual(
            len({turn["id"] for turn in evaluation.SCENARIOS}),
            len(evaluation.SCENARIOS),
        )
        self.assertTrue(
            all(
                turn["conversation"] in evaluation.QUALITY_RUBRIC
                for turn in evaluation.SCENARIOS
            )
        )
        forbidden_keys = {"expected_answer", "gold_answer", "expected_response", "target_response"}
        for turn in evaluation.SCENARIOS:
            self.assertTrue(forbidden_keys.isdisjoint(turn))
        metadata = evaluation.suite_metadata()
        self.assertTrue(metadata["evaluation_only"])
        self.assertFalse(metadata["runtime_string_matching"])
        self.assertFalse(metadata["answer_templates"])
        self.assertFalse(metadata["web_search_query_quality_evaluated"])
        self.assertTrue(metadata["web_evidence_is_frozen_synthetic_fixture"])
        self.assertEqual(metadata["acceptance_baseline_model"], "qwen3.5:2b")
        for field in (
            "scenario_sha256",
            "fixture_sha256",
            "rubric_sha256",
            "judge_system_prompt_sha256",
            "evaluator_script_sha256",
        ):
            self.assertEqual(len(metadata[field]), 64)

    def test_nine_billion_pin_is_eval_only_and_matches_manifest(self) -> None:
        self.assertEqual(
            evaluation.EXPERIMENTAL_9B.pinned_source_manifest_sha256,
            evaluation.EXPERIMENTAL_9B_SOURCE_MANIFEST_SHA256,
        )
        self.assertNotIn(
            evaluation.EXPERIMENTAL_9B.model_id,
            {option.model_id for option in evaluation.supported_models()},
        )

    def test_nine_billion_registration_is_removed_after_evaluation_scope(self) -> None:
        model_id = evaluation.EXPERIMENTAL_9B.model_id
        original = model_installation._MODEL_SPECS.get(model_id)
        with evaluation._experimental_9b_registration():
            self.assertEqual(
                model_installation._MODEL_SPECS[model_id],
                evaluation.EXPERIMENTAL_9B,
            )
        self.assertEqual(model_installation._MODEL_SPECS.get(model_id), original)
        self.assertNotIn(model_id, {option.model_id for option in evaluation.supported_models()})

    def test_output_must_be_outside_repository(self) -> None:
        with self.assertRaises(ValueError):
            evaluation.validate_output_directory(evaluation.PROJECT_ROOT / "eval-results")
        outside = evaluation.PROJECT_ROOT.parents[2] / "lumi-evaluation-output"
        output = evaluation.validate_output_directory(outside)
        self.assertEqual(output, outside.resolve())

    def test_runtime_release_manifest_is_hashed_for_reproducibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_bytes = b'{"tag":"test","runtimeWheels":[]}'
            (root / "lumi-release-install.json").write_bytes(manifest_bytes)
            metadata = evaluation.runtime_release_metadata(root)
        self.assertEqual(
            metadata["install_manifest_sha256"],
            hashlib.sha256(manifest_bytes).hexdigest(),
        )
        self.assertEqual(metadata["install_manifest"]["tag"], "test")

    def test_resource_snapshots_are_best_effort_and_structured(self) -> None:
        memory = evaluation.process_memory_snapshot()
        self.assertIsInstance(memory, dict)
        self.assertIn("measurement", memory)
        working_set = memory.get(
            "working_set_bytes",
            memory.get("run_peak_working_set_bytes", memory.get("run_peak_rss_bytes")),
        )
        self.assertTrue(working_set is None or isinstance(working_set, int))
        gpu = evaluation.gpu_memory_snapshot()
        self.assertTrue(gpu is None or isinstance(gpu.get("devices"), list))

    def test_summary_reports_prompt_generation_and_gpu_memory_metrics(self) -> None:
        def row(prompt_ns: int, generation_ns: int, before_mib: int, after_mib: int):
            return {
                "runtime_responses": [{
                    "prompt_duration_ns": prompt_ns,
                    "generation_duration_ns": generation_ns,
                }],
                "gpu_memory_before": {
                    "devices": [{"index": 0, "used_mib": before_mib}],
                },
                "gpu_memory_after": {
                    "devices": [{"index": 0, "used_mib": after_mib}],
                },
                "memory_after": {
                    "working_set_bytes": 1_200_000_000,
                    "run_peak_working_set_bytes": 1_300_000_000,
                },
            }

        summary = evaluation.summarize_model(
            [row(2_000_000, 4_000_000, 1_000, 1_300),
             row(4_000_000, 8_000_000, 1_300, 1_200)],
            runtime_open_ms=0,
        )
        self.assertEqual(summary["median_prompt_eval_ms_per_turn"], 3)
        self.assertEqual(summary["median_generation_ms_per_turn"], 6)
        self.assertEqual(summary["median_warm_prompt_eval_ms_per_turn"], 4)
        self.assertEqual(summary["median_warm_generation_ms_per_turn"], 8)
        self.assertEqual(summary["max_system_nvidia_gpu_memory_used_mib"], 1_300)
        self.assertEqual(summary["max_system_nvidia_gpu_memory_increase_mib"], 300)
        self.assertEqual(summary["max_sampled_process_working_set_bytes"], 1_200_000_000)
        self.assertIsInstance(summary["max_sampled_process_working_set_bytes"], int)
        self.assertNotIn("max_process_peak_memory_bytes", summary)

    def test_visible_generation_rate_uses_stable_visible_stream_window(self) -> None:
        self.assertEqual(evaluation._visible_output_rate(100, 250, 1_250), 100)
        self.assertIsNone(evaluation._visible_output_rate(100, 250, 250))
        self.assertIsNone(evaluation._visible_output_rate(100, None, 1_250))

    async def test_fixtures_are_conversation_scoped_and_web_reads_use_searched_url(self) -> None:
        context = evaluation.ChatContext(
            account_id="synthetic-evaluation-user",
            conversation_id="off-topic",
            model="qwen3.5:2b",
            thinking=False,
            turn_id="EN-34",
            user_message="Show me that series.",
        )
        detail = await evaluation.FixtureClient().item_detail(
            context, entity_id="lab-series-northwind", language=None
        )
        self.assertIsNone(detail["item"])
        self.assertEqual(detail["seasons"], [])

        context = evaluation.ChatContext(
            account_id="synthetic-evaluation-user",
            conversation_id="glass-current",
            model="qwen3.5:2b",
            thinking=False,
            turn_id="EN-11",
            user_message="Verify the announcement.",
        )
        sessions = evaluation.WebResearchSessions()
        config = evaluation.WebResearchConfig(max_results=6, search_min_interval_seconds=0)
        search = evaluation.FixtureWeb(
            evaluation.SearXNGSearchTool(config, sessions), sessions
        )
        reader = evaluation.FixtureWeb(evaluation.WebReadTool(config, sessions), sessions)
        evaluation.SEARCHED_URLS.pop(context.conversation_id, None)
        result = await search.execute(context, {})
        self.assertEqual(result.trust, evaluation.EvidenceTrust.EXTERNAL)
        bad = await reader.execute(context, {"url": "https://attacker.example/path"})
        self.assertEqual(bad.trust, evaluation.EvidenceTrust.LOCAL)
        good = await reader.execute(
            context,
            {"url": "https://studio.example.test/news/glass-harbor-season-two"},
        )
        self.assertEqual(good.trust, evaluation.EvidenceTrust.EXTERNAL)
        self.assertEqual(
            good.sources[0].url,
            "https://studio.example.test/news/glass-harbor-season-two",
        )

    def test_judge_json_is_validated_as_rubric_scores(self) -> None:
        score_values = {dimension: 4 for dimension in evaluation.SCORE_DIMENSIONS}
        parsed = evaluation._parse_judgement(
            '{"scores":' + json.dumps(score_values)
            + ',"overall":4,"critical_failure":false,"notes":"Grounded response."}'
        )
        self.assertEqual(parsed["overall"], 4)
        score_values["grounding"] = 6
        with self.assertRaises(ValueError):
            evaluation._parse_judgement(
                '{"scores":' + json.dumps(score_values)
                + ',"overall":4,"critical_failure":false,"notes":"invalid score"}'
            )

    def test_pairwise_advantage_requires_enough_scored_cases(self) -> None:
        baseline = []
        candidate = []
        for index in range(20):
            base_scores = {dimension: 3 for dimension in evaluation.SCORE_DIMENSIONS}
            candidate_scores = {dimension: 4 for dimension in evaluation.SCORE_DIMENSIONS}
            language = "JA" if index < 3 else "EN"
            baseline.append({
                "id": f"{language}-{index}",
                "conversation": f"chain-{index}",
                "category": "recommendations",
                "judgement": {
                    "status": "scored", "overall": 3,
                    "critical_failure": False, "scores": base_scores,
                },
            })
            candidate.append({
                "id": f"{language}-{index}",
                "conversation": f"chain-{index}",
                "category": "recommendations",
                "judgement": {
                    "status": "scored", "overall": 4,
                    "critical_failure": False, "scores": candidate_scores,
                },
            })
        self.assertTrue(evaluation.compare_models(baseline, candidate)["clear_quality_advantage"])
        self.assertFalse(
            evaluation.compare_models(baseline[:5], candidate[:5])["clear_quality_advantage"]
        )
        partial = evaluation.compare_models(baseline, candidate[:19])
        self.assertFalse(partial["all_selected_cases_scored_in_both_models"])
        self.assertFalse(partial["clear_quality_advantage"])
        language_regression = [dict(row) for row in candidate]
        for row in language_regression[:3]:
            row["judgement"] = dict(row["judgement"], overall=1)
        regression = evaluation.compare_models(baseline, language_regression)
        self.assertTrue(regression["meaningful_language_regression"])
        self.assertFalse(regression["clear_quality_advantage"])
        one_chain_baseline = [
            dict(row, conversation="single-chain") for row in baseline
        ]
        one_chain_candidate = [
            dict(row, conversation="single-chain") for row in candidate
        ]
        clustered = evaluation.compare_models(one_chain_baseline, one_chain_candidate)
        self.assertIsNone(clustered["bootstrap_95_percent_interval"])
        self.assertFalse(clustered["clear_quality_advantage"])


if __name__ == "__main__":
    unittest.main()
