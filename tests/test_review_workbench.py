import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from evaluation.review_workbench import (
    ReviewHTTPServer,
    ReviewManager,
    WorkbenchError,
    _read_cases,
)


def make_case(*, language="en", split="development", tool_scenario=None):
    gold = {
        "decision": "no_action",
        "action": None,
        "arguments": {},
        "requires_clarification": False,
    }
    if tool_scenario is not None:
        gold["tool_trajectory"] = {"calls": [], "completion": "final"}
    return {
        "schema_version": 4,
        "case_id": "case-private-id",
        "family_id": "family-private-id",
        "split": split,
        "review_status": "draft",
        "language": language,
        "categories": ["no_action"],
        "turns": [{"role": "user", "text": "Do not play anything."}],
        "trusted_context": None,
        "tool_scenario": tool_scenario,
        "gold": gold,
        "provenance_record_id": "private-provenance-id",
        "review": {
            "annotation_status": "pending",
            "language_review_status": "pending" if language != "en" else "not_required",
            "reviewer_ids": [],
            "review_record_ids": [],
        },
    }


class ReviewWorkbenchTests(unittest.TestCase):
    def test_browser_projection_excludes_case_labels_and_gold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ReviewManager(
                [make_case()], Path(temp_dir) / "review.jsonl", "rev-reviewer-01", "semantic"
            )
            self.assertEqual(
                set(manager.safe_cases()[0]),
                {"index", "language", "turns", "trusted_context"},
            )
            self.assertNotIn("case-private-id", json.dumps(manager.safe_cases()))
            self.assertNotIn("no_action", json.dumps(manager.safe_cases()))

    def test_semantic_review_appends_a_valid_hash_bound_record_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "review.jsonl"
            manager = ReviewManager([make_case()], output, "rev-reviewer-01", "semantic")
            result = manager.submit({
                "index": 0,
                "proposed_gold": {
                    "decision": "no_action",
                    "action": None,
                    "arguments": {},
                    "requires_clarification": False,
                },
            })
            self.assertEqual(result, {"completed": 1, "total": 1})
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["reviewer_id"], "rev-reviewer-01")
            self.assertEqual(record["record_type"], "independent_annotation")
            with self.assertRaises(WorkbenchError):
                manager.submit({
                    "index": 0,
                    "proposed_gold": {
                        "decision": "no_action",
                        "action": None,
                        "arguments": {},
                        "requires_clarification": False,
                    },
                })

    def test_language_review_rejects_wrong_role_and_unhashable_values_cleanly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ReviewManager(
                [make_case(language="ja")], Path(temp_dir) / "review.jsonl",
                "rev-reviewer-02", "language",
            )
            with self.assertRaises(WorkbenchError):
                manager.submit({
                    "index": 0,
                    "language_review": {
                        "qualification": [],
                        "naturalness_status": "approved",
                        "meaning_preservation_status": "approved",
                    },
                })

    def test_development_loader_rejects_holdout_and_tool_scenarios(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cases.jsonl"
            path.write_text(json.dumps(make_case(split="final_holdout")), encoding="utf-8")
            with self.assertRaises(WorkbenchError):
                _read_cases(path)

            scenario = {
                "schema_version": 1,
                "tools": [{
                    "name": "search",
                    "description": "Search media",
                    "effect": "read_only",
                    "arguments_schema": {"type": "object"},
                }],
                "fixtures": [],
                "max_model_steps": 1,
                "max_tool_calls": 0,
            }
            path.write_text(json.dumps(make_case(tool_scenario=scenario)), encoding="utf-8")
            with self.assertRaisesRegex(WorkbenchError, "tool scenarios are not supported"):
                _read_cases(path)

    def test_http_projection_is_blind_and_cross_origin_submission_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ReviewManager(
                [make_case()], Path(temp_dir) / "review.jsonl", "rev-reviewer-01", "semantic"
            )
            server = ReviewHTTPServer(("127.0.0.1", 0), manager)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_address[1]
            host = f"127.0.0.1:{port}"
            try:
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
                connection.request("GET", "/api/cases", headers={"Host": host})
                response = connection.getresponse()
                body = json.loads(response.read())
                self.assertEqual(response.status, 200)
                self.assertEqual(response.getheader("Cache-Control"), "no-store")
                self.assertNotIn("case-private-id", json.dumps(body))
                self.assertNotIn("no_action", json.dumps(body))
                connection.close()

                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
                connection.request("GET", "/api/cases", headers={"Host": f"localhost:{port}"})
                response = connection.getresponse()
                response.read()
                self.assertEqual(response.status, 403)
                connection.close()

                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
                connection.request("GET", "/api/session", headers={"Host": host})
                response = connection.getresponse()
                session = json.loads(response.read())
                connection.close()
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
                connection.request(
                    "POST",
                    "/api/review",
                    body=json.dumps({"index": 0, "proposed_gold": {}}),
                    headers={
                        "Host": host,
                        "Origin": "http://attacker.example",
                        "Content-Type": "application/json",
                        "X-Lumi-Review-Token": session["csrf_token"],
                    },
                )
                response = connection.getresponse()
                response.read()
                self.assertEqual(response.status, 403)
                connection.close()
                self.assertFalse(manager.output_path.exists())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
