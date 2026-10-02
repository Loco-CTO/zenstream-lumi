import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.coverage_audit import CoverageInputError, audit_cases as _audit_cases, main
from review_fixtures import make_review_ledger


def make_case(
    case_id,
    *,
    split="development",
    review_status="ready",
    language="en",
    category="simple_action",
    family_id=None,
):
    reviewed = review_status == "ready"
    return {
        "schema_version": 2,
        "case_id": case_id,
        "family_id": family_id or case_id,
        "split": split,
        "review_status": review_status,
        "language": language,
        "categories": [category],
        "turns": [{"role": "user", "text": "Play it"}],
        "trusted_context": None,
        "gold": {
            "decision": "act",
            "action": "play",
            "arguments": {"title": "Example"},
            "requires_clarification": False,
        },
        "provenance_record_id": f"prov-{case_id}",
        "review": {
            "annotation_status": "approved" if reviewed else "pending",
            "language_review_status": "approved" if language in {"ja", "en_ja"} else "not_required",
            "reviewer_ids": (
                ["annotator-1", "annotator-2"] + (["language-reviewer"] if language in {"ja", "en_ja"} else [])
                if reviewed else []
            ),
            "review_record_ids": [],
        },
    }


def audit_cases(cases, **kwargs):
    review_records = make_review_ledger(cases)
    return _audit_cases(cases, review_records, **kwargs)


class EvaluationCoverageAuditTests(unittest.TestCase):
    def test_reports_ready_draft_and_language_specific_slice_presence(self):
        cases = [
            make_case("en-ready", category="simple_action"),
            make_case("ja-ready", language="ja", category="simple_action"),
            make_case("mix-draft", language="en_ja", category="simple_action", review_status="draft"),
            make_case("en-no-action", category="no_action_negation", review_status="draft"),
            make_case("en-typo", category="english_typos"),
        ]

        report = audit_cases(cases)

        self.assertEqual(report["case_inventory"]["case_count"], 5)
        self.assertEqual(report["case_inventory"]["ready_case_count"], 3)
        self.assertEqual(report["language_inventory"]["en_ja"]["presence_status"], "draft_only")
        self.assertFalse(report["slice_presence"]["all_required_slices_have_ready_cases"])
        slices = {item["id"]: item for item in report["required_slices"]}
        self.assertEqual(
            slices["simple_action"]["by_language"]["en"]["presence_status"],
            "ready_cases_present",
        )
        self.assertEqual(
            slices["simple_action"]["by_language"]["en_ja"]["presence_status"],
            "draft_only",
        )
        self.assertEqual(
            slices["no_action_negation"]["by_language"]["en"]["presence_status"],
            "draft_only",
        )
        self.assertEqual(
            slices["english_typos"]["by_language"]["en"]["ready_family_count"],
            1,
        )
        self.assertEqual(report["unmapped_category_tags"], [])

    def test_empty_inventory_reports_missing_slices_without_claiming_readiness(self):
        report = audit_cases([])

        self.assertEqual(report["case_inventory"]["presence_status"], "missing")
        self.assertEqual(report["slice_presence"]["ready_slice_language_count"], 0)
        self.assertTrue(report["slice_presence"]["missing_slice_language_count"] > 0)
        self.assertFalse(report["slice_presence"]["all_required_slices_have_ready_cases"])

    def test_cli_reads_jsonl_and_hashes_the_inventory_and_slice_taxonomy(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            case_path = Path(temporary_directory) / "cases.jsonl"
            review_path = Path(temporary_directory) / "review-records.jsonl"
            report_path = Path(temporary_directory) / "coverage.json"
            case = make_case("cli-case")
            review_records = make_review_ledger([case])
            case_path.write_text(json.dumps(case) + "\n", encoding="utf-8")
            review_path.write_text(
                "".join(json.dumps(record) + "\n" for record in review_records),
                encoding="utf-8",
            )

            result = main([
                "--cases", str(case_path), "--review-records", str(review_path),
                "--output", str(report_path),
            ])

            self.assertEqual(result, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            expected_case_hash = "sha256:" + hashlib.sha256(case_path.read_bytes()).hexdigest()
            self.assertEqual(report["case_manifest_sha256"], expected_case_hash)
            self.assertEqual(
                report["review_records_sha256"],
                "sha256:" + hashlib.sha256(review_path.read_bytes()).hexdigest(),
            )
            self.assertTrue(report["required_slices_sha256"].startswith("sha256:"))

    def test_reports_descriptive_categories_separately_from_required_categories(self):
        report = audit_cases([make_case("with-descriptive-tag") | {"categories": ["simple_action", "register:casual"]}])

        self.assertEqual(report["unmapped_category_tags"], ["register:casual"])

    def test_final_holdout_requires_explicit_authorization(self):
        cases = [make_case("holdout", split="final_holdout")]

        with self.assertRaisesRegex(CoverageInputError, "final holdout requires explicit"):
            audit_cases(cases, split="final_holdout")

        report = audit_cases(cases, split="final_holdout", allow_final_holdout=True)
        self.assertTrue(report["final_audit"])
        self.assertEqual(report["case_inventory"]["case_count"], 1)

    def test_rejects_duplicate_ids_and_mixed_split_inventories(self):
        same_id = make_case("duplicate")
        with self.assertRaisesRegex(CoverageInputError, "duplicate case_id"):
            audit_cases([same_id, same_id.copy()])

        with self.assertRaisesRegex(CoverageInputError, "only the requested split"):
            audit_cases([
                make_case("dev-family", family_id="shared-family"),
                make_case("holdout-family", split="final_holdout", family_id="shared-family"),
            ])


if __name__ == "__main__":
    unittest.main()
