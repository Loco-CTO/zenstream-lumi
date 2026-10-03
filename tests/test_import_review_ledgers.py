import json
import tempfile
import unittest
from pathlib import Path

from evaluation.import_review_ledgers import LedgerImportError, import_ledgers
from evaluation.review_records import read_review_records, validate_case_review_records
from evaluation.review_workbench import ReviewManager
from test_provenance_validator import (
    make_generation_manifest,
    make_record,
    make_source_manifest,
)


def make_case(*, language="ja", split="development"):
    return {
        "schema_version": 4,
        "case_id": "case-review-import",
        "family_id": "family-review-import",
        "split": split,
        "review_status": "draft",
        "language": language,
        "categories": ["no_action"],
        "turns": [{"role": "user", "text": "音楽を再生しないで。"}],
        "trusted_context": None,
        "tool_scenario": None,
        "gold": {
            "decision": "no_action",
            "action": None,
            "arguments": {},
            "requires_clarification": False,
        },
        "provenance_record_id": "provenance-review-import",
        "review": {
            "annotation_status": "pending",
            "language_review_status": "not_required" if language == "en" else "pending",
            "reviewer_ids": [],
            "review_record_ids": [],
        },
    }


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def provenance_arguments(root, case, *, approved=True):
    sources_path = root / "sources.json"
    generations_path = root / "generations.json"
    records_path = root / "sample-records.jsonl"
    sources_path.write_text(json.dumps(make_source_manifest()), encoding="utf-8")
    generations_path.write_text(json.dumps(make_generation_manifest()), encoding="utf-8")
    record = make_record(case)
    if not approved:
        record["decision"] = "review_pending"
    write_jsonl(records_path, [record])
    return {
        "provenance_records_path": records_path,
        "source_manifest_path": sources_path,
        "generation_manifest_path": generations_path,
    }


def create_reviewer_ledger(case, path, reviewer_id, role):
    manager = ReviewManager([case], path, reviewer_id, role)
    if role == "semantic":
        manager.submit({
            "index": 0,
            "proposed_gold": {
                "decision": "no_action",
                "action": None,
                "arguments": {},
                "requires_clarification": False,
            },
        })
    else:
        manager.submit({
            "index": 0,
            "language_review": {
                "qualification": "native_japanese",
                "naturalness_status": "approved",
                "meaning_preservation_status": "approved",
            },
        })


class ImportReviewLedgersTests(unittest.TestCase):
    def test_combines_distinct_reviews_and_keeps_case_draft(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = make_case()
            cases_path = root / "cases.jsonl"
            write_jsonl(cases_path, [case])
            ledgers = [
                root / "semantic-1.jsonl",
                root / "semantic-2.jsonl",
                root / "language.jsonl",
            ]
            create_reviewer_ledger(case, ledgers[0], "rev-semantic-01", "semantic")
            create_reviewer_ledger(case, ledgers[1], "rev-semantic-02", "semantic")
            create_reviewer_ledger(case, ledgers[2], "rev-japanese-01", "language")

            output_cases = root / "combined-cases.jsonl"
            output_ledger = root / "combined-reviews.jsonl"
            summary = import_ledgers(
                cases_path,
                ledgers,
                output_cases,
                output_ledger,
                **provenance_arguments(root, case),
            )

            merged_case = json.loads(output_cases.read_text(encoding="utf-8"))
            merged_records = read_review_records(output_ledger)
            self.assertEqual(summary["ready_case_count"], 0)
            self.assertEqual(summary["draft_case_count"], 1)
            self.assertEqual(summary["semantic_annotation_count"], 2)
            self.assertEqual(summary["language_review_count"], 1)
            self.assertEqual(merged_case["review_status"], "draft")
            self.assertEqual(merged_case["review"]["annotation_status"], "pending")
            self.assertEqual(merged_case["review"]["language_review_status"], "pending")
            self.assertEqual(merged_case["gold"], case["gold"])
            self.assertEqual(len(merged_case["review"]["review_record_ids"]), 3)
            self.assertEqual(
                merged_case["review"]["reviewer_ids"],
                ["rev-japanese-01", "rev-semantic-01", "rev-semantic-02"],
            )
            validate_case_review_records([merged_case], merged_records)

    def test_rejects_same_reviewer_in_semantic_and_language_roles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = make_case()
            cases_path = root / "cases.jsonl"
            write_jsonl(cases_path, [case])
            semantic = root / "semantic.jsonl"
            language = root / "language.jsonl"
            create_reviewer_ledger(case, semantic, "rev-same-person", "semantic")
            create_reviewer_ledger(case, language, "rev-same-person", "language")

            with self.assertRaisesRegex(LedgerImportError, "both semantic and language"):
                import_ledgers(
                    cases_path,
                    [semantic, language],
                    root / "combined-cases.jsonl",
                    root / "combined-reviews.jsonl",
                    **provenance_arguments(root, case),
                )

    def test_refuses_holdout_cases_and_preserves_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cases_path = root / "cases.jsonl"
            write_jsonl(cases_path, [make_case(split="final_holdout")])
            ledger = root / "empty.jsonl"
            ledger.write_text("", encoding="utf-8")
            output_cases = root / "combined-cases.jsonl"
            output_ledger = root / "combined-reviews.jsonl"
            output_cases.write_text("keep me", encoding="utf-8")

            with self.assertRaises(LedgerImportError):
                import_ledgers(
                    cases_path,
                    [ledger],
                    output_cases,
                    output_ledger,
                    **provenance_arguments(root, make_case(split="final_holdout")),
                )
            self.assertEqual(output_cases.read_text(encoding="utf-8"), "keep me")
            self.assertFalse(output_ledger.exists())

    def test_rejects_case_output_inside_repository(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cases_path = root / "cases.jsonl"
            write_jsonl(cases_path, [make_case(language="en")])
            ledger = root / "empty.jsonl"
            ledger.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(LedgerImportError, "outside the Git repository"):
                import_ledgers(
                    cases_path,
                    [ledger],
                    Path(__file__).resolve().parents[1] / "unsafe-output.jsonl",
                    root / "combined-reviews.jsonl",
                    **provenance_arguments(root, make_case(language="en")),
                )

    def test_rejects_output_inside_a_sibling_lumi_worktree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cases_path = root / "cases.jsonl"
            write_jsonl(cases_path, [make_case(language="en")])
            ledger = root / "empty.jsonl"
            ledger.write_text("", encoding="utf-8")
            sibling_checkout = Path(__file__).resolve().parents[4] / "zenstream-lumi"
            with self.assertRaisesRegex(LedgerImportError, "outside the Git repository"):
                import_ledgers(
                    cases_path,
                    [ledger],
                    sibling_checkout / "unsafe-output.jsonl",
                    root / "combined-reviews.jsonl",
                    **provenance_arguments(root, make_case(language="en")),
                )

    def test_pending_rights_block_review_ledger_import_without_writing_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = make_case()
            cases_path = root / "cases.jsonl"
            write_jsonl(cases_path, [case])
            ledger = root / "reviewer.jsonl"
            create_reviewer_ledger(case, ledger, "rev-semantic-01", "semantic")
            output_cases = root / "combined-cases.jsonl"
            output_ledger = root / "combined-reviews.jsonl"

            with self.assertRaisesRegex(LedgerImportError, "provenance is not approved"):
                import_ledgers(
                    cases_path,
                    [ledger],
                    output_cases,
                    output_ledger,
                    **provenance_arguments(root, case, approved=False),
                )

            self.assertFalse(output_cases.exists())
            self.assertFalse(output_ledger.exists())


if __name__ == "__main__":
    unittest.main()
