import copy
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.review_records import (
    ReviewRecordError,
    read_review_records,
    validate_case_review_records,
)
from provenance.validate import evaluation_case_sha256


def make_case(case_id="case-1", language="en"):
    reviewers = ["annotator-1", "annotator-2"]
    if language in {"ja", "en_ja"}:
        reviewers.append("language-reviewer")
    return {
        "schema_version": 3,
        "case_id": case_id,
        "family_id": case_id,
        "split": "development",
        "review_status": "ready",
        "language": language,
        "categories": ["simple_action"],
        "turns": [{"role": "user", "text": "Play the album"}],
        "trusted_context": None,
        "gold": {
            "decision": "act",
            "action": "play",
            "arguments": {"title": "Example"},
            "requires_clarification": False,
        },
        "provenance_record_id": f"provenance-{case_id}",
        "review": {
            "annotation_status": "approved",
            "language_review_status": "approved" if language in {"ja", "en_ja"} else "not_required",
            "reviewer_ids": reviewers,
            "review_record_ids": [],
        },
    }


def independent_record(case, index, reviewer_id, proposed_gold=None):
    return {
        "schema_version": 1,
        "review_record_id": f"review-{case['case_id']}-annotation-{index}",
        "case_id": case["case_id"],
        "case_sha256": evaluation_case_sha256(case),
        "reviewer_id": reviewer_id,
        "reviewed_at": "2026-10-02T12:00:00Z",
        "record_type": "independent_annotation",
        "supersedes_record_ids": [],
        "proposed_gold": copy.deepcopy(proposed_gold or case["gold"]),
    }


def language_record(case, *, reviewer_id="language-reviewer", qualification=None):
    return {
        "schema_version": 1,
        "review_record_id": f"review-{case['case_id']}-language",
        "case_id": case["case_id"],
        "case_sha256": evaluation_case_sha256(case),
        "reviewer_id": reviewer_id,
        "reviewed_at": "2026-10-02T12:00:00Z",
        "record_type": "language_review",
        "supersedes_record_ids": [],
        "language": case["language"],
        "qualification": qualification or ("native_japanese" if case["language"] == "ja" else "fluent_bilingual"),
        "naturalness_status": "approved",
        "meaning_preservation_status": "approved",
    }


def ready_ledger(case):
    records = [
        independent_record(case, 1, "annotator-1"),
        independent_record(case, 2, "annotator-2"),
    ]
    if case["language"] in {"ja", "en_ja"}:
        records.append(language_record(case))
    case["review"]["review_record_ids"] = [record["review_record_id"] for record in records]
    return records


class ReviewRecordTests(unittest.TestCase):
    def test_two_independent_matching_annotations_verify_an_english_ready_case(self):
        case = make_case()

        validate_case_review_records([case], ready_ledger(case))

    def test_ready_case_requires_two_distinct_annotation_records(self):
        case = make_case()
        records = [independent_record(case, 1, "annotator-1")]
        case["review"]["review_record_ids"] = [records[0]["review_record_id"]]
        case["review"]["reviewer_ids"] = ["annotator-1"]

        with self.assertRaisesRegex(ReviewRecordError, "at least two independent annotations"):
            validate_case_review_records([case], records)

    def test_active_reviews_must_match_the_current_content_hash(self):
        case = make_case()
        records = ready_ledger(case)
        case["turns"][0]["text"] = "Play a different album"

        with self.assertRaisesRegex(ReviewRecordError, "stale case hash"):
            validate_case_review_records([case], records)

    def test_review_history_for_an_older_hash_can_remain_unreferenced(self):
        case = make_case()
        old_record = independent_record(case, 0, "annotator-old")
        case["turns"][0]["text"] = "Play a different album"
        records = ready_ledger(case)

        validate_case_review_records([case], [old_record, *records])

    def test_same_hash_review_revision_must_supersede_and_preserve_the_prior_record(self):
        case = make_case()
        original = ready_ledger(case)
        revised = independent_record(case, 3, "annotator-1")
        revised["reviewed_at"] = "2026-10-02T13:00:00Z"
        revised["supersedes_record_ids"] = [original[0]["review_record_id"]]
        case["review"]["review_record_ids"] = [
            revised["review_record_id"], original[1]["review_record_id"],
        ]

        validate_case_review_records([case], [original[0], original[1], revised])

    def test_current_unsuperseded_records_cannot_be_omitted_from_case_references(self):
        case = make_case()
        records = ready_ledger(case)
        extra = independent_record(case, 3, "annotator-3")

        with self.assertRaisesRegex(ReviewRecordError, "include every current unsuperseded"):
            validate_case_review_records([case], [*records, extra])

    def test_disagreement_requires_an_adjudication_that_cites_all_independent_reviews(self):
        case = make_case()
        alternate_gold = copy.deepcopy(case["gold"])
        alternate_gold["arguments"]["title"] = "Other"
        records = [
            independent_record(case, 1, "annotator-1"),
            independent_record(case, 2, "annotator-2", alternate_gold),
        ]
        case["review"]["review_record_ids"] = [record["review_record_id"] for record in records]

        with self.assertRaisesRegex(ReviewRecordError, "requires one adjudication"):
            validate_case_review_records([case], records)

        adjudication = {
            "schema_version": 1,
            "review_record_id": "review-case-1-adjudication",
            "case_id": case["case_id"],
            "case_sha256": evaluation_case_sha256(case),
            "reviewer_id": "adjudicator-1",
            "reviewed_at": "2026-10-02T13:00:00Z",
            "record_type": "adjudication",
            "supersedes_record_ids": [],
            "basis_record_ids": [record["review_record_id"] for record in records],
            "adjudicated_gold": case["gold"],
            "rationale": "The supplied turns specify the selected album title.",
        }
        records.append(adjudication)
        case["review"]["review_record_ids"].append(adjudication["review_record_id"])
        case["review"]["reviewer_ids"].append("adjudicator-1")

        validate_case_review_records([case], records)

        adjudication["reviewed_at"] = "2026-10-02T11:00:00Z"
        with self.assertRaisesRegex(ReviewRecordError, "adjudication must follow every independent annotation"):
            validate_case_review_records([case], records)

    def test_adjudication_must_reference_every_independent_record(self):
        case = make_case()
        alternate_gold = copy.deepcopy(case["gold"])
        alternate_gold["arguments"]["title"] = "Other"
        records = [
            independent_record(case, 1, "annotator-1"),
            independent_record(case, 2, "annotator-2", alternate_gold),
        ]
        adjudication = {
            "schema_version": 1,
            "review_record_id": "review-case-1-adjudication",
            "case_id": case["case_id"],
            "case_sha256": evaluation_case_sha256(case),
            "reviewer_id": "adjudicator-1",
            "reviewed_at": "2026-10-02T13:00:00Z",
            "record_type": "adjudication",
            "supersedes_record_ids": [],
            "basis_record_ids": [records[0]["review_record_id"], "other-annotation-record"],
            "adjudicated_gold": case["gold"],
            "rationale": "The supplied turns specify the selected album title.",
        }
        records.append(adjudication)
        case["review"]["review_record_ids"] = [record["review_record_id"] for record in records]
        case["review"]["reviewer_ids"] = ["annotator-1", "annotator-2", "adjudicator-1"]

        with self.assertRaisesRegex(ReviewRecordError, "reference every active independent annotation"):
            validate_case_review_records([case], records)

    def test_japanese_case_needs_a_separate_qualified_reviewer_for_both_language_checks(self):
        case = make_case(language="ja")
        records = ready_ledger(case)

        validate_case_review_records([case], records)

        records[-1]["qualification"] = "fluent_bilingual"
        with self.assertRaisesRegex(ReviewRecordError, "qualification is insufficient"):
            validate_case_review_records([case], records)

    def test_code_switch_case_needs_fluent_bilingual_review(self):
        case = make_case(language="en_ja")
        records = ready_ledger(case)

        validate_case_review_records([case], records)

        records[-1]["meaning_preservation_status"] = "needs_revision"
        with self.assertRaisesRegex(ReviewRecordError, "approved naturalness and meaning-preservation"):
            validate_case_review_records([case], records)

    def test_language_reviewer_must_be_independent_from_semantic_annotators(self):
        case = make_case(language="ja")
        records = ready_ledger(case)
        records[-1]["reviewer_id"] = "annotator-1"
        case["review"]["reviewer_ids"] = ["annotator-1", "annotator-2"]

        with self.assertRaisesRegex(ReviewRecordError, "language reviewer must be independent"):
            validate_case_review_records([case], records)

    def test_duplicate_record_ids_and_records_outside_the_inventory_are_rejected(self):
        case = make_case()
        records = ready_ledger(case)

        with self.assertRaisesRegex(ReviewRecordError, "duplicate review_record_id"):
            validate_case_review_records([case], [*records, records[0]])
        outside_record = records[0] | {
            "review_record_id": "review-outside-case",
            "case_id": "holdout-case",
        }
        with self.assertRaisesRegex(ReviewRecordError, "outside the supplied single-split"):
            validate_case_review_records([case], [*records, outside_record])

    def test_jsonl_reader_rejects_nonstandard_constants_and_nonobject_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "reviews.jsonl"
            path.write_text('{"review_record_id":"x","value":NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(ReviewRecordError, "non-standard JSON constant"):
                read_review_records(path)
            path.write_text('{"review_record_id":"first","review_record_id":"second"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ReviewRecordError, "duplicate JSON object key"):
                read_review_records(path)
            path.write_text('["not", "an", "object"]\n', encoding="utf-8")
            with self.assertRaisesRegex(ReviewRecordError, "must be an object"):
                read_review_records(path)


if __name__ == "__main__":
    unittest.main()
