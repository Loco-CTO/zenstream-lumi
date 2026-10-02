"""Synthetic review-ledger rows for software tests only; not benchmark annotations."""

from __future__ import annotations

from typing import Any

from provenance.validate import evaluation_case_sha256


def make_review_records(case: dict[str, Any]) -> list[dict[str, Any]]:
    review = case["review"]
    if case["review_status"] != "ready":
        review["review_record_ids"] = []
        return []

    reviewer_ids = list(review.get("reviewer_ids", []))
    case_sha256 = evaluation_case_sha256(case)
    records: list[dict[str, Any]] = []
    for index, reviewer_id in enumerate(reviewer_ids[:2], start=1):
        records.append({
            "schema_version": 1,
            "review_record_id": f"review-{case['case_id']}-annotation-{index}",
            "case_id": case["case_id"],
            "case_sha256": case_sha256,
            "reviewer_id": reviewer_id,
            "reviewed_at": "2026-10-02T12:00:00Z",
            "record_type": "independent_annotation",
            "supersedes_record_ids": [],
            "proposed_gold": case["gold"],
        })

    if case["language"] in {"ja", "en_ja"}:
        language_reviewer = reviewer_ids[2] if len(reviewer_ids) > 2 else (reviewer_ids[0] if reviewer_ids else "")
        qualification = "native_japanese" if case["language"] == "ja" else "fluent_bilingual"
        language_status = review["language_review_status"]
        approved = language_status == "approved"
        records.append({
            "schema_version": 1,
            "review_record_id": f"review-{case['case_id']}-language",
            "case_id": case["case_id"],
            "case_sha256": case_sha256,
            "reviewer_id": language_reviewer,
            "reviewed_at": "2026-10-02T12:00:00Z",
            "record_type": "language_review",
            "supersedes_record_ids": [],
            "language": case["language"],
            "qualification": qualification,
            "naturalness_status": "approved" if approved else "needs_revision",
            "meaning_preservation_status": "approved" if approved else "needs_revision",
        })

    review["review_record_ids"] = [record["review_record_id"] for record in records]
    return records


def make_review_ledger(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for case in cases for record in make_review_records(case)]
