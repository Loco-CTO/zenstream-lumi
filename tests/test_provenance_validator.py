import copy
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.scorer import score_files
from provenance.validate import (
    ProvenanceError,
    evaluation_case_sha256,
    validate_bundle,
    validate_evaluation_case_provenance,
)
from review_fixtures import make_review_records


HASH = "sha256:" + "a" * 64
GENERATION_HASH = "sha256:" + "b" * 64


def make_case():
    return {
        "schema_version": 2,
        "case_id": "dev-001",
        "family_id": "family-001",
        "split": "development",
        "review_status": "ready",
        "language": "en",
        "categories": ["simple_action"],
        "turns": [{"role": "user", "text": "play something"}],
        "trusted_context": None,
        "gold": {
            "decision": "act",
            "action": "play",
            "arguments": {"title": "Example"},
            "requires_clarification": False,
        },
        "provenance_record_id": "record-001",
        "review": {
            "annotation_status": "approved",
            "language_review_status": "not_required",
            "reviewer_ids": ["annotator-001", "annotator-002"],
            "review_record_ids": [],
        },
    }


def make_source_manifest():
    return {
        "manifest_version": 1,
        "status": "approved_sources_present",
        "sources": [
            {
                "source_id": "synthetic-source",
                "source_type": "synthetic",
                "name": "Lumi evaluation generation",
                "canonical_identifier": "urn:lumi:test-generation",
                "authors_or_organization": "Test fixture only",
                "version": "1",
                "revision": "r1",
                "release_date": None,
                "access_date": "2026-10-02",
                "license": {
                    "identifier": "Test fixture terms",
                    "url": "https://example.invalid/terms",
                    "copyright_notice": "No external text; software fixture only",
                    "attribution_requirement": "Test fixture only",
                    "terms_review": "Fictional permission values for validator tests; not a rights determination",
                },
                "permissions": {
                    "commercial_use": "permitted",
                    "redistribution": "permitted",
                    "modification": "permitted",
                    "derivative_works": "permitted",
                    "training_use": "prohibited",
                    "evaluation_use": "permitted",
                },
                "lumi_uses": ["evaluation"],
                "decision": "approved_for_use",
                "direct_or_indirect": "direct",
                "transformations": [],
                "filters": [],
                "deduplication": [],
                "document_count": 1,
                "example_count": 1,
                "token_count": 12,
                "checksum": HASH,
                "review": {
                    "reviewed_by": "reviewer-001",
                    "reviewed_date": "2026-10-02",
                    "rationale": "Test fixture only; does not represent an external provider or licensed content",
                },
                "generation_record_id": "generation-001",
            }
        ],
    }


def make_generation_manifest():
    return {
        "manifest_version": 1,
        "status": "records_present",
        "generations": [
            {
                "generation_id": "generation-001",
                "provider": "Test fixture only",
                "model_name": "Fixture model",
                "model_version": "test-version",
                "generation_date": "2026-10-02",
                "purpose": "evaluation",
                "prompt_family": "test-fixture",
                "prompt_template_id": "test-template-v1",
                "prompt_template_sha256": GENERATION_HASH,
                "settings": {},
                "example_count": 1,
                "filtered": True,
                "filter_details": "Fixture only",
                "deduplicated": True,
                "deduplication_details": "Fixture only",
                "automatically_validated": True,
                "automatic_validation_details": "Fixture only",
                "manually_reviewed": True,
                "manual_review_details": "Fixture only",
                "downstream_uses": ["evaluation"],
                "provider_terms_url": "https://example.invalid/terms",
                "output_checksum": HASH,
                "review_notes": "Fixture generation metadata for tests",
            }
        ],
    }


def make_record(case=None):
    case = case or make_case()
    split = {"development": "development", "final_holdout": "sealed_holdout"}[case["split"]]
    return {
        "schema_version": 1,
        "record_id": case["provenance_record_id"],
        "sample_kind": "synthetic",
        "sample_sha256": evaluation_case_sha256(case),
        "language_tags": [case["language"]],
        "lumi_uses": ["evaluation"],
        "decision": "approved_for_use",
        "split": split,
        "contamination_status": "checked_clear",
        "privacy_status": "cleared",
        "source_items": [
            {
                "source_id": "synthetic-source",
                "item_id": case["case_id"],
                "source_uri": None,
                "source_revision": "r1",
                "source_sha256": HASH,
                "rights_basis": "license",
                "license_identifier": "Test fixture terms",
                "license_uri": "https://example.invalid/terms",
                "rights_evidence_uris": ["https://example.invalid/terms"],
                "rights_review_status": "reviewed",
                "rights_reviewed_date": "2026-10-02",
                "permissions": {
                    "training_use": "prohibited",
                    "evaluation_use": "permitted",
                    "commercial_use": "permitted",
                    "modification": "permitted",
                    "redistribution": "permitted",
                },
                "attribution": "Test fixture only; no external content",
            }
        ],
        "transformations": [],
        "parent_record_ids": [],
        "generator": {
            "model_or_tool": "Fixture model",
            "version": "test-version",
            "prompt_template_id": "test-template-v1",
            "prompt_template_sha256": GENERATION_HASH,
        },
        "review": {
            "reviewed_by": "reviewer-001",
            "reviewed_date": "2026-10-02",
            "rationale": "Test fixture only; does not represent a human language review",
        },
    }


class ProvenanceValidatorTests(unittest.TestCase):
    def test_approved_synthetic_evaluation_bundle_passes(self):
        case = make_case()
        case["trusted_context"] = {
            "schema_version": 1,
            "items": [{
                "kind": "watch_history",
                "source": "account.watch_history",
                "status": "success",
                "payload": {"last_played_title": "Example"},
            }],
        }
        report = validate_bundle(
            make_source_manifest(),
            make_generation_manifest(),
            [make_record(case)],
            [case],
        )

        self.assertEqual(report, {
            "status": "valid",
            "source_count": 1,
            "generation_count": 1,
            "sample_record_count": 1,
            "evaluation_case_count": 1,
        })

    def test_context_fixture_bytes_are_bound_to_case_provenance_hash(self):
        case = make_case()
        record = make_record(case)
        case["trusted_context"] = {
            "schema_version": 1,
            "items": [{
                "kind": "playback_state",
                "source": "playback.current_item",
                "status": "success",
                "payload": {"title": "Changed after annotation"},
            }],
        }

        with self.assertRaisesRegex(ProvenanceError, "content hash differs"):
            validate_bundle(make_source_manifest(), make_generation_manifest(), [record], [case])

    def test_evaluation_provenance_requires_v2_case_and_explicit_context(self):
        for change, expected_error in (
            (lambda case: case.update(schema_version=1), "schema_version must be 2"),
            (lambda case: case.pop("trusted_context"), "lacks trusted_context"),
        ):
            with self.subTest(expected_error=expected_error):
                case = make_case()
                change(case)
                record = make_record(case)

                with self.assertRaisesRegex(ProvenanceError, expected_error):
                    validate_bundle(make_source_manifest(), make_generation_manifest(), [record], [case])

    def test_file_scorer_checks_manifests_and_includes_their_hashes(self):
        case = make_case()
        prediction = {
            "schema_version": 1,
            "case_id": case["case_id"],
            "raw_output": json.dumps(case["gold"], ensure_ascii=False),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            cases_path = folder / "cases.jsonl"
            predictions_path = folder / "predictions.jsonl"
            records_path = folder / "records.jsonl"
            review_records_path = folder / "review-records.jsonl"
            sources_path = folder / "sources.json"
            generations_path = folder / "generations.json"
            review_records = make_review_records(case)
            cases_path.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
            predictions_path.write_text(json.dumps(prediction, ensure_ascii=False) + "\n", encoding="utf-8")
            records_path.write_text(json.dumps(make_record(case), ensure_ascii=False) + "\n", encoding="utf-8")
            review_records_path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in review_records),
                encoding="utf-8",
            )
            sources_path.write_text(json.dumps(make_source_manifest()), encoding="utf-8")
            generations_path.write_text(json.dumps(make_generation_manifest()), encoding="utf-8")

            report = score_files(
                cases_path,
                predictions_path,
                records_path,
                review_records_path=review_records_path,
                split="development",
                allow_final_holdout=False,
                source_manifest_path=sources_path,
                generation_manifest_path=generations_path,
            )

        self.assertEqual(report["metrics"]["overall_semantic_exact_match"]["rate"], 1.0)
        self.assertTrue(report["provenance_records_sha256"].startswith("sha256:"))
        self.assertTrue(report["review_records_sha256"].startswith("sha256:"))
        self.assertTrue(report["source_manifest_sha256"].startswith("sha256:"))
        self.assertTrue(report["synthetic_manifest_sha256"].startswith("sha256:"))

    def test_unpermitted_evaluation_use_is_rejected_at_both_rights_levels(self):
        sources = make_source_manifest()
        records = [make_record()]
        records[0]["source_items"][0]["permissions"]["evaluation_use"] = "unknown"

        with self.assertRaisesRegex(ProvenanceError, "does not permit evaluation"):
            validate_bundle(sources, make_generation_manifest(), records)

        sources["sources"][0]["permissions"]["evaluation_use"] = "conditional"
        with self.assertRaisesRegex(ProvenanceError, "source manifest does not permit evaluation"):
            validate_bundle(sources, make_generation_manifest(), [make_record()])

    def test_evaluation_case_content_must_match_record_hash(self):
        case = make_case()
        record = make_record(case)
        changed = copy.deepcopy(case)
        changed["turns"][0]["text"] = "play an example"

        with self.assertRaisesRegex(ProvenanceError, "content hash differs"):
            validate_evaluation_case_provenance([changed], [record])

    def test_synthetic_record_must_match_prompt_and_model_provenance(self):
        record = make_record()
        record["generator"]["prompt_template_sha256"] = HASH

        with self.assertRaisesRegex(ProvenanceError, "generator metadata differs"):
            validate_bundle(make_source_manifest(), make_generation_manifest(), [record])

    def test_unresolved_sample_source_is_rejected(self):
        record = make_record()
        record["source_items"][0]["source_id"] = "missing-source"

        with self.assertRaisesRegex(ProvenanceError, "unknown source_id"):
            validate_bundle(make_source_manifest(), make_generation_manifest(), [record])

    def test_source_item_must_match_schema_shape(self):
        record = make_record()
        record["source_items"][0]["unreviewed_field"] = "unexpected"

        with self.assertRaisesRegex(ProvenanceError, "sample source-item fields"):
            validate_bundle(make_source_manifest(), make_generation_manifest(), [record])

    def test_malformed_nested_enums_return_provenance_errors(self):
        record = make_record()
        record["source_items"][0]["rights_basis"] = []
        with self.assertRaisesRegex(ProvenanceError, "rights_basis"):
            validate_bundle(make_source_manifest(), make_generation_manifest(), [record])

        sources = make_source_manifest()
        sources["sources"][0]["direct_or_indirect"] = []
        with self.assertRaisesRegex(ProvenanceError, "direct_or_indirect"):
            validate_bundle(sources, make_generation_manifest(), [make_record()])

    def test_evaluation_sample_cannot_inherit_training_split(self):
        case = make_case()
        record = make_record(case)
        record["split"] = "train"

        with self.assertRaisesRegex(ProvenanceError, "evaluation use needs"):
            validate_bundle(make_source_manifest(), make_generation_manifest(), [record])

    def test_unassigned_ancestor_cannot_feed_both_train_and_evaluation(self):
        sources = make_source_manifest()
        generation = make_generation_manifest()
        source = sources["sources"][0]
        source["lumi_uses"] = ["evaluation", "pretraining", "other"]
        source["permissions"]["training_use"] = "permitted"
        generation["generations"][0]["downstream_uses"] = ["evaluation", "pretraining", "other"]

        evaluation_record = make_record()
        evaluation_record["parent_record_ids"] = ["shared-unassigned"]

        training_record = copy.deepcopy(evaluation_record)
        training_record["record_id"] = "training-derived"
        training_record["split"] = "train"
        training_record["lumi_uses"] = ["pretraining"]
        training_record["source_items"][0]["item_id"] = "training-derived"
        training_record["source_items"][0]["permissions"]["training_use"] = "permitted"

        shared_parent = copy.deepcopy(evaluation_record)
        shared_parent["record_id"] = "shared-unassigned"
        shared_parent["split"] = "unassigned"
        shared_parent["lumi_uses"] = ["other"]
        shared_parent["decision"] = "review_pending"
        shared_parent["contamination_status"] = "not_checked"
        shared_parent["privacy_status"] = "not_reviewed"
        shared_parent["source_items"][0]["item_id"] = "shared-unassigned"
        shared_parent["parent_record_ids"] = []

        with self.assertRaisesRegex(ProvenanceError, "descendants cross data splits"):
            validate_bundle(sources, generation, [shared_parent, training_record, evaluation_record])

    def test_malformed_manifest_enum_is_reported_without_type_error(self):
        sources = make_source_manifest()
        sources["sources"][0]["source_type"] = []

        with self.assertRaisesRegex(ProvenanceError, "invalid source_type"):
            validate_bundle(sources, make_generation_manifest(), [])


if __name__ == "__main__":
    unittest.main()
