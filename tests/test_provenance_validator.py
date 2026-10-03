import copy
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.scorer import score_files
from provenance.validate import (
    AUTHOR_CONSENTS_DEFAULT,
    ProvenanceError,
    evaluation_case_sha256,
    validate_bundle,
    validate_evaluation_case_provenance,
    validate_files,
)
from review_fixtures import make_review_records


HASH = "sha256:" + "a" * 64
GENERATION_HASH = "sha256:" + "b" * 64


def make_case():
    return {
        "schema_version": 4,
        "case_id": "dev-001",
        "family_id": "family-001",
        "split": "development",
        "review_status": "ready",
        "language": "en",
        "categories": ["simple_action"],
        "turns": [{"role": "user", "text": "play something"}],
        "trusted_context": None,
        "tool_scenario": None,
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


def make_manual_source_manifest():
    manifest = make_source_manifest()
    source = manifest["sources"][0]
    source["source_id"] = "manual-source"
    source["source_type"] = "manually_authored"
    source["name"] = "Manually authored fixture"
    source["canonical_identifier"] = "urn:lumi:manual-fixture"
    source["generation_record_id"] = None
    return manifest


def make_manual_record(case=None, *, uses=("evaluation",), decision="approved_for_use"):
    case = case or make_case()
    record = make_record(case)
    record["sample_kind"] = "manually_authored"
    record.pop("generator")
    record["author_consent_record_id"] = "consent-001"
    record["lumi_uses"] = list(uses)
    record["decision"] = decision
    if "pretraining" in uses:
        record["split"] = "train"
        record["source_items"][0]["permissions"]["training_use"] = "permitted"
    item = record["source_items"][0]
    item["source_id"] = "manual-source"
    item["item_id"] = "authored-item-001"
    return record


def make_author_consent(record=None, *, granted_uses=("evaluation",), distribution="unknown", status="approved"):
    record = record or make_manual_record()
    consent = {
        "consent_record_id": "consent-001",
        "author_pseudonym": "author-001",
        "source_id": "manual-source",
        "item_ids": ["authored-item-001"],
        "sample_sha256": record["sample_sha256"],
        "granted_uses": list(granted_uses),
        "model_artifact_distribution": distribution,
        "status": status,
        "agreement_sha256": HASH if status == "approved" else None,
        "granted_date": "2026-10-02" if status == "approved" else None,
        "expires_date": None,
        "review": {
            "reviewed_by": "reviewer-001",
            "reviewed_date": "2026-10-02" if status == "approved" else None,
            "rationale": "Test fixture only; not a real consent receipt",
        },
    }
    return {"manifest_version": 1, "status": "approved_grants_present" if status == "approved" else "reviewing", "consents": [consent]}


class ProvenanceValidatorTests(unittest.TestCase):
    def test_manually_authored_sample_requires_exact_approved_consent_join(self):
        record = make_manual_record()
        report = validate_bundle(
            make_manual_source_manifest(),
            make_generation_manifest(),
            [record],
            author_consent_manifest=make_author_consent(record),
        )
        self.assertEqual(report["status"], "valid")

    def test_fabricated_or_missing_author_consent_id_is_rejected(self):
        record = make_manual_record()
        with self.assertRaisesRegex(ProvenanceError, "unknown author consent"):
            validate_bundle(make_manual_source_manifest(), make_generation_manifest(), [record])

        consent = make_author_consent(record)
        consent["consents"][0]["consent_record_id"] = "another-id"
        with self.assertRaisesRegex(ProvenanceError, "unknown author consent"):
            validate_bundle(
                make_manual_source_manifest(), make_generation_manifest(), [record],
                author_consent_manifest=consent,
            )

    def test_author_consent_is_bound_to_source_item_and_exact_sample_hash(self):
        for field, value, expected in (
            ("source_id", "different-source", "consent source does not match"),
            ("item_ids", ["different-item"], "outside the author's consent scope"),
            ("sample_sha256", "sha256:" + "c" * 64, "exact sample covered by author consent"),
        ):
            with self.subTest(field=field):
                record = make_manual_record()
                consent = make_author_consent(record)
                consent["consents"][0][field] = value
                with self.assertRaisesRegex(ProvenanceError, expected):
                    validate_bundle(
                        make_manual_source_manifest(), make_generation_manifest(), [record],
                        author_consent_manifest=consent,
                    )

    def test_pending_consent_and_ungranted_use_cannot_approve_manual_sample(self):
        record = make_manual_record()
        pending = make_author_consent(record, status="pending")
        with self.assertRaisesRegex(ProvenanceError, "consent is not approved"):
            validate_bundle(
                make_manual_source_manifest(), make_generation_manifest(), [record],
                author_consent_manifest=pending,
            )

        consent = make_author_consent(record, granted_uses=("other",))
        with self.assertRaisesRegex(ProvenanceError, "exceeds the author's granted uses"):
            validate_bundle(
                make_manual_source_manifest(), make_generation_manifest(), [record],
                author_consent_manifest=consent,
            )

    def test_manual_training_requires_explicit_model_artifact_distribution_permission(self):
        record = make_manual_record(uses=("pretraining",))
        source_manifest = make_manual_source_manifest()
        source = source_manifest["sources"][0]
        source["lumi_uses"] = ["pretraining"]
        source["permissions"]["training_use"] = "permitted"
        consent = make_author_consent(record, granted_uses=("pretraining",), distribution="unknown")

        with self.assertRaisesRegex(ProvenanceError, "does not permit model artifact distribution"):
            validate_bundle(
                source_manifest, make_generation_manifest(), [record], author_consent_manifest=consent
            )

        consent["consents"][0]["model_artifact_distribution"] = "permitted"
        report = validate_bundle(
            source_manifest, make_generation_manifest(), [record], author_consent_manifest=consent
        )
        self.assertEqual(report["status"], "valid")

    def test_derived_sample_inherits_manual_author_use_and_distribution_limits(self):
        parent = make_manual_record(uses=("other",), decision="review_pending")
        parent["split"] = "unassigned"
        source_manifest = make_manual_source_manifest()
        source = source_manifest["sources"][0]
        source["lumi_uses"] = ["other", "pretraining"]
        source["permissions"]["training_use"] = "permitted"
        consent = make_author_consent(parent, granted_uses=("evaluation",), distribution="unknown")
        child = copy.deepcopy(parent)
        child["record_id"] = "derived-training-sample"
        child["sample_kind"] = "transformed_sample"
        child.pop("author_consent_record_id")
        child["sample_sha256"] = "sha256:" + "d" * 64
        child["lumi_uses"] = ["pretraining"]
        child["decision"] = "approved_for_use"
        child["split"] = "train"
        child["source_items"][0]["permissions"]["training_use"] = "permitted"
        child["parent_record_ids"] = [parent["record_id"]]
        child["transformations"] = [{
            "operation_id": "normalize-fixture",
            "revision": "r1",
            "config_sha256": HASH,
        }]

        with self.assertRaisesRegex(ProvenanceError, "exceeds an ancestor author's granted uses"):
            validate_bundle(
                source_manifest, make_generation_manifest(), [parent, child], author_consent_manifest=consent
            )

        consent["consents"][0]["granted_uses"] = ["pretraining"]
        with self.assertRaisesRegex(ProvenanceError, "does not permit model artifact distribution"):
            validate_bundle(
                source_manifest, make_generation_manifest(), [parent, child], author_consent_manifest=consent
            )

        consent["consents"][0]["model_artifact_distribution"] = "permitted"
        self.assertEqual(validate_bundle(
            source_manifest, make_generation_manifest(), [parent, child], author_consent_manifest=consent
        )["status"], "valid")

    def test_author_consent_manifest_rejects_duplicate_ids_and_self_review(self):
        record = make_manual_record()
        consent = make_author_consent(record)
        consent["consents"].append(copy.deepcopy(consent["consents"][0]))
        with self.assertRaisesRegex(ProvenanceError, "duplicate consent_record_id"):
            validate_bundle(
                make_manual_source_manifest(), make_generation_manifest(), [record],
                author_consent_manifest=consent,
            )

        consent = make_author_consent(record)
        consent["consents"][0]["review"]["reviewed_by"] = "author-001"
        with self.assertRaisesRegex(ProvenanceError, "someone other than the author"):
            validate_bundle(
                make_manual_source_manifest(), make_generation_manifest(), [record],
                author_consent_manifest=consent,
            )

    def test_synthetic_samples_cannot_carry_author_consent_ids(self):
        record = make_record()
        record["author_consent_record_id"] = "consent-001"
        with self.assertRaisesRegex(ProvenanceError, "only manually authored records"):
            validate_bundle(make_source_manifest(), make_generation_manifest(), [record])

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

    def test_tool_scenario_and_ordered_results_are_bound_to_case_provenance_hash(self):
        case = make_case()
        expected_call = {"tool": "catalog.search", "arguments": {"query": "Example"}}
        case["tool_scenario"] = {
            "schema_version": 1,
            "tools": [{
                "name": "catalog.search",
                "description": "Search the fixture catalog.",
                "effect": "read_only",
                "arguments_schema": {"type": "object", "required": ["query"]},
            }],
            "fixtures": [{
                **expected_call,
                "result": {"status": "success", "payload": {"items": [{"title": "Example"}]}},
            }],
            "max_model_steps": 3,
            "max_tool_calls": 2,
        }
        case["gold"]["tool_trajectory"] = {"calls": [expected_call], "completion": "final"}
        record = make_record(case)
        case["tool_scenario"]["fixtures"][0]["result"]["payload"]["items"][0]["title"] = "Changed"

        with self.assertRaisesRegex(ProvenanceError, "content hash differs"):
            validate_bundle(make_source_manifest(), make_generation_manifest(), [record], [case])

    def test_evaluation_provenance_requires_v4_case_and_context(self):
        for change, expected_error in (
            (lambda case: case.update(schema_version=1), "schema_version must be 4"),
            (lambda case: case.pop("tool_scenario"), "lacks tool_scenario"),
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
            "schema_version": 3,
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
        self.assertTrue(report["author_consent_manifest_sha256"].startswith("sha256:"))

    def test_file_gate_rejects_nonempty_consent_manifest_inside_git_repository(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".git").write_text("", encoding="utf-8")
            sources_path = root / "sources.json"
            generations_path = root / "generations.json"
            records_path = root / "records.jsonl"
            consents_path = root / "author-consents.json"
            record = make_manual_record()
            sources_path.write_text(json.dumps(make_manual_source_manifest()), encoding="utf-8")
            generations_path.write_text(json.dumps(make_generation_manifest()), encoding="utf-8")
            records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            consents_path.write_text(json.dumps(make_author_consent(record)), encoding="utf-8")

            with self.assertRaisesRegex(ProvenanceError, "must be stored outside Git repositories"):
                validate_files(sources_path, generations_path, records_path, author_consents_path=consents_path)

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
