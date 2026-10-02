import copy
import json
import unittest
from pathlib import Path

from evaluation.capability_registry import (
    CapabilityRegistryError,
    validate_registry,
)


REGISTRY_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "capabilities.json"


class CapabilityRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    def test_checked_in_registry_validates_and_stays_draft(self):
        report = validate_registry(self.registry)

        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["review_status"], "draft")
        self.assertIn("playback.start", report["capability_ids"])

    def test_duplicate_capability_ids_are_rejected(self):
        registry = copy.deepcopy(self.registry)
        registry["capabilities"].append(copy.deepcopy(registry["capabilities"][0]))

        with self.assertRaisesRegex(CapabilityRegistryError, "duplicate capability id"):
            validate_registry(registry)

    def test_approved_registry_requires_every_capability_approved(self):
        registry = copy.deepcopy(self.registry)
        registry["review_status"] = "approved"

        with self.assertRaisesRegex(
            CapabilityRegistryError, "approved registry cannot contain"
        ):
            validate_registry(registry)

    def test_required_arguments_need_a_declared_schema(self):
        registry = copy.deepcopy(self.registry)
        registry["capabilities"][0]["argument_schema"]["required"].append("missing")

        with self.assertRaisesRegex(
            CapabilityRegistryError, "required argument.*lack schemas"
        ):
            validate_registry(registry)

    def test_argument_schemas_reject_unregistered_fields(self):
        registry = copy.deepcopy(self.registry)
        registry["capabilities"][0]["argument_schema"]["additionalProperties"] = True

        with self.assertRaisesRegex(
            CapabilityRegistryError, "must reject additional properties"
        ):
            validate_registry(registry)


if __name__ == "__main__":
    unittest.main()
