import tempfile
import unittest
from pathlib import Path

from experiments import random_init_conversation_response_smoke as response_smoke
from experiments import random_init_intent_smoke as intent_smoke


class SmokeInputPinningTests(unittest.TestCase):
    def test_intent_smoke_rejects_replacement_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data = Path(temp_dir) / "examples.jsonl"
            data.write_text("replacement data\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "frozen intent-smoke dataset"):
                intent_smoke._require_frozen_smoke_dataset(data)

    def test_response_smoke_rejects_replacement_dataset_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data = Path(temp_dir) / "examples.jsonl"
            manifest = Path(temp_dir) / "manifest.json"
            data.write_text("replacement data\n", encoding="utf-8")
            manifest.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "source manifest"):
                response_smoke._require_frozen_hash(
                    manifest,
                    response_smoke.RESPONSE_SMOKE_MANIFEST_SHA256,
                    "source manifest",
                )
            with self.assertRaisesRegex(ValueError, "examples file"):
                response_smoke._require_frozen_hash(
                    data,
                    response_smoke.RESPONSE_SMOKE_DATA_SHA256,
                    "examples file",
                )


if __name__ == "__main__":
    unittest.main()
