from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lumi
from lumi.model_installation import (
    INSTALL_PROGRESS_TOTAL,
    MODEL_MANIFEST_FILENAME,
    InstalledModelArtifact,
    ModelInstallationError,
    Qwen35ModelInstaller,
    Qwen35ModelSpec,
    UnsupportedModelError,
    _SourceFetchResult,
    _SourceFileRecord,
    supported_models,
)

_MODEL_ID = "qwen3.5:0.8b"
_SOURCE_FILES = ("config.json", "tokenizer_config.json", "model.safetensors")


def _git_blob_id(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _test_spec(weight_data: bytes = b"test checkpoint bytes") -> Qwen35ModelSpec:
    return Qwen35ModelSpec(
        model_id=_MODEL_ID,
        directory_name="qwen3.5-0.8b",
        label="Qwen3.5 0.8B",
        repository_id="Qwen/Qwen3.5-0.8B",
        revision="a" * 40,
        architecture="Qwen3_5ForConditionalGeneration",
        model_type="qwen3_5_text",
        source_files=_SOURCE_FILES,
        weight_sha256=(("model.safetensors", hashlib.sha256(weight_data).hexdigest()),),
        max_source_bytes=1024 * 1024,
    )


class FakeSourceFetcher:
    def __init__(self, *, tamper_record: bool = False) -> None:
        self.called = False
        self.tamper_record = tamper_record

    def fetch(self, spec: Qwen35ModelSpec, destination: Path, progress):
        self.called = True
        files = {
            "config.json": json.dumps(
                {
                    "model_type": "qwen3_5",
                    "architectures": ["Qwen3_5ForConditionalGeneration"],
                }
            ).encode(),
            "tokenizer_config.json": b'{"chat_template":"template"}',
            "model.safetensors": b"test checkpoint bytes",
        }
        records = []
        for relative_name in spec.source_files:
            data = files[relative_name]
            path = destination / relative_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            is_weight = relative_name == "model.safetensors"
            records.append(
                _SourceFileRecord(
                    path=relative_name,
                    size_bytes=len(data),
                    sha256="0" * 64 if self.tamper_record and is_weight else digest,
                    blob_id=None if is_weight else _git_blob_id(data),
                    lfs_sha256=digest if is_weight else None,
                )
            )
        progress(4_500)
        return _SourceFetchResult(spec.revision, tuple(records))


class FakeConverter:
    def __init__(self, *, create_symlink: Path | None = None) -> None:
        self.called = False
        self.create_symlink = create_symlink

    def convert(self, spec: Qwen35ModelSpec, source_dir: Path, output_dir: Path, cache_dir: Path):
        self.called = True
        (output_dir / "genai_config.json").write_text(
            json.dumps({"model": {"type": "qwen3_5_text"}}), encoding="utf-8"
        )
        (output_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        graph_dir = output_dir / "decoder"
        graph_dir.mkdir()
        (graph_dir / "model.onnx").write_bytes(b"fake onnx graph")
        if self.create_symlink is not None:
            (output_dir / "escape.txt").symlink_to(self.create_symlink)


class ModelInstallationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "managed-models"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_supported_catalog_is_fixed_and_rejects_unlisted_ids_before_fetch(self) -> None:
        self.assertEqual(lumi.LUMI_PLUGIN_API_VERSION, 1)
        options = supported_models()
        ids = {item.model_id for item in options}
        self.assertEqual(ids, {"qwen3.5:0.8b", "qwen3.5:2b", "qwen3.5:4b"})
        self.assertEqual(
            {item.model_id: item.label for item in options},
            {
                "qwen3.5:0.8b": "Qwen3.5 0.8B",
                "qwen3.5:2b": "Qwen3.5 2B",
                "qwen3.5:4b": "Qwen3.5 4B",
            },
        )
        self.assertTrue(all(item.supports_thinking for item in options))
        from lumi.runtime import DEFAULT_QWEN35_MODELS, SUPPORTED_QWEN35_MODELS

        self.assertEqual(DEFAULT_QWEN35_MODELS, tuple(item.model_id for item in options))
        self.assertEqual(SUPPORTED_QWEN35_MODELS, ids)

        fetcher = FakeSourceFetcher()
        installer = Qwen35ModelInstaller(
            self.root,
            source_fetcher=fetcher,
            converter=FakeConverter(),
        )
        with self.assertRaises(UnsupportedModelError):
            installer.install_model("qwen3.5:9b")
        with self.assertRaises(UnsupportedModelError):
            installer.remove_model("../outside")
        self.assertFalse(fetcher.called)

    def test_install_writes_runtime_compatible_digest_manifest_and_bounded_progress(self) -> None:
        spec = _test_spec()
        fetcher = FakeSourceFetcher()
        converter = FakeConverter()
        events = []
        with patch("lumi.model_installation._MODEL_SPECS", {_MODEL_ID: spec}):
            installer = Qwen35ModelInstaller(self.root, source_fetcher=fetcher, converter=converter)
            artifact = installer.install_model(_MODEL_ID, progress=events.append)
            self.assertIsInstance(artifact, InstalledModelArtifact)
            self.assertEqual(artifact.model_id, _MODEL_ID)
            self.assertEqual(installer.list_models()[0].label, "Qwen3.5 0.8B")
            self.assertTrue(Path(artifact.directory).is_relative_to(self.root.resolve()))
            manifest_path = Path(artifact.directory) / MODEL_MANIFEST_FILENAME
            manifest_bytes = manifest_path.read_bytes()
            self.assertEqual(hashlib.sha256(manifest_bytes).hexdigest(), artifact.manifest_sha256)
            manifest = json.loads(manifest_bytes)
            self.assertEqual(manifest["format"], "onnxruntime-genai")
            self.assertEqual(manifest["schemaVersion"], 1)
            self.assertEqual(manifest["modelId"], _MODEL_ID)
            self.assertEqual(manifest["modelType"], "qwen3_5_text")
            self.assertIn("genai_config.json", {item["path"] for item in manifest["files"]})
            self.assertTrue(all(item.total == INSTALL_PROGRESS_TOTAL for item in events))
            values = [item.current for item in events]
            self.assertEqual(values, sorted(set(values)))
            self.assertEqual(values[-1], INSTALL_PROGRESS_TOTAL)
            self.assertTrue(fetcher.called)
            self.assertTrue(converter.called)
            self.assertFalse((self.root / ".lumi-staging").exists())
            self.assertTrue(installer.list_models()[0].installed)

    def test_installed_artifact_adapts_runtime_verification_identity(self) -> None:
        from lumi.runtime import VerifiedModelArtifact

        artifact = InstalledModelArtifact(
            model_id=_MODEL_ID,
            directory="C:/models/qwen3.5-0.8b",
            manifest_sha256="a" * 64,
            size_bytes=123,
        )
        runtime_artifact = artifact.as_verified_model_artifact()

        self.assertIsInstance(runtime_artifact, VerifiedModelArtifact)
        self.assertEqual(runtime_artifact.model_id, artifact.model_id)
        self.assertEqual(runtime_artifact.directory, Path(artifact.directory).resolve())
        self.assertEqual(runtime_artifact.manifest_sha256, artifact.manifest_sha256)

    def test_failed_source_hash_check_leaves_no_install_or_staging_files(self) -> None:
        spec = _test_spec()
        fetcher = FakeSourceFetcher(tamper_record=True)
        converter = FakeConverter()
        with patch("lumi.model_installation._MODEL_SPECS", {_MODEL_ID: spec}):
            installer = Qwen35ModelInstaller(self.root, source_fetcher=fetcher, converter=converter)
            with self.assertRaises(ModelInstallationError):
                installer.install_model(_MODEL_ID)
            self.assertFalse((self.root / spec.directory_name).exists())
            self.assertFalse(converter.called)
            self.assertFalse((self.root / ".lumi-staging").exists())

    def test_converter_output_links_are_rejected_and_cleaned(self) -> None:
        spec = _test_spec()
        fetcher = FakeSourceFetcher()
        outside = Path(self.temp_dir.name) / "outside.txt"
        outside.write_text("keep", encoding="utf-8")
        link_probe = Path(self.temp_dir.name) / "link-probe"
        try:
            link_probe.symlink_to(outside)
        except OSError:
            self.skipTest("This Windows environment does not permit symlinks")
        link_probe.unlink()
        converter = FakeConverter(create_symlink=outside)
        with patch("lumi.model_installation._MODEL_SPECS", {_MODEL_ID: spec}):
            installer = Qwen35ModelInstaller(self.root, source_fetcher=fetcher, converter=converter)
            with self.assertRaises(ModelInstallationError):
                installer.install_model(_MODEL_ID)
            self.assertEqual(outside.read_text(encoding="utf-8"), "keep")
            self.assertFalse((self.root / spec.directory_name).exists())

    def test_remove_deletes_only_verified_supported_model(self) -> None:
        spec = _test_spec()
        with patch("lumi.model_installation._MODEL_SPECS", {_MODEL_ID: spec}):
            installer = Qwen35ModelInstaller(
                self.root,
                source_fetcher=FakeSourceFetcher(),
                converter=FakeConverter(),
            )
            artifact = installer.install_model(_MODEL_ID)
            self.assertTrue(installer.remove_model(_MODEL_ID))
            self.assertFalse(Path(artifact.directory).exists())
            self.assertFalse(installer.remove_model(_MODEL_ID))

    def test_listing_rehashes_installed_files(self) -> None:
        spec = _test_spec()
        with patch("lumi.model_installation._MODEL_SPECS", {_MODEL_ID: spec}):
            installer = Qwen35ModelInstaller(
                self.root,
                source_fetcher=FakeSourceFetcher(),
                converter=FakeConverter(),
            )
            artifact = installer.install_model(_MODEL_ID)
            model_file = Path(artifact.directory) / "decoder" / "model.onnx"
            model_file.write_bytes(b"FAKE onnx graph")

            option = installer.list_models()[0]

            self.assertFalse(option.installed)

    def test_listing_rejects_manifest_with_wrong_source_pin(self) -> None:
        spec = _test_spec()
        with patch("lumi.model_installation._MODEL_SPECS", {_MODEL_ID: spec}):
            installer = Qwen35ModelInstaller(
                self.root,
                source_fetcher=FakeSourceFetcher(),
                converter=FakeConverter(),
            )
            artifact = installer.install_model(_MODEL_ID)
            manifest_path = Path(artifact.directory) / MODEL_MANIFEST_FILENAME
            manifest = json.loads(manifest_path.read_bytes())
            manifest["source"]["repositoryId"] = "attacker/model"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )

            option = installer.list_models()[0]

            self.assertFalse(option.installed)


if __name__ == "__main__":
    unittest.main()
