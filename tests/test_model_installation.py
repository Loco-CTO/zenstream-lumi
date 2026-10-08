from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import lumi
from lumi.model_installation import (
    GGUF_FORMAT,
    INSTALL_PROGRESS_TOTAL,
    MODEL_MANIFEST_FILENAME,
    MODEL_MANIFEST_SCHEMA_VERSION,
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
_GGUF_FILENAME = "Qwen_Qwen3.5-0.8B-Q4_K_M.gguf"
_GGUF_BYTES = b"pinned test GGUF"


def _test_spec(gguf_data: bytes = _GGUF_BYTES) -> Qwen35ModelSpec:
    return Qwen35ModelSpec(
        model_id=_MODEL_ID,
        directory_name="qwen3.5-0.8b",
        label="Qwen3.5 0.8B",
        repository_id="bartowski/Qwen_Qwen3.5-0.8B-GGUF",
        revision="a" * 40,
        gguf_filename=_GGUF_FILENAME,
        quantization="Q4_K_M",
        gguf_sha256=hashlib.sha256(gguf_data).hexdigest(),
        download_size_bytes=len(gguf_data),
        max_download_bytes=1024 * 1024,
    )


class FakeGGUFSourceFetcher:
    def __init__(self, *, tamper_record: bool = False, add_symlink: Path | None = None) -> None:
        self.called = False
        self.tamper_record = tamper_record
        self.add_symlink = add_symlink

    def fetch(self, spec: Qwen35ModelSpec, destination: Path, progress):
        self.called = True
        path = destination / spec.gguf_filename
        path.write_bytes(_GGUF_BYTES)
        if self.add_symlink is not None:
            (destination / "escape.gguf").symlink_to(self.add_symlink)
        digest = "0" * 64 if self.tamper_record else hashlib.sha256(_GGUF_BYTES).hexdigest()
        progress(4_400)
        return _SourceFetchResult(
            spec.revision,
            (_SourceFileRecord(spec.gguf_filename, len(_GGUF_BYTES), digest),),
        )


class ModelInstallationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "managed-models"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def installer(self, spec: Qwen35ModelSpec, fetcher: FakeGGUFSourceFetcher | None = None):
        return Qwen35ModelInstaller(self.root, source_fetcher=fetcher or FakeGGUFSourceFetcher())

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
        self.assertEqual(options[0].size_bytes, 579_615_840)
        from lumi.runtime import DEFAULT_QWEN35_MODELS, SUPPORTED_QWEN35_MODELS

        self.assertEqual(DEFAULT_QWEN35_MODELS, tuple(item.model_id for item in options))
        self.assertEqual(SUPPORTED_QWEN35_MODELS, ids)

        fetcher = FakeGGUFSourceFetcher()
        installer = Qwen35ModelInstaller(self.root, source_fetcher=fetcher)
        with self.assertRaises(UnsupportedModelError):
            installer.install_model("qwen3.5:9b")
        with self.assertRaises(UnsupportedModelError):
            installer.remove_model("../outside")
        self.assertFalse(fetcher.called)

    def test_install_writes_runtime_compatible_digest_manifest_and_bounded_progress(self) -> None:
        spec = _test_spec()
        fetcher = FakeGGUFSourceFetcher()
        events = []
        with patch("lumi.model_installation._MODEL_SPECS", {_MODEL_ID: spec}):
            installer = self.installer(spec, fetcher)

            def record_progress(event):
                events.append(event)
                if event.stage != "complete":
                    self.assertFalse(installer.list_models()[0].installed)

            artifact = installer.install_model(_MODEL_ID, progress=record_progress)
            self.assertIsInstance(artifact, InstalledModelArtifact)
            self.assertEqual(artifact.model_id, _MODEL_ID)
            self.assertEqual(installer.list_models()[0].label, "Qwen3.5 0.8B")
            self.assertTrue(Path(artifact.directory).is_relative_to(self.root.resolve()))
            manifest_path = Path(artifact.directory) / MODEL_MANIFEST_FILENAME
            manifest_bytes = manifest_path.read_bytes()
            self.assertEqual(hashlib.sha256(manifest_bytes).hexdigest(), artifact.manifest_sha256)
            manifest = json.loads(manifest_bytes)
            self.assertEqual(manifest["format"], GGUF_FORMAT)
            self.assertEqual(manifest["schemaVersion"], MODEL_MANIFEST_SCHEMA_VERSION)
            self.assertEqual(manifest["modelId"], _MODEL_ID)
            self.assertEqual(manifest["quantization"], "Q4_K_M")
            self.assertEqual(manifest["source"]["repositoryId"], spec.repository_id)
            self.assertEqual(manifest["source"]["revision"], spec.revision)
            self.assertEqual(manifest["source"]["sha256"], spec.gguf_sha256)
            self.assertEqual({item["path"] for item in manifest["files"]}, {_GGUF_FILENAME})
            self.assertEqual((Path(artifact.directory) / _GGUF_FILENAME).read_bytes(), _GGUF_BYTES)
            self.assertTrue(all(item.total == INSTALL_PROGRESS_TOTAL for item in events))
            values = [item.current for item in events]
            self.assertEqual(values, sorted(set(values)))
            self.assertEqual(values[-1], INSTALL_PROGRESS_TOTAL)
            self.assertTrue(fetcher.called)
            self.assertFalse((self.root / ".lumi-staging").exists())
            self.assertTrue(installer.list_models()[0].installed)

    def test_install_does_not_require_a_directory_rename(self) -> None:
        spec = _test_spec()
        original_replace = os.replace

        def reject_directory_rename(source, destination):
            if Path(source).is_dir():
                raise PermissionError(5, "directory activation is unavailable")
            return original_replace(source, destination)

        with (
            patch("lumi.model_installation._MODEL_SPECS", {_MODEL_ID: spec}),
            patch(
                "lumi.model_installation.os.replace",
                side_effect=reject_directory_rename,
            ),
        ):
            installer = self.installer(spec)
            artifact = installer.install_model(_MODEL_ID)
            self.assertTrue(Path(artifact.directory).is_dir())
            self.assertTrue(installer.list_models()[0].installed)

    def test_install_recovers_a_stale_legacy_model_directory(self) -> None:
        spec = _test_spec()
        model_directory = self.root / spec.directory_name
        model_directory.mkdir(parents=True)
        (model_directory / "model.onnx").write_bytes(b"legacy ONNX model")

        with patch("lumi.model_installation._MODEL_SPECS", {_MODEL_ID: spec}):
            installer = self.installer(spec)
            artifact = installer.install_model(_MODEL_ID)
            self.assertTrue(installer.list_models()[0].installed)
            self.assertFalse((model_directory / "model.onnx").exists())
            self.assertTrue((Path(artifact.directory) / _GGUF_FILENAME).is_file())

    def test_huggingface_download_cache_survives_retry_and_clears_after_success(self) -> None:
        spec = _test_spec()
        hub = ModuleType("huggingface_hub")
        calls: list[Path] = []

        class FakeHubApi:
            def __init__(self, **_kwargs):
                pass

            def model_info(self, **_kwargs):
                sibling = SimpleNamespace(
                    rfilename=spec.gguf_filename,
                    size=spec.download_size_bytes,
                    lfs=SimpleNamespace(
                        size=spec.download_size_bytes,
                        sha256=spec.gguf_sha256,
                    ),
                )
                return SimpleNamespace(sha=spec.revision, siblings=[sibling])

        def fake_hf_hub_download(**kwargs):
            local_directory = Path(kwargs["local_dir"])
            calls.append(local_directory)
            partial = (
                local_directory / ".cache" / "huggingface" / "download" / "partial.gguf.incomplete"
            )
            partial.parent.mkdir(parents=True, exist_ok=True)
            if len(calls) == 1:
                partial.write_bytes(b"partial transfer")
                raise OSError("simulated interrupted transfer")
            if not partial.is_file():
                raise AssertionError("the retry did not preserve its partial transfer")
            target = local_directory / spec.gguf_filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_GGUF_BYTES)
            returned_path = str(target)
            if os.name == "nt":
                drive, tail = os.path.splitdrive(returned_path)
                returned_path = f"\\\\?\\{drive}{tail}"
            return returned_path

        hub.HfApi = FakeHubApi
        hub.hf_hub_download = fake_hf_hub_download
        tqdm_package = ModuleType("tqdm")
        tqdm_auto = ModuleType("tqdm.auto")
        tqdm_auto.tqdm = type("Tqdm", (), {})
        with (
            patch.dict(
                sys.modules,
                {
                    "huggingface_hub": hub,
                    "tqdm": tqdm_package,
                    "tqdm.auto": tqdm_auto,
                },
            ),
            patch("lumi.model_installation._MODEL_SPECS", {_MODEL_ID: spec}),
        ):
            installer = Qwen35ModelInstaller(self.root)
            with self.assertRaises(ModelInstallationError):
                installer.install_model(_MODEL_ID)
            cache_root = self.root / ".lumi-hub-cache"
            cache_directory = cache_root / spec.directory_name
            partial = (
                cache_directory / ".cache" / "huggingface" / "download" / "partial.gguf.incomplete"
            )
            self.assertTrue(partial.is_file())

            artifact = installer.install_model(_MODEL_ID)

        resolved_cache_directory = cache_directory.resolve()
        self.assertEqual(calls, [resolved_cache_directory, resolved_cache_directory])
        self.assertTrue(Path(artifact.directory).is_dir())
        self.assertFalse(cache_directory.exists())
        self.assertFalse(cache_root.exists())

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
        fetcher = FakeGGUFSourceFetcher(tamper_record=True)
        with patch("lumi.model_installation._MODEL_SPECS", {_MODEL_ID: spec}):
            installer = self.installer(spec, fetcher)
            with self.assertRaises(ModelInstallationError):
                installer.install_model(_MODEL_ID)
            self.assertFalse((self.root / spec.directory_name).exists())
            self.assertFalse((self.root / ".lumi-staging").exists())

    def test_download_links_are_rejected_and_cleaned(self) -> None:
        spec = _test_spec()
        outside = Path(self.temp_dir.name) / "outside.gguf"
        outside.write_bytes(b"keep")
        link_probe = Path(self.temp_dir.name) / "link-probe"
        try:
            link_probe.symlink_to(outside)
        except OSError:
            self.skipTest("This Windows environment does not permit symlinks")
        link_probe.unlink()
        with patch("lumi.model_installation._MODEL_SPECS", {_MODEL_ID: spec}):
            installer = self.installer(spec, FakeGGUFSourceFetcher(add_symlink=outside))
            with self.assertRaises(ModelInstallationError):
                installer.install_model(_MODEL_ID)
            self.assertEqual(outside.read_bytes(), b"keep")
            self.assertFalse((self.root / spec.directory_name).exists())

    def test_remove_deletes_only_verified_supported_model(self) -> None:
        spec = _test_spec()
        with patch("lumi.model_installation._MODEL_SPECS", {_MODEL_ID: spec}):
            installer = self.installer(spec)
            artifact = installer.install_model(_MODEL_ID)
            self.assertTrue(installer.remove_model(_MODEL_ID))
            self.assertFalse(Path(artifact.directory).exists())
            self.assertFalse(installer.remove_model(_MODEL_ID))

    def test_remove_recovers_corrupt_model_files(self) -> None:
        spec = _test_spec()
        with patch("lumi.model_installation._MODEL_SPECS", {_MODEL_ID: spec}):
            installer = self.installer(spec)
            artifact = installer.install_model(_MODEL_ID)
            model_directory = Path(artifact.directory)
            (model_directory / _GGUF_FILENAME).write_bytes(b"incomplete file")

            self.assertFalse(installer.list_models()[0].installed)
            self.assertTrue(installer.remove_model(_MODEL_ID))
            self.assertFalse(model_directory.exists())

    def test_remove_rejects_links_inside_corrupt_model_directory(self) -> None:
        spec = _test_spec()
        outside = Path(self.temp_dir.name) / "outside.gguf"
        outside.write_bytes(b"keep")
        with patch("lumi.model_installation._MODEL_SPECS", {_MODEL_ID: spec}):
            installer = self.installer(spec)
            artifact = installer.install_model(_MODEL_ID)
            model_directory = Path(artifact.directory)
            link = model_directory / "external.gguf"
            try:
                link.symlink_to(outside)
            except OSError:
                self.skipTest("This Windows environment does not permit symlinks")

            with self.assertRaises(ModelInstallationError):
                installer.remove_model(_MODEL_ID)
            self.assertTrue(model_directory.exists())
            self.assertEqual(outside.read_bytes(), b"keep")

    def test_listing_rehashes_installed_gguf(self) -> None:
        spec = _test_spec()
        with patch("lumi.model_installation._MODEL_SPECS", {_MODEL_ID: spec}):
            installer = self.installer(spec)
            artifact = installer.install_model(_MODEL_ID)
            (Path(artifact.directory) / _GGUF_FILENAME).write_bytes(b"tampered model")
            self.assertFalse(installer.list_models()[0].installed)

    def test_listing_rejects_manifest_with_wrong_source_pin(self) -> None:
        spec = _test_spec()
        with patch("lumi.model_installation._MODEL_SPECS", {_MODEL_ID: spec}):
            installer = self.installer(spec)
            artifact = installer.install_model(_MODEL_ID)
            manifest_path = Path(artifact.directory) / MODEL_MANIFEST_FILENAME
            manifest = json.loads(manifest_path.read_bytes())
            manifest["source"]["repositoryId"] = "attacker/model"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            self.assertFalse(installer.list_models()[0].installed)

    def test_old_onnx_installation_is_not_accepted(self) -> None:
        spec = _test_spec()
        model_dir = self.root / spec.directory_name
        model_dir.mkdir(parents=True)
        (model_dir / "model.onnx").write_bytes(b"old model")
        manifest = json.dumps(
            {"schemaVersion": 1, "format": "onnxruntime-genai", "modelId": _MODEL_ID},
            sort_keys=True,
            separators=(",", ":"),
        )
        (model_dir / MODEL_MANIFEST_FILENAME).write_text(manifest, encoding="utf-8")
        with patch("lumi.model_installation._MODEL_SPECS", {_MODEL_ID: spec}):
            self.assertFalse(self.installer(spec).list_models()[0].installed)


if __name__ == "__main__":
    unittest.main()
