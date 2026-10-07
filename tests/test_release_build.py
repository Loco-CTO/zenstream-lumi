from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.build_release import ReleaseBuildError, build_release


class LumiReleaseBuildTests(unittest.TestCase):
    def _wheel(
        self,
        wheelhouse: Path,
        distribution: str,
        version: str,
        python_tag: str,
        abi_tag: str,
        platform_tag: str,
    ) -> Path:
        wheelhouse.mkdir(parents=True, exist_ok=True)
        filename = f"{distribution}-{version}-{python_tag}-{abi_tag}-{platform_tag}.whl"
        path = wheelhouse / filename
        dist_info = f"{distribution.replace('-', '_')}-{version}.dist-info"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                f"{dist_info}/METADATA",
                f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n\n",
            )
            archive.writestr(
                f"{dist_info}/WHEEL",
                "Wheel-Version: 1.0\nGenerator: unit-test\nRoot-Is-Purelib: true\n"
                f"Tag: {python_tag}-{abi_tag}-{platform_tag}\n",
            )
        return path

    def _wheelhouse(self, root: Path) -> Path:
        wheelhouse = root / "wheels"
        targets = (
            ("cp312", "cp312", "win_amd64"),
            ("cp313", "cp313", "win_amd64"),
            ("cp312", "cp312", "manylinux_2_28_x86_64"),
            ("cp313", "cp313", "manylinux_2_28_x86_64"),
            ("cp312", "cp312", "manylinux_2_28_aarch64"),
            ("cp313", "cp313", "manylinux_2_28_aarch64"),
        )
        for python_tag, abi_tag, platform_tag in targets:
            runtime_subdir = wheelhouse / "runtime" / f"{python_tag}-{platform_tag}"
            installer_subdir = wheelhouse / "installer" / f"{python_tag}-{platform_tag}"
            self._wheel(
                runtime_subdir,
                "onnxruntime_genai",
                "0.17.1",
                python_tag,
                abi_tag,
                platform_tag,
            )
            self._wheel(
                runtime_subdir, "numpy", "2.2.6", python_tag, abi_tag, platform_tag
            )
            self._wheel(
                installer_subdir, "torch", "2.11.0+cpu", python_tag, abi_tag, platform_tag
            )
            self._wheel(
                installer_subdir, "numpy", "2.2.6", python_tag, abi_tag, platform_tag
            )
        runtime_universal = wheelhouse / "runtime" / "universal"
        installer_universal = wheelhouse / "installer" / "universal"
        self._wheel(
            installer_universal, "huggingface_hub", "1.10.0", "py3", "none", "any"
        )
        self._wheel(installer_universal, "onnx_ir", "0.2.1", "py3", "none", "any")
        self._wheel(
            installer_universal, "transformers", "5.2.0", "py3", "none", "any"
        )
        self._wheel(installer_universal, "filelock", "3.18.0", "py3", "none", "any")
        self._wheel(runtime_universal, "packaging", "25.0", "py3", "none", "any")
        self._wheel(installer_universal, "packaging", "25.0", "py3", "none", "any")
        return wheelhouse

    def test_builds_runtime_and_deferred_installer_manifest(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheelhouse = self._wheelhouse(root)
            output = root / "release"
            manifest = build_release(project_root, wheelhouse, output, "v0.1.0")

            self.assertEqual(manifest["schemaVersion"], 1)
            self.assertEqual(manifest["tag"], "v0.1.0")
            self.assertEqual(len(manifest["runtimeDependencies"]), 13)
            self.assertEqual(len(manifest["installerDependencies"]), 10)
            runtime_names = {
                entry["distribution"].lower().replace("_", "-")
                for entry in manifest["runtimeDependencies"]
            }
            self.assertEqual(
                runtime_names, {"numpy", "onnxruntime-genai", "packaging"}
            )
            installer_names = {
                entry["distribution"].lower().replace("_", "-")
                for entry in manifest["installerDependencies"]
            }
            self.assertEqual(
                installer_names,
                {"torch", "huggingface-hub", "onnx-ir", "transformers", "filelock"},
            )
            self.assertTrue((output / "lumi-runtime.zip").is_file())
            expected_assets = {
                entry["asset"]
                for entry in manifest["runtimeDependencies"]
                + manifest["installerDependencies"]
            }
            self.assertEqual({path.name for path in output.glob("*.whl")}, expected_assets)
            with zipfile.ZipFile(output / "lumi-runtime.zip") as archive:
                embedded = json.loads(archive.read("lumi-release.json"))
                self.assertEqual(embedded, manifest)
                self.assertIn("lumi/service_factory.py", archive.namelist())
                self.assertFalse(
                    any(
                        name.endswith((".safetensors", ".onnx", ".pt"))
                        for name in archive.namelist()
                    )
                )

    def test_rejects_a_tag_that_does_not_match_project_version(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ReleaseBuildError, "match project.version"):
                build_release(project_root, root, root / "out", "v0.1.1")

    def test_rejects_a_missing_supported_host_wheel(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheelhouse = self._wheelhouse(root)
            missing = (
                wheelhouse
                / "runtime"
                / "cp313-manylinux_2_28_aarch64"
                / "onnxruntime_genai-0.17.1-cp313-cp313-manylinux_2_28_aarch64.whl"
            )
            missing.unlink()
            with self.assertRaisesRegex(ReleaseBuildError, "no onnxruntime-genai wheel"):
                build_release(project_root, wheelhouse, root / "out", "v0.1.0")

    def test_ignores_nested_vendored_distribution_metadata(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheelhouse = self._wheelhouse(root)
            setuptools = self._wheel(
                wheelhouse / "installer" / "universal",
                "setuptools",
                "81.0.0",
                "py3",
                "none",
                "any",
            )
            with zipfile.ZipFile(setuptools, "a") as archive:
                archive.writestr(
                    "setuptools/_vendor/jaraco_text-3.12.1.dist-info/METADATA",
                    "Metadata-Version: 2.1\nName: jaraco-text\nVersion: 3.12.1\n\n",
                )

            manifest = build_release(project_root, wheelhouse, root / "out", "v0.1.0")

        installer_names = {
            entry["distribution"].lower()
            for entry in manifest["installerDependencies"]
        }
        self.assertIn("setuptools", installer_names)


if __name__ == "__main__":
    unittest.main()
