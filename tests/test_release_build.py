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
        wheel_tags = "\n".join(
            f"Tag: {python_tag}-{abi_tag}-{tag}" for tag in platform_tag.split(".")
        )
        with zipfile.ZipFile(path, "w") as archive:
            is_llama = distribution.lower().replace("_", "-") == "llama-cpp-python"
            archive.writestr(
                f"{dist_info}/METADATA",
                f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n\n",
            )
            archive.writestr(
                f"{dist_info}/WHEEL",
                f"Wheel-Version: 1.0\nGenerator: unit-test\nRoot-Is-Purelib: true\n{wheel_tags}\n",
            )
            if is_llama:
                native_filename = "llama.dll" if platform_tag.startswith("win_") else "libllama.so"
                archive.writestr(f"llama_cpp/lib/{native_filename}", b"native fixture")
        return path

    def _wheelhouse(self, root: Path) -> Path:
        wheelhouse = root / "wheels"
        targets = (
            ("cp312", "cp312", "win_amd64"),
            ("cp313", "cp313", "win_amd64"),
            ("cp314", "cp314", "win_amd64"),
            ("cp312", "cp312", "manylinux_2_28_x86_64"),
            ("cp313", "cp313", "manylinux_2_28_x86_64"),
            ("cp314", "cp314", "manylinux_2_28_x86_64"),
            ("cp312", "cp312", "manylinux_2_28_aarch64"),
            ("cp313", "cp313", "manylinux_2_28_aarch64"),
            ("cp314", "cp314", "manylinux_2_28_aarch64"),
        )
        native_targets = (
            ("win_amd64", "windows-x64", "win_amd64"),
            (
                "manylinux_2_28_x86_64",
                "linux-x64",
                "manylinux_2_27_x86_64.manylinux_2_28_x86_64",
            ),
            (
                "manylinux_2_28_aarch64",
                "linux-arm64",
                "manylinux_2_27_aarch64.manylinux_2_28_aarch64",
            ),
        )
        for _target_platform, artifact_name, wheel_platform_tag in native_targets:
            self._wheel(
                wheelhouse / "runtime" / f"llama-cpp-{artifact_name}",
                "llama_cpp_python",
                "0.3.35",
                "py3",
                "none",
                wheel_platform_tag,
            )
        for python_tag, abi_tag, platform_tag in targets:
            runtime_subdir = wheelhouse / "runtime" / f"{python_tag}-{platform_tag}"
            numpy_platform_tag = {
                "manylinux_2_28_x86_64": ("manylinux_2_17_x86_64.manylinux2014_x86_64"),
                "manylinux_2_28_aarch64": ("manylinux_2_17_aarch64.manylinux2014_aarch64"),
            }.get(platform_tag, platform_tag)
            lxml_platform_tag = {
                "manylinux_2_28_x86_64": "manylinux_2_26_x86_64.manylinux_2_28_x86_64",
                "manylinux_2_28_aarch64": "manylinux_2_17_aarch64.manylinux2014_aarch64",
            }.get(platform_tag, platform_tag)
            self._wheel(
                runtime_subdir,
                "numpy",
                "2.5.3",
                python_tag,
                abi_tag,
                numpy_platform_tag,
            )
            self._wheel(
                runtime_subdir,
                "lxml",
                "6.1.3",
                python_tag,
                abi_tag,
                lxml_platform_tag,
            )
        primp_platform_tags = {
            "win_amd64": "win_amd64",
            "manylinux_2_28_x86_64": "manylinux_2_17_x86_64.manylinux2014_x86_64",
            "manylinux_2_28_aarch64": "manylinux_2_17_aarch64.manylinux2014_aarch64",
        }
        for platform_tag, artifact_name, _wheel_platform_tag in native_targets:
            self._wheel(
                wheelhouse / "runtime" / f"primp-{artifact_name}",
                "primp",
                "2.0.1",
                "cp310",
                "abi3",
                primp_platform_tags[platform_tag],
            )
        for platform_tag, _artifact_name, _wheel_platform_tag in native_targets:
            hf_xet_platform_tag = {
                "manylinux_2_28_x86_64": "manylinux_2_17_x86_64",
                "manylinux_2_28_aarch64": "manylinux_2_17_aarch64",
            }.get(platform_tag, platform_tag)
            self._wheel(
                wheelhouse / "installer" / f"native-{platform_tag}",
                "hf_xet",
                "1.1.5",
                "cp38",
                "abi3",
                hf_xet_platform_tag,
            )
        runtime_universal = wheelhouse / "runtime" / "universal"
        installer_universal = wheelhouse / "installer" / "universal"
        self._wheel(runtime_universal, "click", "8.5.0", "py3", "none", "any")
        self._wheel(runtime_universal, "ddgs", "9.16.0", "py3", "none", "any")
        self._wheel(runtime_universal, "diskcache", "5.6.3", "py3", "none", "any")
        self._wheel(runtime_universal, "jinja2", "3.1.6", "py3", "none", "any")
        self._wheel(runtime_universal, "markupsafe", "3.0.2", "py3", "none", "any")
        self._wheel(runtime_universal, "typing_extensions", "4.16.0", "py3", "none", "any")
        self._wheel(installer_universal, "huggingface_hub", "1.10.0", "py3", "none", "any")
        self._wheel(installer_universal, "filelock", "3.18.0", "py3", "none", "any")
        self._wheel(installer_universal, "typing_extensions", "4.16.0", "py3", "none", "any")
        return wheelhouse

    def test_builds_runtime_and_deferred_installer_manifest(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheelhouse = self._wheelhouse(root)
            output = root / "release"
            manifest = build_release(project_root, wheelhouse, output, "v0.3.2")

            self.assertEqual(manifest["schemaVersion"], 1)
            self.assertEqual(manifest["tag"], "v0.3.2")
            self.assertEqual(len(manifest["runtimeDependencies"]), 30)
            self.assertEqual(len(manifest["installerDependencies"]), 5)
            runtime_names = {
                entry["distribution"].lower().replace("_", "-")
                for entry in manifest["runtimeDependencies"]
            }
            self.assertEqual(
                runtime_names,
                {
                    "click",
                    "ddgs",
                    "diskcache",
                    "jinja2",
                    "llama-cpp-python",
                    "lxml",
                    "markupsafe",
                    "numpy",
                    "primp",
                    "typing-extensions",
                },
            )
            native_runtime = [
                entry
                for entry in manifest["runtimeDependencies"]
                if entry["distribution"].lower().replace("_", "-") == "llama-cpp-python"
            ]
            self.assertEqual(len(native_runtime), 3)
            self.assertTrue(all(entry["pythonTag"] == "py3" for entry in native_runtime))
            self.assertTrue(all(entry["abiTag"] == "none" for entry in native_runtime))
            numpy_platforms = {
                entry["platformTag"]
                for entry in manifest["runtimeDependencies"]
                if entry["distribution"].lower() == "numpy"
            }
            self.assertIn("manylinux_2_17_x86_64.manylinux2014_x86_64", numpy_platforms)
            self.assertIn("manylinux_2_17_aarch64.manylinux2014_aarch64", numpy_platforms)
            installer_names = {
                entry["distribution"].lower().replace("_", "-")
                for entry in manifest["installerDependencies"]
            }
            self.assertEqual(
                installer_names,
                {"huggingface-hub", "filelock", "hf-xet"},
            )
            self.assertTrue((output / "lumi-runtime.zip").is_file())
            expected_assets = {
                entry["asset"]
                for entry in manifest["runtimeDependencies"] + manifest["installerDependencies"]
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

    def test_builds_release_with_more_than_128_wheel_assets(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheelhouse = self._wheelhouse(root)
            for index in range(108):
                self._wheel(
                    wheelhouse / "installer" / "universal",
                    f"release_cap_{index:03d}",
                    "1.0.0",
                    "py3",
                    "none",
                    "any",
                )

            self.assertGreater(len(list(wheelhouse.rglob("*.whl"))), 128)
            manifest = build_release(project_root, wheelhouse, root / "out", "v0.3.2")

        self.assertEqual(manifest["tag"], "v0.3.2")
        self.assertGreater(len(manifest["installerDependencies"]), 2)

    def test_rejects_a_tag_that_does_not_match_project_version(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ReleaseBuildError, "match project.version"):
                build_release(project_root, root, root / "out", "v0.1.0")

    def test_rejects_a_missing_supported_host_wheel(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheelhouse = self._wheelhouse(root)
            missing = next(
                (wheelhouse / "runtime" / "llama-cpp-linux-arm64").glob(
                    "llama_cpp_python-0.3.35-py3-none-*.whl"
                )
            )
            missing.unlink()
            with self.assertRaisesRegex(ReleaseBuildError, "no llama-cpp-python wheel"):
                build_release(project_root, wheelhouse, root / "out", "v0.3.2")

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

            manifest = build_release(project_root, wheelhouse, root / "out", "v0.3.2")

        installer_names = {
            entry["distribution"].lower() for entry in manifest["installerDependencies"]
        }
        self.assertIn("setuptools", installer_names)

    def test_rejects_llama_cpp_runtime_wheel_without_native_library(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheelhouse = self._wheelhouse(root)
            for wheel in (wheelhouse / "runtime").rglob("llama_cpp_python-*.whl"):
                replacement = wheel.with_suffix(".tmp")
                with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(replacement, "w") as target:
                    for name in source.namelist():
                        if name.startswith("llama_cpp/lib/"):
                            continue
                        target.writestr(name, source.read(name))
                wheel.unlink()
                replacement.replace(wheel)
            with self.assertRaisesRegex(ReleaseBuildError, "no compiled llama library"):
                build_release(project_root, wheelhouse, root / "out", "v0.3.2")


if __name__ == "__main__":
    unittest.main()
