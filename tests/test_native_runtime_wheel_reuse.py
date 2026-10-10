from __future__ import annotations

import hashlib
import json
import tempfile
import tomllib
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from scripts.reuse_native_runtime_wheel import (
    _LATEST_RELEASE_URL,
    _open_url,
    find_reusable_native_wheel,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
BUILD_ID = PROJECT["tool"]["lumi"]["release"]["native-runtime-build-ids"]["windows-x64"]
WHEEL_NAME = "llama_cpp_python-0.3.35-py3-none-win_amd64.whl"


def _wheel_bytes() -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(
            "llama_cpp_python-0.3.35.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: llama_cpp_python\nVersion: 0.3.35\n\n",
        )
        for backend in ("base", "cpu", "cuda", "vulkan"):
            archive.writestr(f"llama_cpp/lib/ggml-{backend}.dll", b"dll")
    return output.getvalue()


class NativeRuntimeWheelReuseTests(unittest.TestCase):
    def _fixture(self, *, build_ids: dict[str, str] | None, tag: str = "v0.3.9"):
        release_url = f"https://github.com/Loco-CTO/zenstream-lumi/releases/download/{tag}"
        wheel_url = f"{release_url}/{WHEEL_NAME}"
        package_url = f"{release_url}/lumi-runtime.zip"
        wheel = _wheel_bytes()
        manifest: dict[str, object] = {
            "schemaVersion": 1,
            "tag": tag,
            "runtimeDependencies": [
                {
                    "distribution": "llama_cpp_python",
                    "version": "0.3.35",
                    "pythonTag": "py3",
                    "abiTag": "none",
                    "platformTag": "win_amd64",
                    "asset": WHEEL_NAME,
                    "sha256": hashlib.sha256(wheel).hexdigest(),
                }
            ],
        }
        if build_ids is not None:
            manifest["nativeRuntimeBuildIds"] = build_ids
        package = BytesIO()
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr("lumi-release.json", json.dumps(manifest))
        release = {
            "tag_name": tag,
            "assets": [
                {
                    "name": "lumi-runtime.zip",
                    "browser_download_url": package_url,
                },
                {"name": WHEEL_NAME, "browser_download_url": wheel_url},
            ],
        }

        def open_url(url: str, _token: str | None, *, accept: str):
            del accept
            if url == _LATEST_RELEASE_URL:
                return BytesIO(json.dumps(release).encode("utf-8"))
            if url == package_url:
                return BytesIO(package.getvalue())
            if url == wheel_url:
                return BytesIO(wheel)
            raise AssertionError(f"unexpected URL: {url}")

        return wheel, open_url

    def test_reuses_verified_wheel_when_native_build_id_matches(self) -> None:
        build_ids = {"windows-x64": BUILD_ID}
        wheel, open_url = self._fixture(build_ids=build_ids)
        with tempfile.TemporaryDirectory() as temporary, patch(
            "scripts.reuse_native_runtime_wheel._open_url", side_effect=open_url
        ):
            path = find_reusable_native_wheel(
                "windows-x64", Path(temporary), project_root=PROJECT_ROOT
            )
            self.assertIsNotNone(path)
            assert path is not None
            self.assertEqual(path.name, WHEEL_NAME)
            self.assertEqual(path.read_bytes(), wheel)

    def test_reuses_the_explicit_bootstrap_release_without_build_ids(self) -> None:
        wheel, open_url = self._fixture(build_ids=None, tag="v0.3.8")
        with tempfile.TemporaryDirectory() as temporary, patch(
            "scripts.reuse_native_runtime_wheel._open_url", side_effect=open_url
        ):
            path = find_reusable_native_wheel(
                "windows-x64", Path(temporary), project_root=PROJECT_ROOT
            )
            self.assertIsNotNone(path)
            assert path is not None
            self.assertEqual(path.read_bytes(), wheel)

    def test_does_not_reuse_bootstrap_wheel_after_native_contract_changes(self) -> None:
        _wheel, open_url = self._fixture(build_ids=None, tag="v0.3.8")
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            (project_root / "pyproject.toml").write_text(
                """
[tool.lumi.release]
native-runtime-bootstrap-tag = "v0.3.8"

[tool.lumi.release.native-runtime-build-ids]
"windows-x64" = "different-build"

[tool.lumi.release.native-runtime-bootstrap-build-ids]
"windows-x64" = "original-build"
""",
                encoding="utf-8",
            )
            with patch(
                "scripts.reuse_native_runtime_wheel._open_url", side_effect=open_url
            ):
                path = find_reusable_native_wheel(
                    "windows-x64", project_root / "wheels", project_root=project_root
                )
        self.assertIsNone(path)

    def test_github_token_is_only_sent_to_the_github_api(self) -> None:
        with patch("scripts.reuse_native_runtime_wheel.urllib.request.urlopen") as open_request:
            _open_url(_LATEST_RELEASE_URL, "secret-token", accept="application/vnd.github+json")
            api_request = open_request.call_args.args[0]
            self.assertEqual(api_request.get_header("Authorization"), "Bearer secret-token")

            _open_url(
                "https://github.com/Loco-CTO/zenstream-lumi/releases/download/v0.3.8/file.whl",
                "secret-token",
                accept="application/octet-stream",
            )
            asset_request = open_request.call_args.args[0]
            self.assertIsNone(asset_request.get_header("Authorization"))

    def test_does_not_reuse_a_wheel_from_a_different_build_contract(self) -> None:
        _wheel, open_url = self._fixture(build_ids={"windows-x64": "different-build"})
        with tempfile.TemporaryDirectory() as temporary, patch(
            "scripts.reuse_native_runtime_wheel._open_url", side_effect=open_url
        ):
            path = find_reusable_native_wheel(
                "windows-x64", Path(temporary), project_root=PROJECT_ROOT
            )
            self.assertIsNone(path)

    def test_rejects_a_wheel_that_does_not_match_the_published_digest(self) -> None:
        wheel, open_url = self._fixture(build_ids={"windows-x64": BUILD_ID})
        del wheel
        original_open = open_url

        def corrupted_wheel(url: str, token: str | None, *, accept: str):
            response = original_open(url, token, accept=accept)
            if url.endswith(f"/{WHEEL_NAME}"):
                return BytesIO(b"corrupted wheel")
            return response

        with tempfile.TemporaryDirectory() as temporary, patch(
            "scripts.reuse_native_runtime_wheel._open_url", side_effect=corrupted_wheel
        ):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                find_reusable_native_wheel(
                    "windows-x64", Path(temporary), project_root=PROJECT_ROOT
                )
