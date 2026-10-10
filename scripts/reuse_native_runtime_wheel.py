"""Reuse a verified native wheel from the latest compatible Lumi release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tomllib
import urllib.request
import zipfile
from email.parser import BytesParser
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

try:
    from scripts.build_release import _canonical_distribution, _supports_platform_target
except ModuleNotFoundError as error:
    if error.name not in {"scripts", "scripts.build_release"}:
        raise
    from build_release import _canonical_distribution, _supports_platform_target

_REPOSITORY = "Loco-CTO/zenstream-lumi"
_LATEST_RELEASE_URL = f"https://api.github.com/repos/{_REPOSITORY}/releases/latest"
_PLATFORM_TAGS = {
    "windows-x64": "win_amd64",
    "linux-x64": "manylinux_2_28_x86_64",
    "linux-arm64": "manylinux_2_28_aarch64",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHUNK_SIZE = 1024 * 1024


def _open_url(url: str, token: str | None, *, accept: str) -> Any:
    headers = {
        "Accept": accept,
        "User-Agent": "zenstream-lumi-release-builder",
    }
    if token and urlsplit(url).hostname == "api.github.com":
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)


def _release_asset_url(tag: str, filename: str) -> str:
    return (
        f"https://github.com/{_REPOSITORY}/releases/download/"
        f"{quote(tag, safe='')}/{quote(filename, safe='')}"
    )


def _load_release_settings(project_root: Path) -> dict[str, Any]:
    try:
        project = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
        return project["tool"]["lumi"]["release"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as error:
        raise ValueError("Lumi native runtime release settings are missing") from error


def _latest_release(token: str | None) -> dict[str, Any]:
    with _open_url(_LATEST_RELEASE_URL, token, accept="application/vnd.github+json") as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("assets"), list):
        raise ValueError("GitHub returned invalid Lumi release metadata")
    return payload


def _read_release_manifest(release: dict[str, Any], token: str | None) -> dict[str, Any]:
    asset = next(
        (item for item in release["assets"] if item.get("name") == "lumi-runtime.zip"),
        None,
    )
    if not isinstance(asset, dict):
        raise ValueError("the latest Lumi release has no package archive")
    url = asset.get("browser_download_url")
    tag = release.get("tag_name")
    if not isinstance(tag, str) or url != _release_asset_url(tag, "lumi-runtime.zip"):
        raise ValueError("the latest Lumi package archive has an unexpected download URL")
    with _open_url(url, token, accept="application/octet-stream") as response:
        package_archive = response.read()
    try:
        with zipfile.ZipFile(BytesIO(package_archive)) as archive:
            manifest = json.loads(archive.read("lumi-release.json").decode("utf-8"))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("the latest Lumi package archive has an invalid manifest") from error
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schemaVersion")) is not int
        or manifest.get("schemaVersion") != 1
        or manifest.get("tag") != release.get("tag_name")
        or not isinstance(manifest.get("runtimeDependencies"), list)
    ):
        raise ValueError("the latest Lumi release manifest is invalid")
    return manifest


def _native_wheel_entry(
    manifest: dict[str, Any], platform_tag: str
) -> dict[str, Any] | None:
    candidates = []
    for item in manifest["runtimeDependencies"]:
        if not isinstance(item, dict):
            continue
        if _canonical_distribution(str(item.get("distribution", ""))) != "llama-cpp-python":
            continue
        candidate_platform = item.get("platformTag")
        if isinstance(candidate_platform, str) and _supports_platform_target(
            candidate_platform, platform_tag
        ):
            candidates.append(item)
    if len(candidates) != 1:
        return None
    entry = candidates[0]
    if entry.get("pythonTag") != "py3" or entry.get("abiTag") != "none":
        return None
    return entry


def _download_verified_wheel(
    release: dict[str, Any],
    manifest: dict[str, Any],
    target: str,
    platform_tag: str,
    output_dir: Path,
    token: str | None,
) -> Path | None:
    entry = _native_wheel_entry(manifest, platform_tag)
    if entry is None:
        return None
    filename = entry.get("asset")
    expected_digest = entry.get("sha256")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not filename.endswith(".whl")
        or not isinstance(expected_digest, str)
        or not _SHA256_RE.fullmatch(expected_digest)
    ):
        raise ValueError("the native wheel manifest entry is invalid")
    asset = next(
        (item for item in release["assets"] if item.get("name") == filename),
        None,
    )
    if not isinstance(asset, dict):
        raise ValueError("the native wheel recorded in the package manifest is missing")
    url = asset.get("browser_download_url")
    tag = release.get("tag_name")
    if not isinstance(tag, str) or url != _release_asset_url(tag, filename):
        raise ValueError("the native wheel has an unexpected download URL")

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / filename
    temporary = destination.with_suffix(destination.suffix + ".partial")
    digest = hashlib.sha256()
    try:
        with _open_url(url, token, accept="application/octet-stream") as response:
            with temporary.open("wb") as stream:
                while chunk := response.read(_CHUNK_SIZE):
                    stream.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != expected_digest:
            raise ValueError("the downloaded native wheel failed its release SHA-256 check")
        with zipfile.ZipFile(temporary) as wheel:
            names = wheel.namelist()
            metadata_names = [
                name
                for name in names
                if name.endswith(".dist-info/METADATA") and name.count("/") == 1
            ]
            if len(metadata_names) != 1:
                raise ValueError("the downloaded native wheel has invalid package metadata")
            metadata = BytesParser().parsebytes(wheel.read(metadata_names[0]))
            if (
                _canonical_distribution(metadata.get("Name", "")) != "llama-cpp-python"
                or metadata.get("Version") != entry.get("version")
            ):
                raise ValueError("the native wheel identity does not match its release manifest")
            required_backends = ["ggml-base", "ggml-cpu"]
            if target != "linux-arm64":
                required_backends.extend(("ggml-cuda", "ggml-vulkan"))
            missing = [
                backend
                for backend in required_backends
                if not any(backend in name.casefold() for name in names)
            ]
            if missing:
                raise ValueError(f"the native wheel is missing required backends: {missing}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def find_reusable_native_wheel(
    target: str,
    output_dir: Path,
    *,
    project_root: Path,
    token: str | None = None,
) -> Path | None:
    if target not in _PLATFORM_TAGS:
        raise ValueError(f"unsupported native runtime target: {target}")
    settings = _load_release_settings(project_root)
    build_ids = settings.get("native-runtime-build-ids")
    expected_id = build_ids.get(target) if isinstance(build_ids, dict) else None
    if not isinstance(expected_id, str) or not expected_id:
        raise ValueError(f"native runtime build ID is missing for {target}")
    bootstrap_tag = settings.get("native-runtime-bootstrap-tag")
    release = _latest_release(token)
    tag = release.get("tag_name")
    if not isinstance(tag, str):
        return None
    manifest = _read_release_manifest(release, token)
    recorded_ids = manifest.get("nativeRuntimeBuildIds")
    if isinstance(recorded_ids, dict):
        if recorded_ids.get(target) != expected_id:
            print(f"Native runtime build contract differs from {tag}; compiling a fresh wheel")
            return None
    else:
        bootstrap_ids = settings.get("native-runtime-bootstrap-build-ids")
        bootstrap_id = (
            bootstrap_ids.get(target) if isinstance(bootstrap_ids, dict) else None
        )
        if tag != bootstrap_tag or expected_id != bootstrap_id:
            print(f"Release {tag} has no matching native build contract; compiling a fresh wheel")
            return None
        print(f"Using the verified {bootstrap_tag} wheel as the initial native runtime baseline")

    return _download_verified_wheel(
        release,
        manifest,
        target,
        _PLATFORM_TAGS[target],
        output_dir,
        token,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(_PLATFORM_TAGS), required=True)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    try:
        path = find_reusable_native_wheel(
            args.target,
            args.wheel_dir,
            project_root=project_root,
            token=token,
        )
    except (OSError, ValueError) as error:
        print(f"Could not safely reuse the previous native wheel: {error}", file=sys.stderr)
        path = None
    hit = path is not None
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a", encoding="utf-8") as stream:
            stream.write(f"hit={'true' if hit else 'false'}\n")
    if hit:
        print(f"Reusing verified native runtime wheel: {path.name}")
    else:
        print("No compatible published native runtime wheel; the source build will run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
