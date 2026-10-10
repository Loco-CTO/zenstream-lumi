"""Assemble Lumi's immutable package and dependency assets for a tagged release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tomllib
import zipfile
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path
from typing import Any

_TAG_RE = re.compile(r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_MANYLINUX_TAG_RE = re.compile(r"^manylinux_(\d+)_(\d+)_(.+)$")
_LEGACY_MANYLINUX_BASELINES = {
    "manylinux1": (2, 5),
    "manylinux2010": (2, 12),
    "manylinux2014": (2, 17),
}
_SUPPORTED_TARGETS = (
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
_MAX_WHEEL_ASSETS = 512


class ReleaseBuildError(RuntimeError):
    """A Lumi release cannot be assembled from the supplied source and wheels."""


@dataclass(frozen=True, slots=True)
class WheelAsset:
    path: Path
    distribution: str
    version: str
    python_tag: str
    abi_tag: str
    platform_tag: str
    sha256: str

    def manifest_entry(self) -> dict[str, str]:
        return {
            "distribution": self.distribution,
            "version": self.version,
            "pythonTag": self.python_tag,
            "abiTag": self.abi_tag,
            "platformTag": self.platform_tag,
            "asset": self.path.name,
            "sha256": self.sha256,
        }


def _canonical_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        with (project_root / "pyproject.toml").open("rb") as stream:
            config = tomllib.load(stream)
        project = config["project"]
        release = config["tool"]["lumi"]["release"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as error:
        raise ReleaseBuildError("Lumi release metadata is missing or invalid") from error
    return project, release


def validate_tag(project_root: Path, tag: str) -> None:
    project, _release = _load_config(project_root)
    if not isinstance(tag, str) or not _TAG_RE.fullmatch(tag):
        raise ReleaseBuildError("A stable vMAJOR.MINOR.PATCH tag is required")
    if tag != f"v{project.get('version')}":
        raise ReleaseBuildError("The release tag must match project.version in pyproject.toml")


def _wheel_identity(path: Path) -> WheelAsset:
    if not path.name.endswith(".whl"):
        raise ReleaseBuildError(f"dependency asset is not a wheel: {path.name}")
    parts = path.name[:-4].split("-")
    if len(parts) < 5:
        raise ReleaseBuildError(f"wheel filename is invalid: {path.name}")
    distribution, version, python_tag, abi_tag, platform_tag = parts[-5:]
    if not all(re.fullmatch(r"[A-Za-z0-9_.+]+", item) for item in (distribution, version)):
        raise ReleaseBuildError(f"wheel identity is invalid: {path.name}")
    if not all(
        re.fullmatch(r"[A-Za-z0-9_.]+", item) for item in (python_tag, abi_tag, platform_tag)
    ):
        raise ReleaseBuildError(f"wheel tags are invalid: {path.name}")

    try:
        with zipfile.ZipFile(path) as wheel:
            metadata_names = [
                name
                for name in wheel.namelist()
                if name.endswith(".dist-info/METADATA") and name.count("/") == 1
            ]
            if len(metadata_names) != 1:
                raise ReleaseBuildError(f"wheel has no unique package metadata: {path.name}")
            metadata = BytesParser().parsebytes(wheel.read(metadata_names[0]))
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise ReleaseBuildError(f"wheel archive is invalid: {path.name}") from error

    metadata_name = metadata.get("Name")
    metadata_version = metadata.get("Version")
    if (
        not isinstance(metadata_name, str)
        or not isinstance(metadata_version, str)
        or _canonical_distribution(metadata_name) != _canonical_distribution(distribution)
        or metadata_version != version
    ):
        raise ReleaseBuildError(f"wheel filename does not match its metadata: {path.name}")
    return WheelAsset(
        path,
        metadata_name,
        metadata_version,
        python_tag,
        abi_tag,
        platform_tag,
        _sha256_file(path),
    )


def _supports_target(
    wheel: WheelAsset,
    python_tag: str,
    abi_tag: str,
    platform_tag: str,
) -> bool:
    if wheel.python_tag not in {python_tag, "py3"} or wheel.abi_tag not in {abi_tag, "none"}:
        wheel_minor = re.fullmatch(r"cp3(\d+)", wheel.python_tag)
        target_minor = re.fullmatch(r"cp3(\d+)", python_tag)
        stable_abi_compatible = (
            wheel.abi_tag == "abi3"
            and wheel_minor is not None
            and target_minor is not None
            and abi_tag == python_tag
            and int(wheel_minor.group(1)) <= int(target_minor.group(1))
        )
        if not stable_abi_compatible:
            return False
    return _supports_platform_target(wheel.platform_tag, platform_tag)


def _manylinux_platform(platform_tag: str) -> tuple[tuple[int, int], str] | None:
    match = _MANYLINUX_TAG_RE.fullmatch(platform_tag)
    if match:
        return (int(match.group(1)), int(match.group(2))), match.group(3)
    for legacy_tag, baseline in _LEGACY_MANYLINUX_BASELINES.items():
        prefix = f"{legacy_tag}_"
        if platform_tag.startswith(prefix):
            return baseline, platform_tag.removeprefix(prefix)
    return None


def _supports_platform_target(wheel_platform_tag: str, target_platform_tag: str) -> bool:
    wheel_platform_tags = wheel_platform_tag.split(".")
    if target_platform_tag in wheel_platform_tags or "any" in wheel_platform_tags:
        return True

    target = _manylinux_platform(target_platform_tag)
    if target is None:
        return False
    target_baseline, target_architecture = target
    for wheel_tag in wheel_platform_tags:
        candidate = _manylinux_platform(wheel_tag)
        if candidate is None:
            continue
        candidate_baseline, candidate_architecture = candidate
        if candidate_architecture == target_architecture and candidate_baseline <= target_baseline:
            return True
    return False


def _collect_wheels(
    wheelhouse: Path,
    release: dict[str, Any],
) -> tuple[list[WheelAsset], list[WheelAsset]]:
    if not wheelhouse.is_dir():
        raise ReleaseBuildError("the release wheelhouse directory is missing")
    assets_by_group: dict[str, dict[str, WheelAsset]] = {
        "runtime": {},
        "installer": {},
    }
    for path in sorted(wheelhouse.rglob("*.whl")):
        relative_parts = path.relative_to(wheelhouse).parts[:-1]
        groups = [part for part in relative_parts if part in assets_by_group]
        if len(groups) != 1:
            raise ReleaseBuildError(
                f"wheel must be under exactly one runtime or installer directory: {path.name}"
            )
        group = groups[0]
        asset = _wheel_identity(path)
        assets_by_name = assets_by_group[group]
        previous = assets_by_name.get(path.name)
        if previous is not None:
            if previous.sha256 != asset.sha256:
                raise ReleaseBuildError(f"duplicate wheel assets differ: {path.name}")
            continue
        assets_by_name[path.name] = asset
    if not any(assets_by_group.values()):
        raise ReleaseBuildError("the release wheelhouse is empty or exceeds the asset limit")

    runtime_names = {_canonical_distribution(name) for name in release["runtime-distributions"]}
    native_runtime_names = {
        _canonical_distribution(name) for name in release.get("runtime-native-distributions", [])
    }
    installer_roots = {_canonical_distribution(name) for name in release["installer-distributions"]}
    runtime = list(assets_by_group["runtime"].values())
    installer = list(assets_by_group["installer"].values())
    runtime_found = {_canonical_distribution(wheel.distribution) for wheel in runtime}
    installer_found = {_canonical_distribution(wheel.distribution) for wheel in installer}
    if not runtime_names.issubset(runtime_found) or not installer_roots.issubset(installer_found):
        raise ReleaseBuildError("the wheelhouse is missing a required Lumi dependency")
    if not native_runtime_names.issubset(runtime_names):
        raise ReleaseBuildError(
            "native runtime dependencies must be declared as runtime dependencies"
        )
    for wheel in runtime:
        if _canonical_distribution(wheel.distribution) in native_runtime_names:
            _validate_native_runtime_wheel(wheel)

    if len(runtime) + len(installer) > _MAX_WHEEL_ASSETS:
        raise ReleaseBuildError("the release wheelhouse is empty or exceeds the asset limit")

    for python_tag, abi_tag, platform_tag in _SUPPORTED_TARGETS:
        for group, required in ((runtime, runtime_names), (installer, installer_roots)):
            for distribution in required:
                if not any(
                    _canonical_distribution(wheel.distribution) == distribution
                    and _supports_target(wheel, python_tag, abi_tag, platform_tag)
                    for wheel in group
                ):
                    raise ReleaseBuildError(
                        f"no {distribution} wheel for {python_tag}/{platform_tag}"
                    )
        for distribution in native_runtime_names:
            if not any(
                _canonical_distribution(wheel.distribution) == distribution
                and wheel.platform_tag != "any"
                and _supports_target(wheel, python_tag, abi_tag, platform_tag)
                for wheel in runtime
            ):
                raise ReleaseBuildError(f"no host-specific {distribution} wheel for {platform_tag}")

    runtime_versions: dict[str, set[str]] = {}
    installer_versions: dict[str, set[str]] = {}
    for wheel in runtime:
        runtime_versions.setdefault(_canonical_distribution(wheel.distribution), set()).add(
            wheel.version
        )
    for wheel in installer:
        installer_versions.setdefault(_canonical_distribution(wheel.distribution), set()).add(
            wheel.version
        )
    for distribution in runtime_versions.keys() & installer_versions.keys():
        if runtime_versions[distribution] != installer_versions[distribution]:
            raise ReleaseBuildError(
                f"runtime and installer dependency versions differ for {distribution}"
            )
        installer = [
            wheel
            for wheel in installer
            if _canonical_distribution(wheel.distribution) != distribution
        ]

    runtime_asset_names = {wheel.path.name for wheel in runtime}
    installer_by_name: dict[str, WheelAsset] = {}
    for wheel in installer:
        previous = installer_by_name.get(wheel.path.name)
        if previous is not None and previous.sha256 != wheel.sha256:
            raise ReleaseBuildError(f"duplicate wheel assets differ: {wheel.path.name}")
        installer_by_name[wheel.path.name] = wheel
    if runtime_asset_names & installer_by_name.keys():
        raise ReleaseBuildError("runtime and installer wheel assets overlap")

    runtime.sort(key=lambda wheel: wheel.path.name)
    installer = sorted(installer_by_name.values(), key=lambda wheel: wheel.path.name)
    return runtime, installer


def _validate_native_runtime_wheel(wheel: WheelAsset) -> None:
    try:
        with zipfile.ZipFile(wheel.path) as archive:
            has_runtime_library = any(
                name.startswith("llama_cpp/lib/") and _is_llama_runtime_library(name)
                for name in archive.namelist()
            )
    except (OSError, zipfile.BadZipFile) as error:
        raise ReleaseBuildError(
            f"native runtime wheel archive is invalid: {wheel.path.name}"
        ) from error
    if not has_runtime_library:
        raise ReleaseBuildError(
            f"native runtime wheel has no compiled llama library: {wheel.path.name}"
        )


def _is_llama_runtime_library(path: str) -> bool:
    name = path.rsplit("/", 1)[-1].lower()
    if not name.startswith(("llama.", "libllama.")):
        return False
    return name.endswith((".dll", ".dylib", ".so")) or ".so." in name


def _package_files(package_root: Path) -> dict[str, tuple[bytes, str]]:
    files: dict[str, tuple[bytes, str]] = {}
    if not package_root.is_dir():
        raise ReleaseBuildError("the Lumi package directory is missing")
    for path in sorted(package_root.rglob("*")):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ReleaseBuildError("Lumi package release files cannot be symlinks")
        if path.is_dir():
            continue
        if not path.is_file() or path.suffix != ".py":
            raise ReleaseBuildError(f"unexpected non-source file in Lumi package: {path}")
        payload = path.read_bytes()
        relative = path.relative_to(package_root.parent).as_posix()
        files[relative] = (payload, hashlib.sha256(payload).hexdigest())
    if "lumi/__init__.py" not in files or "lumi/runtime/__init__.py" not in files:
        raise ReleaseBuildError("the Lumi runtime package entrypoints are missing")
    return files


def build_release(
    project_root: Path,
    wheelhouse: Path,
    output: Path,
    tag: str,
) -> dict[str, Any]:
    validate_tag(project_root, tag)
    _project, release = _load_config(project_root)
    package_files = _package_files(project_root / "lumi")
    runtime, installer = _collect_wheels(wheelhouse, release)
    native_build_ids = release.get("native-runtime-build-ids")
    if (
        not isinstance(native_build_ids, dict)
        or set(native_build_ids) != {"windows-x64", "linux-x64", "linux-arm64"}
        or any(not isinstance(value, str) or not value for value in native_build_ids.values())
    ):
        raise ReleaseBuildError("native runtime build IDs are missing or invalid")
    manifest: dict[str, Any] = {
        "schemaVersion": 1,
        "tag": tag,
        "runtimeApiVersion": release["runtime-api-version"],
        "minimumOrchestratorVersion": release["minimum-orchestrator-version"],
        "maximumOrchestratorVersionExclusive": release["maximum-orchestrator-version-exclusive"],
        "files": {
            relative: {"size": len(payload), "sha256": digest}
            for relative, (payload, digest) in package_files.items()
        },
        "runtimeDependencies": [wheel.manifest_entry() for wheel in runtime],
        "installerDependencies": [wheel.manifest_entry() for wheel in installer],
        "nativeRuntimeBuildIds": native_build_ids,
    }
    output.mkdir(parents=True, exist_ok=True)
    for wheel in (*runtime, *installer):
        destination = output / wheel.path.name
        if destination.exists() and _sha256_file(destination) != wheel.sha256:
            raise ReleaseBuildError(
                f"release output already contains a different {wheel.path.name}"
            )
        if not destination.exists():
            shutil.copyfile(wheel.path, destination)

    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    archive_path = output / "lumi-runtime.zip"
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.writestr("lumi-release.json", manifest_bytes)
        for relative, (payload, _digest) in package_files.items():
            archive.writestr(relative, payload)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Stable release tag, for example v0.2.0")
    parser.add_argument(
        "--check-tag",
        action="store_true",
        help="Validate the tag without building assets",
    )
    parser.add_argument("--wheelhouse", type=Path, help="Directory containing target wheels")
    parser.add_argument("--output", type=Path, help="Release output directory")
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    try:
        validate_tag(project_root, args.tag)
        if args.check_tag:
            return 0
        if args.wheelhouse is None or args.output is None:
            parser.error("--wheelhouse and --output are required when building a release")
        manifest = build_release(project_root, args.wheelhouse, args.output, args.tag)
    except ReleaseBuildError as error:
        print(f"release build error: {error}", file=sys.stderr)
        return 1
    print(
        f"Built Lumi {args.tag}: {len(manifest['files'])} source files, "
        f"{len(manifest['runtimeDependencies'])} runtime wheels, "
        f"{len(manifest['installerDependencies'])} deferred installer wheels"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
