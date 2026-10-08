"""Install pinned, verified Qwen3.5 GGUF models for in-process llama.cpp.

The module has no download or conversion side effects at import time. Install-time
dependencies are optional so Lumi can still be used without model installation enabled.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from lumi.runtime import VerifiedModelArtifact


INSTALL_PROGRESS_TOTAL = 10_000
MODEL_INSTALL_API_VERSION = 1
MODEL_MANIFEST_FILENAME = "lumi-model-manifest.json"
MODEL_MANIFEST_SCHEMA_VERSION = 2
GGUF_FORMAT = "llama.cpp-gguf"


class ModelInstallationError(RuntimeError):
    """A model could not be installed or the installed model failed verification."""


class UnsupportedModelError(ModelInstallationError):
    """The requested model ID is not in Lumi's pinned installation catalog."""


class ModelInstallationUnavailableError(ModelInstallationError):
    """An optional installer dependency or required platform feature is unavailable."""


@dataclass(frozen=True, slots=True)
class ModelInstallProgress:
    """A bounded install progress event suitable for forwarding through a host job API."""

    model_id: str
    stage: str
    current: int
    total: int = INSTALL_PROGRESS_TOTAL


@dataclass(frozen=True, slots=True)
class InstalledModelArtifact:
    """Verified files that a host can pass to Lumi's runtime adapter."""

    model_id: str
    directory: str
    manifest_sha256: str
    size_bytes: int

    def as_verified_model_artifact(self) -> VerifiedModelArtifact:
        """Build the Lumi runtime's digest-bound model artifact value.

        The runtime is an optional package component, so importing the installer
        does not require it. The returned runtime artifact carries the exact
        directory and manifest digest verified during installation.
        """

        try:
            from lumi.runtime import VerifiedModelArtifact
        except ImportError as error:
            raise ModelInstallationUnavailableError(
                "The Lumi verified model runtime API is unavailable"
            ) from error
        return VerifiedModelArtifact(self.model_id, self.directory, self.manifest_sha256)


@dataclass(frozen=True, slots=True)
class ModelInstallOption:
    """Lumi-owned model choice metadata and its local installation state."""

    model_id: str
    label: str
    repository_id: str
    revision: str
    installed: bool
    supports_thinking: bool = True
    directory: str | None = None
    manifest_sha256: str | None = None
    size_bytes: int = 0


@dataclass(frozen=True, slots=True)
class Qwen35ModelSpec:
    """Immutable source and artifact pins for one supported Qwen3.5 GGUF."""

    model_id: str
    directory_name: str
    label: str
    repository_id: str
    revision: str
    gguf_filename: str
    quantization: str
    gguf_sha256: str
    download_size_bytes: int
    max_download_bytes: int
    supports_thinking: bool = True

    @property
    def pinned_source_manifest_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.source_manifest())).hexdigest()

    def source_manifest(self) -> dict[str, object]:
        return {
            "repositoryId": self.repository_id,
            "revision": self.revision,
            "filename": self.gguf_filename,
            "quantization": self.quantization,
            "sha256": self.gguf_sha256,
            "size": self.download_size_bytes,
        }


_MODEL_SPECS: dict[str, Qwen35ModelSpec] = {
    "qwen3.5:0.8b": Qwen35ModelSpec(
        model_id="qwen3.5:0.8b",
        directory_name="qwen3.5-0.8b",
        label="Qwen3.5 0.8B",
        repository_id="bartowski/Qwen_Qwen3.5-0.8B-GGUF",
        revision="167243f271bba42ffec2e50e982cb3614d6a0b05",
        gguf_filename="Qwen_Qwen3.5-0.8B-Q4_K_M.gguf",
        quantization="Q4_K_M",
        gguf_sha256="fb044e93939a70469c905781334f5de1e6c8b608ced6cbc8c9249bd4127d9526",
        download_size_bytes=579_615_840,
        max_download_bytes=600_000_000,
    ),
    "qwen3.5:2b": Qwen35ModelSpec(
        model_id="qwen3.5:2b",
        directory_name="qwen3.5-2b",
        label="Qwen3.5 2B",
        repository_id="bartowski/Qwen_Qwen3.5-2B-GGUF",
        revision="0719ef0c2bc06b5da2cccfec9dc26b8328f8afbd",
        gguf_filename="Qwen_Qwen3.5-2B-Q4_K_M.gguf",
        quantization="Q4_K_M",
        gguf_sha256="57a1085840f497d764a7fc5d346922dbde961efb54cc792ea81d694fd846a1d8",
        download_size_bytes=1_396_198_496,
        max_download_bytes=1_450_000_000,
    ),
    "qwen3.5:4b": Qwen35ModelSpec(
        model_id="qwen3.5:4b",
        directory_name="qwen3.5-4b",
        label="Qwen3.5 4B",
        repository_id="bartowski/Qwen_Qwen3.5-4B-GGUF",
        revision="ba06320255db2dbec194dad738d066be90dabf29",
        gguf_filename="Qwen_Qwen3.5-4B-Q4_K_M.gguf",
        quantization="Q4_K_M",
        gguf_sha256="13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983",
        download_size_bytes=3_013_027_808,
        max_download_bytes=3_100_000_000,
    ),
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HASH_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class _SourceFileRecord:
    path: str
    size_bytes: int
    sha256: str
    blob_id: str | None = None
    lfs_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class _SourceFetchResult:
    revision: str
    files: tuple[_SourceFileRecord, ...]


class _SourceFetcher(Protocol):
    def fetch(
        self,
        spec: Qwen35ModelSpec,
        destination: Path,
        progress: Callable[[int], None],
    ) -> _SourceFetchResult: ...


class _ProgressReporter:
    def __init__(
        self,
        model_id: str,
        callback: Callable[[ModelInstallProgress], None] | None,
    ) -> None:
        self._model_id = model_id
        self._callback = callback
        self._current = 0

    def report(self, stage: str, current: int) -> None:
        bounded = min(INSTALL_PROGRESS_TOTAL, max(self._current, int(current)))
        if self._callback is not None and bounded != self._current:
            try:
                self._callback(ModelInstallProgress(self._model_id, stage, bounded))
            except Exception:
                # Progress is observational; a host reporting failure must not abort install.
                pass
        self._current = bounded


class Qwen35ModelInstaller:
    """Download and activate only Lumi's pinned Qwen3.5 GGUF model catalog.

    Use one installer instance for a host process. Its methods are synchronous because
    source transfer and hashing are blocking; async hosts should run them through their
    bounded worker/control lane.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        source_fetcher: _SourceFetcher | None = None,
    ) -> None:
        self._root = Path(root).expanduser().resolve()
        self._validate_root()
        self._root.mkdir(parents=True, exist_ok=True)
        self._download_cache_root = self._root / ".lumi-hub-cache"
        self._source_fetcher = source_fetcher or _HuggingFaceSourceFetcher(
            self._download_cache_root
        )
        self._lock = threading.RLock()

    def list_models(self) -> tuple[ModelInstallOption, ...]:
        """List the fixed supported IDs and locally verifiable installations."""

        with self._lock:
            return tuple(self._option(spec) for spec in _MODEL_SPECS.values())

    def install_model(
        self,
        model_id: str,
        *,
        progress: Callable[[ModelInstallProgress], None] | None = None,
    ) -> InstalledModelArtifact:
        """Install one pinned GGUF and return a digest-bound artifact."""

        spec = _require_spec(model_id)
        reporter = _ProgressReporter(spec.model_id, progress)
        with self._lock:
            reporter.report("validating", 100)
            existing = self._read_installed_artifact(spec, verify_files=True)
            if existing is not None:
                reporter.report("complete", INSTALL_PROGRESS_TOTAL)
                return existing

            target = self._model_path(spec)
            if target.exists() or target.is_symlink():
                raise ModelInstallationError(
                    "A Lumi model directory exists but does not match its verified manifest"
                )

            stage_root = self._root / ".lumi-staging"
            if stage_root.is_symlink() or _is_junction(stage_root) or os.path.ismount(stage_root):
                raise ModelInstallationError("The Lumi model staging directory is unsafe")
            stage_root.mkdir(parents=True, exist_ok=True)
            if not _is_within(stage_root.resolve(), self._root):
                raise ModelInstallationError("Lumi staging escaped the model root")
            stage = Path(tempfile.mkdtemp(prefix=f"{spec.directory_name}-", dir=stage_root))
            source_dir = stage / "source"
            output_dir = stage / "output"
            source_dir.mkdir()
            output_dir.mkdir()
            try:
                reporter.report("downloading", 300)
                source_result = self._source_fetcher.fetch(
                    spec,
                    source_dir,
                    lambda fraction: reporter.report(
                        "downloading",
                        300 + min(4_500, max(0, int(fraction))),
                    ),
                )
                self._clear_download_cache(spec)
                reporter.report("verifying-source", 4_900)
                self._verify_source(spec, source_dir, source_result)
                reporter.report("installing", 5_100)
                source_path = source_dir / spec.gguf_filename
                output_path = output_dir / spec.gguf_filename
                os.replace(source_path, output_path)
                reporter.report("verifying-install", 8_800)
                file_records, output_size = self._verify_gguf_output(
                    spec,
                    output_dir,
                    lambda complete, total: reporter.report(
                        "verifying-output",
                        8_800 if total <= 0 else 8_800 + (complete * 900 // total),
                    ),
                )
                manifest = {
                    "schemaVersion": MODEL_MANIFEST_SCHEMA_VERSION,
                    "format": GGUF_FORMAT,
                    "modelId": spec.model_id,
                    "quantization": spec.quantization,
                    "source": {
                        **spec.source_manifest(),
                        "manifestSha256": spec.pinned_source_manifest_sha256,
                    },
                    "files": file_records,
                }
                manifest_bytes = _canonical_json(manifest)
                manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
                manifest_path = output_dir / MODEL_MANIFEST_FILENAME
                with manifest_path.open("xb") as stream:
                    stream.write(manifest_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())

                reporter.report("activating", 9_850)
                if target.exists() or target.is_symlink():
                    raise ModelInstallationError("The Lumi model directory appeared during install")
                os.replace(output_dir, target)
                installed_size = output_size + len(manifest_bytes)
                result = InstalledModelArtifact(
                    model_id=spec.model_id,
                    directory=str(target.resolve()),
                    manifest_sha256=manifest_sha256,
                    size_bytes=installed_size,
                )
                reporter.report("complete", INSTALL_PROGRESS_TOTAL)
                return result
            except ModelInstallationError:
                raise
            except Exception as error:
                raise ModelInstallationError(
                    "The selected Qwen3.5 model could not be installed"
                ) from error
            finally:
                _remove_tree_no_follow(stage)
                try:
                    stage_root.rmdir()
                except OSError:
                    pass

    def remove_model(self, model_id: str) -> bool:
        """Remove the exact managed model directory, including corrupt installations."""

        spec = _require_spec(model_id)
        with self._lock:
            target = self._model_path(spec)
            if not target.exists():
                return False

            if (
                target.is_symlink()
                or _is_junction(target)
                or not target.is_dir()
                or os.path.ismount(target)
            ):
                raise ModelInstallationError("The Lumi model directory failed its path check")
            _walk_regular_files(target)
            _remove_tree_no_follow(target)
            return True

    def _validate_root(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        if _is_within(self._root, package_root):
            raise ValueError("Qwen3.5 model files must be stored outside the Lumi package")

    def _clear_download_cache(self, spec: Qwen35ModelSpec) -> None:
        cache_root = self._download_cache_root
        cache_directory = cache_root / spec.directory_name
        if cache_root.is_symlink() or _is_junction(cache_root) or os.path.ismount(cache_root):
            raise ModelInstallationError("The Lumi download cache directory is unsafe")
        if not cache_root.exists():
            return
        if (
            cache_directory.is_symlink()
            or _is_junction(cache_directory)
            or os.path.ismount(cache_directory)
            or not _is_within(cache_directory.resolve(), self._root)
        ):
            raise ModelInstallationError("The Lumi model download cache is unsafe")
        _remove_tree_no_follow(cache_directory)
        try:
            cache_root.rmdir()
        except OSError:
            pass

    def _model_path(self, spec: Qwen35ModelSpec) -> Path:
        candidate = self._root / spec.directory_name
        if not _is_within(candidate.resolve(), self._root):
            raise ModelInstallationError("The Lumi model directory is outside its managed root")
        return candidate

    def _option(self, spec: Qwen35ModelSpec) -> ModelInstallOption:
        artifact = self._read_installed_artifact(spec, verify_files=True)
        return ModelInstallOption(
            model_id=spec.model_id,
            label=spec.label,
            repository_id=spec.repository_id,
            revision=spec.revision,
            installed=artifact is not None,
            supports_thinking=spec.supports_thinking,
            directory=artifact.directory if artifact else None,
            manifest_sha256=artifact.manifest_sha256 if artifact else None,
            size_bytes=artifact.size_bytes if artifact else spec.download_size_bytes,
        )

    def _read_installed_artifact(
        self,
        spec: Qwen35ModelSpec,
        *,
        verify_files: bool,
    ) -> InstalledModelArtifact | None:
        target = self._model_path(spec)
        if target.is_symlink() or _is_junction(target) or not target.is_dir():
            return None
        manifest_path = target / MODEL_MANIFEST_FILENAME
        if manifest_path.is_symlink() or not manifest_path.is_file():
            return None
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("schemaVersion") != MODEL_MANIFEST_SCHEMA_VERSION
            or manifest.get("format") != GGUF_FORMAT
            or manifest.get("modelId") != spec.model_id
            or manifest.get("quantization") != spec.quantization
        ):
            return None
        source = manifest.get("source")
        if (
            not isinstance(source, Mapping)
            or source.get("manifestSha256") != spec.pinned_source_manifest_sha256
            or source.get("repositoryId") != spec.repository_id
            or source.get("revision") != spec.revision
            or source.get("filename") != spec.gguf_filename
            or source.get("quantization") != spec.quantization
            or source.get("sha256") != spec.gguf_sha256
            or source.get("size") != spec.download_size_bytes
        ):
            return None
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        try:
            files, total_size = self._read_manifest_files(target, manifest, verify_files)
        except (OSError, ValueError, ModelInstallationError):
            return None
        if set(files) != {spec.gguf_filename}:
            return None
        manifest_size = manifest_path.stat(follow_symlinks=False).st_size
        return InstalledModelArtifact(
            model_id=spec.model_id,
            directory=str(target.resolve()),
            manifest_sha256=manifest_sha256,
            size_bytes=total_size + manifest_size,
        )

    def _read_manifest_files(
        self,
        model_dir: Path,
        manifest: Mapping[str, object],
        verify_hashes: bool,
    ) -> tuple[dict[str, Path], int]:
        entries = manifest.get("files")
        if not isinstance(entries, list) or not entries:
            raise ModelInstallationError("The installed Qwen3.5 file manifest is invalid")
        listed: dict[str, Path] = {}
        expected: set[str] = set()
        total_size = 0
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ModelInstallationError("The installed Qwen3.5 file manifest is invalid")
            relative_name = entry.get("path")
            size = entry.get("size")
            digest = entry.get("sha256")
            if (
                not isinstance(relative_name, str)
                or not _valid_relative_path(relative_name)
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                or not isinstance(digest, str)
                or not _SHA256_RE.fullmatch(digest)
                or relative_name in listed
            ):
                raise ModelInstallationError("The installed Qwen3.5 file manifest is invalid")
            path = model_dir.joinpath(*PurePosixPath(relative_name).parts)
            if not _is_within(path.resolve(), model_dir.resolve()):
                raise ModelInstallationError(
                    "An installed Qwen3.5 file path escapes its model directory"
                )
            file_stat = path.stat(follow_symlinks=False)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
                raise ModelInstallationError(
                    "An installed Qwen3.5 file is not a private regular file"
                )
            if file_stat.st_size != size:
                raise ModelInstallationError("An installed Qwen3.5 model file changed size")
            if verify_hashes and _sha256_file(path) != digest:
                raise ModelInstallationError(
                    "An installed Qwen3.5 model file failed its hash check"
                )
            listed[relative_name] = path
            expected.add(relative_name)
            total_size += size

        actual: set[str] = set()
        for relative_name, _path in _walk_regular_files(model_dir):
            if relative_name != MODEL_MANIFEST_FILENAME:
                actual.add(relative_name)
                if relative_name not in expected:
                    raise ModelInstallationError("The installed Qwen3.5 model has unlisted files")
        if actual != expected:
            raise ModelInstallationError("The installed Qwen3.5 model does not match its manifest")
        return listed, total_size

    def _verify_source(
        self,
        spec: Qwen35ModelSpec,
        source_dir: Path,
        result: _SourceFetchResult,
    ) -> None:
        if result.revision != spec.revision:
            raise ModelInstallationError(
                "The downloaded model source revision did not match its pin"
            )
        if len(result.files) != 1 or result.files[0].path != spec.gguf_filename:
            raise ModelInstallationError(
                "The downloaded GGUF did not match Lumi's pinned source manifest"
            )
        actual_paths = {name for name, _ in _walk_regular_files(source_dir)}
        if actual_paths != {spec.gguf_filename}:
            raise ModelInstallationError("The downloaded GGUF directory contains unexpected files")
        record = result.files[0]
        if record.size_bytes != spec.download_size_bytes:
            raise ModelInstallationError("The downloaded GGUF size did not match Lumi's pin")
        if record.size_bytes > spec.max_download_bytes:
            raise ModelInstallationError("The downloaded GGUF exceeded its size limit")
        path = source_dir.joinpath(*_safe_relative_path(spec.gguf_filename).parts)
        digest, size = _hash_file_with_size(path)
        if size != record.size_bytes or digest != spec.gguf_sha256:
            raise ModelInstallationError("The downloaded Qwen3.5 GGUF failed Lumi's pinned hash")
        if record.sha256 and record.sha256 != digest:
            raise ModelInstallationError("The downloaded GGUF changed after transfer")

    def _verify_gguf_output(
        self,
        spec: Qwen35ModelSpec,
        output_dir: Path,
        progress: Callable[[int, int], None],
    ) -> tuple[list[dict[str, object]], int]:
        names_and_paths = _walk_regular_files(output_dir)
        if len(names_and_paths) != 1 or names_and_paths[0][0] != spec.gguf_filename:
            raise ModelInstallationError("The installed output must contain only the pinned GGUF")
        relative_name, path = names_and_paths[0]
        digest, size = _hash_file_with_size(path)
        if size != spec.download_size_bytes or size > spec.max_download_bytes:
            raise ModelInstallationError("The installed Qwen3.5 GGUF has an unexpected size")
        if digest != spec.gguf_sha256:
            raise ModelInstallationError("The installed Qwen3.5 GGUF failed Lumi's pinned hash")
        progress(size, size)
        return [{"path": relative_name, "size": size, "sha256": digest}], size


class _HuggingFaceSourceFetcher:
    """Download exactly one GGUF from its commit-pinned Hugging Face repository."""

    def __init__(self, cache_root: Path) -> None:
        self._cache_root = cache_root

    def fetch(
        self,
        spec: Qwen35ModelSpec,
        destination: Path,
        progress: Callable[[int], None],
    ) -> _SourceFetchResult:
        try:
            from huggingface_hub import HfApi, hf_hub_download
            from tqdm.auto import tqdm
        except ImportError as error:
            raise ModelInstallationUnavailableError(
                "Install Lumi's model-install extra to download Qwen3.5 GGUF models"
            ) from error

        try:
            info = HfApi(endpoint="https://huggingface.co").model_info(
                repo_id=spec.repository_id,
                revision=spec.revision,
                files_metadata=True,
                token=False,
            )
        except Exception as error:
            raise ModelInstallationError(
                "The pinned Qwen3.5 GGUF source could not be read"
            ) from error
        if getattr(info, "sha", None) != spec.revision:
            raise ModelInstallationError(
                "The official GGUF repository returned a different revision"
            )
        siblings = {item.rfilename: item for item in (getattr(info, "siblings", None) or ())}
        sibling = siblings.get(spec.gguf_filename)
        if sibling is None:
            raise ModelInstallationError("The pinned Qwen3.5 GGUF file is missing")

        size = getattr(sibling, "size", None)
        lfs = getattr(sibling, "lfs", None)
        lfs_size = getattr(lfs, "size", None) if lfs is not None else None
        if size is None:
            size = lfs_size
        digest = getattr(lfs, "sha256", None) if lfs is not None else None
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size != spec.download_size_bytes
            or size > spec.max_download_bytes
            or not isinstance(digest, str)
            or digest.lower() != spec.gguf_sha256
        ):
            raise ModelInstallationError(
                "The pinned GGUF metadata no longer matches Lumi's size and hash"
            )

        completed_bytes = 0
        last_reported = -100
        progress_lock = threading.Lock()

        def report_bytes(amount: int) -> None:
            nonlocal completed_bytes, last_reported
            with progress_lock:
                completed_bytes += max(0, amount)
                fraction = min(4_400, completed_bytes * 4_400 // size)
                if fraction >= last_reported + 100:
                    last_reported = fraction
                    progress(fraction)

        class DownloadProgress(tqdm):
            """Capture Hub download progress without writing to a terminal."""

            def __init__(self, *args, **kwargs):
                kwargs["disable"] = False
                super().__init__(*args, **kwargs)
                self._reported_n = 0
                if self.n:
                    report_bytes(self.n)
                    self._reported_n = self.n

            def display(self, *args, **kwargs):
                return False

            def update(self, amount=1):
                result = super().update(amount)
                delta = self.n - self._reported_n
                if delta > 0:
                    report_bytes(delta)
                    self._reported_n = self.n
                return result

        target = destination.joinpath(*_safe_relative_path(spec.gguf_filename).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        cache_directory = self._cache_root / spec.directory_name
        if (
            self._cache_root.is_symlink()
            or _is_junction(self._cache_root)
            or os.path.ismount(self._cache_root)
        ):
            raise ModelInstallationError("The Lumi download cache directory is unsafe")
        if (
            cache_directory.is_symlink()
            or _is_junction(cache_directory)
            or os.path.ismount(cache_directory)
        ):
            raise ModelInstallationError("The Lumi model download cache is unsafe")
        self._cache_root.mkdir(parents=True, exist_ok=True)
        cache_directory.mkdir(parents=True, exist_ok=True)
        if not _is_within(cache_directory.resolve(), self._cache_root.resolve()):
            raise ModelInstallationError("The Lumi model download cache escaped its root")
        cached_target = cache_directory.joinpath(*_safe_relative_path(spec.gguf_filename).parts)
        try:
            downloaded_path = Path(
                hf_hub_download(
                    repo_id=spec.repository_id,
                    filename=spec.gguf_filename,
                    revision=spec.revision,
                    local_dir=cache_directory,
                    cache_dir=cache_directory,
                    token=False,
                    endpoint="https://huggingface.co",
                    tqdm_class=DownloadProgress,
                )
            )
        except Exception as error:
            raise ModelInstallationError(
                "The pinned Qwen3.5 GGUF file could not be downloaded"
            ) from error
        try:
            resolved_download = downloaded_path.resolve()
            resolved_cached_target = cached_target.resolve()
            resolved_cache_directory = cache_directory.resolve()
        except OSError as error:
            raise ModelInstallationError(
                "The GGUF source returned an unsafe local file path"
            ) from error
        resolved_download = _normalise_windows_path_for_comparison(resolved_download)
        resolved_cached_target = _normalise_windows_path_for_comparison(resolved_cached_target)
        resolved_cache_directory = _normalise_windows_path_for_comparison(resolved_cache_directory)
        if (
            downloaded_path.is_symlink()
            or not downloaded_path.is_file()
            or resolved_download != resolved_cached_target
            or not _is_within(resolved_download, resolved_cache_directory)
        ):
            raise ModelInstallationError("The GGUF source returned an unexpected local file path")
        try:
            os.replace(downloaded_path, target)
        except OSError as error:
            raise ModelInstallationError(
                "The downloaded GGUF could not be moved into Lumi staging"
            ) from error
        progress(4_500)
        return _SourceFetchResult(
            spec.revision,
            (_SourceFileRecord(spec.gguf_filename, size, spec.gguf_sha256),),
        )


def supported_models() -> tuple[ModelInstallOption, ...]:
    """Return the fixed allowlist of pinned Qwen3.5 GGUF model choices."""

    return tuple(
        ModelInstallOption(
            model_id=spec.model_id,
            label=spec.label,
            repository_id=spec.repository_id,
            revision=spec.revision,
            installed=False,
            supports_thinking=spec.supports_thinking,
            size_bytes=spec.download_size_bytes,
        )
        for spec in _MODEL_SPECS.values()
    )


def _require_spec(model_id: str) -> Qwen35ModelSpec:
    if not isinstance(model_id, str):
        raise UnsupportedModelError("The selected Qwen3.5 model is unavailable")
    spec = _MODEL_SPECS.get(model_id)
    if spec is None:
        raise UnsupportedModelError("The selected Qwen3.5 model is unavailable")
    return spec


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _valid_relative_path(value: str) -> bool:
    if not value or "\\" in value or "\x00" in value or ":" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and "/".join(path.parts) == value
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _safe_relative_path(value: str) -> PurePosixPath:
    if not _valid_relative_path(value):
        raise ModelInstallationError("A model source contains an unsafe relative path")
    return PurePosixPath(value)


def _hash_file_with_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
                size += len(chunk)
                digest.update(chunk)
    except OSError as error:
        raise ModelInstallationError("A Qwen3.5 model file could not be read") from error
    return digest.hexdigest(), size


def _hash_file(path: Path) -> str:
    return _hash_file_with_size(path)[0]


def _sha256_file(path: Path) -> str:
    return _hash_file(path)


def _walk_regular_files(root: Path) -> list[tuple[str, Path]]:
    if (
        root.is_symlink()
        or _is_junction(root)
        or not root.is_dir()
        or os.path.ismount(root)
    ):
        raise ModelInstallationError("A model directory is not a private directory")
    root = root.resolve()
    result: list[tuple[str, Path]] = []
    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        safe_subdirectories: list[str] = []
        for name in subdirectories:
            path = current / name
            if path.is_symlink() or _is_junction(path) or os.path.ismount(path):
                raise ModelInstallationError("A model directory contains a link")
            if not path.is_dir():
                raise ModelInstallationError("A model directory contains an unsafe entry")
            safe_subdirectories.append(name)
        subdirectories[:] = safe_subdirectories
        for name in filenames:
            path = current / name
            if path.is_symlink() or _is_junction(path):
                raise ModelInstallationError("A model directory contains a link")
            file_stat = path.stat(follow_symlinks=False)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
                raise ModelInstallationError("A model directory contains a non-regular file")
            relative = path.relative_to(root).as_posix()
            if not _valid_relative_path(relative):
                raise ModelInstallationError("A model directory contains an unsafe relative path")
            if not _is_within(path.resolve(), root):
                raise ModelInstallationError("A model file resolves outside its managed directory")
            result.append((relative, path))
    return sorted(result)


def _is_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _normalise_windows_path_for_comparison(path: Path) -> Path:
    """Compare equivalent ordinary and extended-length Windows paths safely."""

    value = os.fspath(path)
    if os.name != "nt":
        return Path(value)
    lowered = value.lower()
    if lowered.startswith("\\\\?\\unc\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(os.path.normcase(value))


def _remove_tree_no_follow(path: Path) -> None:
    if path.is_symlink() or _is_junction(path):
        path.unlink(missing_ok=True)
        return
    if not path.exists():
        return
    shutil.rmtree(path)
