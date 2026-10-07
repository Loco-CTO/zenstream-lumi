"""Install pinned, official Qwen3.5 checkpoints as verified ORT GenAI models.

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
MODEL_MANIFEST_SCHEMA_VERSION = 1
ORT_GENAI_BUILDER_VERSION = "0.17.1"


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
    """Immutable source pins for one text-only Qwen3.5 CPU export."""

    model_id: str
    directory_name: str
    label: str
    repository_id: str
    revision: str
    architecture: str
    model_type: str
    source_files: tuple[str, ...]
    weight_sha256: tuple[tuple[str, str], ...]
    max_source_bytes: int
    supports_thinking: bool = True

    @property
    def pinned_source_manifest_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.source_manifest())).hexdigest()

    def source_manifest(self) -> dict[str, object]:
        return {
            "repositoryId": self.repository_id,
            "revision": self.revision,
            "files": list(self.source_files),
            "weightSha256": dict(self.weight_sha256),
        }


_COMMON_SOURCE_FILES = (
    "LICENSE",
    "chat_template.jinja",
    "config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)

_MODEL_SPECS: dict[str, Qwen35ModelSpec] = {
    "qwen3.5:0.8b": Qwen35ModelSpec(
        model_id="qwen3.5:0.8b",
        directory_name="qwen3.5-0.8b",
        label="Qwen3.5 0.8B",
        repository_id="Qwen/Qwen3.5-0.8B",
        revision="2fc06364715b967f1860aea9cf38778875588b17",
        architecture="Qwen3_5ForConditionalGeneration",
        model_type="qwen3_5_text",
        source_files=(
            *_COMMON_SOURCE_FILES,
            "model.safetensors-00001-of-00001.safetensors",
        ),
        weight_sha256=(
            (
                "model.safetensors-00001-of-00001.safetensors",
                "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
            ),
        ),
        max_source_bytes=2_000_000_000,
    ),
    "qwen3.5:2b": Qwen35ModelSpec(
        model_id="qwen3.5:2b",
        directory_name="qwen3.5-2b",
        label="Qwen3.5 2B",
        repository_id="Qwen/Qwen3.5-2B",
        revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
        architecture="Qwen3_5ForConditionalGeneration",
        model_type="qwen3_5_text",
        source_files=(
            *_COMMON_SOURCE_FILES,
            "model.safetensors-00001-of-00001.safetensors",
        ),
        weight_sha256=(
            (
                "model.safetensors-00001-of-00001.safetensors",
                "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1",
            ),
        ),
        max_source_bytes=5_000_000_000,
    ),
    "qwen3.5:4b": Qwen35ModelSpec(
        model_id="qwen3.5:4b",
        directory_name="qwen3.5-4b",
        label="Qwen3.5 4B",
        repository_id="Qwen/Qwen3.5-4B",
        revision="c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97",
        architecture="Qwen3_5ForConditionalGeneration",
        model_type="qwen3_5_text",
        source_files=(
            *_COMMON_SOURCE_FILES,
            "generation_config.json",
            "model.safetensors-00001-of-00002.safetensors",
            "model.safetensors-00002-of-00002.safetensors",
        ),
        weight_sha256=(
            (
                "model.safetensors-00001-of-00002.safetensors",
                "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
            ),
            (
                "model.safetensors-00002-of-00002.safetensors",
                "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
            ),
        ),
        max_source_bytes=10_000_000_000,
    ),
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_BLOB_RE = re.compile(r"^[0-9a-f]{40}$")
_MAX_MODEL_OUTPUT_BYTES = 20_000_000_000
_HASH_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class _SourceFileRecord:
    path: str
    size_bytes: int
    sha256: str
    blob_id: str | None
    lfs_sha256: str | None


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


class _ModelConverter(Protocol):
    def convert(
        self,
        spec: Qwen35ModelSpec,
        source_dir: Path,
        output_dir: Path,
        cache_dir: Path,
    ) -> None: ...


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
    """Download and convert only Lumi's pinned Qwen3.5 CPU model catalog.

    Use one installer instance for a host process. Its methods are synchronous because
    source transfer and CPU conversion are blocking; async hosts should run them through
    their bounded worker/control lane.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        source_fetcher: _SourceFetcher | None = None,
        converter: _ModelConverter | None = None,
    ) -> None:
        self._root = Path(root).expanduser().resolve()
        self._validate_root()
        self._root.mkdir(parents=True, exist_ok=True)
        self._source_fetcher = source_fetcher or _HuggingFaceSourceFetcher()
        self._converter = converter or _OrtGenAIModelConverter()
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
        """Install an official pinned checkpoint and return a digest-bound artifact."""

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
            cache_dir = stage / "builder-cache"
            source_dir.mkdir()
            output_dir.mkdir()
            cache_dir.mkdir()
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
                reporter.report("verifying-source", 4_900)
                source_records = self._verify_source(spec, source_dir, source_result)
                reporter.report("converting", 5_100)
                self._converter.convert(spec, source_dir, output_dir, cache_dir)
                reporter.report("verifying-output", 8_800)
                file_records, output_size = self._verify_conversion_output(
                    spec,
                    output_dir,
                    lambda complete, total: reporter.report(
                        "verifying-output",
                        8_800 if total <= 0 else 8_800 + (complete * 900 // total),
                    ),
                )
                manifest = {
                    "schemaVersion": MODEL_MANIFEST_SCHEMA_VERSION,
                    "format": "onnxruntime-genai",
                    "modelId": spec.model_id,
                    "modelType": spec.model_type,
                    "builder": {
                        "distribution": "onnxruntime-genai",
                        "version": ORT_GENAI_BUILDER_VERSION,
                        "executionProvider": "cpu",
                        "precision": "int4",
                        "linearAttentionOp": "linear_attention",
                    },
                    "source": {
                        **spec.source_manifest(),
                        "manifestSha256": spec.pinned_source_manifest_sha256,
                        "verifiedFiles": source_records,
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
        """Remove only an exact, manifest-verified model directory owned by this installer."""

        spec = _require_spec(model_id)
        with self._lock:
            artifact = self._read_installed_artifact(spec, verify_files=False)
            if artifact is None:
                return False
            target = self._model_path(spec)
            if target.resolve() != Path(artifact.directory) or os.path.ismount(target):
                raise ModelInstallationError("The Lumi model directory failed its path check")
            _remove_tree_no_follow(target)
            return True

    def _validate_root(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        if _is_within(self._root, package_root):
            raise ValueError("Qwen3.5 model files must be stored outside the Lumi package")

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
            size_bytes=artifact.size_bytes if artifact else 0,
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
            or manifest.get("format") != "onnxruntime-genai"
            or manifest.get("modelId") != spec.model_id
            or manifest.get("modelType") != spec.model_type
        ):
            return None
        source = manifest.get("source")
        if (
            not isinstance(source, Mapping)
            or source.get("manifestSha256") != spec.pinned_source_manifest_sha256
            or source.get("repositoryId") != spec.repository_id
            or source.get("revision") != spec.revision
            or source.get("files") != list(spec.source_files)
            or source.get("weightSha256") != dict(spec.weight_sha256)
            or not _source_records_match_pin(source.get("verifiedFiles"), spec)
        ):
            return None
        builder = manifest.get("builder")
        if (
            not isinstance(builder, Mapping)
            or builder.get("distribution") != "onnxruntime-genai"
            or builder.get("version") != ORT_GENAI_BUILDER_VERSION
            or builder.get("executionProvider") != "cpu"
            or builder.get("precision") != "int4"
            or builder.get("linearAttentionOp") != "linear_attention"
        ):
            return None
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        try:
            files, total_size = self._read_manifest_files(target, manifest, verify_files)
        except (OSError, ValueError, ModelInstallationError):
            return None
        if not files or "genai_config.json" not in files:
            return None
        if not ("chat_template.jinja" in files or "tokenizer_config.json" in files):
            return None
        if not any(name.lower().endswith(".onnx") for name in files):
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
    ) -> list[dict[str, object]]:
        if result.revision != spec.revision:
            raise ModelInstallationError(
                "The downloaded model source revision did not match its pin"
            )
        records = {record.path: record for record in result.files}
        if set(records) != set(spec.source_files):
            raise ModelInstallationError(
                "The downloaded model files did not match Lumi's source manifest"
            )
        actual_paths = {name for name, _ in _walk_regular_files(source_dir)}
        if actual_paths != set(spec.source_files):
            raise ModelInstallationError("The downloaded model directory contains unexpected files")
        expected_weights = dict(spec.weight_sha256)
        verified: list[dict[str, object]] = []
        total_size = 0
        for relative_name in spec.source_files:
            record = records[relative_name]
            path = source_dir.joinpath(*PurePosixPath(relative_name).parts)
            if record.path != relative_name or record.size_bytes < 0:
                raise ModelInstallationError("The downloaded model file metadata is invalid")
            digest, blob_id, size = _hash_source_file(path)
            if size != record.size_bytes or (record.sha256 and digest != record.sha256):
                raise ModelInstallationError(
                    "A downloaded Qwen3.5 source file changed after transfer"
                )
            if record.lfs_sha256 is not None:
                if not _SHA256_RE.fullmatch(record.lfs_sha256) or digest != record.lfs_sha256:
                    raise ModelInstallationError(
                        "A downloaded Qwen3.5 source file failed its LFS hash check"
                    )
            else:
                if record.blob_id is None or not _GIT_BLOB_RE.fullmatch(record.blob_id):
                    raise ModelInstallationError(
                        "A downloaded Qwen3.5 source file has no pinned Git blob"
                    )
                if blob_id != record.blob_id:
                    raise ModelInstallationError(
                        "A downloaded Qwen3.5 source file failed its Git blob check"
                    )
            expected_weight = expected_weights.get(relative_name)
            if expected_weight is not None and digest != expected_weight:
                raise ModelInstallationError(
                    "A Qwen3.5 checkpoint file failed Lumi's pinned SHA-256"
                )
            total_size += size
            if total_size > spec.max_source_bytes:
                raise ModelInstallationError(
                    "The downloaded Qwen3.5 source exceeded its size limit"
                )
            verified.append(
                {
                    "path": relative_name,
                    "size": size,
                    "sha256": digest,
                    "gitBlobId": record.blob_id,
                    "lfsSha256": record.lfs_sha256,
                }
            )

        try:
            config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ModelInstallationError("The pinned Qwen3.5 config is invalid") from error
        if (
            not isinstance(config, Mapping)
            or config.get("model_type") != "qwen3_5"
            or config.get("architectures") != [spec.architecture]
        ):
            raise ModelInstallationError(
                "The official Qwen3.5 model architecture did not match the pin"
            )
        return verified

    def _verify_conversion_output(
        self,
        spec: Qwen35ModelSpec,
        output_dir: Path,
        progress: Callable[[int, int], None],
    ) -> tuple[list[dict[str, object]], int]:
        names_and_paths = _walk_regular_files(output_dir)
        if not names_and_paths or any(
            name == MODEL_MANIFEST_FILENAME for name, _ in names_and_paths
        ):
            raise ModelInstallationError(
                "The ONNX Runtime GenAI builder produced an invalid output directory"
            )
        config_path = output_dir / "genai_config.json"
        if not config_path.is_file() or config_path.is_symlink():
            raise ModelInstallationError(
                "The ONNX Runtime GenAI builder did not produce its runtime config"
            )
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ModelInstallationError(
                "The generated ONNX Runtime GenAI config is invalid"
            ) from error
        model_config = config.get("model") if isinstance(config, Mapping) else None
        if not isinstance(model_config, Mapping) or model_config.get("type") != spec.model_type:
            raise ModelInstallationError(
                "The builder output did not identify the pinned Qwen3.5 text model"
            )
        if not any(name.lower().endswith(".onnx") for name, _ in names_and_paths):
            raise ModelInstallationError(
                "The ONNX Runtime GenAI builder produced no ONNX model graph"
            )
        if not (output_dir / "chat_template.jinja").is_file() and not (
            output_dir / "tokenizer_config.json"
        ).is_file():
            raise ModelInstallationError(
                "The ONNX Runtime GenAI builder produced no tokenizer template"
            )

        total_size = sum(path.stat(follow_symlinks=False).st_size for _, path in names_and_paths)
        if total_size > _MAX_MODEL_OUTPUT_BYTES:
            raise ModelInstallationError("The generated Qwen3.5 model exceeded its size limit")
        verified: list[dict[str, object]] = []
        complete = 0
        for relative_name, path in names_and_paths:
            digest, size = _hash_file_with_size(path)
            complete += size
            verified.append({"path": relative_name, "size": size, "sha256": digest})
            progress(complete, total_size)
        return verified, total_size


class _HuggingFaceSourceFetcher:
    """Download the allowlisted files from a commit-pinned official HF repo."""

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
                "Install Lumi's model-install extra to download and convert Qwen3.5 models"
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
                "The pinned Qwen3.5 source manifest could not be read"
            ) from error
        if getattr(info, "sha", None) != spec.revision:
            raise ModelInstallationError(
                "The official Qwen3.5 source returned a different revision"
            )
        siblings = {item.rfilename: item for item in (getattr(info, "siblings", None) or ())}
        if not set(spec.source_files).issubset(siblings):
            raise ModelInstallationError("The pinned Qwen3.5 source is missing required files")

        sizes: dict[str, int] = {}
        upstream: dict[str, tuple[str | None, str | None]] = {}
        total_size = 0
        pinned_weights = dict(spec.weight_sha256)
        for relative_name in spec.source_files:
            sibling = siblings[relative_name]
            size_value = getattr(sibling, "size", None)
            lfs_info = getattr(sibling, "lfs", None)
            lfs_sha256 = getattr(lfs_info, "sha256", None) if lfs_info else None
            if size_value is None and lfs_info is not None:
                size_value = getattr(lfs_info, "size", None)
            blob_id = getattr(sibling, "blob_id", None)
            if isinstance(size_value, bool) or not isinstance(size_value, int) or size_value < 0:
                raise ModelInstallationError("The pinned Qwen3.5 source has invalid file sizes")
            if lfs_sha256 is not None:
                lfs_sha256 = str(lfs_sha256).lower()
                if not _SHA256_RE.fullmatch(lfs_sha256):
                    raise ModelInstallationError(
                        "The pinned Qwen3.5 source has invalid LFS metadata"
                    )
            if relative_name in pinned_weights and lfs_sha256 != pinned_weights[relative_name]:
                raise ModelInstallationError(
                    "The official Qwen3.5 checkpoint hash no longer matches Lumi's pin"
                )
            if relative_name not in pinned_weights and lfs_sha256 is None:
                blob_id = str(blob_id or "").lower()
                if not _GIT_BLOB_RE.fullmatch(blob_id):
                    raise ModelInstallationError("A pinned Qwen3.5 source file has no Git blob ID")
            if relative_name in pinned_weights and lfs_sha256 is None:
                raise ModelInstallationError(
                    "A Qwen3.5 checkpoint file has no verifiable LFS digest"
                )
            sizes[relative_name] = size_value
            upstream[relative_name] = (str(blob_id).lower() if blob_id else None, lfs_sha256)
            total_size += size_value
        if total_size > spec.max_source_bytes:
            raise ModelInstallationError("The pinned Qwen3.5 source exceeded its size limit")
        if total_size <= 0:
            raise ModelInstallationError("The pinned Qwen3.5 source manifest is empty")

        completed_bytes = 0
        last_reported = -100
        progress_lock = threading.Lock()

        def report_bytes(amount: int) -> None:
            nonlocal completed_bytes, last_reported
            with progress_lock:
                completed_bytes += max(0, amount)
                fraction = min(4_400, completed_bytes * 4_400 // total_size)
                if fraction >= last_reported + 100:
                    last_reported = fraction
                    progress(fraction)

        class DownloadProgress(tqdm):
            """Capture Hub file progress without writing to a terminal."""

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

        records: list[_SourceFileRecord] = []
        for relative_name in spec.source_files:
            target = destination.joinpath(*_safe_relative_path(relative_name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                downloaded_path = Path(
                    hf_hub_download(
                        repo_id=spec.repository_id,
                        filename=relative_name,
                        revision=spec.revision,
                        local_dir=destination,
                        token=False,
                        endpoint="https://huggingface.co",
                        tqdm_class=DownloadProgress,
                    )
                )
            except Exception as error:
                raise ModelInstallationError(
                    "A pinned Qwen3.5 source file could not be downloaded"
                ) from error
            if downloaded_path != target:
                raise ModelInstallationError(
                    "The model source returned an unexpected local file path"
                )
            blob_id, lfs_sha256 = upstream[relative_name]
            records.append(
                _SourceFileRecord(
                    path=relative_name,
                    size_bytes=sizes[relative_name],
                    sha256=lfs_sha256 or "",
                    blob_id=blob_id,
                    lfs_sha256=lfs_sha256,
                )
            )

        cache_metadata = destination / ".cache"
        if cache_metadata.exists() or cache_metadata.is_symlink():
            _remove_tree_no_follow(cache_metadata)
        progress(4_500)
        return _SourceFetchResult(spec.revision, tuple(records))


class _OrtGenAIModelConverter:
    def convert(
        self,
        spec: Qwen35ModelSpec,
        source_dir: Path,
        output_dir: Path,
        cache_dir: Path,
    ) -> None:
        try:
            from importlib.metadata import PackageNotFoundError, version

            installed_version = version("onnxruntime-genai")
        except (ImportError, PackageNotFoundError) as error:
            raise ModelInstallationUnavailableError(
                "Install Lumi's model-install extra to convert Qwen3.5 models"
            ) from error
        if installed_version != ORT_GENAI_BUILDER_VERSION:
            raise ModelInstallationUnavailableError(
                f"Qwen3.5 installation requires ONNX Runtime GenAI {ORT_GENAI_BUILDER_VERSION}"
            )
        try:
            from onnxruntime_genai.models.builder import create_model, parse_extra_options
        except ImportError as error:
            raise ModelInstallationUnavailableError(
                "The ONNX Runtime GenAI model builder dependencies are unavailable"
            ) from error

        model_name = None
        precision = "int4"
        execution_provider = "cpu"
        cache_path = str(cache_dir)
        extra_options = parse_extra_options(
            model_name,
            str(source_dir),
            str(output_dir),
            precision,
            execution_provider,
            cache_path,
            [
                "hf_remote=false",
                "linear_attn_op=linear_attention",
                "use_paged_attention=false",
            ],
        )
        create_model(
            model_name,
            str(source_dir),
            str(output_dir),
            precision,
            execution_provider,
            cache_path,
            **extra_options,
        )


def supported_models() -> tuple[ModelInstallOption, ...]:
    """Return only the model IDs with pinned source and official CPU builder support."""

    return tuple(
        ModelInstallOption(
            model_id=spec.model_id,
            label=spec.label,
            repository_id=spec.repository_id,
            revision=spec.revision,
            installed=False,
            supports_thinking=spec.supports_thinking,
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


def _source_records_match_pin(entries: object, spec: Qwen35ModelSpec) -> bool:
    if not isinstance(entries, list):
        return False
    records: dict[str, Mapping[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            return False
        path = entry.get("path")
        size = entry.get("size")
        digest = entry.get("sha256")
        git_blob = entry.get("gitBlobId")
        lfs_digest = entry.get("lfsSha256")
        if (
            not isinstance(path, str)
            or path in records
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or not _SHA256_RE.fullmatch(digest)
        ):
            return False
        if lfs_digest is not None:
            if not isinstance(lfs_digest, str) or not _SHA256_RE.fullmatch(lfs_digest):
                return False
            if digest != lfs_digest:
                return False
        elif not isinstance(git_blob, str) or not _GIT_BLOB_RE.fullmatch(git_blob):
            return False
        records[path] = entry

    if set(records) != set(spec.source_files):
        return False
    for filename, expected_digest in spec.weight_sha256:
        entry = records[filename]
        if entry.get("sha256") != expected_digest or entry.get("lfsSha256") != expected_digest:
            return False
    return True


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
        raise ModelInstallationError("A generated Qwen3.5 file could not be read") from error
    return digest.hexdigest(), size


def _hash_file(path: Path) -> str:
    return _hash_file_with_size(path)[0]


def _hash_source_file(path: Path) -> tuple[str, str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        file_stat = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            raise ModelInstallationError(
                "A downloaded Qwen3.5 source file is not a private regular file"
            )
        size = file_stat.st_size
        git_blob = hashlib.sha1(f"blob {size}\0".encode("ascii"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
                git_blob.update(chunk)
    except OSError as error:
        raise ModelInstallationError(
            "A downloaded Qwen3.5 source file could not be read"
        ) from error
    return digest.hexdigest(), git_blob.hexdigest(), size


def _sha256_file(path: Path) -> str:
    return _hash_file(path)


def _walk_regular_files(root: Path) -> list[tuple[str, Path]]:
    if root.is_symlink() or _is_junction(root) or not root.is_dir():
        raise ModelInstallationError("A model directory is not a private directory")
    root = root.resolve()
    result: list[tuple[str, Path]] = []
    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        safe_subdirectories: list[str] = []
        for name in subdirectories:
            path = current / name
            if path.is_symlink() or _is_junction(path):
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


def _remove_tree_no_follow(path: Path) -> None:
    if path.is_symlink() or _is_junction(path):
        path.unlink(missing_ok=True)
        return
    if not path.exists():
        return
    shutil.rmtree(path)
