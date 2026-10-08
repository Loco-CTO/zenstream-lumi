"""In-process llama.cpp adapter for installed Qwen3.5 GGUF models.

The native runtime is imported lazily, so Lumi's core package stays lightweight.
Orchestrator installs this Lumi release only when an administrator enables the
integration, then passes the managed model directories to this adapter.
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import os
import re
import stat
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from lumi.contracts import (
    ChatMessage,
    ChatRuntime,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolDefinition,
)
from lumi.model_installation import (
    GGUF_FORMAT,
    MODEL_MANIFEST_SCHEMA_VERSION,
    _require_spec,
    supported_models,
)

LUMI_RUNTIME_API_VERSION = 1
SUPPORTED_QWEN35_MODELS = frozenset(option.model_id for option in supported_models())

_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([A-Za-z0-9_-]{1,64})>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(r"<parameter=([A-Za-z0-9_-]{1,64})>(.*?)</parameter>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_CHANNEL_RE = re.compile(r"<\|channel\|>(analysis|final|commentary|summary|justify|confidence)\b")
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+\|>")


class LlamaCppRuntimeError(RuntimeError):
    """The embedded inference runtime or selected model is unavailable."""


class LlamaCppProtocolError(LlamaCppRuntimeError):
    """The model runtime returned output outside the supported Qwen3.5 format."""


class _CombinedStoppingCriteria(list[Callable[[Any, Any], bool]]):
    """Combine the chat-template and cancellation checks in llama.cpp's callable API."""

    def __call__(self, tokens: Any, logits: Any) -> bool:
        return any(criterion(tokens, logits) for criterion in self)


@dataclass(frozen=True, slots=True)
class VerifiedModelArtifact:
    """A model directory whose files were verified by the Orchestrator installer.

    The installer records the manifest digest after verifying every downloaded
    file. Lumi rechecks that manifest and its declared file layout before native
    loading, without hashing multi-gigabyte weights on every model load.
    """

    model_id: str
    directory: str | Path
    manifest_sha256: str

    def __post_init__(self) -> None:
        if self.model_id not in SUPPORTED_QWEN35_MODELS:
            raise ValueError("Only supported vanilla Qwen3.5 model artifacts may be configured")
        if not isinstance(self.directory, (str, Path)):
            raise ValueError("A model artifact directory must be a filesystem path")
        if not isinstance(self.manifest_sha256, str) or not _SHA256_RE.fullmatch(
            self.manifest_sha256
        ):
            raise ValueError("A model artifact manifest digest must be a lowercase SHA-256")
        object.__setattr__(self, "directory", Path(self.directory).expanduser().absolute())


@dataclass(frozen=True, slots=True)
class LlamaCppConfig:
    """Supported verified model artifacts and bounded CPU inference settings."""

    model_artifacts: Mapping[str, VerifiedModelArtifact]
    idle_unload_seconds: int = 300
    max_context_tokens: int = 8192
    max_output_tokens: int = 2048
    max_messages: int = 64
    max_tools: int = 32
    max_message_chars: int = 32_000
    max_request_bytes: int = 2_000_000
    max_response_chars: int = 200_000
    n_ctx: int = 8192
    n_batch: int = 512
    n_ubatch: int = 512
    n_threads: int = 0
    n_threads_batch: int = 0
    n_gpu_layers: int = 0
    use_mmap: bool = True
    use_mlock: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model_artifacts, Mapping):
            raise ValueError("model_artifacts must map supported IDs to verified artifacts")
        artifacts: dict[str, VerifiedModelArtifact] = {}
        for model, artifact in self.model_artifacts.items():
            if model not in SUPPORTED_QWEN35_MODELS:
                raise ValueError("Only the supported vanilla Qwen3.5 model IDs may be configured")
            if not isinstance(artifact, VerifiedModelArtifact) or artifact.model_id != model:
                raise ValueError("Each configured model must match its verified artifact identity")
            artifacts[model] = artifact
        object.__setattr__(self, "model_artifacts", MappingProxyType(artifacts))

        bounded_values = (
            ("idle_unload_seconds", self.idle_unload_seconds, 0, 86_400),
            ("max_context_tokens", self.max_context_tokens, 256, 32_768),
            ("max_output_tokens", self.max_output_tokens, 1, 8_192),
            ("max_messages", self.max_messages, 1, 128),
            ("max_tools", self.max_tools, 0, 64),
            ("max_message_chars", self.max_message_chars, 256, 1_000_000),
            ("max_request_bytes", self.max_request_bytes, 1_024, 16_000_000),
            ("max_response_chars", self.max_response_chars, 1, 1_000_000),
            ("n_ctx", self.n_ctx, 256, 32_768),
            ("n_batch", self.n_batch, 1, 4096),
            ("n_ubatch", self.n_ubatch, 1, 4096),
            ("n_threads", self.n_threads, 0, 256),
            ("n_threads_batch", self.n_threads_batch, 0, 256),
            ("n_gpu_layers", self.n_gpu_layers, 0, 256),
        )
        for name, value, minimum, maximum in bounded_values:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise ValueError(f"{name} must be between {minimum} and {maximum}")
        if not isinstance(self.use_mmap, bool) or not isinstance(self.use_mlock, bool):
            raise ValueError("use_mmap and use_mlock must be booleans")
        if self.n_ubatch > self.n_batch:
            raise ValueError("n_ubatch cannot exceed n_batch")


@dataclass(slots=True)
class _LoadedModel:
    model_id: str
    model: Any
    chat_template: str
    eos_token: str
    bos_token: str


class LlamaCppChatRuntime(ChatRuntime):
    """Run one lazily loaded Qwen3.5 model in the current Orchestrator process.

    Generation runs on a worker thread so synchronous native inference does not block
    the event loop. One generator runs at a time, and a single model stays resident;
    switching models unloads the previous model first. The native dependency is optional
    and imported only when an installed model is first selected.
    """

    def __init__(self, config: LlamaCppConfig, *, api: Any | None = None) -> None:
        self.config = config
        self._api = api
        self._loaded: _LoadedModel | None = None
        self._verified_models: dict[
            tuple[str, str], tuple[tuple[str, tuple[int, int, int, int, int]], ...]
        ] = {}
        self._resident_model_id: str | None = None
        self._last_used = time.monotonic()
        self._slot = asyncio.Semaphore(1)
        self._opened = False
        self._closing = False
        self._closed = False
        self._idle_task: asyncio.Task[None] | None = None

    async def open(self) -> None:
        """Make the runtime available without importing llama.cpp or loading a model."""

        if self._closed:
            raise RuntimeError("A closed Lumi runtime cannot be reopened")
        if self._opened:
            return
        self._opened = True
        self._closing = False
        self._idle_task = asyncio.create_task(self._unload_when_idle())

    async def close(self) -> None:
        """Stop idle management and release the resident model after active inference."""

        if self._closed:
            return
        self._closing = True
        idle_task = self._idle_task
        self._idle_task = None
        if idle_task is not None and idle_task is not asyncio.current_task():
            idle_task.cancel()
            try:
                await idle_task
            except asyncio.CancelledError:
                pass

        await self._slot.acquire()
        try:
            await asyncio.to_thread(self._unload_sync)
            self._resident_model_id = None
            self._opened = False
            self._closed = True
        finally:
            self._slot.release()

    async def unload(self, model: str | None = None) -> None:
        """Unload the current model, optionally only when it matches ``model``."""

        if model is not None and model not in self.config.model_artifacts:
            raise ValueError("The selected model is not installed and supported")
        if not self._opened or self._closing or self._closed:
            raise RuntimeError("Open the Lumi runtime before unloading a model")
        await self._slot.acquire()
        try:
            if model is None or self._resident_model_id == model:
                await asyncio.to_thread(self._unload_sync)
                self._resident_model_id = None
        finally:
            self._last_used = time.monotonic()
            self._slot.release()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Generate one bounded assistant answer or normalized native tool call."""

        if not self._opened or self._closing or self._closed:
            raise RuntimeError("Open the Lumi runtime before making requests")
        self._validate_request(request)
        await self._slot.acquire()
        if not self._opened or self._closing or self._closed:
            self._slot.release()
            raise RuntimeError("The Lumi runtime is not available")

        cancellation = threading.Event()
        worker = asyncio.create_task(asyncio.to_thread(self._complete_sync, request, cancellation))
        release_in_callback = False
        try:
            response = await asyncio.shield(worker)
            return response
        except asyncio.CancelledError:
            # Stop between native token steps. Keep the single-flight slot until the
            # worker exits because a native step already in progress cannot be killed.
            cancellation.set()
            release_in_callback = True
            worker.add_done_callback(self._finish_cancelled_worker)
            raise
        finally:
            if not release_in_callback:
                self._resident_model_id = (
                    self._loaded.model_id if self._loaded is not None else None
                )
                self._last_used = time.monotonic()
                self._slot.release()

    def _finish_cancelled_worker(self, task: asyncio.Task[ModelResponse]) -> None:
        try:
            task.exception()
        except asyncio.CancelledError:
            pass
        self._resident_model_id = self._loaded.model_id if self._loaded is not None else None
        self._last_used = time.monotonic()
        self._slot.release()

    async def _unload_when_idle(self) -> None:
        interval = min(max(self.config.idle_unload_seconds, 0.1), 30)
        while not self._closing:
            await asyncio.sleep(interval)
            model_id = self._resident_model_id
            if model_id is None:
                continue
            if time.monotonic() - self._last_used < self.config.idle_unload_seconds:
                continue
            await self._slot.acquire()
            try:
                if (
                    self._resident_model_id is not None
                    and time.monotonic() - self._last_used >= self.config.idle_unload_seconds
                ):
                    await asyncio.to_thread(self._unload_sync)
                    self._resident_model_id = None
            finally:
                self._slot.release()

    def _validate_request(self, request: ModelRequest) -> None:
        if request.model not in self.config.model_artifacts:
            raise ValueError("The selected Qwen3.5 model is not installed and enabled")
        if not isinstance(request.thinking, bool):
            raise ValueError("thinking must be a boolean")
        if not isinstance(request.context_size, int) or isinstance(request.context_size, bool):
            raise ValueError("context_size must be a positive integer")
        if not isinstance(request.output_tokens, int) or isinstance(request.output_tokens, bool):
            raise ValueError("output_tokens must be a positive integer")
        if request.context_size < 1 or request.output_tokens < 1:
            raise ValueError("context_size and output_tokens must be positive")
        if not request.messages or len(request.messages) > self.config.max_messages:
            raise ValueError("The message count is outside the configured Lumi limit")
        if len(request.tools) > self.config.max_tools:
            raise ValueError("The tool count is outside the configured Lumi limit")

        messages = [self._serialize_message(message) for message in request.messages]
        tools = [self._serialize_tool(tool) for tool in request.tools]
        try:
            serialized = json.dumps(
                {"messages": messages, "tools": tools},
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("The Lumi request contains non-JSON data") from error
        if len(serialized.encode("utf-8")) > self.config.max_request_bytes:
            raise ValueError("The Lumi request exceeds the configured size limit")

    def _serialize_message(self, message: ChatMessage) -> dict[str, Any]:
        if message.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Unsupported chat message role")
        if (
            not isinstance(message.content, str)
            or len(message.content) > self.config.max_message_chars
        ):
            raise ValueError("A chat message exceeds the configured size limit")
        result: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.role == "tool" and message.name is not None:
            if not _TOOL_NAME_RE.fullmatch(message.name):
                raise ValueError("Tool result messages must include a valid tool name")
            result["name"] = message.name
        if message.role == "assistant" and message.tool_calls:
            if any(not _TOOL_NAME_RE.fullmatch(call.name) for call in message.tool_calls):
                raise ValueError("Tool call name is invalid")
            result["tool_calls"] = [
                {
                    "function": {
                        "name": call.name,
                        "arguments": self._json_object(call.arguments, "Tool call arguments"),
                    }
                }
                for call in message.tool_calls
            ]
        return result

    def _serialize_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        if not tool.read_only:
            raise ValueError("Only explicitly read-only tools may be sent to Qwen3.5")
        if not _TOOL_NAME_RE.fullmatch(tool.name):
            raise ValueError("Tool name is invalid")
        if not isinstance(tool.description, str) or len(tool.description) > 2_000:
            raise ValueError("Tool description exceeds the configured size limit")
        parameters = self._json_object(tool.parameters, "Tool parameters")
        if parameters.get("type") != "object":
            raise ValueError("Tool parameter schemas must describe a JSON object")
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            },
        }

    @staticmethod
    def _json_object(value: Any, label: str) -> dict[str, Any]:
        try:
            normalized = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} must be JSON serializable") from error
        if not isinstance(normalized, dict):
            raise ValueError(f"{label} must be a JSON object")
        return normalized

    def _complete_sync(self, request: ModelRequest, cancellation: threading.Event) -> ModelResponse:
        if cancellation.is_set():
            raise asyncio.CancelledError
        loaded = self._loaded
        if loaded is None or loaded.model_id != request.model:
            # Drop this local reference before clearing the runtime-owned reference;
            # otherwise the old native weights remain alive during the next load.
            loaded = None
            self._unload_sync()
            self._loaded = self._load_model(request.model)
            loaded = self._loaded
        assert loaded is not None

        tools_by_name = {tool.name: tool for tool in request.tools}
        messages = [self._serialize_message(message) for message in request.messages]
        tools = [self._serialize_tool(tool) for tool in request.tools]
        try:
            formatter = self._api_module().Jinja2ChatFormatter(
                template=loaded.chat_template,
                eos_token=loaded.eos_token,
                bos_token=loaded.bos_token,
                add_generation_prompt=True,
            )
            formatted = formatter(
                messages=messages,
                tools=tools or None,
                enable_thinking=request.thinking,
            )
            prompt = getattr(formatted, "prompt", None)
            added_special = bool(getattr(formatted, "added_special", False))
            stop = getattr(formatted, "stop", None)
            formatter_stopping_criteria = getattr(formatted, "stopping_criteria", None) or []
            if not isinstance(formatter_stopping_criteria, (list, tuple)) or any(
                not callable(criterion) for criterion in formatter_stopping_criteria
            ):
                raise LlamaCppProtocolError(
                    "The Qwen3.5 chat template returned invalid stopping criteria"
                )
            if not isinstance(prompt, str) or not prompt:
                raise LlamaCppProtocolError("The Qwen3.5 chat template returned invalid text")
            encoded = loaded.model.tokenize(
                prompt.encode("utf-8"),
                add_bos=not added_special,
                special=True,
            )
            prompt_tokens = _token_count(encoded)
            context_limit = min(
                request.context_size,
                self.config.max_context_tokens,
                self.config.n_ctx,
            )
            if prompt_tokens <= 0 or prompt_tokens >= context_limit:
                raise ValueError("The prompt leaves no room for a Qwen3.5 response")
            output_limit = min(
                request.output_tokens,
                self.config.max_output_tokens,
                context_limit - prompt_tokens,
            )
            if output_limit <= 0:
                raise ValueError("The prompt leaves no room for a Qwen3.5 response")

            def should_stop(_tokens: Any, _logits: Any) -> bool:
                return cancellation.is_set()

            stopping_criteria = _CombinedStoppingCriteria(
                [*formatter_stopping_criteria, should_stop]
            )
            result = loaded.model.create_completion(
                prompt=encoded,
                max_tokens=output_limit,
                temperature=0.0,
                stop=stop,
                stopping_criteria=stopping_criteria,
                stream=False,
            )
            if cancellation.is_set():
                raise asyncio.CancelledError
            choices = result.get("choices") if isinstance(result, Mapping) else None
            text = choices[0].get("text") if isinstance(choices, list) and choices else None
        except (ValueError, LlamaCppRuntimeError):
            raise
        except Exception as error:
            raise LlamaCppRuntimeError("Embedded Qwen3.5 inference failed") from error

        if not isinstance(text, str) or len(text) > self.config.max_response_chars:
            raise LlamaCppProtocolError("The Qwen3.5 response is malformed or too large")
        content, calls = _parse_model_response(text, tools_by_name, self.config.max_tools)
        return ModelResponse(ChatMessage("assistant", content, tool_calls=calls))

    def _load_model(self, model_id: str) -> _LoadedModel:
        artifact = self.config.model_artifacts[model_id]
        model_dir = _verify_model_artifact(artifact, self._verified_models)
        spec = _require_spec(model_id)
        model_path = model_dir / spec.gguf_filename
        api = self._api_module()
        try:
            model = api.Llama(
                model_path=str(model_path),
                n_ctx=self.config.n_ctx,
                n_batch=self.config.n_batch,
                n_ubatch=self.config.n_ubatch,
                n_threads=self.config.n_threads,
                n_threads_batch=self.config.n_threads_batch,
                n_gpu_layers=self.config.n_gpu_layers,
                use_mmap=self.config.use_mmap,
                use_mlock=self.config.use_mlock,
                verbose=False,
            )
            template = _read_chat_template(getattr(model, "metadata", None))
            eos_token = _model_token_text(model, "token_eos")
            bos_token = _model_token_text(model, "token_bos")
        except LlamaCppRuntimeError:
            raise
        except Exception as error:
            raise LlamaCppRuntimeError("The selected Qwen3.5 model could not be loaded") from error
        return _LoadedModel(model_id, model, template, eos_token, bos_token)

    def _api_module(self) -> Any:
        if self._api is not None:
            return self._api
        try:
            from types import SimpleNamespace

            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        except (ImportError, OSError) as error:  # pragma: no cover - optional native dependency
            raise LlamaCppRuntimeError(
                "Lumi's embedded runtime dependency is unavailable in this installation"
            ) from error
        self._api = SimpleNamespace(Llama=Llama, Jinja2ChatFormatter=Jinja2ChatFormatter)
        return self._api

    def _unload_sync(self) -> None:
        self._loaded = None
        gc.collect()


def _read_chat_template(metadata: Any) -> str:
    if not isinstance(metadata, Mapping):
        raise LlamaCppRuntimeError("The selected Qwen3.5 model has no GGUF metadata")
    template = metadata.get("tokenizer.chat_template")
    if not isinstance(template, str) or not template.strip():
        raise LlamaCppRuntimeError("The selected Qwen3.5 GGUF has no chat template")
    return template


def _model_token_text(model: Any, token_method: str) -> str:
    get_id = getattr(model, token_method, None)
    detokenize = getattr(model, "detokenize", None)
    if not callable(get_id) or not callable(detokenize):
        raise LlamaCppRuntimeError("The selected Qwen3.5 GGUF has no special-token metadata")
    try:
        token = detokenize([get_id()], special=True)
        if isinstance(token, bytes):
            token = token.decode("utf-8")
    except Exception as error:
        raise LlamaCppRuntimeError(
            "The selected Qwen3.5 special-token metadata is invalid"
        ) from error
    if not isinstance(token, str) or not token:
        raise LlamaCppRuntimeError("The selected Qwen3.5 special-token metadata is invalid")
    return token


def _verify_model_artifact(
    artifact: VerifiedModelArtifact,
    verified_cache: dict[tuple[str, str], tuple[tuple[str, tuple[int, int, int, int, int]], ...]]
    | None = None,
) -> Path:
    """Validate the pinned GGUF manifest and hash weights when file identity changes."""

    spec = _require_spec(artifact.model_id)
    supplied_dir = Path(artifact.directory)
    cursor = supplied_dir
    while cursor != cursor.parent:
        if cursor.exists() and (cursor.is_symlink() or _is_junction(cursor)):
            raise LlamaCppRuntimeError("The selected Qwen3.5 model path contains a link")
        cursor = cursor.parent
    try:
        model_dir = supplied_dir.resolve(strict=True)
    except OSError as error:
        raise LlamaCppRuntimeError("The selected Qwen3.5 model files are not installed") from error
    if not model_dir.is_dir() or model_dir.name != spec.directory_name:
        raise LlamaCppRuntimeError("The selected Qwen3.5 model directory is invalid")

    manifest_path = model_dir / "lumi-model-manifest.json"
    try:
        manifest_stat = manifest_path.stat(follow_symlinks=False)
        if (
            manifest_path.is_symlink()
            or _is_junction(manifest_path)
            or not stat.S_ISREG(manifest_stat.st_mode)
            or manifest_stat.st_nlink != 1
        ):
            raise LlamaCppRuntimeError("The selected Qwen3.5 model manifest is unsafe")
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise LlamaCppRuntimeError("The selected Qwen3.5 model manifest is unavailable") from error
    if hashlib.sha256(manifest_bytes).hexdigest() != artifact.manifest_sha256:
        raise LlamaCppRuntimeError("The selected Qwen3.5 model manifest changed after verification")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LlamaCppRuntimeError("The selected Qwen3.5 model manifest is invalid") from error

    expected_source = {
        **spec.source_manifest(),
        "manifestSha256": spec.pinned_source_manifest_sha256,
    }
    if (
        not isinstance(manifest, Mapping)
        or set(manifest)
        != {
            "schemaVersion",
            "format",
            "modelId",
            "quantization",
            "source",
            "files",
        }
        or isinstance(manifest.get("schemaVersion"), bool)
        or manifest.get("schemaVersion") != MODEL_MANIFEST_SCHEMA_VERSION
        or manifest.get("format") != GGUF_FORMAT
        or manifest.get("modelId") != artifact.model_id
        or manifest.get("quantization") != spec.quantization
        or manifest.get("source") != expected_source
    ):
        raise LlamaCppRuntimeError("The selected Qwen3.5 GGUF manifest identity is invalid")

    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], Mapping):
        raise LlamaCppRuntimeError("The selected Qwen3.5 GGUF file manifest is invalid")
    entry = entries[0]
    expected_file = {
        "path": spec.gguf_filename,
        "size": spec.download_size_bytes,
        "sha256": spec.gguf_sha256,
    }
    if dict(entry) != expected_file:
        raise LlamaCppRuntimeError("The selected Qwen3.5 GGUF file manifest does not match its pin")

    model_path = model_dir / spec.gguf_filename
    try:
        model_stat = model_path.stat(follow_symlinks=False)
        if (
            model_path.is_symlink()
            or _is_junction(model_path)
            or not stat.S_ISREG(model_stat.st_mode)
            or model_stat.st_nlink != 1
            or model_stat.st_size != spec.download_size_bytes
        ):
            raise LlamaCppRuntimeError("The selected Qwen3.5 GGUF is missing or changed")
        model_path.resolve(strict=True).relative_to(model_dir)
    except (OSError, ValueError) as error:
        raise LlamaCppRuntimeError("The selected Qwen3.5 GGUF path is invalid") from error

    actual_files: set[str] = set()
    for directory, subdirectories, filenames in os.walk(model_dir, followlinks=False):
        current_dir = Path(directory)
        for name in subdirectories:
            child = current_dir / name
            if child.is_symlink() or _is_junction(child):
                raise LlamaCppRuntimeError("The selected Qwen3.5 model contains a symbolic link")
        for name in filenames:
            child = current_dir / name
            if child.is_symlink() or _is_junction(child):
                raise LlamaCppRuntimeError("The selected Qwen3.5 model contains a symbolic link")
            if child != manifest_path:
                actual_files.add(child.relative_to(model_dir).as_posix())
    if actual_files != {spec.gguf_filename}:
        raise LlamaCppRuntimeError("The selected Qwen3.5 GGUF files do not match their manifest")

    fingerprints = ((spec.gguf_filename, _model_file_fingerprint(model_path)),)
    cache_key = (str(model_dir), artifact.manifest_sha256)
    if verified_cache is None or verified_cache.get(cache_key) != fingerprints:
        if _sha256_file(model_path) != spec.gguf_sha256:
            raise LlamaCppRuntimeError("The selected Qwen3.5 GGUF failed its integrity check")
        if verified_cache is not None:
            verified_cache[cache_key] = fingerprints
    return model_dir


def _is_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


def _model_file_fingerprint(path: Path) -> tuple[int, int, int, int, int]:
    try:
        file_stat = path.stat(follow_symlinks=False)
    except OSError as error:
        raise LlamaCppRuntimeError("A verified Qwen3.5 GGUF is unavailable") from error
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
        raise LlamaCppRuntimeError("A verified Qwen3.5 GGUF is not a private regular file")
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise LlamaCppRuntimeError("A verified Qwen3.5 GGUF is unavailable") from error
    return digest.hexdigest()


def _parse_model_response(
    text: str,
    tools: Mapping[str, ToolDefinition],
    max_tools: int,
) -> tuple[str, tuple[ToolCall, ...]]:
    matches = list(_TOOL_CALL_RE.finditer(text))
    if text.count("<tool_call>") != len(matches) or text.count("</tool_call>") != len(matches):
        raise LlamaCppProtocolError("The Qwen3.5 tool-call output is malformed")
    if len(matches) > max_tools:
        raise LlamaCppProtocolError("Qwen3.5 returned too many tool calls")

    calls: list[ToolCall] = []
    pieces: list[str] = []
    previous_end = 0
    for match in matches:
        pieces.append(text[previous_end : match.start()])
        name = match.group(1)
        tool = tools.get(name)
        if tool is None:
            raise LlamaCppProtocolError("Qwen3.5 returned an unregistered tool call")
        if not _TOOL_NAME_RE.fullmatch(name) or not tool.read_only:
            raise LlamaCppProtocolError("Qwen3.5 returned an unsupported tool call")

        body = match.group(2)
        arguments: dict[str, Any] = {}
        cursor = 0
        for parameter in _PARAMETER_RE.finditer(body):
            if body[cursor : parameter.start()].strip():
                raise LlamaCppProtocolError("The Qwen3.5 tool-call arguments are malformed")
            key = parameter.group(1)
            if key in arguments:
                raise LlamaCppProtocolError("Qwen3.5 returned a duplicate tool argument")
            value = parameter.group(2).strip("\r\n")
            arguments[key] = _parse_argument(value, tool, key)
            cursor = parameter.end()
        if body[cursor:].strip():
            raise LlamaCppProtocolError("The Qwen3.5 tool-call arguments are malformed")
        required = tool.parameters.get("required", [])
        if isinstance(required, list) and any(key not in arguments for key in required):
            raise LlamaCppProtocolError("The Qwen3.5 tool call is missing a required argument")
        if not isinstance(arguments, dict):
            raise LlamaCppProtocolError("Qwen3.5 tool arguments must be a JSON object")
        calls.append(ToolCall(uuid4().hex, name, arguments))
        previous_end = match.end()
    pieces.append(text[previous_end:])

    content = "".join(pieces)
    # Qwen3.5 may mark analysis and final channels with special tokens in a GGUF
    # completion. Only keep the final channel when one is present.
    channel_markers = list(_CHANNEL_RE.finditer(content))
    final_markers = [marker for marker in channel_markers if marker.group(1) == "final"]
    if final_markers:
        content = content[final_markers[-1].end() :]
    elif any(marker.group(1) == "analysis" for marker in channel_markers):
        first_analysis = next(marker for marker in channel_markers if marker.group(1) == "analysis")
        content = content[: first_analysis.start()]

    content = _THINK_RE.sub("", content)
    if "<think>" in content:
        content = content.split("<think>", 1)[0]
    content = _SPECIAL_TOKEN_RE.sub("", content).strip()
    return content, tuple(calls)


def _parse_argument(value: str, tool: ToolDefinition, name: str) -> Any:
    properties = tool.parameters.get("properties", {})
    schema = properties.get(name, {}) if isinstance(properties, Mapping) else {}
    expected_type = schema.get("type") if isinstance(schema, Mapping) else None
    allowed_types = (
        {item for item in expected_type if isinstance(item, str)}
        if isinstance(expected_type, list)
        else {expected_type} if isinstance(expected_type, str) else set()
    )
    if "string" in allowed_types:
        if "null" in allowed_types and value.strip().casefold() == "null":
            return None
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise LlamaCppProtocolError("Qwen3.5 returned a malformed tool argument") from error


def _token_count(tokens: Any) -> int:
    shape = getattr(tokens, "shape", None)
    if shape is not None:
        if not shape:
            return 0
        if len(shape) == 1:
            return int(shape[0])
        if len(shape) == 2 and shape[0] == 1:
            return int(shape[1])
        raise ValueError("The Qwen3.5 tokenizer returned an unsupported token shape")
    try:
        return len(tokens)
    except TypeError as error:
        raise ValueError("The Qwen3.5 tokenizer returned invalid tokens") from error
