"""In-process ONNX Runtime GenAI adapter for installed Qwen3.5 models.

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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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
from lumi.model_installation import supported_models

LUMI_RUNTIME_API_VERSION = 1
SUPPORTED_QWEN35_MODELS = frozenset(option.model_id for option in supported_models())

_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_THINKING_CONDITION_RE = re.compile(
    r"if\s+enable_thinking\s+is\s+defined\s+and\s+enable_thinking\s+is\s+true"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([A-Za-z0-9_-]{1,64})>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r"<parameter=([A-Za-z0-9_-]{1,64})>(.*?)</parameter>", re.DOTALL
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class OrtGenAIRuntimeError(RuntimeError):
    """The embedded inference runtime or selected model is unavailable."""


class OrtGenAIProtocolError(OrtGenAIRuntimeError):
    """The model runtime returned output outside the supported Qwen3.5 format."""


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
        object.__setattr__(self, "directory", Path(self.directory).expanduser().resolve())


@dataclass(frozen=True, slots=True)
class OrtGenAIConfig:
    """Supported verified model artifacts and bounded inference/lifecycle settings."""

    model_artifacts: Mapping[str, VerifiedModelArtifact]
    idle_unload_seconds: int = 300
    max_context_tokens: int = 8192
    max_output_tokens: int = 2048
    max_messages: int = 64
    max_tools: int = 32
    max_message_chars: int = 32_000
    max_request_bytes: int = 2_000_000
    max_response_chars: int = 200_000

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
        )
        for name, value, minimum, maximum in bounded_values:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise ValueError(f"{name} must be between {minimum} and {maximum}")


@dataclass(slots=True)
class _LoadedModel:
    model_id: str
    model: Any
    tokenizer: Any
    chat_template: str


class OrtGenAIChatRuntime(ChatRuntime):
    """Run one lazily loaded Qwen3.5 model in the current Orchestrator process.

    Generation runs on a worker thread so synchronous native inference does not block
    the event loop. One generator runs at a time, and a single model stays resident;
    switching models unloads the previous model first. The native dependency is optional
    and imported only when an installed model is first selected.
    """

    def __init__(self, config: OrtGenAIConfig, *, api: Any | None = None) -> None:
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
        """Make the runtime available without importing ORT or loading a model."""

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
        worker = asyncio.create_task(
            asyncio.to_thread(self._complete_sync, request, cancellation)
        )
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

    def _complete_sync(
        self, request: ModelRequest, cancellation: threading.Event
    ) -> ModelResponse:
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
        template = _template_with_thinking(loaded.chat_template, request.thinking)
        try:
            prompt = loaded.tokenizer.apply_chat_template(
                template,
                messages=json.dumps(messages, ensure_ascii=False, separators=(",", ":")),
                tools=(
                    json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
                    if tools
                    else None
                ),
                add_generation_prompt=True,
            )
            if not isinstance(prompt, str):
                raise OrtGenAIProtocolError("The Qwen3.5 chat template returned invalid text")
            encoded = loaded.tokenizer.encode(prompt)
            prompt_tokens = _token_count(encoded)
            context_limit = min(request.context_size, self.config.max_context_tokens)
            output_limit = min(request.output_tokens, self.config.max_output_tokens)
            if prompt_tokens <= 0 or prompt_tokens >= context_limit:
                raise ValueError("The prompt leaves no room for a Qwen3.5 response")
            maximum_length = min(context_limit, prompt_tokens + output_limit)

            params = self._api_module().GeneratorParams(loaded.model)
            params.set_search_options(max_length=maximum_length, do_sample=False)
            generator = self._api_module().Generator(loaded.model, params)
            generator.append_tokens(encoded)
            while not generator.is_done():
                if cancellation.is_set():
                    raise asyncio.CancelledError
                generator.generate_next_token()
            sequence = generator.get_sequence(0)
            text = loaded.tokenizer.decode(sequence[prompt_tokens:])
        except (ValueError, OrtGenAIRuntimeError):
            raise
        except Exception as error:
            raise OrtGenAIRuntimeError("Embedded Qwen3.5 inference failed") from error

        if not isinstance(text, str) or len(text) > self.config.max_response_chars:
            raise OrtGenAIProtocolError("The Qwen3.5 response is malformed or too large")
        content, calls = _parse_model_response(text, tools_by_name, self.config.max_tools)
        return ModelResponse(ChatMessage("assistant", content, tool_calls=calls))

    def _load_model(self, model_id: str) -> _LoadedModel:
        artifact = self.config.model_artifacts[model_id]
        model_dir = _verify_model_artifact(artifact, self._verified_models)
        config_file = model_dir / "genai_config.json"
        api = self._api_module()
        try:
            model = api.Model(str(config_file))
            tokenizer = api.Tokenizer(model)
            template = _read_chat_template(model_dir)
        except OrtGenAIRuntimeError:
            raise
        except Exception as error:
            raise OrtGenAIRuntimeError("The selected Qwen3.5 model could not be loaded") from error
        return _LoadedModel(model_id, model, tokenizer, template)

    def _api_module(self) -> Any:
        if self._api is not None:
            return self._api
        try:
            import onnxruntime_genai
        except (ImportError, OSError) as error:  # pragma: no cover - optional native dependency
            raise OrtGenAIRuntimeError(
                "Lumi's embedded runtime dependency is unavailable in this installation"
            ) from error
        self._api = onnxruntime_genai
        return onnxruntime_genai

    def _unload_sync(self) -> None:
        self._loaded = None
        gc.collect()


def _read_chat_template(model_dir: Path) -> str:
    template_file = model_dir / "chat_template.jinja"
    if template_file.is_file():
        template = template_file.read_text(encoding="utf-8")
    else:
        tokenizer_config = model_dir / "tokenizer_config.json"
        if not tokenizer_config.is_file():
            raise OrtGenAIRuntimeError("The selected Qwen3.5 model has no chat template")
        try:
            config = json.loads(tokenizer_config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise OrtGenAIRuntimeError("The selected Qwen3.5 chat template is invalid") from error
        template = config.get("chat_template") if isinstance(config, Mapping) else None
    if not isinstance(template, str) or not template.strip():
        raise OrtGenAIRuntimeError("The selected Qwen3.5 model has no supported chat template")
    return template


def _verify_model_artifact(
    artifact: VerifiedModelArtifact,
    verified_cache: dict[
        tuple[str, str], tuple[tuple[str, tuple[int, int, int, int, int]], ...]
    ] | None = None,
) -> Path:
    """Check the manifest, file hashes and exact local file layout.

    Model weights are hashed on first load, then that verification is reused while
    every file's identity and change timestamps remain the same. This avoids
    rescanning multi-gigabyte models after an ordinary idle unload/reload cycle.
    """

    model_dir = Path(artifact.directory).resolve()
    if not model_dir.is_dir():
        raise OrtGenAIRuntimeError("The selected Qwen3.5 model files are not installed")
    manifest_path = model_dir / "lumi-model-manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise OrtGenAIRuntimeError("The selected Qwen3.5 model manifest is unavailable") from error
    if hashlib.sha256(manifest_bytes).hexdigest() != artifact.manifest_sha256:
        raise OrtGenAIRuntimeError("The selected Qwen3.5 model manifest changed after verification")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OrtGenAIRuntimeError("The selected Qwen3.5 model manifest is invalid") from error
    if (
        not isinstance(manifest, Mapping)
        or isinstance(manifest.get("schemaVersion"), bool)
        or manifest.get("schemaVersion") != 1
        or manifest.get("format") != "onnxruntime-genai"
        or manifest.get("modelId") != artifact.model_id
    ):
        raise OrtGenAIRuntimeError("The selected Qwen3.5 model identity is invalid")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise OrtGenAIRuntimeError("The selected Qwen3.5 model file manifest is invalid")

    listed_files: dict[str, tuple[Path, str, int]] = {}
    for item in files:
        if not isinstance(item, Mapping):
            raise OrtGenAIRuntimeError("The selected Qwen3.5 model file manifest is invalid")
        relative_name = item.get("path")
        size = item.get("size")
        digest = item.get("sha256")
        if (
            not isinstance(relative_name, str)
            or not relative_name
            or "\\" in relative_name
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or not _SHA256_RE.fullmatch(digest)
        ):
            raise OrtGenAIRuntimeError("The selected Qwen3.5 model file manifest is invalid")
        relative_path = PurePosixPath(relative_name)
        if (
            relative_path.is_absolute()
            or "/".join(relative_path.parts) != relative_name
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            raise OrtGenAIRuntimeError("The selected Qwen3.5 model file path is invalid")
        if relative_name in listed_files:
            raise OrtGenAIRuntimeError("The selected Qwen3.5 model file manifest has duplicates")
        file_path = (model_dir / Path(*relative_path.parts)).resolve()
        try:
            file_path.relative_to(model_dir)
            file_stat = file_path.stat(follow_symlinks=False)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size != size:
                raise OrtGenAIRuntimeError("A verified Qwen3.5 model file is missing or changed")
        except (OSError, ValueError) as error:
            raise OrtGenAIRuntimeError("The selected Qwen3.5 model file path is invalid") from error
        listed_files[relative_name] = (file_path, digest, size)

    if "genai_config.json" not in listed_files or not (
        "chat_template.jinja" in listed_files or "tokenizer_config.json" in listed_files
    ):
        raise OrtGenAIRuntimeError("The selected Qwen3.5 model is missing required runtime assets")

    actual_files: set[str] = set()
    for directory, subdirectories, filenames in os.walk(model_dir, followlinks=False):
        current_dir = Path(directory)
        for subdirectory in subdirectories:
            if (current_dir / subdirectory).is_symlink():
                raise OrtGenAIRuntimeError("The selected Qwen3.5 model contains a symbolic link")
        for filename in filenames:
            path = current_dir / filename
            if path.is_symlink():
                raise OrtGenAIRuntimeError("The selected Qwen3.5 model contains a symbolic link")
            if path == manifest_path:
                continue
            try:
                relative_name = path.relative_to(model_dir).as_posix()
            except ValueError as error:
                raise OrtGenAIRuntimeError(
                    "The selected Qwen3.5 model file path is invalid"
                ) from error
            actual_files.add(relative_name)
    if actual_files != set(listed_files):
        raise OrtGenAIRuntimeError("The selected Qwen3.5 model files do not match their manifest")

    fingerprints = tuple(
        (name, _model_file_fingerprint(path))
        for name, (path, _digest, _size) in sorted(listed_files.items())
    )
    cache_key = (str(model_dir), artifact.manifest_sha256)
    if verified_cache is None or verified_cache.get(cache_key) != fingerprints:
        for name, (path, digest, _size) in listed_files.items():
            if _sha256_file(path) != digest:
                raise OrtGenAIRuntimeError(
                    f"The selected Qwen3.5 model file failed its integrity check: {name}"
                )
        if verified_cache is not None:
            verified_cache[cache_key] = fingerprints
    return model_dir


def _model_file_fingerprint(path: Path) -> tuple[int, int, int, int, int]:
    try:
        file_stat = path.stat(follow_symlinks=False)
    except OSError as error:
        raise OrtGenAIRuntimeError("A verified Qwen3.5 model file is unavailable") from error
    if not stat.S_ISREG(file_stat.st_mode):
        raise OrtGenAIRuntimeError("A verified Qwen3.5 model file is not a regular file")
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
        raise OrtGenAIRuntimeError("A verified Qwen3.5 model file is unavailable") from error
    return digest.hexdigest()


def _template_with_thinking(template: str, thinking: bool) -> str:
    template, replacements = _THINKING_CONDITION_RE.subn(
        "if true" if thinking else "if false", template
    )
    if replacements != 1:
        raise OrtGenAIProtocolError("The Qwen3.5 chat template has an unsupported thinking control")
    return template


def _parse_model_response(
    text: str,
    tools: Mapping[str, ToolDefinition],
    max_tools: int,
) -> tuple[str, tuple[ToolCall, ...]]:
    # Never surface private reasoning, including an unfinished block at the token limit.
    text = _THINK_RE.sub("", text)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]

    matches = list(_TOOL_CALL_RE.finditer(text))
    if (
        text.count("<tool_call>") != len(matches)
        or text.count("</tool_call>") != len(matches)
    ):
        raise OrtGenAIProtocolError("The Qwen3.5 tool-call output is malformed")
    if len(matches) > max_tools:
        raise OrtGenAIProtocolError("Qwen3.5 returned too many tool calls")
    calls: list[ToolCall] = []
    pieces: list[str] = []
    previous_end = 0
    for match in matches:
        pieces.append(text[previous_end : match.start()])
        name = match.group(1)
        body = match.group(2)
        arguments: dict[str, Any] = {}
        cursor = 0
        for parameter in _PARAMETER_RE.finditer(body):
            if body[cursor : parameter.start()].strip():
                raise OrtGenAIProtocolError("The Qwen3.5 tool-call arguments are malformed")
            key = parameter.group(1)
            if key in arguments:
                raise OrtGenAIProtocolError("Qwen3.5 returned a duplicate tool argument")
            value = parameter.group(2).strip("\r\n")
            arguments[key] = _parse_argument(value, tools.get(name), key)
            cursor = parameter.end()
        if body[cursor:].strip():
            raise OrtGenAIProtocolError("The Qwen3.5 tool-call arguments are malformed")
        tool = tools.get(name)
        if tool is not None:
            required = tool.parameters.get("required", [])
            if isinstance(required, list) and any(key not in arguments for key in required):
                raise OrtGenAIProtocolError("The Qwen3.5 tool call is missing a required argument")
        calls.append(ToolCall(uuid4().hex, name, arguments))
        previous_end = match.end()
    pieces.append(text[previous_end:])
    content = "".join(pieces).strip()
    return content, tuple(calls)


def _parse_argument(value: str, tool: ToolDefinition | None, name: str) -> Any:
    if tool is not None:
        properties = tool.parameters.get("properties", {})
        schema = properties.get(name, {}) if isinstance(properties, Mapping) else {}
        expected_type = schema.get("type") if isinstance(schema, Mapping) else None
        if expected_type == "string":
            return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


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
