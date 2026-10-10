"""In-process llama.cpp adapter for installed Qwen3.5 GGUF models.

The native runtime is imported lazily, so Lumi's core package stays lightweight.
Orchestrator installs this Lumi release only when an administrator enables the
integration, then passes the managed model directories to this adapter.
"""

from __future__ import annotations

import asyncio
import ctypes
import gc
import hashlib
import json
import logging
import os
import re
import stat
import struct
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType, MappingProxyType, SimpleNamespace
from typing import Any
from uuid import uuid4

from lumi.contracts import (
    ChatMessage,
    ChatRuntime,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ToolCall,
    ToolDefinition,
)
from lumi.model_installation import (
    GGUF_FORMAT,
    MODEL_MANIFEST_SCHEMA_VERSION,
    _require_spec,
    supported_models,
)
from lumi.runtime.acceleration import (
    SUPPORTED_ACCELERATION_MODES,
    AccelerationChoice,
    AccelerationMode,
    GpuDevice,
    _cpu_choice,
    choose_acceleration,
    detect_gpu_devices,
)
from lumi.runtime.streaming import VisibleTextFilter

LUMI_RUNTIME_API_VERSION = 1
SUPPORTED_QWEN35_MODELS = frozenset(option.model_id for option in supported_models())

_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([A-Za-z0-9_-]{1,64})>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_TOOL_CONTROL_TOKEN_RE = re.compile(r"<\|(?:tool_[^|>\r\n]*|function_call[^|>\r\n]*)\|>")
_PARAMETER_RE = re.compile(r"<parameter=([A-Za-z0-9_-]{1,64})>(.*?)</parameter>", re.DOTALL)
_CHANNEL_RE = re.compile(r"<\|channel\|>(analysis|final|commentary|summary|justify|confidence)\b")
_BACKEND_REGISTRY_LOCK = threading.Lock()
_LOADED_BACKEND_REGISTRIES: dict[int, Any] = {}
logger = logging.getLogger(__name__)


class LlamaCppRuntimeError(RuntimeError):
    """The embedded inference runtime or selected model is unavailable."""


class LlamaCppProtocolError(LlamaCppRuntimeError):
    """The model runtime returned output outside the supported Qwen3.5 format."""


class LlamaCppBackendInitializationError(LlamaCppRuntimeError):
    """The process-wide llama.cpp backend registry could not be initialized."""


class _CombinedStoppingCriteria(list[Callable[[Any, Any], bool]]):
    """Combine the chat-template and cancellation checks in llama.cpp's callable API."""

    def __call__(self, tokens: Any, logits: Any) -> bool:
        return any(criterion(tokens, logits) for criterion in self)


class _PreparedChatFormatter:
    """Return the already validated GGUF chat template for a binding request."""

    def __init__(self, response: Any) -> None:
        self._response = response

    def __call__(self, **_kwargs: Any) -> Any:
        return self._response


@dataclass(frozen=True, slots=True)
class _StreamWorkerResult:
    response: ModelResponse | None = None
    error: BaseException | None = None


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
    n_gpu_layers: int | None = None
    acceleration_mode: AccelerationMode = "automatic"
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
        if self.n_gpu_layers is not None and (
            isinstance(self.n_gpu_layers, bool)
            or not isinstance(self.n_gpu_layers, int)
            or not 0 <= self.n_gpu_layers <= 256
        ):
            raise ValueError("n_gpu_layers must be between 0 and 256")
        if self.acceleration_mode not in SUPPORTED_ACCELERATION_MODES:
            raise ValueError("acceleration_mode must be automatic, cpu_only, or gpu_preferred")
        if self.n_ubatch > self.n_batch:
            raise ValueError("n_ubatch cannot exceed n_batch")


@dataclass(slots=True)
class _LoadedModel:
    model_id: str
    model: Any
    chat_template: str
    eos_token: str
    bos_token: str
    acceleration: AccelerationChoice
    load_duration_ns: int


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
        self._acceleration = AccelerationChoice(
            backend="cpu",
            device="CPU",
            offloaded_layers=0,
            total_layers=0,
            device_memory_free_bytes=0,
            device_memory_total_bytes=0,
        )
        self._acceleration_state = "not_loaded"
        self._forced_cpu_reason: str | None = None

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

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        """Stream visible assistant-text deltas followed by one normalized response.

        Native inference remains on a worker thread. Cancellation is cooperative at
        token boundaries, and the single-flight slot remains held until that worker
        exits so an abandoned native iterator cannot overlap a later request.
        """

        if not self._opened or self._closing or self._closed:
            raise RuntimeError("Open the Lumi runtime before making requests")
        self._validate_request(request)
        await self._slot.acquire()
        if not self._opened or self._closing or self._closed:
            self._slot.release()
            raise RuntimeError("The Lumi runtime is not available")

        cancellation = threading.Event()
        loop = asyncio.get_running_loop()
        events: asyncio.Queue[ModelStreamEvent | _StreamWorkerResult] = asyncio.Queue()

        def publish(event: ModelStreamEvent) -> None:
            loop.call_soon_threadsafe(events.put_nowait, event)

        def run_worker() -> None:
            try:
                response = self._generate_sync(request, cancellation, publish)
            except BaseException as error:
                loop.call_soon_threadsafe(events.put_nowait, _StreamWorkerResult(error=error))
            else:
                loop.call_soon_threadsafe(
                    events.put_nowait,
                    _StreamWorkerResult(response=response),
                )

        worker = asyncio.create_task(asyncio.to_thread(run_worker))
        release_in_callback = False
        try:
            while True:
                item = await events.get()
                if isinstance(item, _StreamWorkerResult):
                    if item.error is not None:
                        raise item.error
                    assert item.response is not None
                    yield ModelStreamEvent(kind="complete", response=item.response)
                    return
                yield item
        except asyncio.CancelledError:
            cancellation.set()
            release_in_callback = True
            worker.add_done_callback(self._finish_cancelled_worker)
            raise
        finally:
            if not release_in_callback:
                if not worker.done():
                    # Closing an async generator early is also cancellation. Keep
                    # the single-flight lock until its native iterator exits.
                    cancellation.set()
                    worker.add_done_callback(self._finish_cancelled_worker)
                else:
                    self._resident_model_id = (
                        self._loaded.model_id if self._loaded is not None else None
                    )
                    self._last_used = time.monotonic()
                    self._slot.release()

    def acceleration_status(self) -> dict[str, Any]:
        """Return stable JSON-safe acceleration state for an admin status view."""

        acceleration = self._acceleration
        return {
            "state": self._acceleration_state,
            "selectedBackend": acceleration.backend,
            "selectedDevice": acceleration.device,
            "offloadedLayers": acceleration.offloaded_layers,
            "totalLayers": acceleration.total_layers,
            "fallbackReason": acceleration.fallback_reason,
        }

    def _finish_cancelled_worker(self, task: asyncio.Task[Any]) -> None:
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
        return self._generate_sync(request, cancellation, emit_delta=None)

    def _generate_sync(
        self,
        request: ModelRequest,
        cancellation: threading.Event,
        emit_delta: Callable[[ModelStreamEvent], None] | None,
    ) -> ModelResponse:
        if cancellation.is_set():
            raise asyncio.CancelledError
        loaded = self._loaded
        load_duration_ns = 0
        if loaded is None or loaded.model_id != request.model:
            # Drop this local reference before clearing the runtime-owned reference;
            # otherwise the old native weights remain alive during the next load.
            loaded = None
            self._unload_sync()
            self._loaded = self._load_model(request.model)
            loaded = self._loaded
            load_duration_ns = loaded.load_duration_ns
        if cancellation.is_set():
            # A cold native model load cannot be interrupted safely. Do not let a
            # disconnected request continue into prompt preparation or inference.
            raise asyncio.CancelledError
        assert loaded is not None

        tools_by_name = {tool.name: tool for tool in request.tools}
        messages = [self._serialize_message(message) for message in request.messages]
        tools = [self._serialize_tool(tool) for tool in request.tools]
        streamed_visible = False
        prompt_duration_ns: int | None = None
        generation_duration_ns: int | None = None
        total_duration_ns = 0
        result: Any | None = None
        try:
            api = self._api_module()
            formatter = api.Jinja2ChatFormatter(
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
            prepared_response = SimpleNamespace(
                prompt=prompt,
                stop=stop,
                stopping_criteria=stopping_criteria,
                added_special=added_special,
            )
            handler_factory = getattr(api, "chat_formatter_to_chat_completion_handler", None)
            if not callable(handler_factory):
                raise LlamaCppRuntimeError(
                    "Lumi's embedded chat streaming interface is unavailable"
                )
            chat_handler = handler_factory(_PreparedChatFormatter(prepared_response))
            old_chat_handler = getattr(loaded.model, "chat_handler", None)
            started_ns = time.perf_counter_ns()
            loaded.model.chat_handler = chat_handler
            try:
                if cancellation.is_set():
                    raise asyncio.CancelledError
                result = loaded.model.create_chat_completion(
                    messages=messages,
                    tools=tools or None,
                    temperature=0.0,
                    max_tokens=output_limit,
                    stream=emit_delta is not None,
                )
            finally:
                loaded.model.chat_handler = old_chat_handler

            text_parts: list[str] = []
            total_chars = 0
            structured_tool_parts: dict[int, dict[str, Any]] = {}
            filter_visible = VisibleTextFilter(
                require_final_channel=False,
                initially_thinking=request.thinking,
            )
            if emit_delta is None:
                choices = result.get("choices") if isinstance(result, Mapping) else None
                choice = choices[0] if isinstance(choices, list) and choices else None
                message = choice.get("message") if isinstance(choice, Mapping) else None
                if not isinstance(message, Mapping):
                    raise LlamaCppProtocolError(
                        "The Qwen3.5 chat response returned a malformed completion"
                    )
                response_text = message.get("content")
                if response_text is None:
                    response_text = ""
                if not isinstance(response_text, str):
                    raise LlamaCppProtocolError(
                        "The Qwen3.5 chat response returned malformed text"
                    )
                total_chars = len(response_text)
                if total_chars > self.config.max_response_chars:
                    raise LlamaCppProtocolError("The Qwen3.5 response is too large")
                text = response_text
                structured_calls = message.get("tool_calls")
                if structured_calls:
                    calls = _parse_chat_tool_calls(
                        structured_calls,
                        tools_by_name,
                        self.config.max_tools,
                    )
                    content = ""
                elif message.get("function_call"):
                    raise LlamaCppProtocolError(
                        "The Qwen3.5 response used an unsupported legacy function call"
                    )
                else:
                    content, calls = _parse_model_response(
                        text,
                        tools_by_name,
                        self.config.max_tools,
                        initially_thinking=request.thinking,
                    )
            else:
                first_generation_token_ns: int | None = None
                structured_control = False

                def discard_partial_turn() -> None:
                    nonlocal streamed_visible
                    if streamed_visible:
                        emit_delta(ModelStreamEvent(kind="reset", reason="intermediate"))
                        streamed_visible = False

                for chunk in result:
                    if cancellation.is_set():
                        raise asyncio.CancelledError
                    choices = chunk.get("choices") if isinstance(chunk, Mapping) else None
                    choice = choices[0] if isinstance(choices, list) and choices else None
                    delta = choice.get("delta") if isinstance(choice, Mapping) else None
                    if not isinstance(delta, Mapping):
                        raise LlamaCppProtocolError(
                            "The Qwen3.5 chat stream returned a malformed chunk"
                        )
                    tool_calls = delta.get("tool_calls")
                    if tool_calls:
                        structured_control = True
                        discard_partial_turn()
                        _collect_chat_tool_call_deltas(
                            tool_calls,
                            structured_tool_parts,
                            self.config.max_tools,
                            self.config.max_response_chars,
                        )
                    if delta.get("function_call"):
                        structured_control = True
                        discard_partial_turn()
                        raise LlamaCppProtocolError(
                            "The Qwen3.5 stream used an unsupported legacy function call"
                        )
                    if structured_control:
                        continue

                    # Read exactly `delta.content`; reasoning_content and all other
                    # provider-specific fields remain private and are never joined.
                    chunk_text = delta.get("content")
                    if chunk_text is None:
                        continue
                    if not isinstance(chunk_text, str):
                        raise LlamaCppProtocolError(
                            "The Qwen3.5 chat stream returned malformed text"
                        )
                    if chunk_text and first_generation_token_ns is None:
                        first_generation_token_ns = time.perf_counter_ns()
                        prompt_duration_ns = first_generation_token_ns - started_ns
                    total_chars += len(chunk_text)
                    if total_chars > self.config.max_response_chars:
                        raise LlamaCppProtocolError("The Qwen3.5 response is too large")
                    text_parts.append(chunk_text)
                    visible = filter_visible.feed(chunk_text)
                    if visible:
                        streamed_visible = True
                        emit_delta(ModelStreamEvent(kind="delta", text=visible))
                    if filter_visible.consume_reset_required():
                        discard_partial_turn()
                    if filter_visible.suppressed:
                        discard_partial_turn()
                final_visible = filter_visible.finish()
                if final_visible:
                    streamed_visible = True
                    emit_delta(ModelStreamEvent(kind="delta", text=final_visible))
                if filter_visible.consume_reset_required():
                    discard_partial_turn()
                finished_ns = time.perf_counter_ns()
                total_duration_ns = finished_ns - started_ns
                if first_generation_token_ns is not None:
                    generation_duration_ns = max(0, finished_ns - first_generation_token_ns)
                text = "".join(text_parts)
                calls = (
                    _parse_chat_tool_call_parts(
                        structured_tool_parts,
                        tools_by_name,
                        self.config.max_tools,
                    )
                    if structured_tool_parts
                    else ()
                )
                if calls:
                    content = ""
                else:
                    content, calls = _parse_model_response(
                        text,
                        tools_by_name,
                        self.config.max_tools,
                        initially_thinking=request.thinking,
                    )
                if calls:
                    # Tool rounds are intermediate control flow. The agent will
                    # publish the final post-tool answer as the canonical stream
                    # completion after it clears these deltas.
                    content = ""
                    discard_partial_turn()
            if cancellation.is_set():
                raise asyncio.CancelledError
        except asyncio.CancelledError:
            raise
        except (ValueError, LlamaCppRuntimeError):
            raise
        except Exception as error:
            if loaded.acceleration.uses_gpu:
                fallback_reason = "GPU inference failed; CPU will be used for later turns."
                # Drop traceback frames that retain the failed model through the
                # synchronous binding iterator before attempting a CPU reload.
                error.__traceback__ = None
                self._forced_cpu_reason = fallback_reason
                self._acceleration = _cpu_choice(loaded.acceleration.total_layers, fallback_reason)
                self._acceleration_state = "cpu_fallback"
                close_result = getattr(result, "close", None)
                if callable(close_result):
                    try:
                        close_result()
                    except Exception:
                        pass
                result = None
                self._loaded = None
                loaded = None
                gc.collect()
                if streamed_visible and emit_delta is not None:
                    emit_delta(ModelStreamEvent(kind="reset", reason="cpu_fallback"))
                    streamed_visible = False
                if not cancellation.is_set():
                    self._loaded = self._load_model(request.model)
                    return self._generate_sync(request, cancellation, emit_delta)
                raise LlamaCppRuntimeError(
                    "Embedded Qwen3.5 GPU inference failed; CPU fallback is ready"
                ) from error
            raise LlamaCppRuntimeError("Embedded Qwen3.5 inference failed") from error

        if not isinstance(text, str) or len(text) > self.config.max_response_chars:
            raise LlamaCppProtocolError("The Qwen3.5 response is malformed or too large")
        return ModelResponse(
            ChatMessage("assistant", content, tool_calls=calls),
            prompt_tokens=prompt_tokens,
            prompt_duration_ns=prompt_duration_ns,
            generation_duration_ns=generation_duration_ns,
            load_duration_ns=load_duration_ns,
            total_duration_ns=total_duration_ns,
        )

    def _load_model(self, model_id: str) -> _LoadedModel:
        load_started_ns = time.perf_counter_ns()
        artifact = self.config.model_artifacts[model_id]
        model_dir = _verify_model_artifact(artifact, self._verified_models)
        spec = _require_spec(model_id)
        model_path = model_dir / spec.gguf_filename
        api = self._api_module()
        try:
            total_layers = _read_gguf_layer_count(model_path)
        except LlamaCppRuntimeError:
            total_layers = 0
        try:
            _initialize_backend_registry(api)
        except LlamaCppBackendInitializationError as error:
            self._acceleration = _cpu_choice(total_layers, str(error))
            self._acceleration_state = "unavailable"
            raise
        if self.config.acceleration_mode == "cpu_only" or self._forced_cpu_reason is not None:
            # Avoid GPU enumeration during device selection while still preparing
            # llama-cpp-python's process-wide backend registry exactly once.
            choice = _cpu_choice(total_layers, self._forced_cpu_reason)
        else:
            devices = _available_gpu_devices(api)
            choice = choose_acceleration(
                self.config.acceleration_mode,
                devices,
                model_size_bytes=spec.download_size_bytes,
                total_layers=total_layers,
                max_layers=self.config.n_gpu_layers,
            )
        self._acceleration = choice
        self._acceleration_state = "cpu_fallback" if choice.fallback_reason else "ready"

        def load_native(n_gpu_layers: int, native_device_name: str | None) -> Any:
            options = dict(
                model_path=str(model_path),
                n_ctx=self.config.n_ctx,
                n_batch=self.config.n_batch,
                n_ubatch=self.config.n_ubatch,
                n_threads=self.config.n_threads,
                n_threads_batch=self.config.n_threads_batch,
                n_gpu_layers=n_gpu_layers,
                use_mmap=self.config.use_mmap,
                use_mlock=self.config.use_mlock,
                verbose=False,
            )
            if n_gpu_layers == 0:
                # Keep all model operations and cache placement on the CPU.
                options["offload_kqv"] = False
                options["op_offload"] = False
            return _load_llama_with_device(api, options, native_device_name or "CPU")

        try:
            model = load_native(choice.offloaded_layers, choice.native_device_name)
        except Exception as error:
            if not choice.uses_gpu:
                self._acceleration_state = "unavailable"
                self._acceleration = _cpu_choice(total_layers, "The model could not be loaded.")
                raise LlamaCppRuntimeError(
                    "The selected Qwen3.5 model could not be loaded"
                ) from error
            fallback_reason = "The selected GPU backend could not initialize; using CPU."
            self._forced_cpu_reason = fallback_reason
            self._acceleration = _cpu_choice(total_layers, fallback_reason)
            self._acceleration_state = "cpu_fallback"
            try:
                model = load_native(0, "CPU")
            except Exception as cpu_error:
                self._acceleration_state = "unavailable"
                raise LlamaCppRuntimeError(
                    "The selected Qwen3.5 model could not be loaded"
                ) from cpu_error
        try:
            template = _read_chat_template(getattr(model, "metadata", None))
            eos_token = _model_token_text(model, "token_eos")
            bos_token = _model_token_text(model, "token_bos")
            metadata_layers = _metadata_layer_count(getattr(model, "metadata", None))
            if metadata_layers is not None:
                total_layers = metadata_layers
                if not self._acceleration.uses_gpu:
                    self._acceleration = _cpu_choice(
                        total_layers,
                        self._acceleration.fallback_reason,
                    )
            self._acceleration_state = (
                "cpu_fallback" if self._acceleration.fallback_reason else "ready"
            )
        except LlamaCppRuntimeError:
            raise
        except Exception as error:
            raise LlamaCppRuntimeError("The selected Qwen3.5 model could not be loaded") from error
        return _LoadedModel(
            model_id,
            model,
            template,
            eos_token,
            bos_token,
            self._acceleration,
            time.perf_counter_ns() - load_started_ns,
        )

    def _api_module(self) -> Any:
        if self._api is not None:
            return self._api
        try:
            from types import SimpleNamespace

            import llama_cpp
            import llama_cpp._ggml as llama_ggml
            from llama_cpp import llama_cpp as llama_cpp_bindings
            from llama_cpp._ctypes_extensions import load_shared_library
            from llama_cpp.llama_chat_format import (
                Jinja2ChatFormatter,
                chat_formatter_to_chat_completion_handler,
            )
        except (ImportError, OSError) as error:  # pragma: no cover - optional native dependency
            raise LlamaCppRuntimeError(
                "Lumi's embedded runtime dependency is unavailable in this installation"
            ) from error
        library_directory = Path(llama_cpp.__file__).resolve().parent / "lib"
        self._api = SimpleNamespace(
            Llama=llama_cpp.Llama,
            Jinja2ChatFormatter=Jinja2ChatFormatter,
            chat_formatter_to_chat_completion_handler=chat_formatter_to_chat_completion_handler,
            llama_cpp=llama_cpp,
            llama_cpp_bindings=llama_cpp_bindings,
            ggml=llama_ggml.libggml,
            ggml_base=load_shared_library("ggml-base", library_directory),
            ggml_backend_directory=library_directory,
        )
        return self._api

    def _unload_sync(self) -> None:
        self._loaded = None
        gc.collect()
        if self._acceleration_state != "unavailable":
            self._acceleration_state = "not_loaded"
            self._acceleration = _cpu_choice(0, None)


def _load_llama_with_device(api: Any, options: dict[str, Any], device_name: str) -> Any:
    """Restrict llama.cpp model loading to one selected backend device.

    llama-cpp-python 0.3.35 exposes ``llama_model_params.devices`` in its ctypes
    structure but omits it from the high-level ``Llama`` constructor. Install a
    constructor-local binding proxy so the list is applied before native model
    loading without changing the process-wide binding module.
    """

    bindings = getattr(api, "llama_cpp_bindings", None)
    default_params = getattr(bindings, "llama_model_default_params", None)
    llama_type = getattr(api, "Llama", None)
    original_init = getattr(llama_type, "__init__", None)
    original_globals = getattr(original_init, "__globals__", None)
    if (
        not callable(default_params)
        or not isinstance(llama_type, type)
        or not isinstance(original_globals, dict)
        or original_globals.get("llama_cpp") is not bindings
    ):
        raise LlamaCppRuntimeError("The embedded runtime cannot restrict model devices")

    device_lists: list[Any] = []

    def default_params_for_device() -> Any:
        params = default_params()
        device_handle = _backend_device_handle(api, device_name)
        device_list = (ctypes.c_void_p * 2)(device_handle, None)
        params.devices = ctypes.cast(device_list, ctypes.c_void_p).value
        device_lists.append(device_list)
        return params

    class DevicePinnedBindings:
        def __getattr__(self, name: str) -> Any:
            if name == "llama_model_default_params":
                return default_params_for_device
            return getattr(bindings, name)

    init_globals = dict(original_globals)
    init_globals["llama_cpp"] = DevicePinnedBindings()
    device_pinned_init = FunctionType(
        original_init.__code__,
        init_globals,
        name=original_init.__name__,
        argdefs=original_init.__defaults__,
        closure=original_init.__closure__,
    )
    device_pinned_init.__kwdefaults__ = original_init.__kwdefaults__
    device_pinned_init.__annotations__ = original_init.__annotations__.copy()
    device_pinned_type = type(
        f"_LumiDevicePinned{llama_type.__name__}",
        (llama_type,),
        {"__init__": device_pinned_init},
    )
    model = device_pinned_type(**options)
    model._lumi_device_lists = tuple(device_lists)
    return model


def _backend_device_handle(api: Any, device_name: str) -> int:
    libraries = tuple(
        library
        for library in (getattr(api, "ggml", None), getattr(api, "ggml_base", None))
        if library is not None
    )
    for library in libraries:
        try:
            get_device = library.ggml_backend_dev_by_name
        except AttributeError:
            continue
        get_device.restype = ctypes.c_void_p
        get_device.argtypes = [ctypes.c_char_p]
        handle = get_device(device_name.encode("utf-8"))
        if isinstance(handle, ctypes.c_void_p):
            handle = handle.value
        if handle:
            return int(handle)
    raise LlamaCppRuntimeError(f"The selected llama.cpp device is unavailable: {device_name}")


def _available_gpu_devices(api: Any) -> tuple[GpuDevice, ...]:
    """Prefer a binding-provided device hook, then use ggml's stable C API."""

    list_devices = getattr(api, "list_gpu_devices", None)
    if callable(list_devices):
        try:
            devices = tuple(list_devices())
            return tuple(device for device in devices if isinstance(device, GpuDevice))
        except Exception:
            return ()
    return detect_gpu_devices(api)


def _initialize_backend_registry(api: Any) -> None:
    """Initialize llama.cpp's process-wide backend registry once, with a clear error.

    llama-cpp-python 0.3.35 calls ``llama_backend_init`` inside ``Llama.__init__``.
    Lumi initializes it before probing devices so a Python-level failure is not
    mistaken for a normal CPU-only host. After success, set the wrapper's matching
    class flag to avoid repeating the same global initialization in its constructor.
    """

    llama_type = getattr(api, "Llama", None)
    backend_initialized_attribute = "_Llama__backend_initialized"
    if (
        isinstance(llama_type, type)
        and getattr(llama_type, backend_initialized_attribute, False)
    ):
        return

    native_module = getattr(api, "llama_cpp", None)
    backend_init = getattr(native_module, "llama_backend_init", None)
    try:
        backend_directory = getattr(api, "ggml_backend_directory", None)
        dependency_path = (
            _windows_backend_dependencies_on_path(Path(backend_directory).resolve())
            if sys.platform == "win32" and backend_directory is not None
            else nullcontext()
        )
        with dependency_path:
            _load_packaged_backend_plugins(api)
            if callable(backend_init):
                backend_init()
    except Exception as error:
        raise LlamaCppBackendInitializationError(
            "The llama.cpp backend registry could not initialize; CPU fallback is "
            "unavailable in this process"
        ) from error
    if isinstance(llama_type, type):
        setattr(llama_type, backend_initialized_attribute, True)


def _load_packaged_backend_plugins(api: Any) -> None:
    """Load dynamic backends from llama-cpp-python's installed package directory.

    The wheel keeps CUDA/Vulkan plugins beside its ggml libraries. llama.cpp's
    default backend search checks the executable and working directories, which
    do not reliably include Python's site-packages directory.
    """

    library = getattr(api, "ggml", None)
    backend_directory = getattr(api, "ggml_backend_directory", None)
    if library is None or backend_directory is None:
        return

    library_id = id(library)
    with _BACKEND_REGISTRY_LOCK:
        if _LOADED_BACKEND_REGISTRIES.get(library_id) is library:
            return
        try:
            directory = Path(backend_directory).resolve()
            if not directory.is_dir():
                raise OSError("llama.cpp package library directory is unavailable")
            load_all = library.ggml_backend_load_all_from_path
            load_all.argtypes = [ctypes.c_char_p]
            load_all.restype = None
            load_all(os.fsencode(directory))
        except Exception:
            logger.warning(
                "Could not load package-local llama.cpp backends; CPU fallback remains available",
                exc_info=True,
            )
            return
        _LOADED_BACKEND_REGISTRIES[library_id] = library


@contextmanager
def _windows_backend_dependencies_on_path(
    backend_directory: Path,
) -> Iterator[None]:
    """Temporarily expose delvewheel DLLs while llama.cpp registers backends.

    llama.cpp uses ``LoadLibraryW`` for dynamic backend plugins. That lookup does
    not search the backend DLL's own directory for its imported dependencies, so
    the repaired wheel's package-local ``llama_cpp_python.libs`` directory must be
    on PATH while the plugin DLLs and their dependencies are loaded. Restoring
    PATH on exit avoids leaking this package-private directory to child processes.
    """

    dependency_directory = backend_directory.parent.parent / "llama_cpp_python.libs"
    if not dependency_directory.is_dir():
        yield
        return

    dependency_path = os.fspath(dependency_directory.resolve())
    normalized_dependency_path = os.path.normcase(os.path.abspath(dependency_path))
    current_entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    if any(
        os.path.normcase(os.path.abspath(entry)) == normalized_dependency_path
        for entry in current_entries
    ):
        yield
        return

    original_path = os.environ.get("PATH")
    os.environ["PATH"] = os.pathsep.join([dependency_path, *current_entries])
    try:
        yield
    finally:
        if original_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = original_path


def _metadata_layer_count(metadata: Any) -> int | None:
    if not isinstance(metadata, Mapping):
        return None
    values = [
        value
        for key, value in metadata.items()
        if isinstance(key, str) and key.endswith(".block_count")
    ]
    if len(values) != 1:
        return None
    value = values[0]
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 512:
        return None
    return value


def _read_gguf_layer_count(path: Path) -> int:
    """Read the pinned GGUF's architecture block count without loading its weights."""

    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            if stream.read(4) != b"GGUF":
                raise ValueError("invalid GGUF header")
            version, = _read_gguf_struct(stream, "<I")
            if version not in {1, 2, 3}:
                raise ValueError("unsupported GGUF version")
            _tensor_count, metadata_count = _read_gguf_struct(stream, "<QQ")
            if metadata_count > 100_000:
                raise ValueError("excessive GGUF metadata count")
            block_count: int | None = None
            for _ in range(metadata_count):
                key = _read_gguf_string(stream, size, decode=True)
                value_type, = _read_gguf_struct(stream, "<I")
                if key.endswith(".block_count") and value_type in {4, 10}:
                    block_count, = _read_gguf_struct(
                        stream,
                        "<I" if value_type == 4 else "<Q",
                    )
                else:
                    _skip_gguf_value(stream, size, value_type)
            if isinstance(block_count, bool) or not isinstance(block_count, int):
                raise ValueError("GGUF block count is missing")
            if not 1 <= block_count <= 512:
                raise ValueError("GGUF block count is outside the supported range")
            return block_count
    except (OSError, EOFError, UnicodeDecodeError, ValueError, struct.error) as error:
        raise LlamaCppRuntimeError(
            "The selected Qwen3.5 model layer layout could not be measured safely"
        ) from error


def _read_gguf_struct(stream: Any, format_string: str) -> tuple[int, ...]:
    size = struct.calcsize(format_string)
    payload = stream.read(size)
    if len(payload) != size:
        raise EOFError("truncated GGUF metadata")
    return struct.unpack(format_string, payload)


def _read_gguf_string(stream: Any, file_size: int, *, decode: bool) -> str:
    length, = _read_gguf_struct(stream, "<Q")
    position = stream.tell()
    if length > file_size - position or (decode and length > 65_535):
        raise ValueError("invalid GGUF string length")
    payload = stream.read(length) if decode else b""
    if not decode:
        stream.seek(length, os.SEEK_CUR)
        return ""
    if len(payload) != length:
        raise EOFError("truncated GGUF string")
    return payload.decode("ascii")


def _skip_gguf_value(stream: Any, file_size: int, value_type: int, *, depth: int = 0) -> None:
    scalar_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if value_type in scalar_sizes:
        _seek_gguf(stream, file_size, scalar_sizes[value_type])
        return
    if value_type == 8:
        _read_gguf_string(stream, file_size, decode=False)
        return
    if value_type != 9 or depth >= 8:
        raise ValueError("unsupported GGUF metadata value")
    item_type, = _read_gguf_struct(stream, "<I")
    count, = _read_gguf_struct(stream, "<Q")
    if count > 1_000_000:
        raise ValueError("excessive GGUF array size")
    for _ in range(count):
        _skip_gguf_value(stream, file_size, item_type, depth=depth + 1)


def _seek_gguf(stream: Any, file_size: int, amount: int) -> None:
    if amount < 0 or stream.tell() + amount > file_size:
        raise EOFError("truncated GGUF metadata")
    stream.seek(amount, os.SEEK_CUR)


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
    *,
    initially_thinking: bool = False,
) -> tuple[str, tuple[ToolCall, ...]]:
    if _TOOL_CONTROL_TOKEN_RE.search(text):
        raise LlamaCppProtocolError("Qwen3.5 returned unsupported tool-control tokens")
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

    raw_content = "".join(pieces)
    has_channels = bool(_CHANNEL_RE.search(raw_content)) or "<|channel|>" in raw_content
    filter_visible = VisibleTextFilter(
        require_final_channel=has_channels,
        initially_thinking=initially_thinking,
    )
    content = filter_visible.feed(raw_content) + filter_visible.finish()
    # The agent will execute tool calls before accepting a user-facing answer.
    # Keeping sanitized preamble text here preserves the non-streaming response
    # contract; streaming clients clear any such text on the intermediate reset.
    return content, tuple(calls)


def _collect_chat_tool_call_deltas(
    deltas: Any,
    collected: dict[int, dict[str, Any]],
    max_tools: int,
    max_chars: int,
) -> None:
    if not isinstance(deltas, list) or len(deltas) > max_tools:
        raise LlamaCppProtocolError("The Qwen3.5 stream returned malformed tool calls")
    for delta in deltas:
        if not isinstance(delta, Mapping):
            raise LlamaCppProtocolError("The Qwen3.5 stream returned malformed tool calls")
        index = delta.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < max_tools:
            raise LlamaCppProtocolError("The Qwen3.5 stream returned an invalid tool index")
        entry = collected.setdefault(index, {"id": "", "name": "", "arguments": ""})
        call_id = delta.get("id")
        if call_id is not None:
            if not isinstance(call_id, str) or len(call_id) > 128:
                raise LlamaCppProtocolError("The Qwen3.5 stream returned an invalid tool ID")
            entry["id"] = call_id
        function = delta.get("function")
        if function is None:
            continue
        if not isinstance(function, Mapping):
            raise LlamaCppProtocolError("The Qwen3.5 stream returned malformed tool calls")
        name = function.get("name")
        if name is not None:
            if not isinstance(name, str) or len(name) > 64:
                raise LlamaCppProtocolError("The Qwen3.5 stream returned an invalid tool name")
            entry["name"] += name
        arguments = function.get("arguments")
        if arguments is not None:
            if not isinstance(arguments, str):
                raise LlamaCppProtocolError(
                    "The Qwen3.5 stream returned malformed tool arguments"
                )
            entry["arguments"] += arguments
        if sum(len(value["arguments"]) for value in collected.values()) > max_chars:
            raise LlamaCppProtocolError("The Qwen3.5 tool response is too large")


def _parse_chat_tool_calls(
    value: Any,
    tools: Mapping[str, ToolDefinition],
    max_tools: int,
) -> tuple[ToolCall, ...]:
    if not isinstance(value, list) or len(value) > max_tools:
        raise LlamaCppProtocolError("The Qwen3.5 chat response returned malformed tool calls")
    collected: dict[int, dict[str, Any]] = {}
    for index, call in enumerate(value):
        if not isinstance(call, Mapping):
            raise LlamaCppProtocolError("The Qwen3.5 chat response returned malformed tool calls")
        function = call.get("function")
        if not isinstance(function, Mapping):
            raise LlamaCppProtocolError("The Qwen3.5 chat response returned malformed tool calls")
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, dict):
            try:
                arguments = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise LlamaCppProtocolError(
                    "The Qwen3.5 chat response returned malformed tool arguments"
                ) from error
        if not isinstance(arguments, str) or len(arguments) > 200_000:
            raise LlamaCppProtocolError(
                "The Qwen3.5 chat response returned malformed tool arguments"
            )
        call_id = call.get("id")
        collected[index] = {
            "id": call_id if isinstance(call_id, str) else "",
            "name": function.get("name", ""),
            "arguments": arguments,
        }
    return _parse_chat_tool_call_parts(collected, tools, max_tools)


def _parse_chat_tool_call_parts(
    collected: Mapping[int, Mapping[str, Any]],
    tools: Mapping[str, ToolDefinition],
    max_tools: int,
) -> tuple[ToolCall, ...]:
    if len(collected) > max_tools:
        raise LlamaCppProtocolError("Qwen3.5 returned too many tool calls")
    calls: list[ToolCall] = []
    for index in sorted(collected):
        entry = collected[index]
        name = entry.get("name")
        tool = tools.get(name) if isinstance(name, str) else None
        if tool is None or not tool.read_only:
            raise LlamaCppProtocolError("Qwen3.5 returned an unregistered tool call")
        raw_arguments = entry.get("arguments")
        if raw_arguments == "":
            arguments: Any = {}
        else:
            try:
                arguments = json.loads(raw_arguments)
            except (TypeError, json.JSONDecodeError) as error:
                raise LlamaCppProtocolError(
                    "Qwen3.5 returned malformed tool arguments"
                ) from error
        if not isinstance(arguments, dict):
            raise LlamaCppProtocolError("Qwen3.5 tool arguments must be a JSON object")
        try:
            json.dumps(arguments, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise LlamaCppProtocolError(
                "Qwen3.5 returned malformed tool arguments"
            ) from error
        required = tool.parameters.get("required", [])
        if isinstance(required, list) and any(key not in arguments for key in required):
            raise LlamaCppProtocolError("The Qwen3.5 tool call is missing a required argument")
        call_id = entry.get("id")
        if not isinstance(call_id, str) or not call_id or len(call_id) > 128:
            call_id = uuid4().hex
        calls.append(ToolCall(call_id, name, arguments))
    return tuple(calls)


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
