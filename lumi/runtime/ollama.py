"""Ollama's native chat API adapter for the approved vanilla Qwen3.5 models.

The client is created only by :meth:`open` and must be closed explicitly. Ollama
loads a model on the first ``/api/chat`` request; this adapter never downloads
models and delegates idle eviction to Ollama's ``keep_alive`` request option.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit
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

DEFAULT_QWEN35_MODELS = tuple(option.model_id for option in supported_models())

_OFFICIAL_MODEL_TAGS = frozenset(
    {
        "0.8b",
        "2b",
        "4b",
        "9b",
        "27b",
        "35b-a3b",
    }
)
_MODEL_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class OllamaRuntimeError(RuntimeError):
    """Base class for a safe, user-presentable local runtime failure."""


class OllamaTransportError(OllamaRuntimeError):
    """Ollama could not be reached or its response could not be read."""


class OllamaProtocolError(OllamaRuntimeError):
    """Ollama returned a response outside the supported API contract."""


class OllamaHTTPError(OllamaRuntimeError):
    """Ollama rejected a request; the response body is intentionally omitted."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"Ollama returned HTTP {status_code}")


@dataclass(frozen=True, slots=True)
class OllamaRuntimeConfig:
    """Bounded transport, model, context, concurrency, and retention settings."""

    base_url: str = "http://127.0.0.1:11434"
    allowed_models: tuple[str, ...] = DEFAULT_QWEN35_MODELS
    idle_unload_seconds: int = 300
    max_concurrent_requests: int = 1
    max_context_tokens: int = 8192
    max_output_tokens: int = 2048
    request_timeout_seconds: float = 120.0
    connect_timeout_seconds: float = 3.0
    max_messages: int = 64
    max_tools: int = 32
    max_message_chars: int = 32_000
    max_request_bytes: int = 2_000_000
    max_response_bytes: int = 2_000_000
    max_status_models: int = 32

    def __post_init__(self) -> None:
        normalized_url = self.base_url.strip()
        parsed = urlsplit(normalized_url)
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError("Ollama base_url has an invalid port") from exc
        if (
            normalized_url != self.base_url
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("Ollama base_url must be an HTTP(S) origin without credentials")
        if not self.allowed_models:
            raise ValueError("At least one approved Qwen3.5 model is required")
        if len(set(self.allowed_models)) != len(self.allowed_models):
            raise ValueError("allowed_models cannot contain duplicates")
        if any(not _is_vanilla_qwen35_model(model) for model in self.allowed_models):
            raise ValueError("Only the approved vanilla Qwen3.5 model tags are supported")
        if not 0 <= self.idle_unload_seconds <= 86_400:
            raise ValueError("idle_unload_seconds must be between 0 and 86400")
        if not 1 <= self.max_concurrent_requests <= 16:
            raise ValueError("max_concurrent_requests must be between 1 and 16")
        if not 256 <= self.max_context_tokens <= 32_768:
            raise ValueError("max_context_tokens must be between 256 and 32768")
        if not 1 <= self.max_output_tokens <= 8192:
            raise ValueError("max_output_tokens must be between 1 and 8192")
        if not 0.1 <= self.request_timeout_seconds <= 600:
            raise ValueError("request_timeout_seconds must be between 0.1 and 600")
        if not 0.1 <= self.connect_timeout_seconds <= 30:
            raise ValueError("connect_timeout_seconds must be between 0.1 and 30")
        if not 1 <= self.max_messages <= 128:
            raise ValueError("max_messages must be between 1 and 128")
        if not 0 <= self.max_tools <= 64:
            raise ValueError("max_tools must be between 0 and 64")
        if not 256 <= self.max_message_chars <= 1_000_000:
            raise ValueError("max_message_chars must be between 256 and 1000000")
        if not 1024 <= self.max_request_bytes <= 16_000_000:
            raise ValueError("max_request_bytes must be between 1024 and 16000000")
        if not 1024 <= self.max_response_bytes <= 16_000_000:
            raise ValueError("max_response_bytes must be between 1024 and 16000000")
        if not 1 <= self.max_status_models <= 128:
            raise ValueError("max_status_models must be between 1 and 128")


@dataclass(frozen=True, slots=True)
class OllamaModelStatus:
    """A model present in Ollama, filtered through the configured allowlist."""

    name: str
    loaded: bool


@dataclass(frozen=True, slots=True)
class OllamaRuntimeStatus:
    """Bounded health and model inventory; status calls never load a model."""

    healthy: bool
    version: str | None
    models: tuple[OllamaModelStatus, ...]
    unavailable_checks: tuple[str, ...] = ()


class _Response(Protocol):
    status_code: int
    headers: Mapping[str, str]

    def json(self) -> Any: ...


class _AsyncClient(Protocol):
    async def get(self, url: str) -> _Response: ...

    async def post(self, url: str, *, json: Mapping[str, Any]) -> _Response: ...

    async def aclose(self) -> None: ...


def _is_vanilla_qwen35_model(model: str) -> bool:
    if not isinstance(model, str) or ":" not in model:
        return False
    family, tag = model.split(":", 1)
    return (
        family == "qwen3.5"
        and bool(_MODEL_TAG_RE.fullmatch(tag))
        and tag in _OFFICIAL_MODEL_TAGS
    )


class OllamaChatRuntime(ChatRuntime):
    """Native ``/api/chat`` runtime for a bounded allowlist of Qwen3.5 models.

    Call :meth:`open` during service startup and :meth:`close` during shutdown,
    or use the runtime as an async context manager. An injected HTTP client is
    useful for tests and is not closed unless ``owns_client`` is true.
    """

    def __init__(
        self,
        config: OllamaRuntimeConfig | None = None,
        *,
        client: _AsyncClient | None = None,
        owns_client: bool = False,
    ) -> None:
        self.config = config or OllamaRuntimeConfig()
        self._client = client
        self._owns_client = client is None or owns_client
        self._opened = False
        self._opening = False
        self._closed = False
        self._gate = asyncio.Condition()
        self._active_generations = 0
        self._active_http_requests = 0
        self._exclusive_operation = False
        self._exclusive_waiters = 0
        self._closing = False

    async def open(self) -> None:
        """Create the HTTP client without loading or pulling a model."""

        async with self._gate:
            while self._opening:
                await self._gate.wait()
            if self._opened:
                return
            if self._closed:
                raise RuntimeError("A closed Ollama runtime cannot be reopened")
            self._opening = True

        client = self._client
        try:
            if client is None:
                try:
                    import httpx
                except ImportError as exc:  # pragma: no cover - depends on deployment extras
                    raise RuntimeError("Install Lumi's 'service' extra to use Ollama") from exc

                client = httpx.AsyncClient(
                    base_url=self.config.base_url.rstrip("/"),
                    timeout=httpx.Timeout(
                        self.config.request_timeout_seconds,
                        connect=self.config.connect_timeout_seconds,
                    ),
                    limits=httpx.Limits(
                        max_connections=self.config.max_concurrent_requests + 4,
                        max_keepalive_connections=self.config.max_concurrent_requests + 2,
                    ),
                    follow_redirects=False,
                    trust_env=False,
                )
                self._client = client
        except BaseException:
            async with self._gate:
                self._opening = False
                self._gate.notify_all()
            raise

        async with self._gate:
            self._opening = False
            if self._closed:
                if self._owns_client:
                    await client.aclose()
                self._gate.notify_all()
                raise RuntimeError("A closed Ollama runtime cannot be reopened")
            self._opened = True
            self._gate.notify_all()

    async def close(self) -> None:
        """Wait for in-flight calls and close an owned HTTP client."""

        async with self._gate:
            while self._opening:
                await self._gate.wait()
            if not self._opened or self._closed:
                return
            self._closing = True
            self._gate.notify_all()
            await self._gate.wait_for(
                lambda: self._active_http_requests == 0
                and self._active_generations == 0
                and not self._exclusive_operation
            )
            client = self._client

        try:
            if self._owns_client and client is not None:
                await client.aclose()
        finally:
            async with self._gate:
                self._opened = False
                self._closed = True
                self._closing = False
                self._client = None if self._owns_client else client
                self._gate.notify_all()

    async def __aenter__(self) -> OllamaChatRuntime:
        await self.open()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Generate one native assistant response, with tool calls normalized."""

        body = self._build_chat_payload(request)
        await self._enter_generation()
        try:
            data = await self._request_json("POST", "/api/chat", body)
        finally:
            await self._leave_generation()

        return self._normalize_response(data)

    async def health(self) -> bool:
        """Return whether Ollama answers its lightweight version endpoint."""

        try:
            payload = await self._request_json("GET", "/api/version")
        except OllamaRuntimeError:
            return False
        return isinstance(payload.get("version"), str) and bool(payload["version"])

    async def model_status(self) -> OllamaRuntimeStatus:
        """Return a bounded view of health, approved installed models, and loaded models."""

        version_result, tags_result, ps_result = await asyncio.gather(
            self._capture_status("/api/version"),
            self._capture_status("/api/tags"),
            self._capture_status("/api/ps"),
        )
        version_payload, version_error = version_result
        tags_payload, tags_error = tags_result
        ps_payload, ps_error = ps_result

        version = None
        if version_payload is not None:
            raw_version = version_payload.get("version")
            if isinstance(raw_version, str) and raw_version:
                version = raw_version[:80]

        configured = set(self.config.allowed_models)
        installed = self._collect_model_names(tags_payload, "models", configured)
        loaded = self._collect_model_names(ps_payload, "models", configured)
        present = (sorted(loaded) + sorted(installed - loaded))[: self.config.max_status_models]
        models = tuple(OllamaModelStatus(name, name in loaded) for name in present)
        errors = tuple(
            name
            for name, error in (
                ("version", version_error),
                ("installed_models", tags_error),
                ("loaded_models", ps_error),
            )
            if error
        )
        return OllamaRuntimeStatus(version is not None, version, models, errors)

    async def unload(self, model: str) -> None:
        """Explicitly ask Ollama to unload an approved model after active chats finish."""

        self._validate_model(model)
        await self._enter_exclusive_operation()
        try:
            await self._request_json(
                "POST",
                "/api/generate",
                {"model": model, "keep_alive": 0, "stream": False},
            )
        finally:
            await self._leave_exclusive_operation()

    async def _capture_status(self, path: str) -> tuple[dict[str, Any] | None, bool]:
        try:
            return await self._request_json("GET", path), False
        except OllamaRuntimeError:
            return None, True

    def _collect_model_names(
        self,
        payload: dict[str, Any] | None,
        key: str,
        configured: set[str],
    ) -> set[str]:
        if payload is None:
            return set()
        records = payload.get(key)
        if not isinstance(records, list):
            return set()
        names: set[str] = set()
        # Ignore unusually large inventories after a bounded prefix.
        for item in records[: self.config.max_status_models * 8]:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name") or item.get("model")
            if isinstance(name, str) and name in configured:
                names.add(name)
        return names

    def _build_chat_payload(self, request: ModelRequest) -> dict[str, Any]:
        self._validate_model(request.model)
        if not isinstance(request.thinking, bool):
            raise ValueError("thinking must be a boolean")
        if not isinstance(request.context_size, int) or request.context_size < 1:
            raise ValueError("context_size must be a positive integer")
        if not isinstance(request.output_tokens, int) or request.output_tokens < 1:
            raise ValueError("output_tokens must be a positive integer")
        if len(request.messages) > self.config.max_messages:
            raise ValueError("Too many messages for the configured Ollama limit")
        if len(request.tools) > self.config.max_tools:
            raise ValueError("Too many tools for the configured Ollama limit")

        messages = [self._serialize_message(message) for message in request.messages]
        body: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": False,
            "think": request.thinking,
            "keep_alive": f"{self.config.idle_unload_seconds}s",
            "options": {
                "num_ctx": min(request.context_size, self.config.max_context_tokens),
                "num_predict": min(request.output_tokens, self.config.max_output_tokens),
            },
        }
        if request.tools:
            body["tools"] = [self._serialize_tool(tool) for tool in request.tools]

        try:
            encoded = json.dumps(body, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("Ollama request contains non-JSON data") from exc
        if len(encoded.encode("utf-8")) > self.config.max_request_bytes:
            raise ValueError("Ollama request exceeds the configured size limit")
        return body

    def _serialize_message(self, message: ChatMessage) -> dict[str, Any]:
        if message.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Unsupported chat message role")
        if (
            not isinstance(message.content, str)
            or len(message.content) > self.config.max_message_chars
        ):
            raise ValueError("Chat message exceeds the configured size limit")

        result: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.role == "tool":
            if not message.name or not _TOOL_NAME_RE.fullmatch(message.name):
                raise ValueError("Tool result messages must include a valid tool name")
            result["tool_name"] = message.name
        elif message.role == "assistant" and message.tool_calls:
            result["tool_calls"] = [self._serialize_tool_call(call) for call in message.tool_calls]
        return result

    def _serialize_tool_call(self, call: ToolCall) -> dict[str, Any]:
        if not _TOOL_NAME_RE.fullmatch(call.name):
            raise ValueError("Tool call name is invalid")
        try:
            arguments = json.loads(json.dumps(call.arguments, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("Tool call arguments must be JSON serializable") from exc
        if not isinstance(arguments, Mapping):
            raise ValueError("Tool call arguments must be a JSON object")
        return {"function": {"name": call.name, "arguments": arguments}}

    def _serialize_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        if not tool.read_only:
            raise ValueError("Only read-only tools may be sent to Ollama")
        if not _TOOL_NAME_RE.fullmatch(tool.name):
            raise ValueError("Tool name is invalid")
        if not isinstance(tool.description, str) or len(tool.description) > 2000:
            raise ValueError("Tool description exceeds the configured limit")
        if not isinstance(tool.parameters, Mapping):
            raise ValueError("Tool parameters must be a JSON schema object")
        try:
            parameters = json.loads(
                json.dumps(tool.parameters, ensure_ascii=False, allow_nan=False)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Tool parameters must be JSON serializable") from exc
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise ValueError("Tool parameter schemas must describe a JSON object")
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            },
        }

    def _validate_model(self, model: str) -> None:
        if model not in self.config.allowed_models:
            raise ValueError("The selected model is not in the configured Qwen3.5 allowlist")

    def _normalize_response(self, payload: dict[str, Any]) -> ModelResponse:
        raw_message = payload.get("message")
        if not isinstance(raw_message, Mapping):
            raise OllamaProtocolError("Ollama response is missing its assistant message")
        content = raw_message.get("content", "")
        if not isinstance(content, str) or len(content) > self.config.max_message_chars:
            raise OllamaProtocolError("Ollama assistant content is malformed or too large")

        role = raw_message.get("role", "assistant")
        if role != "assistant":
            raise OllamaProtocolError("Ollama returned a non-assistant response message")

        calls: list[ToolCall] = []
        raw_calls = raw_message.get("tool_calls", [])
        if raw_calls is None:
            raw_calls = []
        if not isinstance(raw_calls, list) or len(raw_calls) > self.config.max_tools:
            raise OllamaProtocolError("Ollama returned an invalid tool call list")
        for raw_call in raw_calls:
            if not isinstance(raw_call, Mapping):
                raise OllamaProtocolError("Ollama returned a malformed tool call")
            function = raw_call.get("function")
            if not isinstance(function, Mapping):
                raise OllamaProtocolError("Ollama returned a malformed tool function")
            name = function.get("name")
            arguments = function.get("arguments")
            if not isinstance(name, str) or not _TOOL_NAME_RE.fullmatch(name):
                raise OllamaProtocolError("Ollama returned an invalid tool name")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise OllamaProtocolError("Ollama returned invalid tool arguments") from exc
            if not isinstance(arguments, Mapping):
                raise OllamaProtocolError("Ollama tool arguments must be a JSON object")
            calls.append(ToolCall(uuid4().hex, name, dict(arguments)))

        return ModelResponse(
            message=ChatMessage("assistant", content, tool_calls=tuple(calls)),
            prompt_tokens=_optional_nonnegative_int(payload.get("prompt_eval_count")),
            completion_tokens=_optional_nonnegative_int(payload.get("eval_count")),
            load_duration_ns=_optional_nonnegative_int(payload.get("load_duration")),
            total_duration_ns=_optional_nonnegative_int(payload.get("total_duration")),
        )

    async def _request_json(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        async with self._gate:
            if not self._opened or self._closed or self._closing or self._client is None:
                raise RuntimeError("Open the Ollama runtime before making requests")
            self._active_http_requests += 1
            client = self._client

        try:
            try:
                if method == "GET":
                    response = await client.get(path)
                elif method == "POST" and body is not None:
                    response = await client.post(path, json=body)
                else:
                    raise ValueError("Unsupported Ollama HTTP operation")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise OllamaTransportError("Could not reach the local Ollama service") from exc

            if not 200 <= response.status_code < 300:
                raise OllamaHTTPError(response.status_code)
            headers = getattr(response, "headers", {})
            content_length = _header(headers, "content-length")
            if content_length is not None:
                try:
                    if int(content_length) > self.config.max_response_bytes:
                        raise OllamaProtocolError(
                            "Ollama response exceeds the configured size limit"
                        )
                except ValueError:
                    raise OllamaProtocolError(
                        "Ollama returned an invalid response length"
                    ) from None
            content = getattr(response, "content", None)
            if (
                isinstance(content, (bytes, bytearray))
                and len(content) > self.config.max_response_bytes
            ):
                raise OllamaProtocolError(
                    "Ollama response exceeds the configured size limit"
                )
            try:
                payload = response.json()
            except Exception as exc:
                raise OllamaProtocolError("Ollama returned invalid JSON") from exc
            if not isinstance(payload, dict):
                raise OllamaProtocolError("Ollama returned a non-object JSON response")
            return payload
        finally:
            async with self._gate:
                self._active_http_requests -= 1
                self._gate.notify_all()

    async def _enter_generation(self) -> None:
        async with self._gate:
            if not self._opened or self._closed:
                raise RuntimeError("Open the Ollama runtime before making requests")
            await self._gate.wait_for(
                lambda: self._closing
                or self._closed
                or (
                    not self._exclusive_operation
                    and self._exclusive_waiters == 0
                    and self._active_generations < self.config.max_concurrent_requests
                )
            )
            if self._closing or self._closed or not self._opened:
                raise RuntimeError("The Ollama runtime is not available")
            self._active_generations += 1

    async def _leave_generation(self) -> None:
        async with self._gate:
            self._active_generations -= 1
            self._gate.notify_all()

    async def _enter_exclusive_operation(self) -> None:
        async with self._gate:
            if not self._opened or self._closed:
                raise RuntimeError("Open the Ollama runtime before making requests")
            self._exclusive_waiters += 1
            self._gate.notify_all()
            try:
                await self._gate.wait_for(
                    lambda: self._closing
                    or self._closed
                    or (not self._exclusive_operation and self._active_generations == 0)
                )
                if self._closing or self._closed or not self._opened:
                    raise RuntimeError("The Ollama runtime is not available")
                self._exclusive_operation = True
            finally:
                self._exclusive_waiters -= 1
                self._gate.notify_all()

    async def _leave_exclusive_operation(self) -> None:
        async with self._gate:
            self._exclusive_operation = False
            self._gate.notify_all()


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _header(headers: Mapping[str, str], name: str) -> str | None:
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return value
    return None
