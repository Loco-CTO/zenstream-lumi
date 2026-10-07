"""Contract tests for the native Ollama/Qwen3.5 runtime adapter."""

from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any

from lumi.contracts import ChatMessage, ModelRequest, ToolCall, ToolDefinition
from lumi.runtime.ollama import (
    DEFAULT_QWEN35_MODELS,
    OllamaChatRuntime,
    OllamaHTTPError,
    OllamaProtocolError,
    OllamaRuntimeConfig,
)


class FakeResponse:
    def __init__(
        self,
        payload: Any,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.content = json.dumps(payload).encode("utf-8")

    def json(self) -> Any:
        return self._payload


class FakeClient:
    def __init__(self, responses: list[FakeResponse] | None = None) -> None:
        self.responses = list(responses or [])
        self.get_responses: dict[str, FakeResponse] = {}
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []
        self.closed = False

    async def get(self, url: str) -> FakeResponse:
        self.calls.append(("GET", url, None))
        return self.get_responses.get(url, FakeResponse({"version": "0.0.0"}))

    async def post(self, url: str, *, json: dict[str, Any]) -> FakeResponse:
        self.calls.append(("POST", url, json))
        if self.responses:
            return self.responses.pop(0)
        return FakeResponse({"message": {"role": "assistant", "content": "ok"}})

    async def aclose(self) -> None:
        self.closed = True


class BlockingClient(FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.max_active = 0
        self.started = asyncio.Event()
        self.two_started = asyncio.Event()
        self.release = asyncio.Event()

    async def post(self, url: str, *, json: dict[str, Any]) -> FakeResponse:
        self.calls.append(("POST", url, json))
        if url != "/api/chat":
            return FakeResponse({"model": json.get("model", ""), "done": True})
        self.active += 1
        self.started.set()
        self.max_active = max(self.max_active, self.active)
        if self.active >= 2:
            self.two_started.set()
        await self.release.wait()
        self.active -= 1
        return FakeResponse({"message": {"role": "assistant", "content": "done"}})


def make_request(
    *,
    model: str = "qwen3.5:4b",
    messages: tuple[ChatMessage, ...] | None = None,
    tools: tuple[ToolDefinition, ...] = (),
    thinking: bool = True,
    context_size: int = 4096,
    output_tokens: int = 512,
) -> ModelRequest:
    return ModelRequest(
        model=model,
        messages=messages or (ChatMessage("user", "hello"),),
        tools=tools,
        thinking=thinking,
        context_size=context_size,
        output_tokens=output_tokens,
    )


class OllamaRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_tool_calls_thinking_limits_and_metrics_normalize(self) -> None:
        client = FakeClient(
            [
                FakeResponse(
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Searching the catalog now.",
                            "thinking": "private reasoning must not enter chat content",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "catalog_search",
                                        "arguments": {"query": "blue album"},
                                    }
                                }
                            ],
                        },
                        "prompt_eval_count": 341,
                        "eval_count": 16,
                        "load_duration": 1_250_000,
                        "total_duration": 8_500_000,
                    }
                )
            ]
        )
        config = OllamaRuntimeConfig(
            max_context_tokens=4096,
            max_output_tokens=128,
            idle_unload_seconds=75,
        )
        runtime = OllamaChatRuntime(config, client=client)
        await runtime.open()

        result = await runtime.complete(
            make_request(
                messages=(
                    ChatMessage("system", "Use read-only tools."),
                    ChatMessage("user", "Find it"),
                ),
                tools=(
                    ToolDefinition(
                        "catalog_search",
                        "Search the local catalog.",
                        {"type": "object", "properties": {"query": {"type": "string"}}},
                        data_scope="local",
                        read_only=True,
                    ),
                ),
                context_size=12000,
                output_tokens=900,
            )
        )

        method, path, payload = client.calls[0]
        self.assertEqual((method, path), ("POST", "/api/chat"))
        assert payload is not None
        self.assertEqual(payload["think"], True)
        self.assertEqual(payload["keep_alive"], "75s")
        self.assertEqual(payload["stream"], False)
        self.assertEqual(payload["options"], {"num_ctx": 4096, "num_predict": 128})
        self.assertEqual(payload["tools"][0]["type"], "function")
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(result.message.content, "Searching the catalog now.")
        self.assertEqual(result.message.tool_calls[0].name, "catalog_search")
        self.assertEqual(result.message.tool_calls[0].arguments, {"query": "blue album"})
        self.assertTrue(result.message.tool_calls[0].call_id)
        self.assertEqual(result.prompt_tokens, 341)
        self.assertEqual(result.completion_tokens, 16)
        self.assertEqual(result.load_duration_ns, 1_250_000)
        self.assertEqual(result.total_duration_ns, 8_500_000)
        self.assertNotIn("thinking", payload)
        self.assertFalse(any(url == "/api/pull" for _, url, _ in client.calls))

    async def test_tool_assistant_and_result_messages_use_ollama_shape(self) -> None:
        client = FakeClient()
        runtime = OllamaChatRuntime(client=client)
        await runtime.open()
        history = (
            ChatMessage(
                "assistant",
                "",
                tool_calls=(ToolCall("call-1", "catalog_search", {"query": "jazz"}),),
            ),
            ChatMessage("tool", '{"count":2}', name="catalog_search", tool_call_id="call-1"),
            ChatMessage("user", "Which result is the best match?"),
        )

        await runtime.complete(make_request(messages=history, thinking=False))

        payload = client.calls[0][2]
        assert payload is not None
        self.assertEqual(
            payload["messages"][0]["tool_calls"],
            [
                {
                    "function": {
                        "name": "catalog_search",
                        "arguments": {"query": "jazz"},
                    }
                }
            ],
        )
        self.assertEqual(
            payload["messages"][1],
            {"role": "tool", "content": '{"count":2}', "tool_name": "catalog_search"},
        )
        self.assertIs(payload["think"], False)

    async def test_stringified_tool_arguments_are_parsed_and_bad_metrics_ignored(self) -> None:
        client = FakeClient(
            [
                FakeResponse(
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "catalog_search",
                                        "arguments": '{"query":"ambient"}',
                                    }
                                }
                            ],
                        },
                        "prompt_eval_count": -1,
                        "eval_count": True,
                        "load_duration": "fast",
                    }
                )
            ]
        )
        runtime = OllamaChatRuntime(client=client)
        await runtime.open()

        response = await runtime.complete(make_request())

        self.assertEqual(response.message.tool_calls[0].arguments, {"query": "ambient"})
        self.assertIsNone(response.prompt_tokens)
        self.assertIsNone(response.completion_tokens)
        self.assertIsNone(response.load_duration_ns)

    async def test_model_and_request_bounds_are_rejected_before_http(self) -> None:
        client = FakeClient()
        runtime = OllamaChatRuntime(
            OllamaRuntimeConfig(max_tools=0, max_message_chars=256, max_request_bytes=1024),
            client=client,
        )
        await runtime.open()

        with self.assertRaises(ValueError):
            await runtime.complete(make_request(model="llama3.1"))
        with self.assertRaises(ValueError):
            await runtime.complete(make_request(model="qwen3.5:4b-finetuned"))
        with self.assertRaises(ValueError):
            await runtime.complete(make_request(context_size=0))
        with self.assertRaises(ValueError):
            await runtime.complete(
                make_request(
                    tools=(
                        ToolDefinition(
                            "search", "", {"type": "object"}, data_scope="local", read_only=True
                        ),
                    )
                )
            )
        with self.assertRaises(ValueError):
            await runtime.complete(
                make_request(messages=(ChatMessage("user", "x" * 257),))
            )
        self.assertEqual(client.calls, [])

    async def test_non_read_only_or_invalid_tool_schemas_are_rejected(self) -> None:
        runtime = OllamaChatRuntime(client=FakeClient())
        await runtime.open()
        with self.assertRaises(ValueError):
            await runtime.complete(
                make_request(
                    tools=(
                        ToolDefinition(
                            "change_setting",
                            "Unsafe mutation",
                            {"type": "object"},
                            data_scope="local",
                            read_only=False,
                        ),
                    )
                )
            )
        with self.assertRaises(ValueError):
            await runtime.complete(
                make_request(
                    tools=(
                        ToolDefinition(
                            "search",
                            "Search",
                            {"type": "string"},
                            data_scope="local",
                            read_only=True,
                        ),
                    )
                )
            )

    async def test_model_inventory_is_filtered_bounded_and_does_not_load(self) -> None:
        client = FakeClient()
        client.get_responses = {
            "/api/version": FakeResponse({"version": "0.7.1"}),
            "/api/tags": FakeResponse(
                {
                    "models": [
                        {"name": "qwen3.5:0.8b"},
                        {"name": "llama3.1:8b"},
                        {"name": "qwen3.5:4b"},
                    ]
                }
            ),
            "/api/ps": FakeResponse({"models": [{"model": "qwen3.5:4b"}]}),
        }
        runtime = OllamaChatRuntime(OllamaRuntimeConfig(max_status_models=1), client=client)
        await runtime.open()

        status = await runtime.model_status()

        self.assertTrue(status.healthy)
        self.assertEqual(status.version, "0.7.1")
        self.assertEqual(len(status.models), 1)
        self.assertEqual(status.models[0].name, "qwen3.5:4b")
        self.assertTrue(status.models[0].loaded)
        self.assertEqual(
            {url for method, url, _ in client.calls if method == "GET"},
            {"/api/version", "/api/tags", "/api/ps"},
        )
        self.assertTrue(all(method == "GET" for method, _, _ in client.calls))

    async def test_status_reports_partial_failure_without_raw_error_text(self) -> None:
        client = FakeClient()
        client.get_responses = {
            "/api/version": FakeResponse({"version": "0.7.1"}),
            "/api/tags": FakeResponse({"message": "private provider detail"}, status_code=500),
            "/api/ps": FakeResponse({"models": []}),
        }
        runtime = OllamaChatRuntime(client=client)
        await runtime.open()

        status = await runtime.model_status()

        self.assertTrue(status.healthy)
        self.assertEqual(status.unavailable_checks, ("installed_models",))
        self.assertNotIn("private provider detail", repr(status))

    async def test_explicit_unload_waits_for_active_chat_and_uses_keep_alive_zero(self) -> None:
        client = BlockingClient()
        runtime = OllamaChatRuntime(
            OllamaRuntimeConfig(max_concurrent_requests=2),
            client=client,
        )
        await runtime.open()
        active_chats = [
            asyncio.create_task(runtime.complete(make_request())) for _ in range(2)
        ]
        await asyncio.wait_for(client.started.wait(), timeout=1)
        unload = asyncio.create_task(runtime.unload("qwen3.5:4b"))
        await asyncio.sleep(0)
        self.assertFalse(any(url == "/api/generate" for _, url, _ in client.calls))

        client.release.set()
        await asyncio.gather(*active_chats, unload)

        unload_call = next(call for call in client.calls if call[1] == "/api/generate")
        self.assertEqual(unload_call[2], {"model": "qwen3.5:4b", "keep_alive": 0, "stream": False})

    async def test_configured_concurrency_caps_in_flight_chat_requests(self) -> None:
        client = BlockingClient()
        runtime = OllamaChatRuntime(
            OllamaRuntimeConfig(max_concurrent_requests=2),
            client=client,
        )
        await runtime.open()
        tasks = [asyncio.create_task(runtime.complete(make_request())) for _ in range(3)]
        await asyncio.wait_for(client.two_started.wait(), timeout=1)
        self.assertEqual(client.max_active, 2)

        client.release.set()
        await asyncio.gather(*tasks)
        self.assertEqual(client.max_active, 2)

    async def test_waiting_unload_takes_priority_over_new_chat_generations(self) -> None:
        client = BlockingClient()
        runtime = OllamaChatRuntime(
            OllamaRuntimeConfig(max_concurrent_requests=2),
            client=client,
        )
        await runtime.open()
        first_chat = asyncio.create_task(runtime.complete(make_request()))
        await asyncio.wait_for(client.started.wait(), timeout=1)
        unload = asyncio.create_task(runtime.unload("qwen3.5:4b"))
        await asyncio.sleep(0)
        second_chat = asyncio.create_task(runtime.complete(make_request()))
        await asyncio.sleep(0)

        client.release.set()
        await asyncio.gather(first_chat, unload, second_chat)

        request_paths = [url for method, url, _ in client.calls if method == "POST"]
        self.assertEqual(request_paths, ["/api/chat", "/api/generate", "/api/chat"])

    async def test_close_waits_for_in_flight_chat_before_closing_owned_client(self) -> None:
        client = BlockingClient()
        runtime = OllamaChatRuntime(client=client, owns_client=True)
        await runtime.open()
        chat = asyncio.create_task(runtime.complete(make_request()))
        await asyncio.wait_for(client.started.wait(), timeout=1)
        closing = asyncio.create_task(runtime.close())
        await asyncio.sleep(0)
        self.assertFalse(client.closed)

        client.release.set()
        await asyncio.gather(chat, closing)
        self.assertTrue(client.closed)

    async def test_open_is_lazy_and_owned_client_is_closed_explicitly(self) -> None:
        client = FakeClient()
        runtime = OllamaChatRuntime(client=client, owns_client=True)

        await runtime.open()
        self.assertEqual(client.calls, [])
        self.assertFalse(client.closed)
        await runtime.close()

        self.assertTrue(client.closed)
        with self.assertRaises(RuntimeError):
            await runtime.open()

    async def test_request_before_open_fails_without_waiting_forever(self) -> None:
        runtime = OllamaChatRuntime(client=FakeClient())

        with self.assertRaises(RuntimeError):
            await asyncio.wait_for(runtime.complete(make_request()), timeout=0.2)

    async def test_http_and_oversized_response_errors_are_bounded(self) -> None:
        client = FakeClient([FakeResponse({"message": "secret"}, status_code=500)])
        runtime = OllamaChatRuntime(client=client)
        await runtime.open()

        with self.assertRaises(OllamaHTTPError) as error:
            await runtime.complete(make_request())
        self.assertEqual(error.exception.status_code, 500)
        self.assertNotIn("secret", str(error.exception))

        client.responses.append(
            FakeResponse(
                {"message": {"role": "assistant", "content": "ok"}},
                headers={"content-length": "3000000"},
            )
        )
        with self.assertRaises(OllamaProtocolError):
            await runtime.complete(make_request())

    def test_config_restricts_model_family_and_url_shape(self) -> None:
        with self.assertRaises(ValueError):
            OllamaRuntimeConfig(allowed_models=("custom-qwen:4b",))
        with self.assertRaises(ValueError):
            OllamaRuntimeConfig(base_url="http://user:secret@localhost:11434")
        with self.assertRaises(ValueError):
            OllamaRuntimeConfig(base_url="http://localhost:11434/path")
        self.assertEqual(DEFAULT_QWEN35_MODELS, ("qwen3.5:0.8b", "qwen3.5:2b", "qwen3.5:4b"))


if __name__ == "__main__":
    unittest.main()
