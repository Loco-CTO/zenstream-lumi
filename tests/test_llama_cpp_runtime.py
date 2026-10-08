from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import threading
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from lumi.contracts import ChatMessage, ModelRequest, ToolDefinition
from lumi.model_installation import (
    GGUF_FORMAT,
    MODEL_MANIFEST_SCHEMA_VERSION,
    Qwen35ModelSpec,
)
from lumi.runtime.llama_cpp import (
    LUMI_RUNTIME_API_VERSION,
    LlamaCppChatRuntime,
    LlamaCppConfig,
    LlamaCppProtocolError,
    LlamaCppRuntimeError,
    VerifiedModelArtifact,
)

_GGUF_FILENAME = "Qwen_Qwen3.5-test-Q4_K_M.gguf"
_GGUF_BYTES = b"small fake GGUF bytes"


def _spec(model_id: str) -> Qwen35ModelSpec:
    scale = model_id.split(":", 1)[1]
    return Qwen35ModelSpec(
        model_id=model_id,
        directory_name=f"qwen3.5-{scale.replace('.', '-')}",
        label=f"Qwen3.5 {scale.upper()}",
        repository_id="bartowski/test-qwen35-gguf",
        revision="a" * 40,
        gguf_filename=_GGUF_FILENAME,
        quantization="Q4_K_M",
        gguf_sha256=hashlib.sha256(_GGUF_BYTES).hexdigest(),
        download_size_bytes=len(_GGUF_BYTES),
        max_download_bytes=1024,
    )


class FakeFormatter:
    def __init__(self, api: FakeLlamaAPI, **options: Any) -> None:
        self.api = api
        self.options = options
        self.api.formatter_options.append(options.copy())

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.api.formatted.append(kwargs.copy())
        return SimpleNamespace(
            prompt="rendered native Qwen prompt",
            stop=["<|im_end|>"],
            stopping_criteria=self.api.formatter_stopping_criteria,
        )


class FakeModel:
    def __init__(self, api: FakeLlamaAPI, model_path: str, options: dict[str, Any]) -> None:
        self.api = api
        self.model_path = model_path
        self.options = options
        self.metadata = {"tokenizer.chat_template": "Qwen3.5 native Jinja chat template"}

    def token_eos(self) -> int:
        return 2

    def token_bos(self) -> int:
        return 1

    def detokenize(self, tokens: list[int], *, special: bool) -> bytes:
        if not special:
            raise AssertionError("Special token decoding must preserve template markers")
        return {1: b"<|im_start|>", 2: b"<|im_end|>"}[tokens[0]]

    def tokenize(self, prompt: bytes, *, add_bos: bool, special: bool) -> list[int]:
        self.api.tokenized_prompts.append((prompt, add_bos, special))
        return list(range(self.api.prompt_token_count))

    def create_completion(self, **options: Any) -> dict[str, Any]:
        if not callable(options.get("stopping_criteria")):
            raise TypeError("llama.cpp stopping criteria must be callable")
        self.api.generation_started.set()
        self.api.completion_options.append(options)
        if self.api.block_generation:
            self.api.allow_generation.wait(timeout=3)
            stop = options["stopping_criteria"][0]
            if stop([], []):
                return {"choices": [{"text": ""}]}
        if self.api.fail_generation:
            raise RuntimeError("simulated native generation failure")
        self.api.generation_calls += 1
        return {"choices": [{"text": self.api.output}]}


class FakeLlamaAPI:
    def __init__(self, output: str) -> None:
        self.output = output
        self.loads: list[str] = []
        self.model_options: list[dict[str, Any]] = []
        self.formatter_options: list[dict[str, Any]] = []
        self.formatted: list[dict[str, Any]] = []
        self.tokenized_prompts: list[tuple[bytes, bool, bool]] = []
        self.completion_options: list[dict[str, Any]] = []
        self.live_models: weakref.WeakSet[FakeModel] = weakref.WeakSet()
        self.prompt_token_count = 3
        self.fail_generation = False
        self.block_generation = False
        self.generation_calls = 0
        self.formatter_stopping_criteria: list[Any] = []
        self.generation_started = threading.Event()
        self.allow_generation = threading.Event()
        self.allow_generation.set()
        self.Jinja2ChatFormatter = lambda **options: FakeFormatter(self, **options)

    def Llama(self, **options: Any) -> FakeModel:
        model_path = options["model_path"]
        if self.live_models:
            raise AssertionError("The previous native model must unload before switching")
        self.loads.append(model_path)
        self.model_options.append(options)
        model = FakeModel(self, model_path, options)
        self.live_models.add(model)
        return model


class LlamaCppRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.specs = {model: _spec(model) for model in ("qwen3.5:2b", "qwen3.5:4b")}
        self.artifacts = {
            model: self.write_model_artifact(spec) for model, spec in self.specs.items()
        }
        self.spec_patcher = patch("lumi.model_installation._MODEL_SPECS", self.specs)
        self.spec_patcher.start()
        self.addCleanup(self.spec_patcher.stop)
        self.tool = ToolDefinition(
            "catalog_search",
            "Search the local catalog.",
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            data_scope="local",
            read_only=True,
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def write_model_artifact(self, spec: Qwen35ModelSpec) -> VerifiedModelArtifact:
        directory = self.root / spec.directory_name
        directory.mkdir(parents=True)
        model_path = directory / spec.gguf_filename
        model_path.write_bytes(_GGUF_BYTES)
        source = {
            **spec.source_manifest(),
            "manifestSha256": spec.pinned_source_manifest_sha256,
        }
        manifest = json.dumps(
            {
                "schemaVersion": MODEL_MANIFEST_SCHEMA_VERSION,
                "format": GGUF_FORMAT,
                "modelId": spec.model_id,
                "quantization": spec.quantization,
                "source": source,
                "files": [
                    {
                        "path": spec.gguf_filename,
                        "size": spec.download_size_bytes,
                        "sha256": spec.gguf_sha256,
                    }
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        (directory / "lumi-model-manifest.json").write_bytes(manifest)
        return VerifiedModelArtifact(
            spec.model_id,
            directory,
            hashlib.sha256(manifest).hexdigest(),
        )

    def make_runtime(self, api: FakeLlamaAPI, **overrides: Any) -> LlamaCppChatRuntime:
        options = {
            "model_artifacts": {"qwen3.5:2b": self.artifacts["qwen3.5:2b"]},
            "max_context_tokens": 256,
            "max_output_tokens": 32,
            "n_ctx": 256,
        }
        options.update(overrides)
        return LlamaCppChatRuntime(LlamaCppConfig(**options), api=api)

    def make_request(
        self,
        *,
        model: str = "qwen3.5:2b",
        thinking: bool = False,
        context_size: int = 64,
        output_tokens: int = 24,
        tools: tuple[ToolDefinition, ...] = (),
    ) -> ModelRequest:
        return ModelRequest(
            model=model,
            messages=(
                ChatMessage("system", "Use read-only tools."),
                ChatMessage("user", "Find it"),
            ),
            tools=tools,
            thinking=thinking,
            context_size=context_size,
            output_tokens=output_tokens,
        )

    async def test_open_is_lazy_and_inference_uses_native_chat_template(self) -> None:
        api = FakeLlamaAPI("<think>private reasoning</think> A short answer. <|im_end|>")
        runtime = self.make_runtime(api)
        await runtime.open()
        self.assertEqual(api.loads, [])

        response = await runtime.complete(self.make_request(thinking=True))

        self.assertEqual(LUMI_RUNTIME_API_VERSION, 1)
        self.assertEqual(response.message.content, "A short answer.")
        self.assertEqual(api.formatted[0]["enable_thinking"], True)
        self.assertEqual(api.formatter_options[0]["add_generation_prompt"], True)
        self.assertNotIn("add_generation_prompt", api.formatted[0])
        self.assertEqual(api.tokenized_prompts[0][1:], (True, True))
        self.assertEqual(api.completion_options[0]["temperature"], 0.0)
        self.assertEqual(api.completion_options[0]["max_tokens"], 24)
        self.assertEqual(api.model_options[0]["n_gpu_layers"], 0)
        await runtime.close()

    async def test_formatter_stopping_criteria_are_preserved_with_cancellation(self) -> None:
        api = FakeLlamaAPI("Answer")

        def formatter_criterion(_tokens, _logits):
            return False

        api.formatter_stopping_criteria = [formatter_criterion]
        runtime = self.make_runtime(api)
        await runtime.open()

        await runtime.complete(self.make_request())

        criteria = api.completion_options[0]["stopping_criteria"]
        self.assertEqual(criteria[0], formatter_criterion)
        self.assertEqual(len(criteria), 2)
        self.assertTrue(callable(criteria[1]))
        self.assertTrue(callable(criteria))
        self.assertFalse(criteria([], []))
        await runtime.close()

    async def test_thinking_is_forwarded_per_request(self) -> None:
        api = FakeLlamaAPI("Answer")
        runtime = self.make_runtime(api)
        await runtime.open()
        await runtime.complete(self.make_request(thinking=False))
        await runtime.complete(self.make_request(thinking=True))
        self.assertEqual([item["enable_thinking"] for item in api.formatted], [False, True])
        await runtime.close()

    async def test_native_qwen_tool_call_is_normalized_and_reasoning_is_hidden(self) -> None:
        api = FakeLlamaAPI(
            "<|channel|>analysis private thought\n"
            "<tool_call><function=catalog_search>"
            "<parameter=query>Fate/Zero</parameter></function></tool_call>"
            "<|channel|>final Searching now. <|im_end|>"
        )
        runtime = self.make_runtime(api)
        await runtime.open()
        response = await runtime.complete(self.make_request(tools=(self.tool,)))

        self.assertEqual(response.message.content, "Searching now.")
        self.assertEqual(len(response.message.tool_calls), 1)
        self.assertEqual(response.message.tool_calls[0].name, "catalog_search")
        self.assertEqual(response.message.tool_calls[0].arguments, {"query": "Fate/Zero"})
        sent_tools = api.formatted[0]["tools"]
        self.assertEqual(sent_tools[0]["function"]["name"], "catalog_search")
        await runtime.close()

    async def test_multiple_native_tool_calls_are_supported(self) -> None:
        second_tool = ToolDefinition(
            "artist_lookup",
            "Look up a local artist.",
            {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
            data_scope="local",
            read_only=True,
        )
        api = FakeLlamaAPI(
            "<tool_call><function=catalog_search><parameter=query>jazz</parameter></function></tool_call>"
            "<tool_call><function=artist_lookup>"
            "<parameter=name>Yoko Kanno</parameter></function></tool_call>"
        )
        runtime = self.make_runtime(api)
        await runtime.open()
        response = await runtime.complete(self.make_request(tools=(self.tool, second_tool)))
        self.assertEqual(
            [call.name for call in response.message.tool_calls],
            ["catalog_search", "artist_lookup"],
        )
        await runtime.close()

    async def test_unknown_and_malformed_tool_calls_fail_closed(self) -> None:
        outputs = (
            "<tool_call><function=unknown_tool></function></tool_call>",
            "<tool_call><function=catalog_search><parameter=query>jazz</function></tool_call>",
            "<tool_call><function=catalog_search></function></tool_call>",
        )
        for output in outputs:
            with self.subTest(output=output):
                runtime = self.make_runtime(FakeLlamaAPI(output))
                await runtime.open()
                with self.assertRaises(LlamaCppProtocolError):
                    await runtime.complete(self.make_request(tools=(self.tool,)))
                await runtime.close()

    async def test_tool_argument_values_must_be_valid_for_the_registered_schema(self) -> None:
        numeric_tool = ToolDefinition(
            "track_lookup",
            "Look up a track ID.",
            {
                "type": "object",
                "properties": {"track_id": {"type": "integer"}},
                "required": ["track_id"],
            },
            data_scope="local",
            read_only=True,
        )
        runtime = self.make_runtime(
            FakeLlamaAPI(
                "<tool_call><function=track_lookup><parameter=track_id>abc</parameter></function></tool_call>"
            )
        )
        await runtime.open()
        with self.assertRaises(LlamaCppProtocolError):
            await runtime.complete(self.make_request(tools=(numeric_tool,)))
        await runtime.close()

    async def test_rejects_state_changing_tools_before_loading_a_model(self) -> None:
        api = FakeLlamaAPI("should never run")
        runtime = self.make_runtime(api)
        await runtime.open()
        write_tool = ToolDefinition(
            "play_media",
            "Start playback.",
            {"type": "object", "properties": {}},
            data_scope="local",
            read_only=False,
        )
        with self.assertRaisesRegex(ValueError, "read-only"):
            await runtime.complete(self.make_request(tools=(write_tool,)))
        self.assertEqual(api.loads, [])
        await runtime.close()

    async def test_prompt_must_leave_room_for_output_tokens(self) -> None:
        api = FakeLlamaAPI("unused")
        api.prompt_token_count = 64
        runtime = self.make_runtime(api)
        await runtime.open()
        with self.assertRaisesRegex(ValueError, "leaves no room"):
            await runtime.complete(self.make_request(context_size=3))
        self.assertEqual(api.completion_options, [])
        await runtime.close()

    async def test_model_switch_unloads_previous_before_loading_next(self) -> None:
        api = FakeLlamaAPI("Answer")
        runtime = self.make_runtime(
            api,
            model_artifacts=self.artifacts,
        )
        await runtime.open()
        await runtime.complete(self.make_request(model="qwen3.5:2b"))
        await runtime.complete(self.make_request(model="qwen3.5:4b"))
        self.assertEqual(len(api.loads), 2)
        self.assertEqual(Path(api.loads[0]).parent.name, "qwen3.5-2b")
        self.assertEqual(Path(api.loads[0]).name, _GGUF_FILENAME)
        self.assertEqual(Path(api.loads[1]).parent.name, "qwen3.5-4b")
        self.assertEqual(Path(api.loads[1]).name, _GGUF_FILENAME)
        self.assertEqual(len(api.live_models), 1)
        await runtime.close()

    async def test_idle_model_unloads(self) -> None:
        api = FakeLlamaAPI("Answer")
        runtime = self.make_runtime(api, idle_unload_seconds=0)
        await runtime.open()
        await runtime.complete(self.make_request())
        for _ in range(20):
            if runtime._loaded is None and runtime._resident_model_id is None:
                break
            await asyncio.sleep(0.05)
        self.assertIsNone(runtime._loaded)
        self.assertIsNone(runtime._resident_model_id)
        await runtime.close()

    async def test_native_failure_becomes_safe_runtime_error(self) -> None:
        api = FakeLlamaAPI("unused")
        api.fail_generation = True
        runtime = self.make_runtime(api)
        await runtime.open()
        with self.assertRaisesRegex(LlamaCppRuntimeError, "inference failed"):
            await runtime.complete(self.make_request())
        await runtime.close()

    async def test_cancelled_inference_waits_for_native_worker_before_unload(self) -> None:
        api = FakeLlamaAPI("unused")
        api.block_generation = True
        api.allow_generation.clear()
        runtime = self.make_runtime(api)
        await runtime.open()
        request = asyncio.create_task(runtime.complete(self.make_request()))
        self.assertTrue(await asyncio.to_thread(api.generation_started.wait, 1))
        request.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await request
        api.allow_generation.set()
        await asyncio.wait_for(runtime.close(), timeout=1)
        self.assertIsNone(runtime._loaded)

    async def test_tampered_gguf_is_rejected_before_native_load(self) -> None:
        api = FakeLlamaAPI("unused")
        runtime = self.make_runtime(api)
        await runtime.open()
        path = Path(self.artifacts["qwen3.5:2b"].directory) / _GGUF_FILENAME
        path.write_bytes(b"changed with different size")
        with self.assertRaisesRegex(LlamaCppRuntimeError, "missing or changed"):
            await runtime.complete(self.make_request())
        self.assertEqual(api.loads, [])
        await runtime.close()

    async def test_missing_model_returns_runtime_error(self) -> None:
        missing = self.root / "qwen3.5-2b-missing"
        runtime = self.make_runtime(
            FakeLlamaAPI("unused"),
            model_artifacts={"qwen3.5:2b": VerifiedModelArtifact("qwen3.5:2b", missing, "0" * 64)},
        )
        await runtime.open()
        with self.assertRaises(LlamaCppRuntimeError):
            await runtime.complete(self.make_request())
        await runtime.close()

    def test_runtime_config_bounds_native_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_ubatch"):
            LlamaCppConfig(model_artifacts=self.artifacts, n_batch=16, n_ubatch=32)
        with self.assertRaisesRegex(ValueError, "n_ctx"):
            LlamaCppConfig(model_artifacts=self.artifacts, n_ctx=128)


if __name__ == "__main__":
    unittest.main()
