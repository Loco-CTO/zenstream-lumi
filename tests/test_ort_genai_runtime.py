"""Contract tests for Lumi's in-process ONNX Runtime GenAI adapter."""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import threading
import unittest
import weakref
from pathlib import Path
from typing import Any

from lumi.contracts import ChatMessage, ModelRequest, ToolDefinition
from lumi.runtime.ort_genai import (
    LUMI_RUNTIME_API_VERSION,
    OrtGenAIChatRuntime,
    OrtGenAIConfig,
    OrtGenAIProtocolError,
    OrtGenAIRuntimeError,
    VerifiedModelArtifact,
)


class FakeTokenizer:
    def __init__(self, api: FakeOrtGenAI) -> None:
        self.api = api

    def apply_chat_template(
        self,
        template: str,
        *,
        messages: str,
        tools: str | None = None,
        add_generation_prompt: bool,
    ) -> str:
        self.api.template = template
        self.api.messages = json.loads(messages)
        self.api.tools = json.loads(tools) if tools is not None else None
        self.api.add_generation_prompt = add_generation_prompt
        return "rendered prompt"

    def encode(self, prompt: str) -> list[int]:
        self.api.prompt = prompt
        return [11, 12, 13]

    def decode(self, tokens: list[int]) -> str:
        self.api.decoded_tokens = list(tokens)
        return self.api.output


class FakeGeneratorParams:
    def __init__(self, model: FakeModel) -> None:
        self.model_ref = weakref.ref(model)
        self.options: dict[str, Any] = {}

    def set_search_options(self, **options: Any) -> None:
        self.options = options


class FakeGenerator:
    def __init__(self, model: FakeModel, params: FakeGeneratorParams, api: FakeOrtGenAI) -> None:
        self.model_ref = weakref.ref(model)
        self.params = params
        self.api = api
        self.sequence: list[int] = []
        self.done = False
        self.steps = 0

    def append_tokens(self, tokens: list[int]) -> None:
        self.sequence.extend(tokens)

    def is_done(self) -> bool:
        return self.done

    def generate_next_token(self) -> None:
        self.api.generation_started.set()
        if self.api.block_generation:
            self.api.allow_generation.wait(timeout=3)
        if self.api.fail_generation:
            raise RuntimeError("simulated native generation failure")
        self.sequence.append(999)
        self.steps += 1
        self.api.generation_calls += 1
        self.done = self.steps >= self.api.generation_steps

    def get_sequence(self, index: int) -> list[int]:
        if index != 0:
            raise IndexError(index)
        return self.sequence


class FakeModel:
    def __init__(self, path: str) -> None:
        self.path = path


class FakeOrtGenAI:
    def __init__(self, output: str) -> None:
        self.output = output
        self.loads: list[str] = []
        self.template = ""
        self.messages: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] | None = None
        self.add_generation_prompt = False
        self.prompt = ""
        self.decoded_tokens: list[int] = []
        self.params: list[FakeGeneratorParams] = []
        self.generators: list[FakeGenerator] = []
        self.live_models: weakref.WeakSet[FakeModel] = weakref.WeakSet()
        self.fail_generation = False
        self.block_generation = False
        self.generation_steps = 1
        self.generation_calls = 0
        self.generation_started = threading.Event()
        self.allow_generation = threading.Event()
        self.allow_generation.set()

    def Model(self, path: str) -> FakeModel:
        if self.live_models:
            raise AssertionError("the previous model must be released before loading another")
        self.loads.append(path)
        model = FakeModel(path)
        self.live_models.add(model)
        return model

    def Tokenizer(self, model: FakeModel) -> FakeTokenizer:
        return FakeTokenizer(self)

    def GeneratorParams(self, model: FakeModel) -> FakeGeneratorParams:
        params = FakeGeneratorParams(model)
        self.params.append(params)
        return params

    def Generator(self, model: FakeModel, params: FakeGeneratorParams) -> FakeGenerator:
        generator = FakeGenerator(model, params, self)
        self.generators.append(generator)
        return generator


class OrtGenAIRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.tempdir.name)
        self.artifact = self.write_model_artifact(self.model_dir, "qwen3.5:2b")
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

    @staticmethod
    def write_model_artifact(directory: Path, model_id: str) -> VerifiedModelArtifact:
        directory.mkdir(parents=True, exist_ok=True)
        files = {
            "genai_config.json": "{}",
            "chat_template.jinja": (
                "{% if enable_thinking is defined and enable_thinking is true %}"
                "think{% else %}no-think{% endif %}"
            ),
        }
        manifest_files = []
        for name, content in files.items():
            file_bytes = content.encode("utf-8")
            (directory / name).write_bytes(file_bytes)
            manifest_files.append(
                {
                    "path": name,
                    "size": len(file_bytes),
                    "sha256": hashlib.sha256(file_bytes).hexdigest(),
                }
            )
        manifest = json.dumps(
            {
                "schemaVersion": 1,
                "format": "onnxruntime-genai",
                "modelId": model_id,
                "files": manifest_files,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        (directory / "lumi-model-manifest.json").write_bytes(manifest)
        return VerifiedModelArtifact(model_id, directory, hashlib.sha256(manifest).hexdigest())

    def make_runtime(self, api: FakeOrtGenAI, **config_overrides: Any) -> OrtGenAIChatRuntime:
        model_artifacts = config_overrides.pop(
            "model_artifacts", {"qwen3.5:2b": self.artifact}
        )
        config = OrtGenAIConfig(
            model_artifacts=model_artifacts,
            max_context_tokens=256,
            max_output_tokens=32,
            **config_overrides,
        )
        return OrtGenAIChatRuntime(config, api=api)

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

    async def test_lazy_in_process_tool_call_hides_reasoning_and_uses_native_template(self) -> None:
        api = FakeOrtGenAI(
            "<think>private reasoning</think>Searching now.\n"
            "<tool_call><function=catalog_search>\n"
            "<parameter=query>\nFate/Zero\n</parameter>\n"
            "</function></tool_call>"
        )
        runtime = self.make_runtime(api)
        await runtime.open()
        self.assertEqual(api.loads, [])

        response = await runtime.complete(self.make_request(tools=(self.tool,)))

        self.assertEqual(len(api.loads), 1)
        self.assertTrue(api.add_generation_prompt)
        self.assertIn("if false", api.template)
        self.assertNotIn("enable_thinking", api.template)
        self.assertEqual(api.tools[0]["function"]["name"], "catalog_search")
        self.assertEqual(api.messages[0]["role"], "system")
        self.assertEqual(api.params[0].options, {"max_length": 27, "do_sample": False})
        self.assertEqual(response.message.content, "Searching now.")
        self.assertEqual(response.message.tool_calls[0].name, "catalog_search")
        self.assertEqual(response.message.tool_calls[0].arguments, {"query": "Fate/Zero"})
        self.assertNotIn("private reasoning", response.message.content)
        await runtime.close()

    async def test_thinking_can_be_enabled_and_runtime_is_lazy_until_first_completion(self) -> None:
        api = FakeOrtGenAI("A short answer.")
        runtime = self.make_runtime(api)
        await runtime.open()

        response = await runtime.complete(self.make_request(thinking=True))

        self.assertEqual(response.message.content, "A short answer.")
        self.assertIn("if true", api.template)
        await runtime.close()

    async def test_model_switch_unloads_before_loading_the_next_model(self) -> None:
        api = FakeOrtGenAI("A short answer.")
        first_artifact = self.write_model_artifact(
            self.model_dir / "qwen35-2b", "qwen3.5:2b"
        )
        second_artifact = self.write_model_artifact(
            self.model_dir / "qwen35-4b", "qwen3.5:4b"
        )
        runtime = self.make_runtime(
            api,
            model_artifacts={
                "qwen3.5:2b": first_artifact,
                "qwen3.5:4b": second_artifact,
            },
        )
        await runtime.open()

        await runtime.complete(self.make_request(model="qwen3.5:2b"))
        await runtime.complete(self.make_request(model="qwen3.5:4b"))

        self.assertEqual(len(api.loads), 2)
        self.assertEqual(runtime._loaded.model_id, "qwen3.5:4b")
        self.assertEqual(len(api.live_models), 1)
        await runtime.close()

    async def test_idle_model_unloads_without_a_separate_service(self) -> None:
        api = FakeOrtGenAI("A short answer.")
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

    async def test_idle_model_unloads_after_generation_failure(self) -> None:
        api = FakeOrtGenAI("unused")
        api.fail_generation = True
        runtime = self.make_runtime(api, idle_unload_seconds=0)
        await runtime.open()

        with self.assertRaisesRegex(OrtGenAIRuntimeError, "inference failed"):
            await runtime.complete(self.make_request())

        for _ in range(20):
            if runtime._loaded is None and runtime._resident_model_id is None:
                break
            await asyncio.sleep(0.05)

        self.assertIsNone(runtime._loaded)
        self.assertIsNone(runtime._resident_model_id)
        await runtime.close()

    async def test_cancelled_inference_stops_between_native_token_steps(self) -> None:
        api = FakeOrtGenAI("unused")
        api.block_generation = True
        api.generation_steps = 10
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
        self.assertEqual(api.generation_calls, 1)
        self.assertIsNone(runtime._loaded)

    async def test_rejects_state_changing_tools_before_loading_a_model(self) -> None:
        api = FakeOrtGenAI("should never run")
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
        api = FakeOrtGenAI("unused")
        runtime = self.make_runtime(api)
        await runtime.open()

        with self.assertRaisesRegex(ValueError, "leaves no room"):
            await runtime.complete(self.make_request(context_size=3))

        self.assertEqual(api.generators, [])
        await runtime.close()

    async def test_malformed_or_missing_required_tool_arguments_fail_closed(self) -> None:
        for output in (
            "<tool_call><function=catalog_search><parameter=query>jazz</function></tool_call>",
            "<tool_call><function=catalog_search></function></tool_call>",
        ):
            with self.subTest(output=output):
                api = FakeOrtGenAI(output)
                runtime = self.make_runtime(api)
                await runtime.open()
                with self.assertRaises(OrtGenAIProtocolError):
                    await runtime.complete(self.make_request(tools=(self.tool,)))
                await runtime.close()

    async def test_missing_model_files_produce_a_safe_runtime_error(self) -> None:
        api = FakeOrtGenAI("unused")
        missing = self.model_dir / "missing"
        runtime = OrtGenAIChatRuntime(
            OrtGenAIConfig(
                model_artifacts={
                    "qwen3.5:2b": VerifiedModelArtifact(
                        "qwen3.5:2b", missing, "0" * 64
                    )
                }
            ),
            api=api,
        )
        await runtime.open()

        with self.assertRaises(OrtGenAIRuntimeError):
            await runtime.complete(self.make_request())

        await runtime.close()

    async def test_rejects_a_changed_model_file_before_native_load(self) -> None:
        api = FakeOrtGenAI("unused")
        runtime = self.make_runtime(api)
        await runtime.open()
        (self.model_dir / "chat_template.jinja").write_text("changed size", encoding="utf-8")

        with self.assertRaisesRegex(OrtGenAIRuntimeError, "missing or changed"):
            await runtime.complete(self.make_request())

        self.assertEqual(api.loads, [])
        await runtime.close()

    async def test_rejects_same_size_model_file_tampering_before_native_load(self) -> None:
        api = FakeOrtGenAI("A short answer.")
        runtime = self.make_runtime(api)
        await runtime.open()
        template_path = self.model_dir / "chat_template.jinja"
        original = template_path.read_bytes()
        template_path.write_bytes(b"x" + original[1:])

        with self.assertRaisesRegex(OrtGenAIRuntimeError, "integrity check"):
            await runtime.complete(self.make_request())

        self.assertEqual(api.loads, [])
        await runtime.close()

    async def test_rechecks_changed_file_after_cached_verification(self) -> None:
        api = FakeOrtGenAI("A short answer.")
        runtime = self.make_runtime(api)
        await runtime.open()
        await runtime.complete(self.make_request())
        await runtime.unload()

        template_path = self.model_dir / "chat_template.jinja"
        original = template_path.read_bytes()
        template_path.write_bytes(b"x" + original[1:])
        with self.assertRaisesRegex(OrtGenAIRuntimeError, "integrity check"):
            await runtime.complete(self.make_request())

        self.assertEqual(len(api.loads), 1)
        await runtime.close()

    async def test_rejects_unlisted_files_before_native_load(self) -> None:
        api = FakeOrtGenAI("A short answer.")
        runtime = self.make_runtime(api)
        await runtime.open()
        (self.model_dir / "unexpected.bin").write_bytes(b"not in the signed manifest")

        with self.assertRaisesRegex(OrtGenAIRuntimeError, "do not match their manifest"):
            await runtime.complete(self.make_request())

        self.assertEqual(api.loads, [])
        await runtime.close()

    async def test_rejects_a_changed_manifest_before_native_load(self) -> None:
        api = FakeOrtGenAI("unused")
        runtime = self.make_runtime(api)
        await runtime.open()
        manifest_file = self.model_dir / "lumi-model-manifest.json"
        manifest_file.write_bytes(manifest_file.read_bytes() + b" ")

        with self.assertRaisesRegex(OrtGenAIRuntimeError, "manifest changed"):
            await runtime.complete(self.make_request())

        self.assertEqual(api.loads, [])
        await runtime.close()

    def test_config_rejects_an_artifact_mapped_under_a_different_model_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "identity"):
            OrtGenAIConfig(model_artifacts={"qwen3.5:4b": self.artifact})

    def test_runtime_api_version_is_exposed(self) -> None:
        self.assertEqual(LUMI_RUNTIME_API_VERSION, 1)
