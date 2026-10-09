from __future__ import annotations

import asyncio
import ctypes
import hashlib
import json
import os
import struct
import tempfile
import threading
import unittest
import weakref
from pathlib import Path
from types import FunctionType, SimpleNamespace
from typing import Any
from unittest.mock import patch

from lumi.contracts import ChatMessage, ModelRequest, ToolDefinition
from lumi.model_installation import (
    GGUF_FORMAT,
    MODEL_MANIFEST_SCHEMA_VERSION,
    Qwen35ModelSpec,
)
from lumi.runtime.acceleration import GpuDevice, choose_acceleration
from lumi.runtime.llama_cpp import (
    LUMI_RUNTIME_API_VERSION,
    LlamaCppChatRuntime,
    LlamaCppConfig,
    LlamaCppProtocolError,
    LlamaCppRuntimeError,
    VerifiedModelArtifact,
    _initialize_backend_registry,
)

_GGUF_FILENAME = "Qwen_Qwen3.5-test-Q4_K_M.gguf"
# Each fake Llama constructor receives its own binding in a copied globals dict.
llama_cpp: Any = None


class _FakeModelParams(ctypes.Structure):
    _fields_ = [("devices", ctypes.c_void_p)]


class _FakeNativeFunction:
    def __init__(self, callback):
        self.callback = callback
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        return self.callback(*args)


def _fake_llama_init(self, **options: Any) -> None:
    api = self._api
    params = llama_cpp.llama_model_default_params()
    if params.devices:
        devices = ctypes.cast(params.devices, ctypes.POINTER(ctypes.c_void_p))
        api.native_device_lists.append((devices[0], devices[1]))
    else:
        api.native_device_lists.append((None, None))
    model_path = options["model_path"]
    if options["n_gpu_layers"] and api.fail_gpu_init:
        api.loads.append(model_path)
        api.model_options.append(options)
        raise RuntimeError("simulated GPU initialization failure")
    if api.live_models:
        raise AssertionError("The previous native model must unload before switching")
    api.loads.append(model_path)
    api.model_options.append(options)
    FakeModel.__init__(self, api, model_path, options)
    api.live_models.add(self)


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("ascii")
    return struct.pack("<Q", len(encoded)) + encoded


_GGUF_BYTES = (
    b"GGUF"
    + struct.pack("<IQQ", 3, 0, 2)
    + _gguf_string("general.architecture")
    + struct.pack("<I", 8)
    + _gguf_string("qwen3")
    + _gguf_string("qwen3.block_count")
    + struct.pack("<II", 4, 28)
)


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
            added_special=self.api.added_special,
        )


class FakeModel:
    def __init__(self, api: FakeLlamaAPI, model_path: str, options: dict[str, Any]) -> None:
        self.api = api
        self.model_path = model_path
        self.options = options
        self.chat_handler: Any = None
        self.metadata = {
            "tokenizer.chat_template": "Qwen3.5 native Jinja chat template",
            "qwen3.block_count": 28,
        }

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

    def create_completion(self, **options: Any) -> Any:
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
        if options.get("stream"):
            return self._stream_completion(options)
        return {"choices": [{"text": self.api.output}]}

    def create_chat_completion(self, **options: Any) -> Any:
        if not callable(self.chat_handler):
            raise RuntimeError("A chat handler is required")
        self.api.chat_completion_options.append(options.copy())
        return self.chat_handler(llama=self, **options)

    def _stream_completion(self, options: dict[str, Any]):
        if self.api.block_generation:
            self.api.allow_generation.wait(timeout=3)
            if options["stopping_criteria"]([], []):
                return
        chunks = self.api.stream_chunks if self.api.stream_chunks is not None else [self.api.output]
        for index, chunk in enumerate(chunks):
            should_fail = self.api.fail_after_stream_chunks == index
            if self.api.fail_gpu_after_stream_chunks and not self.options.get("n_gpu_layers"):
                should_fail = False
            if should_fail:
                raise RuntimeError("simulated mid-stream native generation failure")
            if index == 0 and self.api.block_after_first_stream_chunk:
                self.api.first_stream_chunk_started.set()
            yield {"choices": [{"text": chunk}]}
            if index == 0 and self.api.block_after_first_stream_chunk:
                self.api.allow_next_stream_chunk.wait(timeout=3)


class FakeLlamaAPI:
    def __init__(self, output: str, *, stream_chunks: list[str] | None = None) -> None:
        self.output = output
        self.stream_chunks = stream_chunks
        self.loads: list[str] = []
        self.model_options: list[dict[str, Any]] = []
        self.formatter_options: list[dict[str, Any]] = []
        self.formatted: list[dict[str, Any]] = []
        self.tokenized_prompts: list[tuple[bytes, bool, bool]] = []
        self.completion_options: list[dict[str, Any]] = []
        self.chat_completion_options: list[dict[str, Any]] = []
        self.chat_function_call: dict[str, Any] | None = None
        self.live_models: weakref.WeakSet[FakeModel] = weakref.WeakSet()
        self.prompt_token_count = 3
        self.fail_generation = False
        self.fail_gpu_init = False
        self.fail_after_stream_chunks: int | None = None
        self.fail_gpu_after_stream_chunks = False
        self.chat_stream_chunks: list[dict[str, Any]] | None = None
        self.block_after_first_stream_chunk = False
        self.gpu_devices: tuple[GpuDevice, ...] = ()
        self.native_device_lists: list[tuple[int | None, int | None]] = []
        self.native_device_handles = {
            b"CPU": 100,
            b"CUDA0": 101,
            b"Vulkan0": 102,
            b"HIP0": 103,
            b"CUDA1": 104,
        }
        self.ggml = SimpleNamespace(
            ggml_backend_dev_by_name=_FakeNativeFunction(
                lambda name: self.native_device_handles.get(name)
            )
        )
        self.ggml_base = None
        self.llama_cpp_bindings = SimpleNamespace(
            llama_model_default_params=self._model_default_params
        )
        llama_globals = dict(globals())
        llama_globals["llama_cpp"] = self.llama_cpp_bindings
        fake_init = FunctionType(
            _fake_llama_init.__code__,
            llama_globals,
            name=_fake_llama_init.__name__,
            argdefs=_fake_llama_init.__defaults__,
            closure=_fake_llama_init.__closure__,
        )
        fake_init.__kwdefaults__ = _fake_llama_init.__kwdefaults__
        fake_init.__annotations__ = _fake_llama_init.__annotations__.copy()
        self.Llama = type(
            "FakeLlama",
            (FakeModel,),
            {"__init__": fake_init, "_api": self},
        )
        self.block_generation = False
        self.generation_calls = 0
        self.formatter_stopping_criteria: list[Any] = []
        self.added_special = True
        self.generation_started = threading.Event()
        self.allow_generation = threading.Event()
        self.allow_generation.set()
        self.first_stream_chunk_started = threading.Event()
        self.allow_next_stream_chunk = threading.Event()
        self.allow_next_stream_chunk.set()
        self.Jinja2ChatFormatter = lambda **options: FakeFormatter(self, **options)
        self.chat_formatter_to_chat_completion_handler = self._make_chat_handler

    def _make_chat_handler(self, formatter: Any):
        def handle(
            *,
            llama: FakeModel,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            temperature: float = 0.0,
            max_tokens: int = 0,
            stream: bool = False,
            **kwargs: Any,
        ):
            formatted = formatter(
                messages=messages,
                tools=tools,
                enable_thinking=kwargs.get("enable_thinking", False),
            )
            prompt = llama.tokenize(
                formatted.prompt.encode("utf-8"),
                add_bos=not formatted.added_special,
                special=True,
            )
            completion = llama.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=formatted.stop,
                stopping_criteria=formatted.stopping_criteria,
                stream=stream,
            )
            if not stream:
                message = {
                    "role": "assistant",
                    "content": completion["choices"][0]["text"],
                }
                if self.chat_function_call is not None:
                    message["function_call"] = self.chat_function_call
                return {
                    "choices": [
                        {
                            "message": message
                        }
                    ]
                }

            if self.chat_stream_chunks is not None:
                # Drive the native iterator for cancellation/failure behavior, then
                # return explicit binding-shaped chunks for protocol edge cases.
                list(completion)
                return iter(self.chat_stream_chunks)

            def stream_chat_chunks():
                for chunk in completion:
                    yield {
                        "choices": [
                            {
                                "delta": {
                                    "content": chunk["choices"][0]["text"],
                                    "reasoning_content": "private reasoning must not appear",
                                }
                            }
                        ]
                    }

            return stream_chat_chunks()

        return handle

    @staticmethod
    def _model_default_params() -> _FakeModelParams:
        return _FakeModelParams()

    def list_gpu_devices(self) -> tuple[GpuDevice, ...]:
        return self.gpu_devices


class LlamaCppBackendRegistryTests(unittest.TestCase):
    def test_packaged_backends_load_before_registry_init_once(self) -> None:
        events: list[tuple[str, object]] = []
        llama_type = type("FakeLlama", (), {})
        backend_loader = _FakeNativeFunction(lambda path: events.append(("load", path)))
        backend_init = _FakeNativeFunction(
            lambda: events.append(("initialize", None))
        )

        backend_directory = Path(__file__).resolve().parents[1] / "lumi"
        encoded_backend_directory = os.fsencode(backend_directory.resolve())
        api = SimpleNamespace(
            Llama=llama_type,
            ggml=SimpleNamespace(ggml_backend_load_all_from_path=backend_loader),
            ggml_backend_directory=backend_directory,
            llama_cpp=SimpleNamespace(llama_backend_init=backend_init),
        )

        _initialize_backend_registry(api)
        _initialize_backend_registry(api)

        self.assertEqual(
            events,
            [("load", encoded_backend_directory), ("initialize", None)],
        )
        self.assertEqual(backend_loader.argtypes, [ctypes.c_char_p])
        self.assertIsNone(backend_loader.restype)


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
        self.assertEqual(api.tokenized_prompts[0][1:], (False, True))
        self.assertEqual(api.completion_options[0]["prompt"], list(range(api.prompt_token_count)))
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

    async def test_thinking_is_applied_by_template_without_unsupported_binding_argument(
        self,
    ) -> None:
        api = FakeLlamaAPI("Answer")
        runtime = self.make_runtime(api)
        await runtime.open()
        await runtime.complete(self.make_request(thinking=False))
        await runtime.complete(self.make_request(thinking=True))
        self.assertEqual([item["enable_thinking"] for item in api.formatted], [False, True])
        self.assertTrue(
            all("enable_thinking" not in options for options in api.chat_completion_options)
        )
        await runtime.close()

    async def test_automatic_acceleration_selects_gpu_and_honors_layer_cap(self) -> None:
        api = FakeLlamaAPI("Answer")
        api.gpu_devices = (
            GpuDevice("cuda", "NVIDIA test device", 2_000_000_000, 8_000_000_000, "CUDA0"),
            GpuDevice("cuda", "NVIDIA test device 2", 4_000_000_000, 8_000_000_000, "CUDA1"),
        )
        default_params = api.llama_cpp_bindings.llama_model_default_params
        runtime = self.make_runtime(api, acceleration_mode="automatic", n_gpu_layers=12)
        await runtime.open()

        await runtime.complete(self.make_request())

        offloaded = api.model_options[0]["n_gpu_layers"]
        self.assertGreater(offloaded, 0)
        self.assertEqual(offloaded, 12)
        self.assertEqual(runtime.acceleration_status()["state"], "ready")
        self.assertEqual(runtime.acceleration_status()["selectedBackend"], "cuda")
        self.assertEqual(api.native_device_lists, [(104, None)])
        self.assertIs(api.llama_cpp_bindings.llama_model_default_params, default_params)
        self.assertEqual(runtime.acceleration_status()["offloadedLayers"], offloaded)
        self.assertEqual(runtime.acceleration_status()["totalLayers"], 28)
        await runtime.close()

    async def test_automatic_acceleration_uses_vram_for_partial_layer_offload(self) -> None:
        choice = choose_acceleration(
            "automatic",
            (
                GpuDevice(
                    "cuda", "NVIDIA test device", 1_500_000_000, 4_000_000_000, "CUDA0"
                ),
            ),
            model_size_bytes=1_400_000_000,
            total_layers=28,
        )

        self.assertEqual(choice.backend, "cuda")
        self.assertGreater(choice.offloaded_layers, 0)
        self.assertLess(choice.offloaded_layers, choice.total_layers)

    async def test_insufficient_vram_falls_back_to_cpu(self) -> None:
        api = FakeLlamaAPI("Answer")
        api.gpu_devices = (
            GpuDevice("vulkan", "AMD test GPU", 700_000_000, 1_000_000_000, "Vulkan0"),
        )
        runtime = self.make_runtime(api, acceleration_mode="gpu_preferred")
        await runtime.open()

        await runtime.complete(self.make_request())

        self.assertEqual(api.model_options[0]["n_gpu_layers"], 0)
        self.assertEqual(api.native_device_lists, [(100, None)])
        status = runtime.acceleration_status()
        self.assertEqual(status["state"], "cpu_fallback")
        self.assertEqual(status["selectedBackend"], "cpu")
        self.assertIn("memory", status["fallbackReason"].casefold())
        await runtime.close()

    async def test_gpu_model_initialization_failure_retries_cpu(self) -> None:
        api = FakeLlamaAPI("Answer")
        api.gpu_devices = (
            GpuDevice("cuda", "NVIDIA test device", 4_000_000_000, 8_000_000_000, "CUDA0"),
        )
        api.fail_gpu_init = True
        runtime = self.make_runtime(api)
        await runtime.open()

        answer = await runtime.complete(self.make_request())

        self.assertEqual(answer.message.content, "Answer")
        self.assertGreater(api.model_options[0]["n_gpu_layers"], 0)
        self.assertEqual(api.model_options[1]["n_gpu_layers"], 0)
        self.assertEqual(api.native_device_lists, [(101, None), (100, None)])
        self.assertFalse(api.model_options[1]["offload_kqv"])
        self.assertFalse(api.model_options[1]["op_offload"])
        self.assertEqual(runtime.acceleration_status()["state"], "cpu_fallback")
        await runtime.close()

    async def test_cpu_only_mode_ignores_detected_gpu(self) -> None:
        api = FakeLlamaAPI("Answer")
        api.gpu_devices = (
            GpuDevice("hip", "AMD Radeon test", 8_000_000_000, 12_000_000_000, "HIP0"),
        )
        runtime = self.make_runtime(api, acceleration_mode="cpu_only")
        await runtime.open()

        with patch(
            "lumi.runtime.llama_cpp._available_gpu_devices",
            side_effect=AssertionError("CPU-only mode must not enumerate GPU devices"),
        ) as detect_devices:
            await runtime.complete(self.make_request())
        detect_devices.assert_not_called()

        self.assertEqual(api.model_options[0]["n_gpu_layers"], 0)
        self.assertEqual(api.native_device_lists, [(100, None)])
        self.assertFalse(api.model_options[0]["offload_kqv"])
        self.assertFalse(api.model_options[0]["op_offload"])
        status = runtime.acceleration_status()
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["selectedBackend"], "cpu")
        self.assertEqual(status["selectedDevice"], "CPU")
        self.assertEqual(status["offloadedLayers"], 0)
        self.assertIsNone(status["fallbackReason"])
        await runtime.close()

    async def test_stream_emits_incremental_visible_text_and_one_completion(self) -> None:
        api = FakeLlamaAPI(
            "unused",
            stream_chunks=["<|channel|>final\nA short", " answer", "."],
        )
        runtime = self.make_runtime(api)
        await runtime.open()

        events = [event async for event in runtime.stream(self.make_request())]

        deltas = [event.text for event in events if event.kind == "delta"]
        completions = [event for event in events if event.kind == "complete"]
        self.assertEqual("".join(deltas), "A short answer.")
        self.assertEqual(len(completions), 1)
        self.assertEqual(completions[0].response.message.content, "A short answer.")
        self.assertNotIn(
            "private reasoning must not appear",
            "".join(item for item in deltas if item is not None),
        )
        self.assertTrue(api.completion_options[0]["stream"])
        self.assertTrue(api.chat_completion_options[0]["stream"])
        await runtime.close()

    async def test_stream_plain_content_without_tools_does_not_require_final_channel(self) -> None:
        api = FakeLlamaAPI(
            "unused",
            stream_chunks=["A plain", " assistant", " response."],
        )
        runtime = self.make_runtime(api)
        await runtime.open()

        events = [event async for event in runtime.stream(self.make_request())]

        deltas = [event.text for event in events if event.kind == "delta"]
        self.assertEqual("".join(deltas), "A plain assistant response.")
        self.assertEqual(events[-1].response.message.content, "A plain assistant response.")
        await runtime.close()

    async def test_tool_enabled_plain_final_response_without_prompt_prefix_is_visible(self) -> None:
        api = FakeLlamaAPI("unused", stream_chunks=["Safe response without a channel."])
        runtime = self.make_runtime(api)
        await runtime.open()

        events = [
            event
            async for event in runtime.stream(self.make_request(tools=(self.tool,)))
        ]

        self.assertEqual([event.kind for event in events], ["delta", "complete"])
        self.assertEqual(events[0].text, "Safe response without a channel.")
        self.assertEqual(events[1].response.message.content, "Safe response without a channel.")
        await runtime.close()

    async def test_thinking_enabled_stream_hides_prompt_prefixed_reasoning_with_tools(self) -> None:
        api = FakeLlamaAPI(
            "unused",
            stream_chunks=["private reasoning</think>\n\nSafe answer after the prompt prefix."],
        )
        runtime = self.make_runtime(api)
        await runtime.open()

        events = [
            event
            async for event in runtime.stream(
                self.make_request(thinking=True, tools=(self.tool,))
            )
        ]

        self.assertEqual([event.kind for event in events], ["delta", "complete"])
        self.assertEqual(events[0].text, "Safe answer after the prompt prefix.")
        self.assertEqual(
            events[1].response.message.content,
            "Safe answer after the prompt prefix.",
        )
        await runtime.close()

    async def test_nonstream_tool_enabled_responses_follow_qwen_prompt_prefix(self) -> None:
        cases = (
            (False, "Safe answer without a prompt prefix.", "Safe answer without a prompt prefix."),
            (
                True,
                "private reasoning</think>\n\nSafe answer after the prompt prefix.",
                "Safe answer after the prompt prefix.",
            ),
        )
        for thinking, output, expected in cases:
            with self.subTest(thinking=thinking):
                api = FakeLlamaAPI(output)
                runtime = self.make_runtime(api)
                await runtime.open()

                response = await runtime.complete(
                    self.make_request(thinking=thinking, tools=(self.tool,))
                )

                self.assertEqual(response.message.content, expected)
                self.assertNotIn("private reasoning", response.message.content)
                await runtime.close()

    async def test_thinking_tool_call_body_without_prompt_prefix_hides_arguments(self) -> None:
        api = FakeLlamaAPI(
            "unused",
            stream_chunks=[
                "private reasoning</think>\n"
                "<tool_call><function=catalog_search>"
                "<parameter=query>Fate/Zero secret</parameter></function></tool_call>"
            ],
        )
        runtime = self.make_runtime(api)
        await runtime.open()

        events = [
            event
            async for event in runtime.stream(
                self.make_request(thinking=True, tools=(self.tool,))
            )
        ]

        self.assertEqual([event.kind for event in events], ["complete"])
        self.assertEqual(events[0].response.message.content, "")
        self.assertEqual(
            events[0].response.message.tool_calls[0].arguments,
            {"query": "Fate/Zero secret"},
        )
        await runtime.close()

    async def test_stream_filters_reasoning_and_tool_call_transitions(self) -> None:
        api = FakeLlamaAPI(
            "unused",
            stream_chunks=[
                "<|channel|>analysis private planning\n",
                "<|channel|>final Checking now.",
                "<tool_call><function=catalog_search>",
                "<parameter=query>Fate/Zero</parameter></function></tool_call>",
                "<|channel|>final This text follows a tool call.",
            ],
        )
        runtime = self.make_runtime(api)
        await runtime.open()

        events = [event async for event in runtime.stream(self.make_request(tools=(self.tool,)))]

        self.assertEqual([event.kind for event in events], ["delta", "reset", "complete"])
        self.assertEqual(events[0].text, "Checking now.")
        self.assertEqual(events[1].reason, "intermediate")
        self.assertEqual(events[2].response.message.content, "")
        self.assertEqual(events[2].response.message.tool_calls[0].name, "catalog_search")
        await runtime.close()

    async def test_stream_ignores_reasoning_and_assembles_structured_tool_deltas(self) -> None:
        api = FakeLlamaAPI("unused")
        api.chat_stream_chunks = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_content": "private thought",
                            "content": "Visible draft before the tool call",
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "function": {
                                        "name": "catalog_search",
                                        "arguments": '{"query":"Fate',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '/Zero"}'},
                                }
                            ]
                        }
                    }
                ]
            },
        ]
        runtime = self.make_runtime(api)
        await runtime.open()

        events = [event async for event in runtime.stream(self.make_request(tools=(self.tool,)))]

        self.assertEqual([event.kind for event in events], ["delta", "reset", "complete"])
        self.assertEqual(events[0].text, "Visible draft before the tool call")
        self.assertEqual(events[1].reason, "intermediate")
        visible_events = "".join(event.text or "" for event in events)
        self.assertNotIn("Fate", visible_events)
        self.assertNotIn("private thought", visible_events)
        response = events[2].response
        self.assertEqual(response.message.content, "")
        self.assertEqual(response.message.tool_calls[0].call_id, "call-1")
        self.assertEqual(response.message.tool_calls[0].name, "catalog_search")
        self.assertEqual(response.message.tool_calls[0].arguments, {"query": "Fate/Zero"})
        self.assertEqual(
            api.chat_completion_options[0]["tools"][0]["function"]["name"],
            "catalog_search",
        )
        await runtime.close()

    async def test_structured_tool_call_resets_visible_preamble_before_control_chunks(self) -> None:
        api = FakeLlamaAPI("unused")
        api.chat_stream_chunks = [
            {
                "choices": [
                    {"delta": {"content": "<|channel|>final Safe preamble"}}
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-private",
                                    "function": {
                                        "name": "catalog_search",
                                        "arguments": '{"query":"secret term"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
        ]
        runtime = self.make_runtime(api)
        await runtime.open()

        events = [event async for event in runtime.stream(self.make_request(tools=(self.tool,)))]

        self.assertEqual([event.kind for event in events], ["delta", "reset", "complete"])
        self.assertEqual(events[0].text, "Safe preamble")
        self.assertEqual(events[1].reason, "intermediate")
        visible_events = "".join(event.text or "" for event in events)
        self.assertNotIn("secret term", visible_events)
        self.assertEqual(events[2].response.message.content, "")
        self.assertEqual(
            events[2].response.message.tool_calls[0].arguments,
            {"query": "secret term"},
        )
        await runtime.close()

    async def test_stream_resets_unmarked_prefix_when_later_channel_changes_visibility(
        self,
    ) -> None:
        api = FakeLlamaAPI(
            "unused",
            stream_chunks=[
                "Unmarked prefix",
                "<|channel|>analysis\nprivate reasoning",
                "<|channel|>final\nVisible final answer",
            ],
        )
        runtime = self.make_runtime(api)
        await runtime.open()

        events = [event async for event in runtime.stream(self.make_request())]

        assembled: list[str] = []
        for event in events:
            if event.kind == "reset":
                assembled.clear()
            elif event.kind == "delta":
                assembled.append(event.text or "")
        completion = events[-1].response.message.content
        self.assertEqual(
            [event.kind for event in events],
            ["delta", "reset", "delta", "complete"],
        )
        self.assertEqual(assembled, ["Visible final answer"])
        self.assertEqual("".join(assembled), completion)
        self.assertNotIn("private reasoning", "".join(assembled))
        await runtime.close()

    async def test_unparsed_qwen_tool_tokens_reset_prefix_and_fail_closed(self) -> None:
        api = FakeLlamaAPI("unused")
        api.chat_stream_chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "content": (
                                '<|channel|>final Safe prefix <|tool_call_begin|>'
                                'catalog_search <|tool_call_argument_begin|>'
                                '{"query":"private argument"}<|tool_call_end|>'
                            )
                        }
                    }
                ]
            }
        ]
        runtime = self.make_runtime(api)
        await runtime.open()
        events = []

        with self.assertRaisesRegex(LlamaCppProtocolError, "unsupported tool-control"):
            async for event in runtime.stream(self.make_request(tools=(self.tool,))):
                events.append(event)

        self.assertEqual([event.kind for event in events], ["delta", "reset"])
        self.assertEqual(events[0].text, "Safe prefix")
        self.assertEqual(events[1].reason, "intermediate")
        self.assertNotIn("private argument", "".join(event.text or "" for event in events))
        await runtime.close()

    async def test_unparsed_qwen_tool_tokens_cannot_escape_nonstream_completion(self) -> None:
        api = FakeLlamaAPI(
            '<|channel|>final Visible <|tool_call_begin|>catalog_search '
            '<|tool_call_argument_begin|>{"query":"private argument"}<|tool_call_end|>'
        )
        runtime = self.make_runtime(api)
        await runtime.open()

        with self.assertRaisesRegex(LlamaCppProtocolError, "unsupported tool-control"):
            await runtime.complete(self.make_request(tools=(self.tool,)))

        await runtime.close()

    async def test_legacy_function_call_is_rejected_without_returning_preamble(self) -> None:
        api = FakeLlamaAPI("Visible preamble")
        api.chat_function_call = {
            "name": "catalog_search",
            "arguments": '{"query":"private term"}',
        }
        runtime = self.make_runtime(api)
        await runtime.open()

        with self.assertRaisesRegex(LlamaCppProtocolError, "unsupported legacy function call"):
            await runtime.complete(self.make_request(tools=(self.tool,)))

        await runtime.close()

    async def test_legacy_stream_function_call_resets_and_rejects_without_arguments(self) -> None:
        api = FakeLlamaAPI("unused")
        api.chat_stream_chunks = [
            {"choices": [{"delta": {"content": "Visible preamble"}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "function_call": {
                                "name": "catalog_search",
                                "arguments": '{"query":"private term"}',
                            }
                        }
                    }
                ]
            },
        ]
        runtime = self.make_runtime(api)
        await runtime.open()
        events = []

        with self.assertRaisesRegex(LlamaCppProtocolError, "unsupported legacy function call"):
            async for event in runtime.stream(self.make_request(tools=(self.tool,))):
                events.append(event)

        self.assertEqual([event.kind for event in events], ["delta", "reset"])
        self.assertEqual(events[0].text, "Visible preamble")
        self.assertEqual(events[1].reason, "intermediate")
        self.assertNotIn("private term", "".join(event.text or "" for event in events))
        await runtime.close()

    async def test_gpu_failure_after_visible_text_resets_then_retries_cpu(self) -> None:
        api = FakeLlamaAPI(
            "unused",
            stream_chunks=["<|channel|>final\npartial GPU text", " finished"],
        )
        api.gpu_devices = (
            GpuDevice("cuda", "NVIDIA test device", 4_000_000_000, 8_000_000_000, "CUDA0"),
        )
        api.fail_after_stream_chunks = 1
        api.fail_gpu_after_stream_chunks = True
        runtime = self.make_runtime(api)
        await runtime.open()

        with patch(
            "lumi.runtime.llama_cpp._available_gpu_devices",
            return_value=api.gpu_devices,
        ) as detect_devices:
            events = [event async for event in runtime.stream(self.make_request())]
        detect_devices.assert_called_once()

        self.assertEqual(
            [event.kind for event in events],
            ["delta", "reset", "delta", "delta", "complete"],
        )
        self.assertEqual(events[0].text, "partial GPU text")
        self.assertIsNone(events[1].text)
        self.assertEqual(events[1].reason, "cpu_fallback")
        self.assertEqual(events[2].text, "partial GPU text")
        self.assertEqual(events[3].text, " finished")
        self.assertEqual(
            "".join(event.text or "" for event in events[2:-1]),
            events[-1].response.message.content,
        )
        self.assertEqual(api.model_options[0]["n_gpu_layers"], 28)
        self.assertEqual(api.model_options[-1]["n_gpu_layers"], 0)
        self.assertEqual(runtime.acceleration_status()["state"], "cpu_fallback")
        await runtime.close()

    async def test_stream_exposes_only_final_channel_when_thinking_is_enabled(self) -> None:
        api = FakeLlamaAPI(
            "unused",
            stream_chunks=[
                "<|channel|>analysis\nprivate reasoning",
                "<|channel|>final\nSafe answer with `code` and **Markdown**.",
                "<|im_end|>",
            ],
        )
        runtime = self.make_runtime(api)
        await runtime.open()

        events = [
            event
            async for event in runtime.stream(self.make_request(thinking=True))
        ]

        deltas = [event.text for event in events if event.kind == "delta"]
        completion = events[-1]
        self.assertEqual("".join(deltas), "Safe answer with `code` and **Markdown**.")
        self.assertEqual(
            completion.response.message.content,
            "Safe answer with `code` and **Markdown**.",
        )
        await runtime.close()

    async def test_stream_cancellation_releases_native_generation_slot(self) -> None:
        api = FakeLlamaAPI("unused")
        api.block_generation = True
        api.allow_generation.clear()
        runtime = self.make_runtime(api)
        await runtime.open()

        async def consume() -> None:
            async for _event in runtime.stream(self.make_request()):
                pass

        consumer = asyncio.create_task(consume())
        await asyncio.to_thread(api.generation_started.wait, 1)
        consumer.cancel()
        api.allow_generation.set()
        with self.assertRaises(asyncio.CancelledError):
            await consumer
        for _ in range(50):
            if runtime._slot._value == 1:
                break
            await asyncio.sleep(0.02)
        self.assertEqual(runtime._slot._value, 1)
        await runtime.close()

    async def test_closing_stream_cancels_blocked_native_iterator_without_slot_leak(self) -> None:
        api = FakeLlamaAPI(
            "unused",
            stream_chunks=["<|channel|>final\nfirst", " second"],
        )
        api.block_after_first_stream_chunk = True
        api.allow_next_stream_chunk.clear()
        runtime = self.make_runtime(api)
        await runtime.open()
        stream = runtime.stream(self.make_request())

        first = await anext(stream)
        self.assertEqual(first.kind, "delta")
        await stream.aclose()
        api.allow_next_stream_chunk.set()
        for _ in range(50):
            if runtime._slot._value == 1:
                break
            await asyncio.sleep(0.02)

        self.assertEqual(runtime._slot._value, 1)
        await runtime.close()

    async def test_tool_capable_stream_emits_delta_before_native_iterator_finishes(self) -> None:
        api = FakeLlamaAPI(
            "unused",
            stream_chunks=["First visible", " second"],
        )
        api.block_after_first_stream_chunk = True
        api.allow_next_stream_chunk.clear()
        runtime = self.make_runtime(api)
        await runtime.open()
        stream = runtime.stream(self.make_request(tools=(self.tool,)))
        first_event = asyncio.create_task(anext(stream))

        self.assertTrue(await asyncio.to_thread(api.first_stream_chunk_started.wait, 1))
        first = await asyncio.wait_for(first_event, timeout=1)
        self.assertEqual(first.kind, "delta")
        self.assertEqual(first.text, "First visible")
        self.assertFalse(api.allow_next_stream_chunk.is_set())

        api.allow_next_stream_chunk.set()
        remainder = [event async for event in stream]
        self.assertEqual([event.kind for event in remainder], ["delta", "complete"])
        self.assertEqual(remainder[0].text, " second")
        self.assertEqual(remainder[-1].response.message.content, "First visible second")
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

    async def test_nullable_web_search_string_arguments_are_parsed_as_raw_strings(self) -> None:
        search_tool = ToolDefinition(
            "web_search",
            "Search public sources.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"},
                    "recency": {"type": ["string", "null"]},
                    "language": {"type": ["string", "null"]},
                },
                "required": ["query"],
            },
            data_scope="external_search",
            read_only=True,
        )
        api = FakeLlamaAPI(
            "<tool_call><function=web_search>"
            "<parameter=query>official season two announcement</parameter>"
            "<parameter=max_results>5</parameter>"
            "<parameter=recency>week</parameter>"
            "<parameter=language>ja</parameter>"
            "</function></tool_call>"
        )
        runtime = self.make_runtime(api)
        await runtime.open()

        response = await runtime.complete(self.make_request(tools=(search_tool,)))

        self.assertEqual(
            response.message.tool_calls[0].arguments,
            {
                "query": "official season two announcement",
                "max_results": 5,
                "recency": "week",
                "language": "ja",
            },
        )
        await runtime.close()

    async def test_nullable_web_search_string_arguments_accept_null(self) -> None:
        search_tool = ToolDefinition(
            "web_search",
            "Search public sources.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "recency": {"type": ["string", "null"]},
                    "language": {"type": ["string", "null"]},
                },
                "required": ["query"],
            },
            data_scope="external_search",
            read_only=True,
        )
        api = FakeLlamaAPI(
            "<tool_call><function=web_search>"
            "<parameter=query>Frieren season two</parameter>"
            "<parameter=recency>null</parameter>"
            "<parameter=language>null</parameter>"
            "</function></tool_call>"
        )
        runtime = self.make_runtime(api)
        await runtime.open()

        response = await runtime.complete(self.make_request(tools=(search_tool,)))

        self.assertEqual(
            response.message.tool_calls[0].arguments,
            {"query": "Frieren season two", "recency": None, "language": None},
        )
        await runtime.close()

    async def test_formatter_without_special_tokens_adds_bos_once(self) -> None:
        api = FakeLlamaAPI("Answer")
        api.added_special = False
        runtime = self.make_runtime(api)
        await runtime.open()

        await runtime.complete(self.make_request())

        self.assertEqual(api.tokenized_prompts[0][1:], (True, True))
        self.assertEqual(api.completion_options[0]["prompt"], list(range(api.prompt_token_count)))
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
