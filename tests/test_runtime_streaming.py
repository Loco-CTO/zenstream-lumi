from __future__ import annotations

import asyncio
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lumi.contracts import ChatMessage, ModelRequest
from lumi.runtime.llama_cpp import LlamaCppChatRuntime, LlamaCppConfig, VerifiedModelArtifact
from lumi.runtime.streaming import VisibleTextFilter


class VisibleTextFilterTests(unittest.TestCase):
    def test_split_channel_and_special_tokens_never_leak(self) -> None:
        filter_ = VisibleTextFilter(require_final_channel=True)
        chunks = (
            "<|chan",
            "nel|>analysis secret <|chan",
            "nel|>final Visible <|im_",
            "end|>",
        )

        visible = "".join(filter_.feed(chunk) for chunk in chunks) + filter_.finish()

        self.assertEqual(visible, "Visible")

    def test_split_thinking_and_tool_markers_are_filtered(self) -> None:
        thinking = VisibleTextFilter(require_final_channel=False)
        thinking_chunks = ("Visible <th", "ink>private thought</think", ">answer")
        visible = "".join(thinking.feed(chunk) for chunk in thinking_chunks) + thinking.finish()
        self.assertEqual(visible, "Visible answer")

    def test_prompt_prefixed_thinking_is_hidden_without_generated_open_marker(self) -> None:
        thinking = VisibleTextFilter(
            require_final_channel=False,
            initially_thinking=True,
        )
        chunks = ("private reasoning</thi", "nk>\n\nSafe answer")

        visible = "".join(thinking.feed(chunk) for chunk in chunks) + thinking.finish()

        self.assertEqual(visible, "Safe answer")

        final_channel = VisibleTextFilter(
            require_final_channel=False,
            initially_thinking=True,
        )
        channel_chunks = ("private reasoning<|chan", "nel|>final\nSafe answer")
        channel_visible = "".join(
            final_channel.feed(chunk) for chunk in channel_chunks
        ) + final_channel.finish()
        self.assertEqual(channel_visible, "Safe answer")

        tool = VisibleTextFilter(require_final_channel=False)
        tool_chunks = ("Before tool <tool_", "call><function=private>", "secret")
        visible_tool_text = "".join(tool.feed(chunk) for chunk in tool_chunks) + tool.finish()
        self.assertEqual(visible_tool_text, "Before tool")
        self.assertTrue(tool.suppressed)

    def test_split_qwen_tool_control_tokens_suppress_the_remaining_content(self) -> None:
        filter_ = VisibleTextFilter(require_final_channel=True)
        chunks = (
            "<|channel|>final Visible prefix <|tool_call_",
            "begin|>catalog_search <|tool_call_argument_begin|>",
            '{"query":"private argument"}<|tool_call_end|>',
        )

        visible = "".join(filter_.feed(chunk) for chunk in chunks) + filter_.finish()

        self.assertEqual(visible, "Visible prefix")
        self.assertTrue(filter_.suppressed)


class RuntimeStreamingCancellationTests(unittest.TestCase):
    def test_disconnect_during_cold_load_skips_prompt_and_generation(self) -> None:
        model_id = "qwen3.5:2b"
        artifact = VerifiedModelArtifact(model_id, Path("unused-model"), "a" * 64)
        runtime = LlamaCppChatRuntime(LlamaCppConfig(model_artifacts={model_id: artifact}))

        load_started = threading.Event()
        allow_load_to_finish = threading.Event()
        api_accessed = threading.Event()

        loaded = SimpleNamespace(
            model_id=model_id,
            model=object(),
            acceleration=SimpleNamespace(uses_gpu=False, total_layers=28),
            load_duration_ns=1,
        )

        def load_model(_model_id: str):
            load_started.set()
            if not allow_load_to_finish.wait(timeout=3):
                raise TimeoutError("test did not release the model load")
            return loaded

        request = ModelRequest(
            model=model_id,
            messages=(ChatMessage("user", "hello"),),
            tools=(),
            thinking=False,
            context_size=1024,
            output_tokens=64,
        )
        cancellation = threading.Event()
        worker_errors: list[BaseException] = []

        def unexpected_api_access():
            api_accessed.set()
            raise AssertionError("disconnected request continued after model load")

        def generate() -> None:
            try:
                runtime._generate_sync(request, cancellation, emit_delta=None)
            except BaseException as error:
                worker_errors.append(error)

        with patch.object(runtime, "_load_model", side_effect=load_model):
            with patch.object(runtime, "_api_module", side_effect=unexpected_api_access):
                worker = threading.Thread(target=generate, daemon=True)
                worker.start()
                self.assertTrue(load_started.wait(timeout=2))
                cancellation.set()

                # The native load itself cannot be interrupted. Once it returns,
                # the worker must observe disconnect before prompt preparation.
                allow_load_to_finish.set()
                worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(len(worker_errors), 1)
        self.assertIsInstance(worker_errors[0], asyncio.CancelledError)
        self.assertFalse(api_accessed.is_set())


if __name__ == "__main__":
    unittest.main()
