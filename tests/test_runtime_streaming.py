from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
