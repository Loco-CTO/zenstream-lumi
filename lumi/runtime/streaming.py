"""Incremental filtering of Qwen control tokens before text reaches the host."""

from __future__ import annotations

import re

_CHANNELS = frozenset({"analysis", "final", "commentary", "summary", "justify", "confidence"})
_SPECIAL_RE = re.compile(r"<\|[^>\r\n]+\|>")
_CHANNEL_MARKER = "<|channel|>"
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"
_CONTROL_MARKERS = (
    _CHANNEL_MARKER,
    _THINK_OPEN,
    _THINK_CLOSE,
    _TOOL_OPEN,
    _TOOL_CLOSE,
)


class VisibleTextFilter:
    """Strip model-control content while preserving text chunk boundaries.

    The loaded Qwen3.5 template leaves generation inside an open ``<think>`` block
    when thinking is enabled. Callers pass that prompt state explicitly because
    llama-cpp-python streams only generated content and does not echo the prefix.
    Explicit channel markers still switch visibility when they are present.
    """

    def __init__(
        self,
        *,
        require_final_channel: bool,
        initially_thinking: bool = False,
    ) -> None:
        self._pending = ""
        self._require_final_channel = require_final_channel
        self._visible = not require_final_channel
        self._reading_channel = False
        self._thinking = initially_thinking
        self._suppressed = False
        self._reset_required = False
        self._started = False
        self._trailing_whitespace = ""

    def feed(self, chunk: str) -> str:
        if not isinstance(chunk, str):
            raise TypeError("A model stream chunk must be text")
        if self._suppressed:
            return ""
        self._pending += chunk
        return self._drain(final=False)

    def finish(self) -> str:
        if self._suppressed:
            return ""
        return self._drain(final=True)

    @property
    def suppressed(self) -> bool:
        """Whether a tool call made this model response non-displayable."""

        return self._suppressed

    def consume_reset_required(self) -> bool:
        """Whether newly parsed channel control invalidated previously visible text."""

        if not self._reset_required:
            return False
        self._reset_required = False
        self._started = False
        self._trailing_whitespace = ""
        return True

    def _drain(self, *, final: bool) -> str:
        output: list[str] = []
        while self._pending and not self._suppressed:
            if self._reading_channel:
                channel = self._read_channel(final=final)
                if channel is None:
                    break
                self._reading_channel = False
                if self._started:
                    # A channel marker means any preceding unmarked prefix is
                    # excluded by the final-channel response parser. Let the
                    # stream caller clear that provisional text before more
                    # channel-specific content is exposed.
                    self._reset_required = True
                self._visible = channel == "final"
                if self._reset_required:
                    break
                continue

            if self._thinking:
                close = self._pending.find(_THINK_CLOSE)
                channel = self._pending.find(_CHANNEL_MARKER)
                if channel >= 0 and (close < 0 or channel < close):
                    # Some handlers include an explicit channel token in the
                    # generated body instead of relying on the open prompt
                    # prefix. Channel parsing provides a safe visibility gate.
                    self._pending = self._pending[channel + len(_CHANNEL_MARKER) :]
                    self._thinking = False
                    self._reading_channel = True
                    continue
                if close < 0:
                    if final:
                        self._pending = ""
                    else:
                        self._pending = _retain_marker_suffix(
                            self._pending,
                            _THINK_CLOSE,
                            _CHANNEL_MARKER,
                        )
                    break
                self._pending = self._pending[close + len(_THINK_CLOSE) :]
                self._thinking = False
                continue

            control = _find_control(self._pending)
            if control is None:
                if final:
                    # An unfinished special token is malformed control syntax.
                    if _is_control_prefix(self._pending):
                        self._pending = ""
                    else:
                        output.append(self._take_visible(self._pending, final=True))
                        self._pending = ""
                else:
                    safe, remainder = _split_partial_control(self._pending)
                    output.append(self._take_visible(safe, final=False))
                    self._pending = remainder
                break

            start, marker = control
            if start:
                output.append(self._take_visible(self._pending[:start], final=False))
            self._pending = self._pending[start + len(marker) :]
            if marker == _CHANNEL_MARKER:
                self._reading_channel = True
            elif marker == _THINK_OPEN:
                self._thinking = True
            elif marker == _TOOL_OPEN or marker.startswith(("<|tool_", "<|function_call")):
                # Even a visible prefix is not a completed assistant turn when a
                # tool call follows. Qwen tokenized tool-call markers and legacy
                # XML markers both fail closed, including their following payload.
                self._suppressed = True
                self._pending = ""
                self._trailing_whitespace = ""
            # All other special tokens are control-only and intentionally omitted.

        return "".join(output)

    def _read_channel(self, *, final: bool) -> str | None:
        match = re.match(r"([A-Za-z][A-Za-z0-9_-]*)", self._pending)
        if match is None:
            if final:
                self._pending = ""
                return ""
            return None
        channel = match.group(1)
        end = match.end()
        if end == len(self._pending) and not final:
            # Wait for a delimiter so a split `final` token cannot be confused
            # with another channel name.
            return None
        if end < len(self._pending) and self._pending[end] not in " \t\r\n<":
            # Unknown channel strings fail closed rather than becoming visible.
            while end < len(self._pending) and self._pending[end] not in " \t\r\n<":
                end += 1
            channel = ""
        self._pending = self._pending[end:]
        return channel if channel in _CHANNELS else ""

    def _take_visible(self, text: str, *, final: bool) -> str:
        if not text or not self._visible:
            return ""
        joined = self._trailing_whitespace + text
        if not self._started:
            joined = joined.lstrip()
        if final:
            self._trailing_whitespace = ""
            visible = joined.rstrip()
        else:
            visible = joined.rstrip()
            self._trailing_whitespace = joined[len(visible) :]
        if visible:
            self._started = True
        return visible


def _find_control(value: str) -> tuple[int, str] | None:
    candidates: list[tuple[int, str]] = []
    for marker in _CONTROL_MARKERS:
        position = value.find(marker)
        if position >= 0:
            candidates.append((position, marker))
    special = _SPECIAL_RE.search(value)
    if special:
        candidates.append((special.start(), special.group(0)))
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: candidate[0])


def _split_partial_control(value: str) -> tuple[str, str]:
    position = value.rfind("<")
    if position < 0:
        return value, ""
    suffix = value[position:]
    if _is_control_prefix(suffix):
        return value[:position], suffix
    return value, ""


def _is_control_prefix(value: str) -> bool:
    if not value:
        return False
    if any(marker.startswith(value) for marker in _CONTROL_MARKERS):
        return True
    return value.startswith("<|") and "|>" not in value


def _retain_marker_suffix(value: str, *markers: str) -> str:
    longest = 0
    for marker in markers:
        maximum = min(len(value), len(marker) - 1)
        for length in range(maximum, longest, -1):
            if value.endswith(marker[:length]):
                longest = length
                break
    return value[-longest:] if longest else ""
