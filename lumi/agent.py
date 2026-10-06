"""Bounded native-tool chat loop, independent of the selected Qwen3.5 runtime."""

from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping

from lumi.contracts import (
    ENTITY_TYPES,
    ChatAnswer,
    ChatContext,
    ChatMessage,
    ChatRuntime,
    EntityReference,
    EvidenceTrust,
    ModelRequest,
    ModelResponse,
    Source,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from lumi.prompts import EXTERNAL_SEARCH_PLANNER_PROMPT, SYSTEM_PROMPT
from lumi.tools import ToolRegistry

_REFERENCE_PATTERN = re.compile(
    r':::zenstream\{type="(?P<type>[^"]+)"\s+id="(?P<id>[^"]+)"\}'
)
_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True, slots=True)
class AgentLimits:
    max_tool_rounds: int = 6
    max_tool_calls: int = 12
    max_context_messages: int = 40
    max_context_chars: int = 8_000
    max_tool_result_chars: int = 2_000
    max_tool_argument_chars: int = 4_096
    max_tool_calls_per_round: int = 4
    max_sources: int = 12
    max_trusted_entities: int = 64
    max_user_input_chars: int = 6_000
    max_answer_chars: int = 12_000
    inference_timeout_seconds: float = 120.0
    tool_timeout_seconds: float = 20.0
    turn_timeout_seconds: float = 720.0
    context_size: int = 8_192
    output_tokens: int = 1_024

    def __post_init__(self) -> None:
        numeric_limits = (
            self.max_tool_rounds,
            self.max_tool_calls,
            self.max_context_messages,
            self.max_context_chars,
            self.max_tool_result_chars,
            self.max_tool_argument_chars,
            self.max_tool_calls_per_round,
            self.max_sources,
            self.max_trusted_entities,
            self.max_user_input_chars,
            self.max_answer_chars,
            self.context_size,
            self.output_tokens,
        )
        if any(limit <= 0 for limit in numeric_limits):
            raise ValueError("Lumi limits must be positive")
        if self.inference_timeout_seconds <= 0 or self.tool_timeout_seconds <= 0:
            raise ValueError("Lumi timeouts must be positive")
        if (
            isinstance(self.turn_timeout_seconds, bool)
            or not isinstance(self.turn_timeout_seconds, (int, float))
            or not math.isfinite(self.turn_timeout_seconds)
            or not 0 < self.turn_timeout_seconds <= 720
        ):
            raise ValueError("Lumi turn timeout must be positive and no greater than 720 seconds")
        if self.max_context_messages < 2 or self.max_context_chars < 2_000:
            raise ValueError("Context limits must leave room for policy and the current question")
        if self.max_tool_result_chars < 128:
            raise ValueError("Tool results must allow room for a bounded evidence label")
        if self.context_size <= self.output_tokens:
            raise ValueError("Model context must exceed the configured output-token limit")


class InferenceError(RuntimeError):
    """A model request failed or exceeded its configured deadline."""


class ChatAgent:
    def __init__(
        self,
        runtime: ChatRuntime,
        tools: ToolRegistry,
        limits: AgentLimits | None = None,
    ) -> None:
        self._runtime = runtime
        self._tools = tools
        self._limits = limits or AgentLimits()

    async def answer(
        self,
        context: ChatContext,
        history: list[ChatMessage],
        user_text: str,
    ) -> ChatAnswer:
        if not user_text.strip():
            raise ValueError("A user message cannot be empty")
        if len(user_text) > self._limits.max_user_input_chars:
            raise ValueError("The user message exceeds Lumi's configured input limit")

        trusted_entities = {
            (entity.type, entity.id): entity
            for entity in context.previous_entities[: self._limits.max_trusted_entities]
        }
        sources: dict[str, Source] = {}
        messages = self._bounded_messages(history, user_text)
        seen_calls: set[tuple[str, str]] = set()
        tool_call_counts: dict[str, int] = {}
        total_calls = 0
        tool_rounds = 0
        local_context_seen = False
        web_search_used = False

        while True:
            response = await self._complete(context, messages, self._tools.definitions)
            assistant = response.message
            if assistant.role != "assistant":
                raise InferenceError("Runtime returned a non-assistant message")
            if not assistant.tool_calls:
                return self._make_answer(
                    assistant.content,
                    trusted_entities,
                    sources,
                    tool_rounds,
                    total_calls,
                )

            if tool_rounds >= self._limits.max_tool_rounds:
                return await self._answer_after_limit(
                    context, messages, trusted_entities, sources, tool_rounds, total_calls
                )

            tool_rounds += 1
            original_calls = assistant.tool_calls[: self._limits.max_tool_calls_per_round]
            prepared_calls = [self._prepare_call(call) for call in original_calls]
            messages.append(
                replace(
                    assistant,
                    content="",
                    tool_calls=tuple(safe_call for _, safe_call, _ in prepared_calls),
                )
            )
            per_round_limit_hit = len(assistant.tool_calls) > len(prepared_calls)
            for original_call, safe_call, valid_arguments in prepared_calls:
                if total_calls >= self._limits.max_tool_calls:
                    messages.append(
                        self._tool_message(
                            safe_call,
                            ToolResult(
                                "The per-turn tool-call limit has been reached.",
                                EvidenceTrust.LOCAL,
                            ),
                            messages,
                        )
                    )
                    return await self._answer_after_limit(
                        context,
                        messages,
                        trusted_entities,
                        sources,
                        tool_rounds,
                        total_calls,
                    )

                if self._remaining_context_chars(messages) < 256:
                    messages.append(
                        self._tool_message(
                            safe_call,
                            ToolResult(
                                "The bounded context budget is full; no more evidence was added.",
                                EvidenceTrust.LOCAL,
                            ),
                            messages,
                        )
                    )
                    return await self._answer_after_limit(
                        context,
                        messages,
                        trusted_entities,
                        sources,
                        tool_rounds,
                        total_calls,
                    )

                total_calls += 1
                tool = self._tools.get(original_call.name)
                data_scope = tool.definition.data_scope if tool is not None else "local"
                if valid_arguments and data_scope == "external_search" and local_context_seen:
                    result = ToolResult(
                        "Web search must happen before any local ZenStream lookup in this turn. "
                        "Use only a query derived from the user's request, or answer from the "
                        "local evidence already collected.",
                        EvidenceTrust.LOCAL,
                    )
                elif valid_arguments and data_scope == "external_search" and web_search_used:
                    result = ToolResult(
                        "The one web-search call allowed per turn has already been used. "
                        "Continue from the result IDs and evidence already collected.",
                        EvidenceTrust.LOCAL,
                    )
                elif valid_arguments:
                    if data_scope == "external_search" and tool is not None:
                        result = await self._dispatch_external_search(
                            context,
                            user_text,
                            original_call,
                            tool,
                            seen_calls,
                            tool_call_counts,
                        )
                    else:
                        result = await self._dispatch(
                            context,
                            original_call,
                            seen_calls,
                            tool_call_counts,
                        )
                    if data_scope == "external_search":
                        web_search_used = True
                    elif tool is not None and data_scope == "local":
                        local_context_seen = True
                else:
                    result = ToolResult(
                        "Tool arguments were rejected because they were not a bounded JSON object.",
                        EvidenceTrust.LOCAL,
                    )
                if result.trust is EvidenceTrust.LOCAL:
                    for entity in result.entities:
                        key = (entity.type, entity.id)
                        if key in trusted_entities:
                            continue
                        if len(trusted_entities) >= self._limits.max_trusted_entities:
                            break
                        trusted_entities[key] = entity
                for source in result.sources:
                    if source.url in sources:
                        if data_scope == "external_fetch":
                            sources[source.url] = source
                        continue
                    if len(sources) >= self._limits.max_sources:
                        break
                    sources.setdefault(source.url, source)
                messages.append(self._tool_message(safe_call, result, messages))

            if per_round_limit_hit:
                return await self._answer_after_limit(
                    context, messages, trusted_entities, sources, tool_rounds, total_calls
                )

    async def _complete(
        self,
        context: ChatContext,
        messages: list[ChatMessage],
        definitions: tuple[ToolDefinition, ...],
    ) -> ModelResponse:
        request = ModelRequest(
            model=context.model,
            messages=tuple(messages),
            tools=definitions,
            thinking=context.thinking,
            context_size=self._limits.context_size,
            output_tokens=self._limits.output_tokens,
        )
        try:
            return await asyncio.wait_for(
                self._runtime.complete(request),
                timeout=self._limits.inference_timeout_seconds,
            )
        except TimeoutError as error:
            raise InferenceError("Model request exceeded the configured time limit") from error
        except InferenceError:
            raise
        except Exception as error:
            raise InferenceError("Model request failed") from error

    async def _dispatch(
        self,
        context: ChatContext,
        call: ToolCall,
        seen_calls: set[tuple[str, str]],
        tool_call_counts: dict[str, int],
    ) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult("This read-only tool is unavailable.", EvidenceTrust.LOCAL)
        if not isinstance(call.arguments, Mapping):
            return ToolResult("Tool arguments must be a JSON object.", EvidenceTrust.LOCAL)

        try:
            arguments = tool.validate_arguments(call.arguments)
            encoded_arguments = json.dumps(
                arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError):
            return ToolResult(
                "Tool arguments were rejected by the tool input schema.", EvidenceTrust.LOCAL
            )
        if len(encoded_arguments) > self._limits.max_tool_argument_chars:
            return ToolResult(
                "Tool arguments exceed the configured size limit.", EvidenceTrust.LOCAL
            )

        fingerprint = (
            call.name,
            encoded_arguments,
        )
        if fingerprint in seen_calls:
            return ToolResult(
                "The same tool query was already run in this turn. "
                "Continue using its prior result.",
                EvidenceTrust.LOCAL,
            )
        seen_calls.add(fingerprint)

        call_limit = getattr(tool, "max_calls_per_turn", None)
        calls_used = tool_call_counts.get(call.name, 0)
        if (
            isinstance(call_limit, int)
            and not isinstance(call_limit, bool)
            and call_limit > 0
            and calls_used >= call_limit
        ):
            return ToolResult(
                "The configured per-turn call limit for this tool has been reached. "
                "Continue using evidence already collected.",
                EvidenceTrust.LOCAL,
            )
        tool_call_counts[call.name] = calls_used + 1

        try:
            result = await asyncio.wait_for(
                tool.execute(context, arguments), timeout=self._limits.tool_timeout_seconds
            )
        except TimeoutError:
            return ToolResult("The read-only lookup exceeded its time limit.", EvidenceTrust.LOCAL)
        except Exception:
            return ToolResult("The read-only lookup failed.", EvidenceTrust.LOCAL)

        if not isinstance(result, ToolResult):
            return ToolResult(
                "The read-only lookup returned an invalid result.", EvidenceTrust.LOCAL
            )
        return result

    def _prepare_call(self, call: ToolCall) -> tuple[ToolCall, ToolCall, bool]:
        """Keep oversized or malformed native arguments out of the next model context."""

        try:
            encoded = json.dumps(
                call.arguments,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            valid = isinstance(call.arguments, Mapping) and (
                len(encoded) <= self._limits.max_tool_argument_chars
            )
        except (TypeError, ValueError):
            valid = False
        safe_call = call if valid else replace(call, arguments={})
        return call, safe_call, valid

    async def _answer_after_limit(
        self,
        context: ChatContext,
        messages: list[ChatMessage],
        trusted_entities: dict[tuple[str, str], EntityReference],
        sources: dict[str, Source],
        tool_rounds: int,
        total_calls: int,
    ) -> ChatAnswer:
        final_messages = [
            *messages,
            ChatMessage(
                role="user",
                content=(
                    "The research limits for this response have been reached. Answer the original "
                    "question now using only evidence already in this conversation. Match the "
                    "user's language, state any remaining uncertainty, and do not request more "
                    "tools."
                ),
            ),
        ]
        try:
            response = await self._complete(context, final_messages, ())
        except InferenceError:
            response = None
        if (
            response is not None
            and response.message.role == "assistant"
            and not response.message.tool_calls
        ):
            return self._make_answer(
                response.message.content,
                trusted_entities,
                sources,
                tool_rounds,
                total_calls,
            )
        fallback = "I reached the research limit before I could finish a reliable answer."
        return self._make_answer(fallback, trusted_entities, sources, tool_rounds, total_calls)

    async def _dispatch_external_search(
        self,
        context: ChatContext,
        user_text: str,
        original_call: ToolCall,
        tool: Any,
        seen_calls: set[tuple[str, str]],
        tool_call_counts: dict[str, int],
    ) -> ToolResult:
        """Plan outbound queries from this turn alone, excluding prior chat evidence."""

        planning_messages = [
            ChatMessage(role="system", content=EXTERNAL_SEARCH_PLANNER_PROMPT),
            ChatMessage(role="user", content=user_text),
        ]
        try:
            response = await self._complete(
                context,
                planning_messages,
                (tool.definition,),
            )
        except InferenceError:
            return ToolResult(
                "No external search was sent because a safe query could not be planned.",
                EvidenceTrust.LOCAL,
            )
        message = response.message
        if (
            message.role != "assistant"
            or len(message.tool_calls) != 1
            or message.tool_calls[0].name != original_call.name
        ):
            return ToolResult(
                "No external search was sent because the current message does not provide a "
                "standalone searchable subject. Ask the user to name the subject if needed.",
                EvidenceTrust.LOCAL,
            )
        return await self._dispatch(
            context,
            message.tool_calls[0],
            seen_calls,
            tool_call_counts,
        )

    def _bounded_messages(
        self,
        history: list[ChatMessage],
        user_text: str,
    ) -> list[ChatMessage]:
        system = ChatMessage(role="system", content=SYSTEM_PROMPT)
        if len(system.content) + len(user_text) > self._limits.max_context_chars:
            raise ValueError("The user message leaves no room for Lumi's safety policy")
        public_history = [
            message
            for message in history
            if message.role in {"user", "assistant"} and not message.tool_calls
        ]
        messages = [system, *public_history, ChatMessage(role="user", content=user_text)]

        current = messages[-1]
        kept = [current]
        reserved = min(self._limits.max_tool_result_chars, self._limits.max_context_chars // 3)
        history_budget = self._limits.max_context_chars - reserved
        char_count = len(system.content) + len(current.content)
        for message in reversed(messages[1:-1]):
            if len(kept) >= self._limits.max_context_messages - 1:
                break
            if char_count + len(message.content) > history_budget:
                continue
            kept.append(message)
            char_count += len(message.content)
        kept.reverse()
        return [system, *kept]

    def _tool_message(
        self, call: ToolCall, result: ToolResult, current_messages: list[ChatMessage]
    ) -> ChatMessage:
        result_budget = max(
            128,
            min(
                self._limits.max_tool_result_chars,
                self._remaining_context_chars(current_messages) - 128,
            ),
        )
        return ChatMessage(
            role="tool",
            name=call.name,
            tool_call_id=call.call_id,
            content=result.for_model(result_budget),
        )

    def _remaining_context_chars(self, messages: list[ChatMessage]) -> int:
        used = sum(len(message.content) for message in messages)
        used += sum(
            len(call.name)
            + len(call.call_id)
            + len(json.dumps(call.arguments, ensure_ascii=False, default=str))
            for message in messages
            for call in message.tool_calls
        )
        return self._limits.max_context_chars - used

    def _make_answer(
        self,
        markdown: str,
        trusted_entities: dict[tuple[str, str], EntityReference],
        sources: dict[str, Source],
        tool_rounds: int,
        total_calls: int,
    ) -> ChatAnswer:
        bounded = _THINK_BLOCK_PATTERN.sub("", markdown).strip()[: self._limits.max_answer_chars]
        used_references: list[EntityReference] = []

        def keep_reference(match: re.Match[str]) -> str:
            entity_type = match.group("type")
            entity_id = match.group("id")
            entity = trusted_entities.get((entity_type, entity_id))
            if entity is None or entity_type not in ENTITY_TYPES:
                return ""
            used_references.append(entity)
            return match.group(0)

        validated_markdown = _REFERENCE_PATTERN.sub(keep_reference, bounded)
        unique_references = {
            (entity.type, entity.id): entity for entity in used_references
        }
        return ChatAnswer(
            markdown=validated_markdown,
            references=tuple(unique_references.values()),
            sources=tuple(sources.values()),
            tool_rounds=tool_rounds,
            tool_calls=total_calls,
        )
