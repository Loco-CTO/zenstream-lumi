"""Bounded native-tool chat loop, independent of the selected Qwen3.5 runtime."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

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
_RECOMMENDATION_MARKERS = (
    "recommend",
    "suggest",
    "what should i watch",
    "what to watch",
    "pick something to watch",
    "おすすめ",
    "推薦",
    "推奨",
    "何を観",
    "何を見",
    "gợi ý",
    "đề xuất",
    "nên xem",
    "xem gì",
)
_OUTSIDE_LIBRARY_MARKERS = (
    "outside my library",
    "outside the library",
    "outside-library",
    "not in my library",
    "not in my collection",
    "even if i don't have",
    "even if i do not have",
    "i don't own",
    "i do not own",
    "what should i add",
    "what to add",
    "ライブラリ外",
    "ライブラリにない",
    "ライブラリに入っていない",
    "コレクション外",
    "コレクションにない",
    "持っていない作品",
    "持っていないタイトル",
    "追加すべき",
    "追加した方が",
    "追加する作品",
    "ngoài thư viện",
    "không có trong thư viện",
    "ngoài danh sách",
    "chưa có trong thư viện",
    "tôi chưa có",
    "mình chưa có",
    "nên thêm",
)
_CONSTRAINED_RECOMMENDATION_MARKERS = (
    "similar to",
    "like ",
    "for fans of",
    "same genre",
    "starring",
    "directed by",
    "released in",
    "highly rated",
    "rated",
    "rating",
    "top ",
    "top rated",
    "top-rated",
    "highest rated",
    "best rated",
    "best movie",
    "best film",
    "most popular",
    "popular",
    "award-winning",
    "ranked",
    "from the 90s",
    "from the 1990s",
    "from the 80s",
    "from the 1980s",
    "based on",
    "adapted from",
    "unwatched",
    "haven't seen",
    "not watched",
    "genre",
    "themes",
    "theme",
    "tone",
    "mood",
    "politics",
    "war",
    "dark fantasy",
    "のような",
    "みたいな",
    "似た",
    "ジャンル",
    "テーマ",
    "雰囲気",
    "政治",
    "戦争",
    "高評価",
    "評価",
    "人気",
    "ランキング",
    "giống như",
    "thể loại",
    "chủ đề",
    "chính trị",
    "chiến tranh",
    "cozy",
    "funny",
    "comedy",
    "horror",
    "romantic",
    "action",
    "thriller",
    "science fiction",
    "sci-fi",
    "short movie",
    "under 2 hours",
    "less than",
    "字幕",
    "吹替",
    "短い",
    "新作",
    "最近",
    "コメディ",
    "ホラー",
    "恋愛",
    "アクション",
    "phụ đề",
    "lồng tiếng",
    "gần đây",
    "kinh dị",
    "hài",
    "lãng mạn",
    "dưới",
)
_FOLLOW_UP_RECOMMENDATION_MARKERS = (
    "another",
    "different",
    "new one",
    "something else",
    "instead",
    "alternative",
    "this one",
    "that one",
    "the first",
    "the second",
    "the one you mentioned",
    "which one",
    "それ",
    "これ",
    "その作品",
    "さっきの",
    "二番目",
    "2番目",
    "前の作品",
    "別の作品",
    "別の",
    "違う作品",
    "もう一つ",
    "cái đó",
    "phim đó",
    "một phim khác",
    "khác",
    "thứ hai",
)
_MOVIE_MARKERS = (
    "movie",
    "movies",
    "film",
    "films",
    "映画",
    "phim lẻ",
    "phim điện ảnh",
)
_SERIES_MARKERS = (
    "series",
    "tv show",
    "show",
    "シリーズ",
    "ドラマ",
    "phim bộ",
    "phim truyền hình",
)
_ANIME_MARKERS = ("anime", "アニメ")
_GENERIC_WATCH_MARKERS = (
    "what should i watch",
    "what to watch",
    "something to watch",
    "watch tonight",
    "何を観",
    "何を見",
    "観たい",
    "見たい",
    "xem gì",
    "bộ phim",
    "phim trong thư viện",
)
_UNSUPPORTED_MEDIA_MARKERS = (
    "album",
    "song",
    "track",
    "music",
    "artist",
    "podcast",
    "book",
    "playlist",
    "アルバム",
    "曲",
    "音楽",
    "歌",
    "アーティスト",
    "ポッドキャスト",
    "本を",
    "小説",
    "プレイリスト",
    "bài hát",
    "ca khúc",
    "nhạc",
    "nghệ sĩ",
    "sách",
)
_LOCAL_RECOMMENDATIONS_TOOL = "zenstream_home_recommendations"


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


def _simple_local_recommendation_kind(
    user_text: str, history: list[ChatMessage]
) -> tuple[bool, str | None]:
    """Identify generic recommendations that can be served from trusted local picks."""

    if history:
        return False, None
    text = re.sub(r"\s+", " ", user_text.casefold())
    if any(marker in text for marker in _OUTSIDE_LIBRARY_MARKERS):
        return False, None
    if not any(marker in text for marker in _RECOMMENDATION_MARKERS):
        return False, None
    if any(marker in text for marker in _UNSUPPORTED_MEDIA_MARKERS):
        return False, None
    if any(
        marker in text
        for marker in (
            *_CONSTRAINED_RECOMMENDATION_MARKERS,
            *_FOLLOW_UP_RECOMMENDATION_MARKERS,
        )
    ):
        return False, None

    # The local shortcut returns a single trusted Home pick. Keep requests for
    # ranked lists, a particular year, or a mixed anime/movie category agentic.
    if re.search(r"\b(?:18|19|20)\d{2}\b", text):
        return False, None
    if re.search(
        r"\b(?:two|three|four|five|six|seven|eight|nine|ten|several|multiple|"
        r"a few|a couple of|\d+)\s+"
        r"(?:movies?|films?|series|shows?|anime)\b",
        text,
    ):
        return False, None
    if re.search(
        r"(?:[2-9]\d*|[二三四五六七八九十百千][二三四五六七八九十百千]*)"
        r"\s*(?:本|作品|タイトル|番組|映画|ドラマ|アニメ)",
        text,
    ):
        return False, None

    wants_movie = any(marker in text for marker in _MOVIE_MARKERS)
    wants_series = any(marker in text for marker in _SERIES_MARKERS)
    wants_anime = any(marker in text for marker in _ANIME_MARKERS)
    explicit_anime_movie = bool(
        re.search(r"\banime\s+(?:movies?|films?)\b", text)
        or "アニメ映画" in text
    )
    explicit_anime_series = bool(
        re.search(r"\banime\s+(?:series|shows?|tv)\b", text)
        or "アニメシリーズ" in text
    )
    if wants_movie and wants_series:
        return False, None
    if wants_anime and wants_movie and not explicit_anime_movie:
        return False, None
    if wants_anime and wants_series and not explicit_anime_series:
        return False, None
    if wants_anime and not (explicit_anime_movie or explicit_anime_series):
        return False, None
    if wants_movie:
        return True, "movie"
    if wants_series:
        return True, "series"
    if any(marker in text for marker in _GENERIC_WATCH_MARKERS):
        return True, None
    return False, None


def _recommendation_locale(user_text: str) -> str:
    """Choose a short deterministic fallback in the language of common Lumi prompts."""

    folded = user_text.casefold()
    if any("\u3040" <= char <= "\u30ff" for char in user_text):
        return "ja"
    if any(
        marker in folded
        for marker in ("gợi ý", "đề xuất", "thư viện", "phim", "xem", "bạn", "mình")
    ):
        return "vi"
    return "en"


def _collapse_consecutive_duplicate_paragraphs(markdown: str) -> str:
    """Stop a decoding loop from repeating the same paragraph in the final answer."""

    paragraphs = re.split(r"(?:\r?\n){2,}", markdown.strip())
    unique: list[str] = []
    previous: str | None = None
    for paragraph in paragraphs:
        normalized = re.sub(r"\s+", " ", paragraph).strip().casefold()
        if normalized and normalized == previous:
            continue
        unique.append(paragraph.strip())
        previous = normalized
    return "\n\n".join(unique)


def _external_search_planning_messages(
    conversation_messages: list[ChatMessage],
) -> list[ChatMessage]:
    """Keep only a small recent user/assistant window for local follow-up resolution."""

    candidates = [
        message
        for message in conversation_messages
        if message.role in {"user", "assistant"}
        and not message.tool_calls
        and message.content.strip()
    ][-6:]
    budget = 2_400
    bounded: list[ChatMessage] = []
    for message in reversed(candidates):
        remaining = budget - sum(len(item.content) for item in bounded)
        if remaining <= 0:
            break
        content = message.content[-min(900, remaining) :]
        bounded.append(ChatMessage(role=message.role, content=content))
    bounded.reverse()
    return [
        ChatMessage(role="system", content=EXTERNAL_SEARCH_PLANNER_PROMPT),
        *bounded,
    ]


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

        local_recommendation, requested_type = _simple_local_recommendation_kind(
            user_text, history
        )
        if local_recommendation:
            return await self._answer_from_local_recommendations(
                context,
                user_text,
                requested_type,
            )

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
                if valid_arguments:
                    if data_scope == "external_search" and tool is not None:
                        result = await self._dispatch_external_search(
                            context,
                            messages,
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

    async def _answer_from_local_recommendations(
        self,
        context: ChatContext,
        user_text: str,
        requested_type: str | None,
    ) -> ChatAnswer:
        """Answer generic recommendations only from the authenticated local Home row."""

        result = await self._dispatch(
            context,
            ToolCall("local-recommendation", _LOCAL_RECOMMENDATIONS_TOOL, {}),
            set(),
            {},
        )
        try:
            payload = json.loads(result.content)
        except (TypeError, ValueError):
            payload = None
        catalog_result_available = (
            result.trust is EvidenceTrust.LOCAL
            and isinstance(payload, dict)
            and isinstance(payload.get("items"), list)
        )
        candidates = [
            entity
            for entity in result.entities
            if entity.type in {"movie", "series"}
            and (requested_type is None or entity.type == requested_type)
        ]
        locale = _recommendation_locale(user_text)
        if not catalog_result_available:
            lead = {
                "en": "I couldn't check your library just now. Please try again.",
                "ja": "今はライブラリを確認できませんでした。もう一度お試しください。",
                "vi": "Hiện mình chưa thể kiểm tra thư viện. Vui lòng thử lại.",
            }[locale]
            markdown = lead
            references: dict[tuple[str, str], EntityReference] = {}
        elif not candidates:
            lead = {
                "en": (
                    "I couldn't retrieve a verified local recommendation right now. "
                    "Try again or search for a title or genre."
                ),
                "ja": "今は確認済みのローカルおすすめを取得できませんでした。もう一度お試しいただくか、作品名やジャンルで検索してください。",
                "vi": "Hiện mình chưa lấy được đề xuất trong thư viện đã xác minh. Hãy thử lại hoặc tìm theo tên phim hay thể loại.",
            }[locale]
            markdown = lead
            references = {}
        else:
            selected = candidates[0]
            lead = {
                "en": (
                    "I recommend this title because it appears in your permission-filtered "
                    "ZenStream Home recommendations:"
                ),
                "ja": "アクセス可能なローカルのおすすめ一覧に掲載されているため、この作品をおすすめします:",
                "vi": (
                    "Mình gợi ý phim này vì nó xuất hiện trong mục đề xuất ZenStream mà bạn "
                    "có quyền truy cập:"
                ),
            }[locale]
            markdown = (
                f'{lead} :::zenstream{{type="{selected.type}" id="{selected.id}"}}'
            )
            references = {(selected.type, selected.id): selected}

        sources = {source.url: source for source in result.sources}
        return self._make_answer(
            markdown,
            references,
            sources,
            tool_rounds=1,
            total_calls=1,
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
        conversation_messages: list[ChatMessage],
        original_call: ToolCall,
        tool: Any,
        seen_calls: set[tuple[str, str]],
        tool_call_counts: dict[str, int],
    ) -> ToolResult:
        """Plan a minimal public query from bounded dialogue, excluding tool payloads."""

        planning_messages = _external_search_planning_messages(conversation_messages)
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
                "No external search was sent because a safe, focused public query could not be "
                "resolved from the recent conversation. Ask the user to clarify the subject.",
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
        cleaned = _THINK_BLOCK_PATTERN.sub("", markdown).strip()
        cleaned = _collapse_consecutive_duplicate_paragraphs(cleaned)
        bounded = cleaned[: self._limits.max_answer_chars]
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
