"""Delegation-scoped conversation orchestration for Lumi's internal API."""

from __future__ import annotations

import asyncio
import inspect
import logging
import sqlite3
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from lumi.agent import AgentLimits, ChatAgent, InferenceError
from lumi.contracts import (
    ChatAnswer,
    ChatContext,
    ChatMessage,
    ChatRuntime,
    EntityReference,
    StreamResetReason,
)
from lumi.delegation import DelegationClaims, DelegationError, DelegationVerifier
from lumi.model_catalog import ModelCatalog, ModelConfigurationError, QwenModelOption
from lumi.storage import (
    ConversationNotFound,
    ConversationSnapshot,
    ConversationStore,
    StoredConversation,
    StoredMessage,
    UserModelPreference,
)
from lumi.tools import ToolRegistry

SCOPE_CHAT_WRITE = "lumi.chat.write"
SCOPE_CONVERSATION_CREATE = "lumi.conversations.create"
SCOPE_CONVERSATION_READ = "lumi.conversations.read"
SCOPE_CONVERSATION_LIST = "lumi.conversations.list"
SCOPE_CONVERSATION_CONFIGURE = "lumi.conversations.configure"
SCOPE_PREFERENCE_WRITE = "lumi.preferences.write"
SCOPE_MODELS_READ = "lumi.models.read"
_LOGGER = logging.getLogger(__name__)


def _diagnostic_turn_id(value: str | None) -> str:
    if value is None:
        return uuid4().hex
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 64
        or any(not (character.isascii() and (character.isalnum() or character in "_-")) for character in value)
    ):
        raise ValueError("A diagnostic turn ID must be a short safe identifier")
    return value


class LumiServiceBusy(RuntimeError):
    """Lumi has reached its configured active-conversation lock limit."""


@dataclass(frozen=True, slots=True)
class ChatTurn:
    conversation: StoredConversation
    answer: ChatAnswer


@dataclass(frozen=True, slots=True)
class LumiStreamEvent:
    """A visible delta, partial-turn reset, or authoritative completed chat turn."""

    kind: Literal["delta", "reset", "complete"]
    text: str | None = None
    turn: ChatTurn | None = None
    reason: StreamResetReason | None = None

    def __post_init__(self) -> None:
        if self.kind == "delta":
            if (
                not isinstance(self.text, str)
                or not self.text
                or self.turn is not None
                or self.reason is not None
            ):
                raise ValueError("A Lumi delta must contain only nonempty visible text")
        elif self.kind == "reset":
            if (
                self.text is not None
                or self.turn is not None
                or self.reason not in {"intermediate", "cpu_fallback"}
            ):
                raise ValueError("A Lumi stream reset must contain only a safe reason")
        elif self.kind == "complete":
            if (
                self.text is not None
                or self.reason is not None
                or not isinstance(self.turn, ChatTurn)
            ):
                raise ValueError("A Lumi completion must carry its saved chat turn")
        else:
            raise ValueError("Unsupported Lumi stream event")


@dataclass(frozen=True, slots=True)
class _StreamFailure:
    error: BaseException


class _ConversationLockPool:
    def __init__(self, max_keys: int) -> None:
        self._max_keys = max_keys
        self._guard = asyncio.Lock()
        self._entries: dict[str, tuple[asyncio.Lock, int]] = {}

    @asynccontextmanager
    async def hold(self, key: str) -> AsyncIterator[None]:
        lock = await self._retain(key)
        acquired = False
        try:
            await lock.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                lock.release()
            await self._release(key, lock)

    async def _retain(self, key: str) -> asyncio.Lock:
        async with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                if len(self._entries) >= self._max_keys:
                    raise LumiServiceBusy("Too many conversations are active")
                lock = asyncio.Lock()
                self._entries[key] = (lock, 1)
                return lock
            lock, references = entry
            self._entries[key] = (lock, references + 1)
            return lock

    async def _release(self, key: str, lock: asyncio.Lock) -> None:
        async with self._guard:
            entry = self._entries.get(key)
            if entry is None or entry[0] is not lock:
                return
            references = entry[1] - 1
            if references <= 0:
                del self._entries[key]
            else:
                self._entries[key] = (lock, references)


class LumiConversationService:
    """Own chats behind either a verified delegation or a trusted embedded host.

    Delegated methods preserve Lumi's standalone private API. The ``*_for_account``
    methods are for an in-process host that has already authenticated and authorized
    the request; every storage call remains explicitly scoped to that account ID.
    """

    def __init__(
        self,
        store: ConversationStore,
        runtime: ChatRuntime,
        tools: ToolRegistry,
        delegation_verifier: DelegationVerifier | None,
        models: ModelCatalog,
        *,
        agent_limits: AgentLimits | None = None,
        max_concurrent_chats: int = 1,
        max_active_conversations: int = 128,
    ) -> None:
        if not 1 <= max_concurrent_chats <= 16:
            raise ValueError("max_concurrent_chats must be between 1 and 16")
        if not 1 <= max_active_conversations <= 2_048:
            raise ValueError("max_active_conversations must be between 1 and 2048")
        self._store = store
        self._runtime = runtime
        self._tools = tools
        self._delegations = delegation_verifier
        self._models = models
        self._agent_limits = agent_limits or AgentLimits()
        self._inference_slots = asyncio.Semaphore(max_concurrent_chats)
        self._conversation_locks = _ConversationLockPool(max_active_conversations)

    async def start(self) -> None:
        opener = getattr(self._runtime, "open", None)
        if opener is not None:
            result = opener()
            if inspect.isawaitable(result):
                await result

    async def close(self) -> None:
        closer = getattr(self._runtime, "close", None)
        if closer is not None:
            result = closer()
            if inspect.isawaitable(result):
                await result

    async def chat(
        self,
        delegation_token: str,
        conversation_id: str,
        user_text: str,
        *,
        model: str | None = None,
        thinking: bool | None = None,
    ) -> ChatTurn:
        claims = self._authorize(delegation_token, SCOPE_CHAT_WRITE, conversation_id)
        return await self._run_chat(
            claims.subject,
            conversation_id,
            user_text,
            create_if_missing=SCOPE_CONVERSATION_CREATE in claims.scopes,
            delegation_token=delegation_token,
            model=model,
            thinking=thinking,
        )

    async def chat_for_account(
        self,
        account_id: str,
        conversation_id: str,
        user_text: str,
        *,
        model: str | None = None,
        thinking: bool | None = None,
    ) -> ChatTurn:
        """Run a turn for an account authenticated by the embedding host.

        The account ID must come from the host's trusted authentication boundary, never
        from model-generated arguments or an untrusted client field.
        """

        return await self._run_chat(
            _validate_account_id(account_id),
            conversation_id,
            user_text,
            create_if_missing=True,
            delegation_token=None,
            model=model,
            thinking=thinking,
        )

    async def stream_chat_for_account(
        self,
        account_id: str,
        conversation_id: str,
        user_text: str,
        *,
        model: str | None = None,
        thinking: bool | None = None,
        turn_id: str | None = None,
    ) -> AsyncIterator[LumiStreamEvent]:
        """Stream safe text while running the same single chat turn as ``chat_for_account``."""

        owner_id = _validate_account_id(account_id)
        turn_id = _diagnostic_turn_id(turn_id)
        started = time.perf_counter()
        queue: asyncio.Queue[LumiStreamEvent | _StreamFailure] = asyncio.Queue()
        _LOGGER.info("Lumi stream relay started turn_id=%s", turn_id)
        event_count = 0
        delta_count = 0
        visible_chars = 0

        async def on_delta(text: str) -> None:
            await queue.put(LumiStreamEvent(kind="delta", text=text))

        async def on_reset(reason: StreamResetReason) -> None:
            await queue.put(LumiStreamEvent(kind="reset", reason=reason))

        async def run_turn() -> None:
            try:
                turn = await self._run_chat(
                    owner_id,
                    conversation_id,
                    user_text,
                    create_if_missing=True,
                    delegation_token=None,
                    model=model,
                    thinking=thinking,
                    turn_id=turn_id,
                    on_delta=on_delta,
                    on_reset=on_reset,
                )
            except asyncio.CancelledError:
                _LOGGER.info(
                    "Lumi stream turn task cancelled turn_id=%s duration_ms=%.1f",
                    turn_id,
                    (time.perf_counter() - started) * 1000,
                )
                raise
            except Exception as error:
                _LOGGER.error(
                    "Lumi stream turn task failed turn_id=%s category=%s duration_ms=%.1f",
                    turn_id,
                    type(error).__name__,
                    (time.perf_counter() - started) * 1000,
                )
                await queue.put(_StreamFailure(error))
            else:
                _LOGGER.info(
                    "Lumi stream turn task completed turn_id=%s duration_ms=%.1f",
                    turn_id,
                    (time.perf_counter() - started) * 1000,
                )
                await queue.put(LumiStreamEvent(kind="complete", turn=turn))

        task = asyncio.create_task(run_turn())
        try:
            while True:
                event = await queue.get()
                if isinstance(event, _StreamFailure):
                    raise event.error
                event_count += 1
                if event.kind == "delta":
                    delta_count += 1
                    visible_chars += len(event.text or "")
                    if delta_count == 1 or delta_count % 64 == 0:
                        _LOGGER.info(
                            "Lumi stream relay progress turn_id=%s event_count=%d delta_count=%d visible_chars=%d elapsed_ms=%.1f",
                            turn_id,
                            event_count,
                            delta_count,
                            visible_chars,
                            (time.perf_counter() - started) * 1000,
                        )
                else:
                    _LOGGER.info(
                        "Lumi stream relay event turn_id=%s kind=%s event_count=%d elapsed_ms=%.1f",
                        turn_id,
                        event.kind,
                        event_count,
                        (time.perf_counter() - started) * 1000,
                    )
                yield event
                if event.kind == "complete":
                    return
        finally:
            if not task.done():
                _LOGGER.info(
                    "Lumi stream relay cancelling turn task turn_id=%s event_count=%d delta_count=%d",
                    turn_id,
                    event_count,
                    delta_count,
                )
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            _LOGGER.info(
                "Lumi stream relay closed turn_id=%s event_count=%d delta_count=%d visible_chars=%d duration_ms=%.1f",
                turn_id,
                event_count,
                delta_count,
                visible_chars,
                (time.perf_counter() - started) * 1000,
            )

    async def _run_chat(
        self,
        account_id: str,
        conversation_id: str,
        user_text: str,
        *,
        create_if_missing: bool,
        delegation_token: str | None,
        model: str | None,
        thinking: bool | None,
        turn_id: str | None = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reset: Callable[[StreamResetReason], Awaitable[None]] | None = None,
    ) -> ChatTurn:
        account_id = _validate_account_id(account_id)
        turn_id = _diagnostic_turn_id(turn_id)
        started = time.perf_counter()
        _LOGGER.info(
            "Lumi turn started turn_id=%s user_message_chars=%d deadline_seconds=%.1f",
            turn_id,
            len(user_text),
            self._agent_limits.turn_timeout_seconds,
        )
        try:
            async with asyncio.timeout(self._agent_limits.turn_timeout_seconds):
                result = await self._complete_turn(
                    account_id,
                    delegation_token,
                    conversation_id,
                    create_if_missing,
                    user_text,
                    model,
                    thinking,
                    turn_id,
                    on_delta,
                    on_reset,
                )
            _LOGGER.info(
                "Lumi turn completed turn_id=%s duration_ms=%.1f tool_rounds=%d tool_calls=%d answer_chars=%d",
                turn_id,
                (time.perf_counter() - started) * 1000,
                result.answer.tool_rounds,
                result.answer.tool_calls,
                len(result.answer.markdown),
            )
            return result
        except asyncio.CancelledError:
            _LOGGER.info(
                "Lumi turn cancelled turn_id=%s duration_ms=%.1f",
                turn_id,
                (time.perf_counter() - started) * 1000,
            )
            raise
        except TimeoutError as error:
            _LOGGER.error(
                "Lumi turn failed turn_id=%s phase=turn_deadline category=%s duration_ms=%.1f",
                turn_id,
                type(error).__name__,
                (time.perf_counter() - started) * 1000,
            )
            raise InferenceError("The chat turn exceeded Lumi's configured deadline") from error
        except Exception as error:
            _LOGGER.error(
                "Lumi turn failed turn_id=%s phase=turn category=%s duration_ms=%.1f",
                turn_id,
                type(error).__name__,
                (time.perf_counter() - started) * 1000,
            )
            raise

    async def _complete_turn(
        self,
        account_id: str,
        delegation_token: str | None,
        conversation_id: str,
        create_if_missing: bool,
        user_text: str,
        model: str | None,
        thinking: bool | None,
        turn_id: str,
        on_delta: Callable[[str], Awaitable[None]] | None,
        on_reset: Callable[[StreamResetReason], Awaitable[None]] | None,
    ) -> ChatTurn:
        lock_started = time.perf_counter()
        _LOGGER.info("Lumi turn stage started turn_id=%s stage=conversation_lock", turn_id)
        async with self._conversation_locks.hold(
            _conversation_lock_key(account_id, conversation_id)
        ):
            _LOGGER.info(
                "Lumi turn stage completed turn_id=%s stage=conversation_lock wait_ms=%.1f",
                turn_id,
                (time.perf_counter() - lock_started) * 1000,
            )
            snapshot_started = time.perf_counter()
            _LOGGER.info("Lumi turn stage started turn_id=%s stage=conversation_read", turn_id)
            try:
                snapshot = await asyncio.to_thread(
                    self._store.get_snapshot,
                    account_id,
                    conversation_id,
                    self._history_limit,
                )
            except ConversationNotFound:
                if not create_if_missing:
                    raise
                snapshot = await self._create_conversation(
                    account_id,
                    conversation_id,
                    model=model,
                    thinking=thinking,
                )
                _LOGGER.info(
                    "Lumi conversation created turn_id=%s duration_ms=%.1f",
                    turn_id,
                    (time.perf_counter() - snapshot_started) * 1000,
                )
            else:
                _LOGGER.info(
                    "Lumi turn stage completed turn_id=%s stage=conversation_read duration_ms=%.1f history_messages=%d",
                    turn_id,
                    (time.perf_counter() - snapshot_started) * 1000,
                    len(snapshot.messages),
                )

            conversation = snapshot.conversation
            if model is not None or thinking is not None:
                if model is None or thinking is None:
                    raise ModelConfigurationError(
                        "Model and thinking must be supplied together"
                    )
                requested = self._models.validate_choice(model, thinking)
                if (conversation.model, conversation.thinking) != (
                    requested.id,
                    thinking,
                ):
                    raise ModelConfigurationError(
                        "A conversation model choice can only be set when it is created"
                    )
            option = self._models.validate_choice(conversation.model, conversation.thinking)
            history = tuple(
                ChatMessage(message.role, message.content) for message in snapshot.messages
            )
            context = ChatContext(
                account_id=account_id,
                conversation_id=conversation_id,
                model=option.id,
                thinking=conversation.thinking,
                user_message=user_text,
                delegation_token=delegation_token,
                previous_entities=_previous_entities(
                    snapshot.messages,
                    self._agent_limits.max_trusted_entities,
                ),
                turn_id=turn_id,
            )
            slot_started = time.perf_counter()
            _LOGGER.info("Lumi turn stage started turn_id=%s stage=inference_slot_wait", turn_id)
            async with self._inference_slots:
                _LOGGER.info(
                    "Lumi turn stage completed turn_id=%s stage=inference_slot_wait wait_ms=%.1f",
                    turn_id,
                    (time.perf_counter() - slot_started) * 1000,
                )
                inference_started = time.perf_counter()
                answer = await ChatAgent(
                    self._runtime,
                    self._tools,
                    self._agent_limits,
                ).answer(
                    context,
                    list(history),
                    user_text,
                    on_delta=on_delta,
                    on_reset=on_reset,
                )
                _LOGGER.info(
                    "Lumi turn stage completed turn_id=%s stage=agent_answer duration_ms=%.1f tool_rounds=%d tool_calls=%d",
                    turn_id,
                    (time.perf_counter() - inference_started) * 1000,
                    answer.tool_rounds,
                    answer.tool_calls,
                )
            if not answer.markdown.strip():
                raise InferenceError("The model returned no user-visible answer")
            store_started = time.perf_counter()
            _LOGGER.info("Lumi turn stage started turn_id=%s stage=conversation_write", turn_id)
            updated = await asyncio.to_thread(
                self._store.append_turn,
                account_id,
                conversation_id,
                user_text,
                answer,
            )
            _LOGGER.info(
                "Lumi turn stage completed turn_id=%s stage=conversation_write duration_ms=%.1f",
                turn_id,
                (time.perf_counter() - store_started) * 1000,
            )
            return ChatTurn(updated, answer)

    async def list_conversations(
        self,
        delegation_token: str,
        limit: int = 50,
    ) -> tuple[StoredConversation, ...]:
        claims = self._authorize(delegation_token, SCOPE_CONVERSATION_LIST, None)
        return await asyncio.to_thread(self._store.list_conversations, claims.subject, limit)

    async def list_conversations_for_account(
        self,
        account_id: str,
        limit: int = 50,
    ) -> tuple[StoredConversation, ...]:
        owner_id = _validate_account_id(account_id)
        return await asyncio.to_thread(self._store.list_conversations, owner_id, limit)

    async def get_conversation(
        self,
        delegation_token: str,
        conversation_id: str,
    ) -> ConversationSnapshot:
        claims = self._authorize(delegation_token, SCOPE_CONVERSATION_READ, conversation_id)
        return await asyncio.to_thread(
            self._store.get_snapshot,
            claims.subject,
            conversation_id,
            self._history_limit,
        )

    async def get_conversation_for_account(
        self,
        account_id: str,
        conversation_id: str,
    ) -> ConversationSnapshot:
        owner_id = _validate_account_id(account_id)
        return await asyncio.to_thread(
            self._store.get_snapshot,
            owner_id,
            conversation_id,
            self._history_limit,
        )

    async def update_choice(
        self,
        delegation_token: str,
        conversation_id: str,
        model: str,
        thinking: bool,
    ) -> StoredConversation:
        claims = self._authorize(
            delegation_token,
            SCOPE_CONVERSATION_CONFIGURE,
            conversation_id,
        )
        self._models.validate_choice(model, thinking)
        async with self._conversation_locks.hold(
            _conversation_lock_key(claims.subject, conversation_id)
        ):
            return await asyncio.to_thread(
                self._store.update_choice,
                claims.subject,
                conversation_id,
                model,
                thinking,
            )

    async def update_choice_for_account(
        self,
        account_id: str,
        conversation_id: str,
        model: str,
        thinking: bool,
    ) -> StoredConversation:
        owner_id = _validate_account_id(account_id)
        self._models.validate_choice(model, thinking)
        async with self._conversation_locks.hold(_conversation_lock_key(owner_id, conversation_id)):
            return await asyncio.to_thread(
                self._store.update_choice,
                owner_id,
                conversation_id,
                model,
                thinking,
            )

    async def update_user_preference(
        self,
        delegation_token: str,
        model: str,
        thinking: bool,
    ) -> UserModelPreference:
        claims = self._authorize(delegation_token, SCOPE_PREFERENCE_WRITE, None)
        self._models.validate_choice(model, thinking)
        preference = UserModelPreference(model=model, thinking=thinking)
        await asyncio.to_thread(
            self._store.set_user_preference,
            claims.subject,
            model,
            thinking,
        )
        return preference

    async def update_user_preference_for_account(
        self,
        account_id: str,
        model: str,
        thinking: bool,
    ) -> UserModelPreference:
        owner_id = _validate_account_id(account_id)
        self._models.validate_choice(model, thinking)
        preference = UserModelPreference(model=model, thinking=thinking)
        await asyncio.to_thread(
            self._store.set_user_preference,
            owner_id,
            model,
            thinking,
        )
        return preference

    def available_models(self, delegation_token: str) -> tuple[QwenModelOption, ...]:
        self._authorize(delegation_token, SCOPE_MODELS_READ, None)
        return self._models.selectable_models

    def available_models_for_account(self, account_id: str) -> tuple[QwenModelOption, ...]:
        _validate_account_id(account_id)
        return self._models.selectable_models

    @property
    def default_model(self) -> str:
        return self._models.default_model

    @property
    def default_thinking(self) -> bool:
        return self._models.default_thinking

    @property
    def _history_limit(self) -> int:
        return min(200, max(2, self._agent_limits.max_context_messages * 2))

    def _authorize(
        self,
        delegation_token: str,
        required_scope: str,
        conversation_id: str | None,
    ) -> DelegationClaims:
        if self._delegations is None:
            raise DelegationError("Delegated access is not configured.")
        claims = self._delegations.verify(delegation_token)
        if required_scope not in claims.scopes:
            raise DelegationError("Invalid Lumi delegation.")
        if claims.conversation_id != conversation_id:
            raise DelegationError("Invalid Lumi delegation.")
        return claims

    async def _create_conversation(
        self,
        account_id: str,
        conversation_id: str,
        *,
        model: str | None = None,
        thinking: bool | None = None,
    ) -> ConversationSnapshot:
        if model is not None or thinking is not None:
            if model is None or thinking is None:
                raise ModelConfigurationError("Model and thinking must be supplied together")
            option = self._models.validate_choice(model, thinking)
            selected_model = option.id
            selected_thinking = thinking
        else:
            preference = await asyncio.to_thread(
                self._store.get_user_preference,
                account_id,
            )
            if preference is not None:
                try:
                    option = self._models.require(preference.model)
                    selected_model = option.id
                    selected_thinking = bool(preference.thinking and option.supports_thinking)
                except ModelConfigurationError:
                    selected_model = self._models.default_model
                    selected_thinking = self._models.default_thinking
            else:
                selected_model = self._models.default_model
                selected_thinking = self._models.default_thinking
        try:
            await asyncio.to_thread(
                self._store.create_conversation,
                account_id,
                selected_model,
                default_thinking=selected_thinking,
                conversation_id=conversation_id,
                use_user_preference=False,
            )
        except sqlite3.IntegrityError as error:
            # A conversation ID already owned by another account is indistinguishable
            # from any other conversation the delegated account cannot access.
            raise ConversationNotFound("Conversation not found.") from error
        return await asyncio.to_thread(
            self._store.get_snapshot,
            account_id,
            conversation_id,
            self._history_limit,
        )


def _validate_account_id(account_id: str) -> str:
    if (
        not isinstance(account_id, str)
        or not account_id
        or account_id != account_id.strip()
        or len(account_id) > 200
        or any(ord(character) < 32 or ord(character) == 127 for character in account_id)
    ):
        raise ValueError("A bounded trusted account ID is required")
    return account_id


def _conversation_lock_key(account_id: str, conversation_id: str) -> str:
    return f"{len(account_id)}:{account_id}{conversation_id}"


def _previous_entities(
    messages: tuple[StoredMessage, ...], limit: int
) -> tuple[EntityReference, ...]:
    recent: list[EntityReference] = []
    seen: set[tuple[str, str]] = set()
    for message in reversed(messages):
        if message.role != "assistant":
            continue
        for reference in reversed(message.references):
            key = (reference.type, reference.id)
            if key in seen:
                continue
            seen.add(key)
            recent.append(reference)
            if len(recent) >= limit:
                return tuple(reversed(recent))
    return tuple(reversed(recent))
