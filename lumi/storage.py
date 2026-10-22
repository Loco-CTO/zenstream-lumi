"""Local SQLite persistence for Lumi conversations and per-user model defaults."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from lumi.contracts import ChatAnswer, EntityReference, Source

_SCHEMA_VERSION = 2
_MAX_CONVERSATION_LIST = 100
_MAX_STORED_MESSAGES = 50
_MAX_MESSAGE_PAGE = 50
_NEW_CONVERSATION_TITLE = "New conversation"


class ConversationNotFound(LookupError):
    """The conversation does not exist for the requested owner."""


@dataclass(frozen=True, slots=True)
class UserModelPreference:
    model: str
    thinking: bool


@dataclass(frozen=True, slots=True)
class StoredConversation:
    id: str
    owner_id: str
    title: str
    model: str
    thinking: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class StoredMessage:
    id: str
    conversation_id: str
    owner_id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: str
    references: tuple[EntityReference, ...] = ()
    sources: tuple[Source, ...] = ()
    ordinal: int = 0


@dataclass(frozen=True, slots=True)
class ConversationSnapshot:
    conversation: StoredConversation
    messages: tuple[StoredMessage, ...]
    summary: str | None = None
    summary_through_ordinal: int = 0
    has_older_messages: bool = False
    has_unsummarized_older_messages: bool = False


@dataclass(frozen=True, slots=True)
class MessagePage:
    messages: tuple[StoredMessage, ...]
    next_before_ordinal: int | None
    has_more: bool


class ConversationStore:
    """Persist only user-visible messages and validated assistant metadata.

    Callers must pass an account ID obtained from the verified Orchestrator
    delegation. Every conversation query and mutation includes that owner ID.
    """

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)
        if str(database_path) != ":memory:":
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def set_user_preference(self, owner_id: str, model: str, thinking: bool) -> None:
        self._validate_owner(owner_id)
        self._validate_model(model)
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO user_model_preferences(owner_id, model, thinking, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(owner_id) DO UPDATE SET
                    model = excluded.model,
                    thinking = excluded.thinking,
                    updated_at = excluded.updated_at
                """,
                (owner_id, model.strip(), int(thinking), now),
            )

    def get_user_preference(self, owner_id: str) -> UserModelPreference | None:
        self._validate_owner(owner_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT model, thinking FROM user_model_preferences WHERE owner_id = ?",
                (owner_id,),
            ).fetchone()
        if row is None:
            return None
        return UserModelPreference(model=row["model"], thinking=bool(row["thinking"]))

    def create_conversation(
        self,
        owner_id: str,
        default_model: str,
        default_thinking: bool = False,
        title: str = _NEW_CONVERSATION_TITLE,
        conversation_id: str | None = None,
        use_user_preference: bool = True,
    ) -> StoredConversation:
        """Create from the user's saved choice, falling back to the server default."""

        self._validate_owner(owner_id)
        self._validate_model(default_model)
        if conversation_id is None:
            conversation_id = uuid.uuid4().hex
        elif (
            not conversation_id
            or conversation_id != conversation_id.strip()
            or len(conversation_id) > 100
        ):
            raise ValueError("A bounded conversation ID is required")
        created_at = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            preference = (
                connection.execute(
                    "SELECT model, thinking FROM user_model_preferences WHERE owner_id = ?",
                    (owner_id,),
                ).fetchone()
                if use_user_preference
                else None
            )
            model = preference["model"] if preference is not None else default_model.strip()
            thinking = bool(preference["thinking"]) if preference is not None else default_thinking
            connection.execute(
                """
                INSERT INTO conversations(
                    id, owner_id, title, model, thinking, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    owner_id,
                    _bounded_title(title),
                    model,
                    int(thinking),
                    created_at,
                    created_at,
                ),
            )
            connection.commit()
        return StoredConversation(
            id=conversation_id,
            owner_id=owner_id,
            title=_bounded_title(title),
            model=model,
            thinking=thinking,
            created_at=created_at,
            updated_at=created_at,
        )

    def list_conversations(self, owner_id: str, limit: int = 50) -> tuple[StoredConversation, ...]:
        self._validate_owner(owner_id)
        bounded_limit = max(1, min(limit, _MAX_CONVERSATION_LIST))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, owner_id, title, model, thinking, created_at, updated_at
                FROM conversations
                WHERE owner_id = ?
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (owner_id, bounded_limit),
            ).fetchall()
        return tuple(_conversation_from_row(row) for row in rows)

    def get_snapshot(
        self, owner_id: str, conversation_id: str, message_limit: int = 80
    ) -> ConversationSnapshot:
        self._validate_owner(owner_id)
        bounded_limit = max(1, min(message_limit, _MAX_STORED_MESSAGES))
        with self._connect() as connection:
            conversation_row = connection.execute(
                """
                SELECT id, owner_id, title, model, thinking, created_at, updated_at
                FROM conversations
                WHERE id = ? AND owner_id = ?
                """,
                (conversation_id, owner_id),
            ).fetchone()
            if conversation_row is None:
                raise ConversationNotFound("Conversation not found.")
            message_rows = connection.execute(
                """
                SELECT ordinal, id, conversation_id, owner_id, role, content, created_at,
                       references_json, sources_json
                FROM messages
                WHERE conversation_id = ? AND owner_id = ?
                ORDER BY ordinal DESC
                LIMIT ?
                """,
                (conversation_id, owner_id, bounded_limit + 1),
            ).fetchall()
            summary_row = connection.execute(
                """
                SELECT summary, message_ordinal
                FROM conversation_summaries
                WHERE conversation_id = ? AND owner_id = ?
                """,
                (conversation_id, owner_id),
            ).fetchone()
        has_older_messages = len(message_rows) > bounded_limit
        messages = tuple(_message_from_row(row) for row in reversed(message_rows[:bounded_limit]))
        has_unsummarized_older = False
        if has_older_messages and messages:
            has_unsummarized_older = self._has_unsummarized_messages_before(
                owner_id,
                conversation_id,
                messages[0].ordinal,
                int(summary_row["message_ordinal"]) if summary_row else 0,
            )
        return ConversationSnapshot(
            _conversation_from_row(conversation_row),
            messages,
            summary_row["summary"] if summary_row else None,
            int(summary_row["message_ordinal"]) if summary_row else 0,
            has_older_messages,
            has_unsummarized_older,
        )

    def _has_unsummarized_messages_before(
        self,
        owner_id: str,
        conversation_id: str,
        before_ordinal: int,
        summary_through_ordinal: int,
    ) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM messages
                WHERE conversation_id = ? AND owner_id = ?
                  AND ordinal < ? AND ordinal > ?
                LIMIT 1
                """,
                (
                    conversation_id,
                    owner_id,
                    before_ordinal,
                    summary_through_ordinal,
                ),
            ).fetchone()
        return row is not None

    def get_message_page(
        self,
        owner_id: str,
        conversation_id: str,
        *,
        before_ordinal: int | None = None,
        limit: int = _MAX_MESSAGE_PAGE,
    ) -> MessagePage:
        """Return one owner-scoped page, newest first in storage and oldest first to callers."""

        self._validate_owner(owner_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_MESSAGE_PAGE:
            raise ValueError(f"Message page size must be between 1 and {_MAX_MESSAGE_PAGE}")
        if before_ordinal is not None and (
            isinstance(before_ordinal, bool)
            or not isinstance(before_ordinal, int)
            or before_ordinal < 1
        ):
            raise ValueError("Message page cursor is invalid")
        with self._connect() as connection:
            conversation = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND owner_id = ?",
                (conversation_id, owner_id),
            ).fetchone()
            if conversation is None:
                raise ConversationNotFound("Conversation not found.")
            if before_ordinal is None:
                rows = connection.execute(
                    """
                    SELECT ordinal, id, conversation_id, owner_id, role, content, created_at,
                           references_json, sources_json
                    FROM messages WHERE conversation_id = ? AND owner_id = ?
                    ORDER BY ordinal DESC LIMIT ?
                    """,
                    (conversation_id, owner_id, limit + 1),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT ordinal, id, conversation_id, owner_id, role, content, created_at,
                           references_json, sources_json
                    FROM messages
                    WHERE conversation_id = ? AND owner_id = ? AND ordinal < ?
                    ORDER BY ordinal DESC LIMIT ?
                    """,
                    (conversation_id, owner_id, before_ordinal, limit + 1),
                ).fetchall()
        has_more = len(rows) > limit
        page_rows = rows[:limit]
        messages = tuple(_message_from_row(row) for row in reversed(page_rows))
        next_before = messages[0].ordinal if has_more and messages else None
        return MessagePage(messages, next_before, has_more)

    def get_message_batch_after(
        self,
        owner_id: str,
        conversation_id: str,
        after_ordinal: int,
        *,
        before_ordinal: int | None = None,
        limit: int = _MAX_MESSAGE_PAGE,
    ) -> tuple[StoredMessage, ...]:
        """Read a bounded ascending batch for incremental summary construction."""

        self._validate_owner(owner_id)
        if isinstance(after_ordinal, bool) or not isinstance(after_ordinal, int) or after_ordinal < 0:
            raise ValueError("Summary ordinal is invalid")
        if before_ordinal is not None and (
            isinstance(before_ordinal, bool)
            or not isinstance(before_ordinal, int)
            or before_ordinal < 1
            or before_ordinal <= after_ordinal
        ):
            raise ValueError("Summary upper ordinal is invalid")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_MESSAGE_PAGE:
            raise ValueError(f"Message batch size must be between 1 and {_MAX_MESSAGE_PAGE}")
        with self._connect() as connection:
            conversation = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND owner_id = ?",
                (conversation_id, owner_id),
            ).fetchone()
            if conversation is None:
                raise ConversationNotFound("Conversation not found.")
            if before_ordinal is None:
                rows = connection.execute(
                    """
                    SELECT ordinal, id, conversation_id, owner_id, role, content, created_at,
                           references_json, sources_json
                    FROM messages
                    WHERE conversation_id = ? AND owner_id = ? AND ordinal > ?
                    ORDER BY ordinal ASC LIMIT ?
                    """,
                    (conversation_id, owner_id, after_ordinal, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT ordinal, id, conversation_id, owner_id, role, content, created_at,
                           references_json, sources_json
                    FROM messages
                    WHERE conversation_id = ? AND owner_id = ? AND ordinal > ? AND ordinal < ?
                    ORDER BY ordinal ASC LIMIT ?
                    """,
                    (conversation_id, owner_id, after_ordinal, before_ordinal, limit),
                ).fetchall()
        return tuple(_message_from_row(row) for row in rows)

    def save_summary(
        self,
        owner_id: str,
        conversation_id: str,
        message_ordinal: int,
        summary: str,
    ) -> None:
        """Atomically advance a rolling summary without changing transcript rows."""

        self._validate_owner(owner_id)
        if (
            isinstance(message_ordinal, bool)
            or not isinstance(message_ordinal, int)
            or message_ordinal < 1
            or not summary.strip()
        ):
            raise ValueError("A non-empty summary and positive message ordinal are required")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            conversation = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND owner_id = ?",
                (conversation_id, owner_id),
            ).fetchone()
            maximum = connection.execute(
                "SELECT MAX(ordinal) FROM messages WHERE conversation_id = ? AND owner_id = ?",
                (conversation_id, owner_id),
            ).fetchone()[0]
            if conversation is None or maximum is None or message_ordinal > maximum:
                raise ConversationNotFound("Conversation or summarized messages not found.")
            connection.execute(
                """
                INSERT INTO conversation_summaries(
                    conversation_id, owner_id, message_ordinal, summary, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(conversation_id, owner_id) DO UPDATE SET
                    message_ordinal = excluded.message_ordinal,
                    summary = excluded.summary,
                    updated_at = excluded.updated_at
                WHERE excluded.message_ordinal >= conversation_summaries.message_ordinal
                """,
                (conversation_id, owner_id, message_ordinal, summary, _utc_now()),
            )
            connection.commit()

    def rename_conversation(
        self, owner_id: str, conversation_id: str, title: str
    ) -> StoredConversation:
        self._validate_owner(owner_id)
        renamed_title = _renamed_title(title)
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            result = connection.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE id = ? AND owner_id = ?
                """,
                (renamed_title, now, conversation_id, owner_id),
            )
            if result.rowcount != 1:
                raise ConversationNotFound("Conversation not found.")
            row = connection.execute(
                """
                SELECT id, owner_id, title, model, thinking, created_at, updated_at
                FROM conversations WHERE id = ? AND owner_id = ?
                """,
                (conversation_id, owner_id),
            ).fetchone()
            connection.commit()
        return _conversation_from_row(row)

    def delete_conversation(self, owner_id: str, conversation_id: str) -> None:
        self._validate_owner(owner_id)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            result = connection.execute(
                "DELETE FROM conversations WHERE id = ? AND owner_id = ?",
                (conversation_id, owner_id),
            )
            if result.rowcount != 1:
                raise ConversationNotFound("Conversation not found.")
            connection.commit()

    def update_choice(
        self,
        owner_id: str,
        conversation_id: str,
        model: str,
        thinking: bool,
    ) -> StoredConversation:
        self._validate_owner(owner_id)
        self._validate_model(model)
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            result = connection.execute(
                """
                UPDATE conversations
                SET model = ?, thinking = ?, updated_at = ?
                WHERE id = ? AND owner_id = ?
                """,
                (model.strip(), int(thinking), now, conversation_id, owner_id),
            )
            if result.rowcount != 1:
                raise ConversationNotFound("Conversation not found.")
            row = connection.execute(
                """
                SELECT id, owner_id, title, model, thinking, created_at, updated_at
                FROM conversations WHERE id = ? AND owner_id = ?
                """,
                (conversation_id, owner_id),
            ).fetchone()
            connection.commit()
        return _conversation_from_row(row)

    def append_turn(
        self,
        owner_id: str,
        conversation_id: str,
        user_text: str,
        answer: ChatAnswer,
    ) -> StoredConversation:
        """Append the user/assistant pair atomically; never persist internal tool traces."""

        self._validate_owner(owner_id)
        if not user_text.strip() or not answer.markdown.strip():
            raise ValueError("Conversation messages cannot be empty")
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT id, owner_id, title, model, thinking, created_at, updated_at
                FROM conversations
                WHERE id = ? AND owner_id = ?
                """,
                (conversation_id, owner_id),
            ).fetchone()
            if row is None:
                raise ConversationNotFound("Conversation not found.")

            title = row["title"]
            if title == _NEW_CONVERSATION_TITLE:
                title = _bounded_title(user_text)
            message_rows = (
                (
                    uuid.uuid4().hex,
                    conversation_id,
                    owner_id,
                    "user",
                    user_text,
                    now,
                    "[]",
                    "[]",
                ),
                (
                    uuid.uuid4().hex,
                    conversation_id,
                    owner_id,
                    "assistant",
                    answer.markdown,
                    now,
                    _json_references(answer.references),
                    _json_sources(answer.sources),
                ),
            )
            connection.executemany(
                """
                INSERT INTO messages(
                    id, conversation_id, owner_id, role, content, created_at,
                    references_json, sources_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                message_rows,
            )
            connection.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE id = ? AND owner_id = ?
                """,
                (title, now, conversation_id, owner_id),
            )
            updated_row = connection.execute(
                """
                SELECT id, owner_id, title, model, thinking, created_at, updated_at
                FROM conversations WHERE id = ? AND owner_id = ?
                """,
                (conversation_id, owner_id),
            ).fetchone()
            connection.commit()
        return _conversation_from_row(updated_row)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version > _SCHEMA_VERSION:
                raise RuntimeError("Lumi database schema is newer than this service supports")
            if version == 0:
                connection.executescript(
                    """
                    BEGIN IMMEDIATE;
                    CREATE TABLE user_model_preferences (
                        owner_id TEXT PRIMARY KEY,
                        model TEXT NOT NULL,
                        thinking INTEGER NOT NULL CHECK (thinking IN (0, 1)),
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE conversations (
                        id TEXT PRIMARY KEY,
                        owner_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        model TEXT NOT NULL,
                        thinking INTEGER NOT NULL CHECK (thinking IN (0, 1)),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE (id, owner_id)
                    );
                    CREATE INDEX conversations_owner_updated
                        ON conversations(owner_id, updated_at DESC, id DESC);
                    CREATE TABLE messages (
                        ordinal INTEGER PRIMARY KEY AUTOINCREMENT,
                        id TEXT NOT NULL UNIQUE,
                        conversation_id TEXT NOT NULL,
                        owner_id TEXT NOT NULL,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        references_json TEXT NOT NULL,
                        sources_json TEXT NOT NULL,
                        FOREIGN KEY (conversation_id, owner_id)
                            REFERENCES conversations(id, owner_id) ON DELETE CASCADE
                    );
                    CREATE INDEX messages_conversation_owner_ordinal
                        ON messages(conversation_id, owner_id, ordinal);
                    PRAGMA user_version = 1;
                    COMMIT;
                    """
                )
                version = 1
            if version == 1:
                connection.executescript(
                    """
                    BEGIN IMMEDIATE;
                    CREATE TABLE conversation_summaries (
                        conversation_id TEXT NOT NULL,
                        owner_id TEXT NOT NULL,
                        message_ordinal INTEGER NOT NULL CHECK (message_ordinal > 0),
                        summary TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (conversation_id, owner_id),
                        FOREIGN KEY (conversation_id, owner_id)
                            REFERENCES conversations(id, owner_id) ON DELETE CASCADE
                    );
                    PRAGMA user_version = 2;
                    COMMIT;
                    """
                )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._database_path, timeout=5.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _validate_owner(owner_id: str) -> None:
        if not owner_id.strip() or len(owner_id) > 200:
            raise ValueError("A bounded verified owner ID is required")

    @staticmethod
    def _validate_model(model: str) -> None:
        if not model.strip() or len(model) > 200:
            raise ValueError("A bounded configured model name is required")


def _conversation_from_row(row: sqlite3.Row) -> StoredConversation:
    return StoredConversation(
        id=row["id"],
        owner_id=row["owner_id"],
        title=row["title"],
        model=row["model"],
        thinking=bool(row["thinking"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _message_from_row(row: sqlite3.Row) -> StoredMessage:
    references = json.loads(row["references_json"])
    sources = json.loads(row["sources_json"])
    return StoredMessage(
        id=row["id"],
        conversation_id=row["conversation_id"],
        owner_id=row["owner_id"],
        role=row["role"],
        content=row["content"],
        created_at=row["created_at"],
        references=tuple(EntityReference(**reference) for reference in references),
        sources=tuple(Source(**source) for source in sources),
        ordinal=int(row["ordinal"]) if "ordinal" in row.keys() else 0,
    )


def _json_references(references: tuple[EntityReference, ...]) -> str:
    return json.dumps(
        [{"type": item.type, "id": item.id, "title": item.title} for item in references],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _json_sources(sources: tuple[Source, ...]) -> str:
    return json.dumps(
        [
            {
                "url": item.url,
                "website_name": item.website_name,
                "title": item.title,
                "favicon_url": item.favicon_url,
            }
            for item in sources
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _bounded_title(value: str) -> str:
    title = " ".join(value.strip().split())
    return (title or _NEW_CONVERSATION_TITLE)[:120]


def _renamed_title(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("A conversation title is required")
    title = " ".join(value.split())
    if not title or len(title) > 120:
        raise ValueError("A conversation title must contain 1 to 120 characters")
    return title


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
