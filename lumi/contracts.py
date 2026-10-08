"""Runtime-independent contracts for Lumi's chat and read-only tool loop."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit


class EvidenceTrust(StrEnum):
    LOCAL = "local_catalog_data"
    EXTERNAL = "external_untrusted"


ENTITY_TYPES = frozenset(
    {
        "series",
        "movie",
        "season",
        "episode",
        "album",
        "artist",
        "track",
        "collection",
        "person",
    }
)


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A normalized native function call returned by the selected Qwen runtime."""

    call_id: str
    name: str
    arguments: Any


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A model-facing message. Internal tool messages never leave the agent service."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """The small JSON-schema description exposed for one explicitly read-only tool."""

    name: str
    description: str
    parameters: Mapping[str, Any]
    data_scope: Literal["local", "external_search", "external_fetch"]
    read_only: bool = False

    def __post_init__(self) -> None:
        if self.data_scope not in {"local", "external_search", "external_fetch"}:
            raise ValueError("Lumi tools must declare a supported data scope")


@dataclass(frozen=True, slots=True)
class EntityReference:
    """A ZenStream ID obtained from a trusted local tool result."""

    type: str
    id: str
    title: str

    def __post_init__(self) -> None:
        if self.type not in ENTITY_TYPES:
            raise ValueError(f"Unsupported ZenStream entity type: {self.type}")
        if not self.id.strip():
            raise ValueError("ZenStream entity IDs cannot be empty")


@dataclass(frozen=True, slots=True)
class Source:
    """Structured source metadata collected from a web-research tool result."""

    url: str
    website_name: str
    title: str
    favicon_url: str | None = None

    def __post_init__(self) -> None:
        if (
            len(self.url) > 2_048
            or len(self.website_name) > 120
            or len(self.title) > 500
            or any(ord(char) < 32 or ord(char) == 127 for char in self.url)
            or any(ord(char) < 32 or ord(char) == 127 for char in self.website_name)
            or any(ord(char) < 32 or ord(char) == 127 for char in self.title)
        ):
            raise ValueError("Source metadata exceeds its size or character limits")
        parts = urlsplit(self.url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("Sources must use an absolute HTTP or HTTPS URL")
        if parts.username is not None or parts.password is not None:
            raise ValueError("Source URLs cannot contain embedded credentials")
        try:
            _ = parts.port
        except ValueError as error:
            raise ValueError("Source URLs cannot contain invalid ports") from error
        if self.favicon_url is not None:
            favicon = urlsplit(self.favicon_url)
            if (
                len(self.favicon_url) > 512
                or any(ord(char) < 32 or ord(char) == 127 for char in self.favicon_url)
                or favicon.scheme != "https"
                or not favicon.hostname
                or favicon.username is not None
                or favicon.password is not None
            ):
                raise ValueError("Favicon URLs must use HTTPS")


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Evidence returned by a registered tool, tagged before it enters model context."""

    content: str
    trust: EvidenceTrust
    entities: tuple[EntityReference, ...] = ()
    sources: tuple[Source, ...] = ()

    def __post_init__(self) -> None:
        if self.trust is EvidenceTrust.EXTERNAL and self.entities:
            raise ValueError("External evidence cannot grant trusted ZenStream references")

    def for_model(self, max_chars: int, max_entities: int = 16, max_sources: int = 8) -> str:
        """Serialize evidence with an explicit trust label and a bounded text payload."""

        import json

        if max_chars <= 0:
            return ""

        payload: dict[str, Any] = {
            "evidenceTrust": self.trust.value,
            "content": self.content[:max_chars],
            "entities": [
                {
                    "type": entity.type[:32],
                    "id": entity.id[:128],
                    "title": entity.title[:200],
                }
                for entity in self.entities[: max(0, max_entities)]
            ],
            "sources": [
                {
                    "url": source.url[:512],
                    "websiteName": source.website_name[:120],
                    "title": source.title[:200],
                }
                for source in self.sources[: max(0, max_sources)]
            ],
        }

        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        while len(encoded) > max_chars:
            if not payload["content"]:
                if payload["sources"]:
                    payload["sources"].pop()
                elif payload["entities"]:
                    payload["entities"].pop()
                else:
                    return ""
                encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                continue
            overflow = len(encoded) - max_chars
            payload["content"] = payload["content"][: max(0, len(payload["content"]) - overflow)]
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return encoded


@dataclass(frozen=True, slots=True)
class ChatContext:
    """Account identity and choices supplied by Lumi's trusted host boundary.

    Standalone deployments also carry a short-lived delegation token so catalog tools
    can authenticate their HTTP reads. Embedded hosts already authenticated the user
    and can omit it; tools use ``account_id`` for account-scoped reads instead.
    """

    account_id: str
    conversation_id: str
    model: str
    thinking: bool
    turn_id: str
    user_message: str = field(default="", repr=False, compare=False)
    delegation_token: str | None = field(default=None, repr=False)
    previous_entities: tuple[EntityReference, ...] = ()

    def __post_init__(self) -> None:
        if not self.account_id.strip() or not self.conversation_id.strip():
            raise ValueError("Authenticated account and conversation IDs are required")
        if not self.model.strip():
            raise ValueError("A configured model is required")
        if not self.turn_id.strip():
            raise ValueError("A unique Lumi turn ID is required")
        if self.delegation_token is not None and not self.delegation_token.strip():
            raise ValueError("An Orchestrator delegation cannot be empty")


@dataclass(frozen=True, slots=True)
class ModelRequest:
    model: str
    messages: Sequence[ChatMessage]
    tools: Sequence[ToolDefinition]
    thinking: bool
    context_size: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class ModelResponse:
    message: ChatMessage
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    load_duration_ns: int | None = None
    total_duration_ns: int | None = None


class ChatRuntime(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Generate one assistant message, including any native tool calls."""


class ReadOnlyTool(Protocol):
    definition: ToolDefinition

    def validate_arguments(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        """Validate and normalize untrusted model-generated arguments."""

    async def execute(self, context: ChatContext, arguments: Mapping[str, Any]) -> ToolResult:
        """Run a bounded read-only operation using the trusted account context."""


@dataclass(frozen=True, slots=True)
class ChatAnswer:
    markdown: str
    references: tuple[EntityReference, ...]
    sources: tuple[Source, ...]
    tool_rounds: int
    tool_calls: int
