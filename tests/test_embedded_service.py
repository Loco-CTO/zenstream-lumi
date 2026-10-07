from __future__ import annotations

import tempfile
import unittest
from collections.abc import Mapping
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

from lumi import LUMI_PLUGIN_API_VERSION
from lumi.contracts import (
    ChatContext,
    ChatMessage,
    EntityReference,
    EvidenceTrust,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from lumi.model_catalog import ModelCatalog, QwenModelOption
from lumi.service import LumiConversationService
from lumi.service_factory import create_embedded_service
from lumi.storage import ConversationNotFound, ConversationStore
from lumi.tools import ToolRegistry


class FixtureRuntime:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.responses = [
            ChatMessage(
                "assistant",
                "",
                tool_calls=(ToolCall("call-1", "read_catalog", {"query": "local"}),),
            ),
            ChatMessage("assistant", "I found one local item."),
        ]

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        message = self.responses.pop(0) if self.responses else ChatMessage("assistant", "Answer.")
        return ModelResponse(message)


class FixtureReadOnlyTool:
    definition = ToolDefinition(
        name="read_catalog",
        description="Read the authenticated account's local catalog.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        data_scope="local",
        read_only=True,
    )

    def __init__(self) -> None:
        self.contexts: list[ChatContext] = []

    def validate_arguments(self, arguments: Mapping[str, object]) -> Mapping[str, object]:
        if set(arguments) != {"query"} or not isinstance(arguments["query"], str):
            raise ValueError("Invalid query")
        return {"query": arguments["query"]}

    async def execute(
        self,
        context: ChatContext,
        arguments: Mapping[str, object],
    ) -> ToolResult:
        self.contexts.append(context)
        return ToolResult(
            f"Catalog result for {arguments['query']}",
            EvidenceTrust.LOCAL,
            entities=(EntityReference("movie", "movie-1", "Local Movie"),),
        )


def _models() -> ModelCatalog:
    return ModelCatalog(
        (
            QwenModelOption("qwen3.5:2b", "Qwen3.5 2B"),
            QwenModelOption("qwen3.5:4b", "Qwen3.5 4B"),
        ),
        default_model="qwen3.5:2b",
    )


class EmbeddedServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.runtime = FixtureRuntime()
        self.tool = FixtureReadOnlyTool()
        self.models = _models()

    async def asyncTearDown(self) -> None:
        self.temporary.cleanup()

    def create_service(self) -> LumiConversationService:
        return create_embedded_service(
            runtime=self.runtime,
            models=self.models,
            tools=ToolRegistry((self.tool,)),
            store_path=Path(self.temporary.name) / "managed-lumi.sqlite3",
        )

    async def test_host_calls_lumi_and_read_only_tool_without_delegation_or_http(self) -> None:
        self.assertNotIn("service_token", signature(create_embedded_service).parameters)
        service = self.create_service()
        turn = await service.chat_for_account(
            "account-1",
            "conversation-1",
            "What is in my library?",
        )

        self.assertEqual(turn.answer.markdown, "I found one local item.")
        self.assertEqual(turn.conversation.owner_id, "account-1")
        self.assertEqual(self.runtime.requests[0].model, "qwen3.5:2b")
        self.assertEqual(len(self.tool.contexts), 1)
        self.assertEqual(self.tool.contexts[0].account_id, "account-1")
        self.assertIsNone(self.tool.contexts[0].delegation_token)
        self.assertEqual(LUMI_PLUGIN_API_VERSION, 1)

    async def test_each_embedded_storage_operation_is_scoped_to_the_host_account(self) -> None:
        service = self.create_service()
        created = await service.chat_for_account(
            "account-1",
            "conversation-1",
            "Private conversation",
        )

        self.assertEqual(
            [item.id for item in await service.list_conversations_for_account("account-1")],
            [created.conversation.id],
        )
        self.assertEqual(await service.list_conversations_for_account("account-2"), ())
        with self.assertRaises(ConversationNotFound):
            await service.get_conversation_for_account("account-2", "conversation-1")
        with self.assertRaises(ConversationNotFound):
            await service.update_choice_for_account(
                "account-2",
                "conversation-1",
                "qwen3.5:4b",
                False,
            )
        with self.assertRaises(ConversationNotFound):
            await service.chat_for_account(
                "account-2",
                "conversation-1",
                "Try to claim another account's ID",
            )
        self.assertEqual(await service.list_conversations_for_account("account-2"), ())

        await service.update_user_preference_for_account("account-1", "qwen3.5:4b", False)
        other_account_turn = await service.chat_for_account(
            "account-2",
            "conversation-2",
            "A separate conversation",
        )
        self.assertEqual(other_account_turn.conversation.model, "qwen3.5:2b")
        self.assertEqual(
            service.available_models_for_account("account-2"),
            self.models.selectable_models,
        )

    async def test_account_id_must_be_a_bounded_host_value(self) -> None:
        service = self.create_service()
        for invalid in ("", " account-1", "account-1\n", "x" * 201, None):
            with self.subTest(account_id=invalid), self.assertRaises(ValueError):
                service.available_models_for_account(invalid)  # type: ignore[arg-type]

    async def test_standalone_delegated_service_path_remains_compatible(self) -> None:
        class FixtureVerifier:
            def verify(self, token: str) -> SimpleNamespace:
                if token != "verified-delegation":
                    raise AssertionError("unexpected delegation")
                return SimpleNamespace(
                    subject="account-legacy",
                    conversation_id="conversation-legacy",
                    scopes=frozenset({"lumi.chat.write", "lumi.conversations.create"}),
                )

        service = LumiConversationService(
            ConversationStore(Path(self.temporary.name) / "standalone.sqlite3"),
            self.runtime,
            ToolRegistry((self.tool,)),
            FixtureVerifier(),  # type: ignore[arg-type]
            self.models,
        )

        turn = await service.chat(
            "verified-delegation",
            "conversation-legacy",
            "Keep the standalone integration working",
        )

        self.assertEqual(turn.conversation.owner_id, "account-legacy")
        self.assertEqual(self.tool.contexts[0].delegation_token, "verified-delegation")


if __name__ == "__main__":
    unittest.main()
