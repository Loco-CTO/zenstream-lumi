from __future__ import annotations

import asyncio
import base64
import json
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from lumi.agent import AgentLimits, InferenceError
from lumi.contracts import (
    ChatContext,
    ChatMessage,
    EntityReference,
    EvidenceTrust,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    Source,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from lumi.delegation import Ed25519DelegationVerifier
from lumi.model_catalog import ModelCatalog, QwenModelOption
from lumi.service import (
    SCOPE_CHAT_WRITE,
    SCOPE_CONVERSATION_CONFIGURE,
    SCOPE_CONVERSATION_CREATE,
    SCOPE_CONVERSATION_LIST,
    SCOPE_CONVERSATION_READ,
    SCOPE_MODELS_READ,
    SCOPE_PREFERENCE_WRITE,
    LumiConversationService,
    _conversation_lock_key,
)
from lumi.storage import ConversationStore
from lumi.tools import ToolRegistry

_SERVICE_TOKEN = "test-only-internal-service-token-value"
_ALL_SCOPES = (
    SCOPE_CHAT_WRITE,
    SCOPE_CONVERSATION_CONFIGURE,
    SCOPE_CONVERSATION_CREATE,
    SCOPE_CONVERSATION_LIST,
    SCOPE_CONVERSATION_READ,
    SCOPE_MODELS_READ,
    SCOPE_PREFERENCE_WRITE,
)


def _encode_segment(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


class FixtureRuntime:
    def __init__(self) -> None:
        self.responses: list[ChatMessage] = []
        self.requests: list[ModelRequest] = []
        self.delay_seconds = 0.0
        self.cancelled = False
        self.started_event: asyncio.Event | None = None
        self.complete_calls = 0
        self.stream_rounds: list[list[ModelStreamEvent]] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.complete_calls += 1
        self.requests.append(request)
        if self.started_event is not None:
            self.started_event.set()
        if self.delay_seconds:
            try:
                await asyncio.sleep(self.delay_seconds)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        message = self.responses.pop(0) if self.responses else ChatMessage("assistant", "Answer.")
        return ModelResponse(message)

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        if self.started_event is not None:
            self.started_event.set()
        events = self.stream_rounds.pop(0) if self.stream_rounds else [
            ModelStreamEvent(
                kind="complete",
                response=ModelResponse(ChatMessage("assistant", "Streamed answer.")),
            )
        ]
        for event in events:
            yield event


class FixtureTool:
    def __init__(self, name: str, result: ToolResult) -> None:
        self.definition = ToolDefinition(
            name=name,
            description=f"Read-only fixture for {name}.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "maxLength": 120}},
                "required": ["query"],
                "additionalProperties": False,
            },
            data_scope="local",
            read_only=True,
        )
        self.result = result
        self.calls: list[tuple[ChatContext, Mapping[str, object]]] = []

    def validate_arguments(self, arguments: Mapping[str, object]) -> Mapping[str, object]:
        if set(arguments) != {"query"}:
            raise ValueError("Expected a query")
        query = arguments["query"]
        if not isinstance(query, str) or not query.strip() or len(query) > 120:
            raise ValueError("Invalid query")
        return {"query": query.strip()}

    async def execute(
        self, context: ChatContext, arguments: Mapping[str, object]
    ) -> ToolResult:
        self.calls.append((context, arguments))
        return self.result


class LumiServiceAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.store = ConversationStore(Path(self.temporary.name) / "lumi.sqlite3")
        self.now = 1_800_000_000
        self.private_key = Ed25519PrivateKey.generate()
        public_key = self.private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        verifier = Ed25519DelegationVerifier(public_key, clock=lambda: self.now)
        self.runtime = FixtureRuntime()
        self.catalog_tool = FixtureTool(
            "search_catalog",
            ToolResult(
                "The local catalog has one matching series.",
                EvidenceTrust.LOCAL,
                entities=(EntityReference("series", "series-1", "Frieren"),),
            ),
        )
        self.web_tool = FixtureTool(
            "search_web",
            ToolResult(
                "The page discusses the series themes.",
                EvidenceTrust.EXTERNAL,
                sources=(
                    Source(
                        "https://example.org/frieren",
                        "Example",
                        "Themes in Frieren",
                        "https://example.org/favicon.ico",
                    ),
                ),
            ),
        )
        self.catalog = ModelCatalog(
            (
                QwenModelOption("qwen3.5:0.8b", "Qwen3.5 0.8B", supports_thinking=False),
                QwenModelOption("qwen3.5:2b", "Qwen3.5 2B"),
                QwenModelOption("qwen3.5:4b", "Qwen3.5 4B"),
                QwenModelOption("qwen3.5:9b", "Qwen3.5 9B", enabled=False),
            ),
            default_model="qwen3.5:2b",
        )
        self.service = LumiConversationService(
            self.store,
            self.runtime,
            ToolRegistry((self.catalog_tool, self.web_tool)),
            verifier,
            self.catalog,
        )
        from lumi.api import create_app

        self.client = TestClient(
            create_app(self.service, _SERVICE_TOKEN),
            raise_server_exceptions=False,
        )

    def tearDown(self) -> None:
        self.client.close()
        self.temporary.cleanup()

    def token(
        self,
        *,
        owner: str = "account-1",
        conversation_id: str | None = "conversation-1",
        scopes: tuple[str, ...] = _ALL_SCOPES,
        **overrides: object,
    ) -> str:
        claims: dict[str, object] = {
            "iss": "zenstream-orchestrator",
            "aud": "zenstream-lumi",
            "sub": owner,
            "sid": "session-1",
            "cid": conversation_id,
            "scope": list(scopes),
            "iat": self.now,
            "exp": self.now + 120,
            "jti": "grant-1",
            **overrides,
        }
        header = _encode_segment(b'{"alg":"EdDSA","typ":"JWT"}')
        payload = _encode_segment(json.dumps(claims, separators=(",", ":")).encode("utf-8"))
        signed = f"{header}.{payload}".encode("ascii")
        signature = _encode_segment(self.private_key.sign(signed))
        return f"{header}.{payload}.{signature}"

    def headers(self, delegation: str | None = None) -> dict[str, str]:
        result = {"Authorization": f"Bearer {_SERVICE_TOKEN}"}
        if delegation is not None:
            result["X-Lumi-Delegation"] = delegation
        return result

    def test_internal_api_requires_service_auth_and_delegation(self) -> None:
        self.assertEqual(self.client.get("/healthz").status_code, 200)
        self.assertEqual(self.client.get("/internal/models").status_code, 401)
        self.assertEqual(
            self.client.get(
                "/internal/models",
                headers={"Authorization": "Bearer wrong-token-value-long-enough"},
            ).status_code,
            401,
        )
        self.assertEqual(
            self.client.get("/internal/models", headers=self.headers()).status_code, 422
        )
        self.assertEqual(
            self.client.get(
                "/internal/models",
                headers=self.headers("not-a-signed-delegation"),
            ).status_code,
            401,
        )

    def test_models_and_conversation_lists_require_account_level_scope(self) -> None:
        no_conversation = self.token(conversation_id=None)
        models = self.client.get("/internal/models", headers=self.headers(no_conversation))
        self.assertEqual(models.status_code, 200)
        self.assertEqual(
            [item["id"] for item in models.json()["models"]],
            ["qwen3.5:0.8b", "qwen3.5:2b", "qwen3.5:4b"],
        )
        self.assertEqual(models.json()["defaultModel"], "qwen3.5:2b")

        list_response = self.client.get(
            "/internal/conversations", headers=self.headers(no_conversation)
        )
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(list_response.json(), {"conversations": []})

        scoped_to_conversation = self.token(conversation_id="conversation-1")
        self.assertEqual(
            self.client.get(
                "/internal/conversations", headers=self.headers(scoped_to_conversation)
            ).status_code,
            401,
        )

    def test_chat_creates_delegated_conversation_and_persists_visible_turn(self) -> None:
        response = self.client.post(
            "/internal/conversations/conversation-1/turns",
            headers=self.headers(self.token()),
            json={"message": "What is in my library?"},
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["conversation"]["id"], "conversation-1")
        self.assertEqual(payload["conversation"]["model"], "qwen3.5:2b")
        self.assertFalse(payload["conversation"]["thinking"])
        self.assertNotIn("ownerId", payload["conversation"])
        self.assertEqual(self.runtime.requests[0].model, "qwen3.5:2b")
        self.assertNotIn("account-1", self.runtime.requests[0].messages[0].content)

        read = self.client.get(
            "/internal/conversations/conversation-1",
            headers=self.headers(self.token(scopes=(SCOPE_CONVERSATION_READ,))),
        )
        self.assertEqual(read.status_code, 200, read.text)
        messages = read.json()["messages"]
        self.assertEqual([item["role"] for item in messages], ["user", "assistant"])
        self.assertEqual(messages[0]["content"], "What is in my library?")
        self.assertEqual(messages[1]["content"], "Answer.")

    def test_embedded_stream_forwards_safe_resets_and_runs_one_multi_tool_turn(self) -> None:
        def completion(message: ChatMessage) -> ModelStreamEvent:
            return ModelStreamEvent(kind="complete", response=ModelResponse(message))

        self.runtime.stream_rounds = [
            [
                ModelStreamEvent(kind="delta", text="Checking the first source"),
                completion(
                    ChatMessage(
                        "assistant",
                        "",
                        tool_calls=(ToolCall("call-1", "search_catalog", {"query": "first"}),),
                    )
                ),
            ],
            [
                ModelStreamEvent(kind="delta", text="Checking another source"),
                completion(
                    ChatMessage(
                        "assistant",
                        "",
                        tool_calls=(ToolCall("call-2", "search_catalog", {"query": "second"}),),
                    )
                ),
            ],
            [
                ModelStreamEvent(kind="delta", text="Final streamed answer."),
                completion(ChatMessage("assistant", "Final streamed answer.")),
            ],
        ]

        async def collect():
            return [
                event
                async for event in self.service.stream_chat_for_account(
                    "account-1",
                    "stream-conversation",
                    "Explain the research result.",
                )
            ]

        events = asyncio.run(collect())

        self.assertEqual(
            [event.kind for event in events],
            ["delta", "reset", "delta", "reset", "delta", "complete"],
        )
        self.assertEqual(
            [event.reason for event in events if event.kind == "reset"],
            ["intermediate", "intermediate"],
        )
        turn = events[-1].turn
        self.assertEqual(turn.answer.markdown, "Final streamed answer.")
        self.assertEqual(len(self.runtime.requests), 3)
        self.assertEqual(self.runtime.complete_calls, 0)
        self.assertEqual(len(self.catalog_tool.calls), 2)
        snapshot = self.store.get_snapshot("account-1", "stream-conversation", 10)
        self.assertEqual([message.role for message in snapshot.messages], ["user", "assistant"])
        self.assertEqual(snapshot.messages[-1].content, "Final streamed answer.")

    def test_chat_turn_timeout_cancels_inference_without_persisting_a_turn(self) -> None:
        self.runtime.delay_seconds = 5
        service = LumiConversationService(
            self.store,
            self.runtime,
            ToolRegistry((self.catalog_tool, self.web_tool)),
            Ed25519DelegationVerifier(
                self.private_key.public_key().public_bytes(
                    serialization.Encoding.PEM,
                    serialization.PublicFormat.SubjectPublicKeyInfo,
                ),
                clock=lambda: self.now,
            ),
            self.catalog,
            agent_limits=AgentLimits(turn_timeout_seconds=1.0),
        )

        async def run_until_inference_is_cancelled() -> None:
            self.runtime.started_event = asyncio.Event()
            task = asyncio.create_task(
                service.chat(self.token(), "conversation-1", "Wait forever")
            )
            await asyncio.wait_for(self.runtime.started_event.wait(), timeout=3.0)
            with self.assertRaises(InferenceError):
                await task

        asyncio.run(run_until_inference_is_cancelled())

        snapshot = self.store.get_snapshot("account-1", "conversation-1")
        self.assertEqual(snapshot.messages, ())
        self.assertTrue(self.runtime.cancelled)

    def test_chat_turn_timeout_includes_lock_and_inference_slot_waits(self) -> None:
        verifier = Ed25519DelegationVerifier(
            self.private_key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            ),
            clock=lambda: self.now,
        )
        service = LumiConversationService(
            self.store,
            self.runtime,
            ToolRegistry((self.catalog_tool, self.web_tool)),
            verifier,
            self.catalog,
            agent_limits=AgentLimits(turn_timeout_seconds=0.05),
        )

        async def blocked_on_lock() -> None:
            async with service._conversation_locks.hold(
                _conversation_lock_key("account-1", "lock-wait")
            ):
                with self.assertRaises(InferenceError):
                    await service.chat(
                        self.token(conversation_id="lock-wait"), "lock-wait", "Hi"
                    )

        async def blocked_on_inference_slot() -> None:
            await service._inference_slots.acquire()
            try:
                with self.assertRaises(InferenceError):
                    await service.chat(self.token(conversation_id="slot-wait"), "slot-wait", "Hi")
            finally:
                service._inference_slots.release()

        asyncio.run(blocked_on_lock())
        self.assertEqual(self.store.list_conversations("account-1"), ())
        asyncio.run(blocked_on_inference_slot())
        slot_snapshot = self.store.get_snapshot("account-1", "slot-wait")
        self.assertEqual(slot_snapshot.messages, ())

    def test_turn_persists_only_validated_entity_references_and_structured_sources(self) -> None:
        self.runtime.responses = [
            ChatMessage(
                "assistant",
                "",
                tool_calls=(
                    ToolCall("catalog-call", "search_catalog", {"query": "Frieren"}),
                    ToolCall("web-call", "search_web", {"query": "Frieren themes"}),
                ),
            ),
            ChatMessage(
                "assistant",
                'Frieren is available: :::zenstream{type="series" id="series-1"}.',
            ),
        ]
        response = self.client.post(
            "/internal/conversations/conversation-1/turns",
            headers=self.headers(self.token()),
            json={"message": "Find Frieren and research its themes."},
        )

        self.assertEqual(response.status_code, 200, response.text)
        answer = response.json()["answer"]
        self.assertEqual(
            answer["references"],
            [{"type": "series", "id": "series-1", "title": "Frieren"}],
        )
        self.assertEqual(answer["sources"][0]["url"], "https://example.org/frieren")
        self.assertEqual(self.catalog_tool.calls[0][0].account_id, "account-1")
        self.assertEqual(self.catalog_tool.calls[0][0].delegation_token, self.token())

        read = self.client.get(
            "/internal/conversations/conversation-1",
            headers=self.headers(self.token(scopes=(SCOPE_CONVERSATION_READ,))),
        )
        assistant_message = read.json()["messages"][1]
        self.assertEqual(assistant_message["references"], answer["references"])
        self.assertEqual(assistant_message["sources"], answer["sources"])

    def test_request_rejects_body_identity_and_blank_messages(self) -> None:
        endpoint = "/internal/conversations/conversation-1/turns"
        headers = self.headers(self.token())
        identity = self.client.post(
            endpoint,
            headers=headers,
            json={"message": "Hello", "owner_id": "account-attacker"},
        )
        blank = self.client.post(endpoint, headers=headers, json={"message": "  \n  "})
        oversized = self.client.post(
            endpoint,
            headers=headers,
            json={"message": "x" * 6_001},
        )

        self.assertEqual(identity.status_code, 422)
        self.assertEqual(blank.status_code, 422)
        self.assertEqual(oversized.status_code, 422)
        self.assertEqual(self.store.list_conversations("account-1"), ())

    def test_empty_model_output_returns_a_sanitized_unavailable_response(self) -> None:
        self.runtime.responses = [ChatMessage("assistant", "<think>private reasoning</think>")]
        response = self.client.post(
            "/internal/conversations/conversation-1/turns",
            headers=self.headers(self.token()),
            json={"message": "Hello"},
        )

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private reasoning", response.text)

    def test_first_conversation_inherits_a_valid_user_model_preference(self) -> None:
        preference = self.client.put(
            "/internal/preferences/model",
            headers=self.headers(
                self.token(conversation_id=None, scopes=(SCOPE_PREFERENCE_WRITE,))
            ),
            json={"model": "qwen3.5:4b", "thinking": True},
        )
        self.assertEqual(preference.status_code, 200, preference.text)

        response = self.client.post(
            "/internal/conversations/conversation-1/turns",
            headers=self.headers(self.token()),
            json={"message": "Hello"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["conversation"]["model"], "qwen3.5:4b")
        self.assertTrue(response.json()["conversation"]["thinking"])
        self.assertEqual(self.runtime.requests[0].model, "qwen3.5:4b")
        self.assertTrue(self.runtime.requests[0].thinking)

    def test_first_turn_choice_is_conversation_scoped_and_does_not_change_preference(self) -> None:
        self.store.set_user_preference("account-1", "qwen3.5:4b", True)
        response = self.client.post(
            "/internal/conversations/conversation-1/turns",
            headers=self.headers(self.token()),
            json={
                "message": "Use the small model for this conversation.",
                "model": "qwen3.5:0.8b",
                "thinking": False,
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["conversation"]["model"], "qwen3.5:0.8b")
        self.assertFalse(response.json()["conversation"]["thinking"])
        self.assertEqual(self.runtime.requests[0].model, "qwen3.5:0.8b")
        self.assertFalse(self.runtime.requests[0].thinking)
        preference = self.store.get_user_preference("account-1")
        self.assertEqual((preference.model, preference.thinking), ("qwen3.5:4b", True))

    def test_first_turn_choice_requires_a_complete_available_supported_pair(self) -> None:
        endpoint = "/internal/conversations/conversation-1/turns"
        headers = self.headers(self.token())
        missing_thinking = self.client.post(
            endpoint,
            headers=headers,
            json={"message": "Hello", "model": "qwen3.5:2b"},
        )
        disabled_model = self.client.post(
            endpoint,
            headers=headers,
            json={"message": "Hello", "model": "qwen3.5:9b", "thinking": False},
        )
        unsupported_thinking = self.client.post(
            endpoint,
            headers=headers,
            json={"message": "Hello", "model": "qwen3.5:0.8b", "thinking": True},
        )

        self.assertEqual(missing_thinking.status_code, 422)
        self.assertEqual(disabled_model.status_code, 409)
        self.assertEqual(unsupported_thinking.status_code, 409)
        self.assertEqual(self.store.list_conversations("account-1"), ())

    def test_first_turn_choice_cannot_change_an_existing_conversation(self) -> None:
        endpoint = "/internal/conversations/conversation-1/turns"
        headers = self.headers(self.token())
        created = self.client.post(
            endpoint,
            headers=headers,
            json={"message": "Hello", "model": "qwen3.5:2b", "thinking": False},
        )
        conflicting = self.client.post(
            endpoint,
            headers=headers,
            json={"message": "Try to switch", "model": "qwen3.5:4b", "thinking": True},
        )

        self.assertEqual(created.status_code, 200, created.text)
        self.assertEqual(conflicting.status_code, 409, conflicting.text)
        self.assertEqual(len(self.runtime.requests), 1)
        self.assertEqual(len(self.store.get_snapshot("account-1", "conversation-1").messages), 2)

    def test_disabled_saved_preference_falls_back_to_server_default(self) -> None:
        self.store.set_user_preference("account-1", "qwen3.5:9b", True)
        response = self.client.post(
            "/internal/conversations/conversation-1/turns",
            headers=self.headers(self.token()),
            json={"message": "Hello"},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["conversation"]["model"], "qwen3.5:2b")
        self.assertFalse(response.json()["conversation"]["thinking"])
        self.assertEqual(self.runtime.requests[0].model, "qwen3.5:2b")

    def test_conversation_choice_is_persisted_and_used_by_next_turn(self) -> None:
        self.client.post(
            "/internal/conversations/conversation-1/turns",
            headers=self.headers(self.token()),
            json={"message": "First turn"},
        )
        choice = self.client.patch(
            "/internal/conversations/conversation-1/choice",
            headers=self.headers(self.token(scopes=(SCOPE_CONVERSATION_CONFIGURE,))),
            json={"model": "qwen3.5:4b", "thinking": True},
        )
        self.assertEqual(choice.status_code, 200, choice.text)

        next_turn = self.client.post(
            "/internal/conversations/conversation-1/turns",
            headers=self.headers(self.token()),
            json={"message": "Second turn"},
        )
        self.assertEqual(next_turn.status_code, 200, next_turn.text)
        self.assertEqual(self.runtime.requests[-1].model, "qwen3.5:4b")
        self.assertTrue(self.runtime.requests[-1].thinking)

    def test_unavailable_models_and_unsupported_thinking_are_rejected(self) -> None:
        headers = self.headers(self.token(scopes=(SCOPE_CONVERSATION_CONFIGURE,)))
        unavailable = self.client.patch(
            "/internal/conversations/conversation-1/choice",
            headers=headers,
            json={"model": "qwen3.5:9b", "thinking": False},
        )
        unsupported = self.client.patch(
            "/internal/conversations/conversation-1/choice",
            headers=headers,
            json={"model": "qwen3.5:0.8b", "thinking": True},
        )

        self.assertEqual(unavailable.status_code, 409)
        self.assertEqual(unsupported.status_code, 409)

    def test_delegation_scope_and_signed_conversation_id_are_enforced(self) -> None:
        no_chat_scope = self.token(scopes=(SCOPE_CONVERSATION_CREATE,))
        wrong_conversation = self.token(conversation_id="another-conversation")
        endpoint = "/internal/conversations/conversation-1/turns"

        self.assertEqual(
            self.client.post(
                endpoint, headers=self.headers(no_chat_scope), json={"message": "Hi"}
            ).status_code,
            401,
        )
        self.assertEqual(
            self.client.post(
                endpoint, headers=self.headers(wrong_conversation), json={"message": "Hi"}
            ).status_code,
            401,
        )

    def test_foreign_owner_cannot_read_or_create_over_an_existing_conversation_id(self) -> None:
        created = self.client.post(
            "/internal/conversations/conversation-1/turns",
            headers=self.headers(self.token(owner="account-1")),
            json={"message": "Private"},
        )
        self.assertEqual(created.status_code, 200, created.text)

        foreign_read = self.client.get(
            "/internal/conversations/conversation-1",
            headers=self.headers(
                self.token(owner="account-2", scopes=(SCOPE_CONVERSATION_READ,))
            ),
        )
        foreign_chat = self.client.post(
            "/internal/conversations/conversation-1/turns",
            headers=self.headers(self.token(owner="account-2")),
            json={"message": "Attempt to collide"},
        )
        self.assertEqual(foreign_read.status_code, 404)
        self.assertEqual(foreign_chat.status_code, 404)
        self.assertEqual(self.store.list_conversations("account-2"), ())


if __name__ == "__main__":
    unittest.main()
