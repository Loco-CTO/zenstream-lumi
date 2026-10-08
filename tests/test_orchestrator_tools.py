from __future__ import annotations

import json
import unittest
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

from lumi.agent import ChatAgent
from lumi.contracts import (
    ChatContext,
    ChatMessage,
    EntityReference,
    EvidenceTrust,
    ModelRequest,
    ModelResponse,
    ToolCall,
)
from lumi.service_factory import build_zenstream_tool_registry
from lumi.web_research import WebResearchConfig

SERVICE_TOKEN = "s" * 40
DELEGATION = "signed.local.delegation"


def make_context() -> ChatContext:
    return ChatContext(
        account_id="account-1",
        conversation_id="conversation-1",
        model="qwen3.5:2b",
        thinking=False,
        delegation_token=DELEGATION,
        turn_id="turn-1",
    )


class FakeRuntime:
    def __init__(self, messages: list[ChatMessage]) -> None:
        self._messages = list(messages)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(self._messages.pop(0))


def media_item(
    entity_id: str = "series-7",
    *,
    entity_type: str = "series",
    title: str = "Frieren",
) -> dict[str, Any]:
    return {
        "id": entity_id,
        "type": entity_type,
        "title": title,
        "overview": "A bounded local summary.",
        "genres": ["Adventure"],
    }


class OrchestratorToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_agent_uses_delegated_search_and_returns_validated_reference(self) -> None:
        app = FastAPI()
        captured: list[tuple[str, str, dict[str, str], dict[str, Any]]] = []

        @app.post("/api/internal/lumi/tools/catalog-search")
        async def catalog_search(request: Request) -> JSONResponse:
            body = await request.json()
            captured.append(
                (
                    request.method,
                    request.url.path,
                    dict(request.headers),
                    body,
                )
            )
            return JSONResponse({"items": [media_item()], "total": 1})

        registry = build_zenstream_tool_registry(
            "http://orchestrator.test:9090",
            SERVICE_TOKEN,
            transport=httpx.ASGITransport(app=app),
        )
        runtime = FakeRuntime(
            [
                ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=(
                        ToolCall(
                            "call-1",
                            "zenstream_catalog_search",
                            {"query": "Frieren", "type": "series"},
                        ),
                    ),
                ),
                ChatMessage(
                    role="assistant",
                    content='Open :::zenstream{type="series" id="series-7"}.',
                ),
            ]
        )

        answer = await ChatAgent(runtime, registry).answer(
            make_context(), [], "Show me Frieren"
        )

        self.assertEqual(len(captured), 1)
        method, path, headers, body = captured[0]
        self.assertEqual(method, "POST")
        self.assertEqual(path, "/api/internal/lumi/tools/catalog-search")
        self.assertEqual(headers["authorization"], f"Bearer {SERVICE_TOKEN}")
        self.assertEqual(headers["x-lumi-delegation"], DELEGATION)
        self.assertEqual(body, {"query": "Frieren", "type": "series", "limit": 8})
        self.assertEqual(
            answer.references,
            (EntityReference(type="series", id="series-7", title="Frieren"),),
        )
        self.assertEqual(answer.markdown, 'Open :::zenstream{type="series" id="series-7"}.')
        self.assertEqual(
            tuple(item.name for item in runtime.requests[0].tools),
            (
                "zenstream_catalog_search",
                "zenstream_catalog_resolve",
                "zenstream_catalog_item_detail",
                "zenstream_home_recommendations",
                "zenstream_continue_watching",
                "zenstream_next_up",
                "zenstream_favorites",
                "web_search",
                "web_read",
            ),
        )
        self.assertEqual(runtime.requests[1].messages[-1].role, "tool")
        self.assertIn(EvidenceTrust.LOCAL.value, runtime.requests[1].messages[-1].content)

    async def test_release_results_preserve_trusted_references_and_bounded_progress(self) -> None:
        app = FastAPI()

        @app.post("/api/internal/lumi/tools/catalog-search")
        async def catalog_search(_: Request) -> JSONResponse:
            return JSONResponse(
                {
                    "items": [
                        {
                            **media_item(
                                "release-9",
                                entity_type="release",
                                title="A local soundtrack",
                            ),
                            "userState": {
                                "favorite": True,
                                "played": False,
                                "playCount": 2,
                                "positionSeconds": 12.5,
                                "durationSeconds": 184.0,
                                "lastPlayedAt": "2026-10-07T19:00:00Z",
                                "privatePath": "C:/private/media.mkv",
                            },
                        }
                    ],
                    "total": 1,
                }
            )

        registry = build_zenstream_tool_registry(
            "http://orchestrator.test:9090",
            SERVICE_TOKEN,
            transport=httpx.ASGITransport(app=app),
        )
        search = registry.get("zenstream_catalog_search")
        assert search is not None
        result = await search.execute(
            make_context(),
            search.validate_arguments({"query": "soundtrack", "type": "release"}),
        )

        self.assertEqual(
            result.entities,
            (EntityReference("release", "release-9", "A local soundtrack"),),
        )
        payload = json.loads(result.content)
        self.assertEqual(
            payload["items"][0]["userState"],
            {
                "favorite": True,
                "played": False,
                "playCount": 2,
                "positionSeconds": 12.5,
                "durationSeconds": 184.0,
                "lastPlayedAt": "2026-10-07T19:00:00Z",
            },
        )

    async def test_registry_includes_zero_config_search_and_page_reader(self) -> None:
        app = FastAPI()
        observed: list[tuple[str, str]] = []

        @app.api_route("/{path:path}", methods=["GET", "POST"])
        async def fixed_endpoint(request: Request, path: str) -> JSONResponse:
            observed.append((request.method, request.url.path))
            if request.method == "POST":
                return JSONResponse({"items": [media_item()], "total": 1})
            if "/catalog-items/" in request.url.path:
                return JSONResponse(
                    {
                        "item": media_item(),
                        "backgroundItem": None,
                        "seasons": [
                            media_item("season-3", entity_type="season", title="Season 3")
                        ],
                    }
                )
            return JSONResponse({"items": [media_item()]})

        registry = build_zenstream_tool_registry(
            "http://orchestrator.test:9090",
            SERVICE_TOKEN,
            transport=httpx.ASGITransport(app=app),
            web_research_config=WebResearchConfig(),
        )
        expected = {
            ("POST", "/api/internal/lumi/tools/catalog-search"),
            ("GET", "/api/internal/lumi/tools/catalog-items/series-7"),
            ("GET", "/api/internal/lumi/tools/home/recommendations"),
            ("GET", "/api/internal/lumi/tools/home/continue-watching"),
            ("GET", "/api/internal/lumi/tools/home/next-up"),
            ("GET", "/api/internal/lumi/tools/favorites"),
        }
        arguments = {
            "zenstream_catalog_search": {"query": "Frieren"},
            "zenstream_catalog_resolve": {
                "candidates": [
                    {"titles": [{"title": "Frieren", "language": "en"}], "type": "series"}
                ]
            },
            "zenstream_catalog_item_detail": {"entity_id": "series-7"},
            "zenstream_home_recommendations": {},
            "zenstream_continue_watching": {},
            "zenstream_next_up": {},
            "zenstream_favorites": {},
        }

        for name, tool_arguments in arguments.items():
            tool = registry.get(name)
            self.assertIsNotNone(tool)
            assert tool is not None
            validated = tool.validate_arguments(tool_arguments)
            result = await tool.execute(make_context(), validated)
            self.assertIs(result.trust, EvidenceTrust.LOCAL)
            self.assertTrue(result.entities, name)
            self.assertEqual(tool.definition.data_scope, "local")

        self.assertEqual(set(observed), expected)
        self.assertEqual(len(observed), len(expected) + 1)
        search = registry.get("web_search")
        page_reader = registry.get("web_read")
        self.assertIsNotNone(search)
        self.assertIsNotNone(page_reader)
        assert search is not None and page_reader is not None
        self.assertEqual(search.definition.data_scope, "external_search")
        self.assertTrue(search.definition.read_only)
        self.assertEqual(page_reader.definition.data_scope, "external_fetch")
        self.assertTrue(page_reader.definition.read_only)
        self.assertIsNone(registry.get("open_web_result"))

    async def test_resolver_matches_english_japanese_and_chinese_titles_with_provider_ids(
        self,
    ) -> None:
        app = FastAPI()
        searches: list[dict[str, Any]] = []

        @app.post("/api/internal/lumi/tools/catalog-search")
        async def catalog_search(request: Request) -> JSONResponse:
            body = await request.json()
            searches.append(body)
            item = {
                **media_item("series-12", title="Frieren: Beyond Journey's End"),
                "year": "2023",
                "originalTitle": "葬送のフリーレン",
                "aliases": ["葬送のフリーレン", "葬送的芙莉莲", "Frieren"],
                "providerIds": [{"provider": "tmdb", "id": "12345"}],
            }
            return JSONResponse({"items": [item], "total": 1})

        registry = build_zenstream_tool_registry(
            "http://orchestrator.test:9090",
            SERVICE_TOKEN,
            transport=httpx.ASGITransport(app=app),
        )
        tool = registry.get("zenstream_catalog_resolve")
        self.assertIsNotNone(tool)
        assert tool is not None
        args = tool.validate_arguments(
            {
                "candidates": [
                    {
                        "titles": [
                            {"title": "Frieren", "language": "en"},
                            {"title": "葬送のフリーレン", "language": "ja"},
                            {"title": "葬送的芙莉莲", "language": "zh"},
                        ],
                        "type": "series",
                        "year": 2023,
                        "providerIds": {"tmdb": "12345"},
                    }
                ]
            }
        )
        result = await tool.execute(make_context(), args)

        payload = json.loads(result.content)
        self.assertEqual([item["language"] for item in searches], ["en", "ja", "zh"])
        self.assertEqual(payload["matches"][0]["status"], "matched")
        self.assertEqual(payload["matches"][0]["item"]["id"], "series-12")
        self.assertEqual(
            payload["matches"][0]["matchedBy"],
            ["media_type", "provider_id", "localized_or_alias_title", "release_year"],
        )
        self.assertEqual(
            result.entities,
            (EntityReference("series", "series-12", "Frieren: Beyond Journey's End"),),
        )

        runtime = FakeRuntime(
            [
                ChatMessage(
                    "assistant",
                    "",
                    tool_calls=(ToolCall("resolve-1", "zenstream_catalog_resolve", args),),
                ),
                ChatMessage(
                    "assistant",
                    'I recommend Frieren: Beyond Journey\'s End because it has political themes '
                    ':::zenstream{type="series" id="series-12"}.',
                ),
            ]
        )
        answer = await ChatAgent(runtime, registry).answer(
            make_context(), [], "Recommend an anime like Code Geass with political intrigue."
        )
        self.assertEqual(
            answer.references,
            (EntityReference("series", "series-12", "Frieren: Beyond Journey's End"),),
        )
        self.assertIn("Frieren: Beyond Journey's End", answer.markdown)
        self.assertNotIn("A local match", answer.markdown)

    async def test_resolver_uses_alias_and_year_and_rejects_conflicting_provider_id(self) -> None:
        app = FastAPI()

        @app.post("/api/internal/lumi/tools/catalog-search")
        async def catalog_search(_: Request) -> JSONResponse:
            return JSONResponse(
                {
                    "items": [
                        {
                            **media_item("series-12", title="Frieren: Beyond Journey's End"),
                            "year": "2023",
                            "aliases": ["葬送のフリーレン"],
                            "providerIds": {"tmdb": "different-id"},
                        }
                    ],
                    "total": 1,
                }
            )

        registry = build_zenstream_tool_registry(
            "http://orchestrator.test:9090",
            SERVICE_TOKEN,
            transport=httpx.ASGITransport(app=app),
        )
        tool = registry.get("zenstream_catalog_resolve")
        self.assertIsNotNone(tool)
        assert tool is not None
        args = tool.validate_arguments(
            {
                "candidates": [
                    {
                        "titles": [
                            {"title": "葬送のフリーレン", "language": "ja"},
                            {"title": "葬送的芙莉莲", "language": "zh"},
                        ],
                        "type": "series",
                        "year": 2023,
                    }
                ]
            }
        )
        alias_result = await tool.execute(make_context(), args)
        alias_match = json.loads(alias_result.content)["matches"][0]
        self.assertEqual(alias_match["status"], "matched")
        self.assertEqual(alias_match["item"]["id"], "series-12")
        self.assertEqual(
            alias_match["matchedBy"], ["media_type", "localized_or_alias_title", "release_year"]
        )

        conflicting_args = tool.validate_arguments(
            {
                "candidates": [
                    {
                        "titles": [{"title": "葬送のフリーレン", "language": "ja"}],
                        "type": "series",
                        "year": 2023,
                        "providerIds": {"tmdb": "expected-id"},
                    }
                ]
            }
        )
        conflict_result = await tool.execute(make_context(), conflicting_args)

        self.assertEqual(json.loads(conflict_result.content)["matches"], [{"status": "no_match"}])
        self.assertEqual(conflict_result.entities, ())

    async def test_factory_uses_a_separate_configured_web_transport(self) -> None:
        orchestrator_requests: list[str] = []
        search_requests: list[tuple[str, str, dict[str, str]]] = []
        orchestrator = FastAPI()
        web = FastAPI()

        @orchestrator.api_route("/{path:path}", methods=["GET", "POST"])
        async def local_route(request: Request, path: str) -> JSONResponse:
            del path
            orchestrator_requests.append(request.url.path)
            return JSONResponse({"items": []})

        @web.post("/search")
        async def web_search(request: Request) -> JSONResponse:
            body = (await request.body()).decode("utf-8")
            search_requests.append((request.method, body, dict(request.headers)))
            return JSONResponse(
                {
                    "results": [
                        {
                            "title": "Current source",
                            "url": "https://example.org/current-source",
                            "content": "A recent, untrusted snippet.",
                        }
                    ]
                }
            )

        registry = build_zenstream_tool_registry(
            "http://orchestrator.test:9090",
            SERVICE_TOKEN,
            transport=httpx.ASGITransport(app=orchestrator),
            web_research_config=WebResearchConfig(searxng_url="http://search.test:8080"),
            web_transport=httpx.ASGITransport(app=web),
        )
        tool = registry.get("web_search")
        assert tool is not None
        result = await tool.execute(
            make_context(),
            tool.validate_arguments({"query": "current media news"}),
        )

        self.assertIs(result.trust, EvidenceTrust.EXTERNAL)
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(search_requests[0][0], "POST")
        self.assertIn("format=json", search_requests[0][1])
        self.assertNotIn("authorization", search_requests[0][2])
        self.assertNotIn("x-lumi-delegation", search_requests[0][2])
        self.assertEqual(orchestrator_requests, [])

    def test_argument_schemas_reject_extra_fields_and_out_of_bounds_values(self) -> None:
        registry = build_zenstream_tool_registry("http://127.0.0.1:9090", SERVICE_TOKEN)
        search = registry.get("zenstream_catalog_search")
        detail = registry.get("zenstream_catalog_item_detail")
        home = registry.get("zenstream_next_up")
        assert search is not None and detail is not None and home is not None

        type_schema = search.definition.parameters["properties"]["type"]
        supported_types = {"movie", "series", "collection", "release", "artist", "track"}
        self.assertEqual(type_schema["type"], ["string", "null"])
        self.assertEqual(set(type_schema["enum"]), supported_types | {None})
        for search_type in supported_types:
            self.assertEqual(
                search.validate_arguments({"query": "x", "type": search_type})["type"],
                search_type,
            )
        self.assertIsNone(search.validate_arguments({"query": "x", "type": None})["type"])

        for invalid in (
            {"query": "x", "path": "/arbitrary"},
            {"query": "x", "limit": True},
            {"query": "x", "limit": 11},
            {"query": " "},
            {"query": "x" * 121},
            {"query": "x", "type": "season"},
            {"query": "x", "type": "episode"},
            {"query": "x", "type": "person"},
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    search.validate_arguments(invalid)
        with self.assertRaises(ValueError):
            detail.validate_arguments({"entity_id": "x" * 129})
        with self.assertRaises(ValueError):
            home.validate_arguments({"url": "http://attacker.test"})

    async def test_unsupported_search_types_are_rejected_before_http(self) -> None:
        app = FastAPI()
        received: list[str] = []

        @app.post("/api/internal/lumi/tools/catalog-search")
        async def catalog_search(request: Request) -> JSONResponse:
            received.append((await request.json())["type"])
            return JSONResponse({"items": []})

        registry = build_zenstream_tool_registry(
            "http://orchestrator.test:9090",
            SERVICE_TOKEN,
            transport=httpx.ASGITransport(app=app),
        )
        invalid_types = ("season", "episode", "person")
        responses = [
            message
            for index, search_type in enumerate(invalid_types)
            for message in (
                ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=(
                        ToolCall(
                            f"call-{index}",
                            "zenstream_catalog_search",
                            {"query": "x", "type": search_type},
                        ),
                    ),
                ),
                ChatMessage(role="assistant", content="That search type is unavailable."),
            )
        ]
        runtime = FakeRuntime(responses)
        agent = ChatAgent(runtime, registry)

        for search_type in invalid_types:
            await agent.answer(make_context(), [], f"Search for a {search_type}")

        self.assertEqual(received, [])

    async def test_response_size_is_bounded_and_redirects_are_not_followed(self) -> None:
        app = FastAPI()
        requested: list[str] = []

        @app.post("/api/internal/lumi/tools/catalog-search")
        async def oversized(_: Request) -> JSONResponse:
            requested.append("search")
            return JSONResponse({"items": [media_item()]})

        @app.get("/api/internal/lumi/tools/home/next-up")
        async def redirected() -> RedirectResponse:
            requested.append("redirect-source")
            return RedirectResponse("/api/internal/lumi/tools/favorites", status_code=307)

        @app.get("/api/internal/lumi/tools/favorites")
        async def redirect_target() -> JSONResponse:
            requested.append("redirect-target")
            return JSONResponse({"items": [media_item()]})

        registry = build_zenstream_tool_registry(
            "http://orchestrator.test:9090",
            SERVICE_TOKEN,
            max_response_bytes=64,
            transport=httpx.ASGITransport(app=app),
        )
        search = registry.get("zenstream_catalog_search")
        next_up = registry.get("zenstream_next_up")
        assert search is not None and next_up is not None
        search_result = await search.execute(
            make_context(), search.validate_arguments({"query": "Frieren"})
        )
        next_up_result = await next_up.execute(make_context(), {})

        self.assertIn("oversized", search_result.content)
        self.assertFalse(search_result.entities)
        self.assertIn("HTTP 307", next_up_result.content)
        self.assertEqual(requested, ["search", "redirect-source"])

    def test_invalid_orchestrator_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_zenstream_tool_registry("http://user:pass@localhost:9090", SERVICE_TOKEN)
        with self.assertRaises(ValueError):
            build_zenstream_tool_registry(
                "http://localhost:9090/api?path=elsewhere", SERVICE_TOKEN
            )
        with self.assertRaises(ValueError):
            build_zenstream_tool_registry("http://localhost:9090", "short")


if __name__ == "__main__":
    unittest.main()
