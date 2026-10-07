"""Composition helpers for standalone and embedded Lumi deployments."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from lumi.agent import AgentLimits
from lumi.contracts import ChatRuntime
from lumi.model_catalog import ModelCatalog
from lumi.service import LumiConversationService
from lumi.storage import ConversationStore
from lumi.tools import ToolRegistry

if TYPE_CHECKING:
    import httpx

    from lumi.web_research import WebResearchConfig


def create_embedded_service(
    *,
    runtime: ChatRuntime,
    models: ModelCatalog,
    tools: ToolRegistry,
    store_path: str | Path,
    agent_limits: AgentLimits | None = None,
    max_concurrent_chats: int = 1,
    max_active_conversations: int = 128,
) -> LumiConversationService:
    """Create the in-process Lumi API for an authenticated embedding host.

    The host injects the embedded model runtime, its enabled model catalog, and a
    registry of fixed read-only tools. Calls into ``*_for_account`` service methods
    must pass the account ID from the host's authenticated principal. This path creates
    no delegation verifier or Orchestrator HTTP client and needs no service token.
    """

    return LumiConversationService(
        ConversationStore(store_path),
        runtime,
        tools,
        delegation_verifier=None,
        models=models,
        agent_limits=agent_limits,
        max_concurrent_chats=max_concurrent_chats,
        max_active_conversations=max_active_conversations,
    )


def build_zenstream_tool_registry(
    orchestrator_base_url: str,
    service_token: str,
    *,
    timeout_seconds: float = 8.0,
    max_response_bytes: int = 1_000_000,
    transport: httpx.AsyncBaseTransport | None = None,
    web_research_config: WebResearchConfig | None = None,
    web_transport: httpx.AsyncBaseTransport | None = None,
) -> ToolRegistry:
    """Build six fixed local catalog tools and two bounded web-research tools.

    The resulting registry can be passed directly to ``LumiConversationService``. The
    optional transports are intended for in-process tests. Production uses HTTPX's default
    network transport with environment proxy settings disabled. Web search stays disabled
    until service configuration supplies a SearXNG URL.
    """

    from lumi.orchestrator_tools import (
        CatalogItemDetailTool,
        CatalogSearchTool,
        ContinueWatchingTool,
        FavoritesTool,
        HomeRecommendationsTool,
        NextUpTool,
        OrchestratorReadOnlyClient,
    )
    from lumi.web_research import WebResearchConfig, build_web_research_tools

    client = OrchestratorReadOnlyClient(
        orchestrator_base_url,
        service_token,
        timeout_seconds=timeout_seconds,
        max_response_bytes=max_response_bytes,
        transport=transport,
    )
    web_tools = build_web_research_tools(
        web_research_config or WebResearchConfig(),
        transport=web_transport,
    )
    local_tools = (
        CatalogSearchTool(client),
        CatalogItemDetailTool(client),
        HomeRecommendationsTool(client),
        ContinueWatchingTool(client),
        NextUpTool(client),
        FavoritesTool(client),
    )
    return ToolRegistry((*local_tools, *web_tools))

