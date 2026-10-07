"""Composition helpers for the local Lumi service."""

from __future__ import annotations

import httpx

from lumi.orchestrator_tools import (
    MAX_ORCHESTRATOR_RESPONSE_BYTES,
    CatalogItemDetailTool,
    CatalogSearchTool,
    ContinueWatchingTool,
    FavoritesTool,
    HomeRecommendationsTool,
    NextUpTool,
    OrchestratorReadOnlyClient,
)
from lumi.tools import ToolRegistry
from lumi.web_research import WebResearchConfig, build_web_research_tools


def build_zenstream_tool_registry(
    orchestrator_base_url: str,
    service_token: str,
    *,
    timeout_seconds: float = 8.0,
    max_response_bytes: int = MAX_ORCHESTRATOR_RESPONSE_BYTES,
    transport: httpx.AsyncBaseTransport | None = None,
    web_research_config: WebResearchConfig | None = None,
    web_transport: httpx.AsyncBaseTransport | None = None,
) -> ToolRegistry:
    """Build six fixed local tools and optional bounded web-research tools.

    The resulting registry can be passed directly to ``LumiConversationService``. The
    optional transports are intended for in-process tests. Production uses HTTPX's default
    network transport with environment proxy settings disabled. Web tools are not registered
    unless service configuration supplies a SearXNG URL.
    """

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
    return ToolRegistry(
        (
            CatalogSearchTool(client),
            CatalogItemDetailTool(client),
            HomeRecommendationsTool(client),
            ContinueWatchingTool(client),
            NextUpTool(client),
            FavoritesTool(client),
            *web_tools,
        )
    )
