"""Typed, bounded clients for Lumi's fixed Orchestrator catalog reads."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from lumi.contracts import (
    ENTITY_TYPES,
    ChatContext,
    EntityReference,
    EvidenceTrust,
    ToolDefinition,
    ToolResult,
)

MAX_ORCHESTRATOR_RESPONSE_BYTES = 1_000_000
MAX_TOOL_RESULT_CHARS = 12_000
MAX_TOOL_ITEMS = 18
_LANGUAGE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,15}$")
_SEARCH_TYPES = frozenset({"movie", "series", "collection", "release", "artist", "track"})


class _OrchestratorReadError(RuntimeError):
    """Safe-to-show failure from a read-only Orchestrator lookup."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON field")
        value[key] = item
    return value


def _reject_json_constant(_value: str) -> None:
    raise ValueError("Invalid JSON number")


def _valid_base_url(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 2_048 or value != value.strip():
        raise ValueError("Orchestrator base URL is invalid")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise ValueError("Orchestrator base URL is invalid") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or port is None
    ):
        raise ValueError("Orchestrator base URL must be an absolute HTTP URL with a port")
    return value.rstrip("/")


def _validate_service_token(value: str) -> str:
    if (
        not isinstance(value, str)
        or not 32 <= len(value) <= 4_096
        or any(ord(character) < 33 or ord(character) > 126 for character in value)
    ):
        raise ValueError("Orchestrator service token is invalid")
    return value


class OrchestratorReadOnlyClient:
    """A fixed-route HTTP client; callers cannot provide a URL or HTTP method."""

    def __init__(
        self,
        base_url: str,
        service_token: str,
        *,
        timeout_seconds: float = 8.0,
        max_response_bytes: int = MAX_ORCHESTRATOR_RESPONSE_BYTES,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if (
            not isinstance(timeout_seconds, (float, int))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or not 0.1 <= timeout_seconds <= 30.0
        ):
            raise ValueError("Orchestrator timeout must be between 0.1 and 30 seconds")
        if (
            not isinstance(max_response_bytes, int)
            or isinstance(max_response_bytes, bool)
            or not 1 <= max_response_bytes <= MAX_ORCHESTRATOR_RESPONSE_BYTES
        ):
            raise ValueError("Orchestrator response limit is invalid")
        self._base_url = _valid_base_url(base_url)
        self._service_token = _validate_service_token(service_token)
        self._timeout_seconds = float(timeout_seconds)
        self._max_response_bytes = max_response_bytes
        self._transport = transport

    async def search(
        self,
        context: ChatContext,
        *,
        query: str,
        item_type: str | None,
        limit: int,
        language: str | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"query": query, "limit": limit}
        if item_type is not None:
            body["type"] = item_type
        if language is not None:
            body["language"] = language
        return await self._request_json(
            "POST", "/api/internal/lumi/tools/catalog-search", context, body=body
        )

    async def item_detail(
        self,
        context: ChatContext,
        *,
        entity_id: str,
        language: str | None,
    ) -> dict[str, Any]:
        path = f"/api/internal/lumi/tools/catalog-items/{quote(entity_id, safe='')}"
        params = {"language": language} if language is not None else None
        return await self._request_json("GET", path, context, params=params)

    async def home_recommendations(self, context: ChatContext) -> dict[str, Any]:
        return await self._request_json(
            "GET", "/api/internal/lumi/tools/home/recommendations", context
        )

    async def continue_watching(self, context: ChatContext) -> dict[str, Any]:
        return await self._request_json(
            "GET", "/api/internal/lumi/tools/home/continue-watching", context
        )

    async def next_up(self, context: ChatContext) -> dict[str, Any]:
        return await self._request_json(
            "GET", "/api/internal/lumi/tools/home/next-up", context
        )

    async def favorites(self, context: ChatContext) -> dict[str, Any]:
        return await self._request_json("GET", "/api/internal/lumi/tools/favorites", context)

    async def _request_json(
        self,
        method: str,
        path: str,
        context: ChatContext,
        *,
        params: Mapping[str, str] | None = None,
        body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        delegation = context.delegation_token
        if (
            not delegation
            or len(delegation) > 8_192
            or not delegation.isascii()
            or any(ord(character) < 33 or ord(character) > 126 for character in delegation)
        ):
            raise _OrchestratorReadError("The Orchestrator delegation is invalid.")
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._service_token}",
            "X-Lumi-Delegation": delegation,
        }
        timeout = httpx.Timeout(
            self._timeout_seconds,
            connect=min(2.0, self._timeout_seconds),
            read=self._timeout_seconds,
            write=min(5.0, self._timeout_seconds),
            pool=min(2.0, self._timeout_seconds),
        )
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=timeout,
                limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
                trust_env=False,
                follow_redirects=False,
                transport=self._transport,
            ) as client:
                async with client.stream(
                    method,
                    path,
                    headers=headers,
                    params=params,
                    json=body,
                ) as response:
                    if response.status_code != 200:
                        raise _OrchestratorReadError(
                            f"The local catalog lookup returned HTTP {response.status_code}."
                        )
                    content_length = response.headers.get("content-length")
                    if content_length and content_length.isdecimal():
                        if int(content_length) > self._max_response_bytes:
                            raise _OrchestratorReadError(
                                "The local catalog returned an oversized response."
                            )
                    content = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                        if len(content) + len(chunk) > self._max_response_bytes:
                            raise _OrchestratorReadError(
                                "The local catalog returned an oversized response."
                            )
                        content.extend(chunk)
        except _OrchestratorReadError:
            raise
        except httpx.HTTPError as error:
            # Do not include exception text: it can contain request details.
            raise _OrchestratorReadError("The local catalog lookup is unavailable.") from error

        try:
            result = json.loads(
                content,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeError, ValueError, TypeError) as error:
            raise _OrchestratorReadError("The local catalog returned invalid JSON.") from error
        if not isinstance(result, dict):
            raise _OrchestratorReadError("The local catalog returned an invalid response.")
        return result


def _keys(arguments: Mapping[str, Any], allowed: set[str], required: set[str]) -> None:
    if any(not isinstance(key, str) for key in arguments):
        raise ValueError("Tool argument keys must be strings")
    supplied = set(arguments)
    if not required <= supplied or not supplied <= allowed:
        raise ValueError("Tool arguments do not match the fixed schema")


def _bounded_text(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split())
    return normalized[:limit] if normalized else None


def _provider_ids(value: object) -> dict[str, str]:
    identities: dict[str, str] = {}
    if isinstance(value, Mapping):
        pairs = value.items()
    elif isinstance(value, list):
        pairs = (
            (identity.get("provider"), identity.get("id") or identity.get("providerId"))
            for identity in value
            if isinstance(identity, Mapping)
        )
    else:
        return identities
    for raw_provider, raw_id in pairs:
        provider = _bounded_text(raw_provider, 32)
        provider_id = _bounded_text(raw_id, 96)
        if (
            provider
            and provider_id
            and re.fullmatch(r"[A-Za-z0-9_-]+", provider)
            and not any(ord(character) < 32 or ord(character) == 127 for character in provider_id)
        ):
            identities[provider.casefold()] = provider_id
        if len(identities) >= 8:
            break
    return identities


def _title_variants(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    values: list[str] = []
    seen: set[str] = set()
    for entry in value[:12]:
        title = _bounded_text(entry, 160)
        if title is None:
            continue
        key = _normalize_identity_text(title)
        if key and key not in seen:
            seen.add(key)
            values.append(title)
    return values


def _normalize_identity_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _identity_year(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    match = re.search(r"(?:18|19|20|21)\d{2}", str(value))
    return int(match.group(0)) if match else None


def _match_catalog_candidate(
    candidate: Mapping[str, Any], item: Mapping[str, Any]
) -> tuple[int, tuple[str, ...]] | None:
    if item.get("type") != candidate.get("type"):
        return None
    candidate_ids = candidate.get("providerIds", {})
    local_ids = item.get("providerIds", {})
    candidate_ids = candidate_ids if isinstance(candidate_ids, Mapping) else {}
    local_ids = local_ids if isinstance(local_ids, Mapping) else {}
    shared_providers = set(candidate_ids) & set(local_ids)
    if any(
        str(candidate_ids[provider]).casefold() != str(local_ids[provider]).casefold()
        for provider in shared_providers
    ):
        return None
    provider_match = any(
        str(candidate_ids[provider]).casefold() == str(local_ids[provider]).casefold()
        for provider in shared_providers
    )

    candidate_titles = {
        _normalize_identity_text(entry["title"])
        for entry in candidate.get("titles", [])
        if isinstance(entry, Mapping) and isinstance(entry.get("title"), str)
    }
    local_aliases = item.get("aliases", [])
    local_titles = {
        _normalize_identity_text(title)
        for title in (
            item.get("title"),
            item.get("originalTitle"),
            *(local_aliases if isinstance(local_aliases, list) else []),
        )
        if isinstance(title, str)
    }
    title_match = bool(candidate_titles & local_titles)
    if not provider_match and not title_match:
        return None

    candidate_year = candidate.get("year")
    local_year = _identity_year(
        item.get("year") or item.get("releaseDate") or item.get("date")
    )
    if (
        type(candidate_year) is int
        and local_year is not None
        and abs(candidate_year - local_year) > 1
    ):
        return None

    evidence: list[str] = ["media_type"]
    score = 20
    if provider_match:
        score += 1_000
        evidence.append("provider_id")
    if title_match:
        score += 100
        evidence.append("localized_or_alias_title")
    if candidate_year is not None and local_year is not None:
        score += 20 if candidate_year == local_year else 10
        evidence.append("release_year")
    return score, tuple(evidence)


def _valid_entity_id(value: object) -> str | None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        return None
    return value


def _compact_item(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    entity_id = _valid_entity_id(value.get("id"))
    entity_type = value.get("type")
    title = _bounded_text(value.get("title") or value.get("name"), 200)
    if (
        entity_id is None
        or not isinstance(entity_type, str)
        or entity_type not in ENTITY_TYPES
        or not title
    ):
        return None

    compact: dict[str, Any] = {"id": entity_id, "type": entity_type, "title": title}
    raw_metadata = value.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    for key, limit in (
        ("overview", 800),
        ("description", 800),
        ("year", 16),
        ("date", 32),
        ("releaseDate", 32),
        ("album", 200),
        ("albumArtist", 200),
        ("originalTitle", 160),
    ):
        text = _bounded_text(
            value.get(key) if value.get(key) is not None else metadata.get(key),
            limit,
        )
        if text:
            compact[key] = text
    aliases = _title_variants(
        value.get("aliases") if value.get("aliases") is not None else metadata.get("aliases")
    )
    if aliases:
        compact["aliases"] = aliases
    provider_ids = _provider_ids(value.get("providerIds"))
    if provider_ids:
        compact["providerIds"] = provider_ids
    for key, upper_bound in (("communityRating", 10), ("runtimeMinutes", 100_000)):
        number = value.get(key)
        if type(number) is int and 0 <= number <= upper_bound:
            compact[key] = number
        elif (
            type(number) is float
            and math.isfinite(number)
            and 0 <= number <= upper_bound
        ):
            compact[key] = number
    for key in ("genres", "artists", "tags"):
        entries = value.get(key)
        if entries is None:
            entries = metadata.get(key)
        if isinstance(entries, list):
            selected = [text for entry in entries[:8] if (text := _bounded_text(entry, 80))]
            if selected:
                compact[key] = selected
    user_state = value.get("userState")
    if isinstance(user_state, dict):
        state: dict[str, bool | int | float | str] = {}
        for key in ("favorite", "played"):
            if type(user_state.get(key)) is bool:
                state[key] = user_state[key]
        play_count = user_state.get("playCount")
        if type(play_count) is int and 0 <= play_count <= 1_000_000_000:
            state["playCount"] = play_count
        for key in ("positionSeconds", "durationSeconds"):
            seconds = user_state.get(key)
            if (
                type(seconds) in {int, float}
                and math.isfinite(seconds)
                and 0 <= seconds <= 10_000_000
            ):
                state[key] = seconds
        last_played = _bounded_text(user_state.get("lastPlayedAt"), 40)
        if last_played:
            state["lastPlayedAt"] = last_played
        if state:
            compact["userState"] = state
    return compact


def _entity_references(items: list[dict[str, Any]]) -> tuple[EntityReference, ...]:
    references: list[EntityReference] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (item["type"], item["id"])
        if key in seen:
            continue
        seen.add(key)
        references.append(
            EntityReference(type=item["type"], id=item["id"], title=item["title"])
        )
    return tuple(references)


def _count(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 1_000_000_000:
        return value
    return 0


def _bounded_json(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    for key in ("items", "seasons", "matches"):
        values = value.get(key)
        while len(encoded) > MAX_TOOL_RESULT_CHARS and isinstance(values, list) and values:
            values.pop()
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(encoded) > MAX_TOOL_RESULT_CHARS:
        value.pop("backgroundItem", None)
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(encoded) > MAX_TOOL_RESULT_CHARS:
        return '{"notice":"Local catalog result was too large to include."}'
    return encoded


def _items_result(payload: dict[str, Any], *, include_total: bool = False) -> ToolResult:
    values = payload.get("items")
    items = (
        [
            item
            for value in values[:MAX_TOOL_ITEMS]
            if (item := _compact_item(value)) is not None
        ]
        if isinstance(values, list)
        else []
    )
    normalized: dict[str, Any] = {"items": items}
    if include_total:
        normalized["total"] = _count(payload.get("total"))
    return ToolResult(
        content=_bounded_json(normalized),
        trust=EvidenceTrust.LOCAL,
        entities=_entity_references(items),
    )


def _detail_result(payload: dict[str, Any]) -> ToolResult:
    item = _compact_item(payload.get("item"))
    background = _compact_item(payload.get("backgroundItem"))
    raw_seasons = payload.get("seasons")
    seasons = (
        [
            season
            for value in raw_seasons[:12]
            if (season := _compact_item(value)) is not None
        ]
        if isinstance(raw_seasons, list)
        else []
    )
    items = ([item] if item else []) + ([background] if background else []) + seasons
    normalized: dict[str, Any] = {"item": item, "seasons": seasons}
    if background:
        normalized["backgroundItem"] = background
    if not item:
        normalized["notice"] = "No accessible catalog item was returned."
    return ToolResult(
        content=_bounded_json(normalized),
        trust=EvidenceTrust.LOCAL,
        entities=_entity_references(items),
    )


class _ReadTool:
    definition: ToolDefinition

    def __init__(self, client: OrchestratorReadOnlyClient) -> None:
        self._client = client

    @staticmethod
    def _failed(error: _OrchestratorReadError) -> ToolResult:
        return ToolResult(str(error), EvidenceTrust.LOCAL)


class CatalogSearchTool(_ReadTool):
    definition = ToolDefinition(
        name="zenstream_catalog_search",
        description="Search the authenticated user's accessible ZenStream catalog.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 120},
                "type": {
                    "type": ["string", "null"],
                    "enum": [*sorted(_SEARCH_TYPES), None],
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                "language": {"type": "string", "maxLength": 16},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        data_scope="local",
        read_only=True,
    )

    def validate_arguments(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        _keys(arguments, {"query", "type", "limit", "language"}, {"query"})
        query = arguments["query"]
        if not isinstance(query, str) or not query.strip() or len(query) > 120:
            raise ValueError("Search query must contain 1 to 120 characters")
        result: dict[str, Any] = {"query": query.strip()}
        if "type" in arguments:
            value = arguments["type"]
            if value is not None and (
                not isinstance(value, str) or value not in _SEARCH_TYPES
            ):
                raise ValueError("Unsupported catalog type")
            result["type"] = value
        if "limit" in arguments:
            value = arguments["limit"]
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 10:
                raise ValueError("Search limit must be from 1 to 10")
            result["limit"] = value
        if "language" in arguments:
            value = arguments["language"]
            if not isinstance(value, str) or not _LANGUAGE_PATTERN.fullmatch(value):
                raise ValueError("Language code is invalid")
            result["language"] = value
        return result

    async def execute(self, context: ChatContext, arguments: Mapping[str, Any]) -> ToolResult:
        try:
            payload = await self._client.search(
                context,
                query=arguments["query"],
                item_type=arguments.get("type"),
                limit=arguments.get("limit", 8),
                language=arguments.get("language"),
            )
        except _OrchestratorReadError as error:
            return self._failed(error)
        return _items_result(payload, include_total=True)


class CatalogEntityResolveTool(_ReadTool):
    definition = ToolDefinition(
        name="zenstream_catalog_resolve",
        description=(
            "Resolve externally researched movie or series candidates against the authenticated "
            "local catalog. Supply English, Japanese, Chinese, or other known title variants, "
            "release year, type, and provider IDs when available. Only returned local matches "
            "are available to recommend by default."
        ),
        parameters={
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "titles": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 3,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {
                                            "type": "string",
                                            "minLength": 1,
                                            "maxLength": 120,
                                        },
                                        "language": {"type": "string", "maxLength": 16},
                                    },
                                    "required": ["title"],
                                    "additionalProperties": False,
                                },
                            },
                            "type": {"type": "string", "enum": ["movie", "series"]},
                            "year": {"type": "integer", "minimum": 1888, "maximum": 2200},
                            "providerIds": {
                                "type": "object",
                                "maxProperties": 8,
                                "additionalProperties": {"type": "string", "maxLength": 96},
                            },
                        },
                        "required": ["titles", "type"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["candidates"],
            "additionalProperties": False,
        },
        data_scope="local",
        read_only=True,
    )

    def validate_arguments(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        _keys(arguments, {"candidates"}, {"candidates"})
        raw_candidates = arguments["candidates"]
        if not isinstance(raw_candidates, list) or not 1 <= len(raw_candidates) <= 3:
            raise ValueError("Supply one to three media candidates")
        candidates: list[dict[str, Any]] = []
        for raw in raw_candidates:
            if not isinstance(raw, Mapping):
                raise ValueError("Each candidate must be an object")
            _keys(raw, {"titles", "type", "year", "providerIds"}, {"titles", "type"})
            item_type = raw["type"]
            if item_type not in {"movie", "series"}:
                raise ValueError("Candidate type must be movie or series")
            raw_titles = raw["titles"]
            if not isinstance(raw_titles, list) or not 1 <= len(raw_titles) <= 3:
                raise ValueError("Each candidate must include one to three title variants")
            titles: list[dict[str, str]] = []
            seen_titles: set[tuple[str, str]] = set()
            for raw_title in raw_titles:
                if not isinstance(raw_title, Mapping):
                    raise ValueError("Each title variant must be an object")
                _keys(raw_title, {"title", "language"}, {"title"})
                title = _bounded_text(raw_title["title"], 120)
                if title is None:
                    raise ValueError("Candidate titles must contain 1 to 120 characters")
                language = raw_title.get("language")
                if language is not None and (
                    not isinstance(language, str)
                    or not _LANGUAGE_PATTERN.fullmatch(language)
                ):
                    raise ValueError("Title language code is invalid")
                key = (_normalize_identity_text(title), (language or "").casefold())
                if key not in seen_titles:
                    seen_titles.add(key)
                    titles.append({"title": title, **({"language": language} if language else {})})
            if not titles:
                raise ValueError("Each candidate must include a usable title")
            year = raw.get("year")
            if year is not None and (type(year) is not int or not 1888 <= year <= 2200):
                raise ValueError("Candidate year is invalid")
            raw_provider_ids = raw.get("providerIds")
            if raw_provider_ids is not None and not isinstance(raw_provider_ids, Mapping):
                raise ValueError("Candidate provider IDs are invalid")
            provider_ids = _provider_ids(raw_provider_ids)
            if isinstance(raw_provider_ids, Mapping) and raw_provider_ids and not provider_ids:
                raise ValueError("Candidate provider IDs are invalid")
            candidate: dict[str, Any] = {
                "titles": titles,
                "type": item_type,
                "providerIds": provider_ids,
            }
            if year is not None:
                candidate["year"] = year
            candidates.append(candidate)
        return {"candidates": candidates}

    async def execute(self, context: ChatContext, arguments: Mapping[str, Any]) -> ToolResult:
        import asyncio

        arguments = self.validate_arguments(arguments)
        candidates = arguments["candidates"]
        semaphore = asyncio.Semaphore(3)

        async def search_variant(
            candidate: Mapping[str, Any], title: Mapping[str, str]
        ) -> list[dict[str, Any]]:
            async with semaphore:
                payload = await self._client.search(
                    context,
                    query=title["title"],
                    item_type=candidate["type"],
                    limit=10,
                    language=title.get("language"),
                )
            values = payload.get("items")
            return (
                [
                    item
                    for value in values[:10]
                    if (item := _compact_item(value)) is not None
                ]
                if isinstance(values, list)
                else []
            )

        try:
            results = await asyncio.gather(
                *(
                    search_variant(candidate, title)
                    for candidate in candidates
                    for title in candidate["titles"]
                )
            )
        except _OrchestratorReadError as error:
            return self._failed(error)

        matches: list[dict[str, Any]] = []
        local_entities: dict[tuple[str, str], EntityReference] = {}
        result_index = 0
        for candidate in candidates:
            local_items: dict[tuple[str, str], dict[str, Any]] = {}
            for _title in candidate["titles"]:
                for item in results[result_index]:
                    local_items[(item["type"], item["id"])] = item
                result_index += 1
            ranked = [
                (score, item, evidence)
                for item in local_items.values()
                if (ranked_match := _match_catalog_candidate(candidate, item)) is not None
                for score, evidence in (ranked_match,)
            ]
            ranked.sort(key=lambda row: row[0], reverse=True)
            if not ranked:
                matches.append({"status": "no_match"})
                continue
            highest = ranked[0][0]
            best = [row for row in ranked if row[0] == highest]
            if len(best) != 1:
                matches.append(
                    {
                        "status": "ambiguous",
                        "items": [item for _score, item, _evidence in best[:3]],
                    }
                )
                continue
            _score, item, evidence = best[0]
            matches.append({"status": "matched", "item": item, "matchedBy": list(evidence)})
            reference = EntityReference(type=item["type"], id=item["id"], title=item["title"])
            local_entities[(reference.type, reference.id)] = reference

        payload = {"matches": matches}
        content = _bounded_json(payload)
        return ToolResult(
            content=content,
            trust=EvidenceTrust.LOCAL,
            entities=tuple(local_entities.values()),
        )


class CatalogItemDetailTool(_ReadTool):
    definition = ToolDefinition(
        name="zenstream_catalog_item_detail",
        description="Read one accessible ZenStream catalog item by its exact local ID.",
        parameters={
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "minLength": 1, "maxLength": 128},
                "language": {"type": "string", "maxLength": 16},
            },
            "required": ["entity_id"],
            "additionalProperties": False,
        },
        data_scope="local",
        read_only=True,
    )

    def validate_arguments(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        _keys(arguments, {"entity_id", "language"}, {"entity_id"})
        entity_id = _valid_entity_id(arguments["entity_id"])
        if entity_id is None:
            raise ValueError("Catalog item ID is invalid")
        result: dict[str, Any] = {"entity_id": entity_id}
        if "language" in arguments:
            language = arguments["language"]
            if not isinstance(language, str) or not _LANGUAGE_PATTERN.fullmatch(language):
                raise ValueError("Language code is invalid")
            result["language"] = language
        return result

    async def execute(self, context: ChatContext, arguments: Mapping[str, Any]) -> ToolResult:
        try:
            payload = await self._client.item_detail(
                context,
                entity_id=arguments["entity_id"],
                language=arguments.get("language"),
            )
        except _OrchestratorReadError as error:
            return self._failed(error)
        return _detail_result(payload)


class _HomeReadTool(_ReadTool):
    def validate_arguments(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        _keys(arguments, set(), set())
        return {}

    async def _load(self, context: ChatContext) -> dict[str, Any]:
        raise NotImplementedError

    async def execute(self, context: ChatContext, arguments: Mapping[str, Any]) -> ToolResult:
        self.validate_arguments(arguments)
        try:
            payload = await self._load(context)
        except _OrchestratorReadError as error:
            return self._failed(error)
        return _items_result(payload)


class HomeRecommendationsTool(_HomeReadTool):
    definition = ToolDefinition(
        name="zenstream_home_recommendations",
        description="Read the user's local, permission-filtered Home recommendations.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        data_scope="local",
        read_only=True,
    )

    async def _load(self, context: ChatContext) -> dict[str, Any]:
        return await self._client.home_recommendations(context)


class ContinueWatchingTool(_HomeReadTool):
    definition = ToolDefinition(
        name="zenstream_continue_watching",
        description="Read the user's local Continue Watching row.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        data_scope="local",
        read_only=True,
    )

    async def _load(self, context: ChatContext) -> dict[str, Any]:
        return await self._client.continue_watching(context)


class NextUpTool(_HomeReadTool):
    definition = ToolDefinition(
        name="zenstream_next_up",
        description="Read the user's local Next Up episode row.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        data_scope="local",
        read_only=True,
    )

    async def _load(self, context: ChatContext) -> dict[str, Any]:
        return await self._client.next_up(context)


class FavoritesTool(_ReadTool):
    definition = ToolDefinition(
        name="zenstream_favorites",
        description="Read the user's accessible local favorites.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        data_scope="local",
        read_only=True,
    )

    def validate_arguments(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        _keys(arguments, set(), set())
        return {}

    async def execute(self, context: ChatContext, arguments: Mapping[str, Any]) -> ToolResult:
        self.validate_arguments(arguments)
        try:
            payload = await self._client.favorites(context)
        except _OrchestratorReadError as error:
            return self._failed(error)
        return _items_result(payload, include_total=True)
