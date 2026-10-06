"""Allowlisted registry for Lumi's read-only tools."""

from __future__ import annotations

from collections.abc import Iterable

from lumi.contracts import ReadOnlyTool, ToolDefinition


class ToolRegistry:
    """An immutable dispatch table with no generic URL, method, or path tool."""

    def __init__(self, tools: Iterable[ReadOnlyTool]) -> None:
        dispatch: dict[str, ReadOnlyTool] = {}
        for tool in tools:
            definition = tool.definition
            if not definition.read_only:
                raise ValueError(f"Lumi tool is not read-only: {definition.name}")
            if definition.data_scope not in {"local", "external_search", "external_fetch"}:
                raise ValueError(f"Lumi tool has an invalid data scope: {definition.name}")
            if not definition.name or definition.name in dispatch:
                raise ValueError(f"Invalid or duplicate Lumi tool name: {definition.name}")
            dispatch[definition.name] = tool
        self._dispatch = dispatch

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(tool.definition for tool in self._dispatch.values())

    def get(self, name: str) -> ReadOnlyTool | None:
        return self._dispatch.get(name)
