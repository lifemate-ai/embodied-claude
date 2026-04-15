"""FastMCP server for compact self narrative tools."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from mcp.server.fastmcp import FastMCP

from .store import SelfNarrativeStore

mcp = FastMCP("self-narrative-mcp")


@lru_cache(maxsize=1)
def _store() -> SelfNarrativeStore:
    return SelfNarrativeStore()


@mcp.tool()
def append_daybook(day: str | None = None) -> dict[str, Any]:
    """Create or refresh a compact daybook entry from the shared event store."""

    return _store().append_daybook(day=day).model_dump(mode="json")


@mcp.tool()
def get_self_summary() -> dict[str, Any]:
    """Return a compact self summary for prompt injection."""

    return _store().get_self_summary().model_dump(mode="json")


@mcp.tool()
def list_active_arcs() -> list[dict[str, Any]]:
    """List currently active narrative arcs."""

    return [arc.model_dump(mode="json") for arc in _store().list_active_arcs()]


@mcp.tool()
def reflect_on_change(horizon_days: int = 7) -> dict[str, Any]:
    """Summarize change across a recent horizon."""

    return _store().reflect_on_change(horizon_days=horizon_days).model_dump(mode="json")


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
