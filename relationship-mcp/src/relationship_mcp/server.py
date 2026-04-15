"""FastMCP server for relationship abstractions."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from mcp.server.fastmcp import FastMCP

from .store import RelationshipStore

mcp = FastMCP("relationship-mcp")


@lru_cache(maxsize=1)
def _store() -> RelationshipStore:
    return RelationshipStore()


@mcp.tool()
def upsert_person(
    person_id: str,
    canonical_name: str,
    aliases: list[str] | None = None,
    role: str | None = None,
) -> dict[str, str]:
    """Create or update a compact person record."""

    return _store().upsert_person(
        person_id=person_id,
        canonical_name=canonical_name,
        aliases=aliases,
        role=role,
    )


@mcp.tool()
def ingest_interaction(
    person_id: str,
    channel: str,
    direction: str,
    text: str,
    ts: str,
) -> dict[str, str]:
    """Append a relationship-relevant interaction and update open-loop heuristics."""

    return _store().ingest_interaction(
        person_id=person_id,
        channel=channel,
        direction=direction,
        text=text,
        ts=ts,
    )


@mcp.tool()
def get_person_model(person_id: str) -> dict[str, Any]:
    """Return a compact relationship abstraction for one person."""

    return _store().get_person_model(person_id=person_id).model_dump(mode="json")


@mcp.tool()
def create_commitment(
    person_id: str,
    text: str,
    due_at: str | None = None,
    source: str = "conversation",
) -> dict[str, str]:
    """Create a reminder or promise that should persist across restarts."""

    return _store().create_commitment(person_id=person_id, text=text, due_at=due_at, source=source)


@mcp.tool()
def complete_commitment(commitment_id: str) -> dict[str, str]:
    """Mark a commitment complete."""

    return _store().complete_commitment(commitment_id)


@mcp.tool()
def list_open_loops(person_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """List currently open loops for a person."""

    return [
        loop.model_dump(mode="json")
        for loop in _store().list_open_loops(person_id=person_id, limit=limit)
    ]


@mcp.tool()
def suggest_followup(person_id: str, context: str) -> dict[str, Any]:
    """Suggest a context-aware follow-up."""

    suggestions = _store().suggest_followup(person_id=person_id, context=context)
    return {"suggestions": [item.model_dump(mode="json") for item in suggestions]}


@mcp.tool()
def record_boundary(person_id: str, kind: str, rule: str, source_text: str) -> dict[str, str]:
    """Record a person-specific communication boundary."""

    return _store().record_boundary(
        person_id=person_id, kind=kind, rule=rule, source_text=source_text
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
