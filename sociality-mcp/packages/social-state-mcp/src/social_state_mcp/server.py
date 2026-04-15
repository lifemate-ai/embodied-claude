"""FastMCP server for present-moment social state inference."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from mcp.server.fastmcp import FastMCP

from .inference import (
    should_interrupt_result,
    turn_taking_state,
)
from .inference import (
    summarize_social_context as build_social_context_summary,
)
from .schemas import (
    ShouldInterruptResult,
    SocialContextSummary,
    SocialStateResult,
    TurnTakingState,
)
from .store import SocialStateStore

mcp = FastMCP("social-state-mcp")


@lru_cache(maxsize=1)
def _store() -> SocialStateStore:
    return SocialStateStore()


@mcp.tool()
def ingest_social_event(event: dict[str, Any]) -> dict[str, str]:
    """Validate and append a social event into the shared store."""

    return _store().ingest_social_event(event)


@mcp.tool()
def get_social_state(
    window_seconds: int = 900,
    person_id: str | None = None,
    include_evidence: bool = True,
) -> dict[str, Any]:
    """Infer compact recent social state from the append-only event stream."""

    state: SocialStateResult = _store().get_social_state(
        window_seconds=window_seconds,
        person_id=person_id,
        include_evidence=include_evidence,
    )
    return state.model_dump(mode="json")


@mcp.tool()
def should_interrupt(
    candidate_action: str,
    urgency: str = "low",
    person_id: str | None = None,
    message_preview: str = "",
) -> dict[str, Any]:
    """Decide whether the candidate interruption is socially appropriate."""

    state = _store().get_social_state(
        window_seconds=900, person_id=person_id, include_evidence=True
    )
    result: ShouldInterruptResult = should_interrupt_result(
        state,
        candidate_action=candidate_action,
        urgency=urgency,
        message_preview=message_preview,
    )
    return result.model_dump(mode="json")


@mcp.tool()
def get_turn_taking_state(person_id: str | None = None) -> dict[str, Any]:
    """Infer whether the current conversational turn belongs to the model or the human."""

    reference_ts = _store().events.get_latest_timestamp(person_id=person_id)
    events = _store().events.fetch_events(person_id=person_id, limit=100)
    state: TurnTakingState = turn_taking_state(events, reference_ts=reference_ts)
    return state.model_dump(mode="json")


@mcp.tool()
def summarize_social_context(person_id: str | None = None, max_chars: int = 180) -> dict[str, Any]:
    """Return a compact summary for prompt injection."""

    state = _store().get_social_state(
        window_seconds=900, person_id=person_id, include_evidence=False
    )
    summary: SocialContextSummary = build_social_context_summary(state, max_chars=max_chars)
    return summary.model_dump(mode="json")


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
