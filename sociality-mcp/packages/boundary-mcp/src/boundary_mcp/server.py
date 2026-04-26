"""FastMCP server for social boundary evaluation."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from mcp.server.fastmcp import FastMCP

from .anomaly_detection import analyze as _analyze_text_anomaly
from .store import BoundaryStore

mcp = FastMCP("boundary-mcp")


@lru_cache(maxsize=1)
def _store() -> BoundaryStore:
    return BoundaryStore()


@mcp.tool()
def evaluate_action(
    action_type: str,
    channel: str | None = None,
    person_id: str | None = None,
    context: dict[str, Any] | None = None,
    payload_preview: dict[str, Any] | None = None,
    urgency: str = "low",
) -> dict[str, Any]:
    """Evaluate whether a proposed action is socially acceptable."""

    return (
        _store()
        .evaluate_action(
            action_type=action_type,
            channel=channel,
            person_id=person_id,
            context=context,
            payload_preview=payload_preview,
            urgency=urgency,
        )
        .model_dump(mode="json")
    )


@mcp.tool()
def review_social_post(
    channel: str,
    text: str,
    scene_contains_face: bool = False,
    person_mentions: list[str] | None = None,
) -> dict[str, Any]:
    """Review a post draft for privacy and tact risk."""

    return (
        _store()
        .review_social_post(
            channel=channel,
            text=text,
            scene_contains_face=scene_contains_face,
            person_mentions=person_mentions,
        )
        .model_dump(mode="json")
    )


@mcp.tool()
def record_consent(person_id: str, consent_type: str, value: bool, source: str) -> dict[str, str]:
    """Record consent or refusal for a boundary-sensitive action."""

    return _store().record_consent(
        person_id=person_id, consent_type=consent_type, value=value, source=source
    )


@mcp.tool()
def get_quiet_mode_state(ts: str) -> dict[str, Any]:
    """Return whether quiet mode is active at the supplied timestamp."""

    return _store().get_quiet_mode_state(ts=ts).model_dump(mode="json")


@mcp.tool()
def analyze_text_anomaly(text: str) -> dict[str, Any]:
    """Score a passage for off-baseline conversational language.

    Encodes the input with the E5 sentence-embedding model that
    memory-mcp uses, compares it to two reference banks (baseline and
    aggressive-style) by cosine similarity, and returns:

    - baseline_similarity / aggressive_similarity (max sim per bank)
    - overall_anomaly_score in [0, 1]
    - interpretation: "low" / "medium" / "high"
    - reference_baseline_count / reference_aggressive_count

    One input among many for the agent's final judgement; false
    positives are explicitly acceptable. Not an autosilencer.
    """

    return _analyze_text_anomaly(text).to_dict()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
