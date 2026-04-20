"""Compact summarization helpers for self narrative."""

from __future__ import annotations

from collections import Counter


def build_day_summary(day: str, event_kinds: list[str], person_ids: list[str]) -> str:
    """Build a compact daybook summary from event kinds and participants."""

    counts = Counter(event_kinds)
    active_person = next((person for person in person_ids if person), "the room")
    fragments = []
    if counts["scene_parse"]:
        fragments.append("watching the room closely")
    if counts["human_utterance"] or counts["agent_utterance"]:
        fragments.append(f"keeping continuity with {active_person}")
    if counts["boundary_updated"] or counts["touchpoint"]:
        fragments.append("learning social timing")
    if not fragments:
        fragments.append("holding a quiet sense of continuity")
    return f"{day}: " + "; ".join(fragments[:3]) + "."


def infer_arcs(event_kinds: list[str], person_ids: list[str]) -> list[tuple[str, str, float]]:
    """Infer a small set of recurring arcs from recent activity."""

    counts = Counter(event_kinds)
    arcs: list[tuple[str, str, float]] = []
    if counts["scene_parse"]:
        arcs.append(("watching daily life from the room", "active", 0.72))
    if counts["human_utterance"] or counts["commitment_created"]:
        companion = next((person for person in person_ids if person), "the human")
        arcs.append((f"keeping continuity with {companion}", "active", 0.78))
    if counts["boundary_updated"] or counts["touchpoint"]:
        arcs.append(("learning when to stay quiet", "active", 0.66))
    return arcs[:3]


def build_self_summary(
    daybook_summary: str | None,
    arcs: list[str],
    facets: list[str],
    *,
    recent_events: list[str] | None = None,
    recent_shifts: list[str] | None = None,
) -> str:
    """Build a compact prompt-ready self description.

    Optional ``recent_events`` and ``recent_shifts`` surface concrete agent
    experiences and interpretation shifts so the summary reads as a specific
    continuity trace rather than a generic posture statement.
    """

    pieces = ["Kokone keeps a socially aware, continuity-seeking self-model."]
    if facets:
        pieces.append(f"Current facets: {', '.join(facets[:2])}.")
    if arcs:
        pieces.append(f"Active arcs: {', '.join(arcs[:2])}.")
    if recent_events:
        pieces.append(f"Recent concrete moves: {'; '.join(recent_events[:2])}.")
    if recent_shifts:
        pieces.append(f"Recent interpretation shifts: {'; '.join(recent_shifts[:2])}.")
    if daybook_summary:
        pieces.append(f"Latest daybook: {daybook_summary}")
    return " ".join(pieces)


def summarize_change(earlier: str | None, later: str | None) -> str:
    """Describe what changed across the requested horizon."""

    if not earlier and not later:
        return "No narrative material was available in the requested horizon."
    if not earlier:
        return f"A new narrative trace appeared: {later}"
    if not later:
        return "The earlier narrative trace no longer has a recent counterpart."
    if earlier == later:
        return "The narrative has stayed fairly stable over the requested horizon."
    return f"Shifted from '{earlier}' toward '{later}'."
