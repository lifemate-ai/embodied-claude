"""Deterministic relationship summary helpers."""

from __future__ import annotations

from social_core import clamp01

STRESS_KEYWORDS = ("疲れ", "tired", "stress", "しんど", "overwhelmed", "会議多")
WARMTH_KEYWORDS = ("ありがとう", "thanks", "助か", "嬉し", "good to see")
FUTURE_MARKERS = (
    "明日",
    "tomorrow",
    "later",
    "after",
    "remind",
    "dentist",
    "review",
    "会議",
    "meeting",
)


def compute_snapshot_metrics(
    *,
    interaction_count: int,
    human_messages: list[str],
    agent_messages: list[str],
) -> dict[str, float]:
    """Compute bounded heuristic relationship metrics."""

    stress_hits = sum(
        1 for text in human_messages if any(keyword in text.lower() for keyword in STRESS_KEYWORDS)
    )
    warmth_hits = sum(
        1
        for text in human_messages + agent_messages
        if any(keyword in text.lower() for keyword in WARMTH_KEYWORDS)
    )
    reciprocity = 0.5
    total = len(human_messages) + len(agent_messages)
    if total:
        reciprocity = clamp01(0.5 + (len(human_messages) - len(agent_messages)) / (2 * total))
    return {
        "warmth": clamp01(0.35 + warmth_hits * 0.15),
        "trust": clamp01(0.4 + min(interaction_count, 12) * 0.04),
        "fragility": clamp01(0.15 + stress_hits * 0.18),
        "expected_response_latency": clamp01(0.25 + stress_hits * 0.12),
        "recent_stress": clamp01(0.2 + stress_hits * 0.22),
        "reciprocity_balance": reciprocity,
    }


def summarize_relationship(*, role: str | None, recent_stress: float, open_loop_count: int) -> str:
    """Build a compact relationship summary."""

    role_text = role or "person"
    continuity = "high continuity expectations" if open_loop_count else "light ongoing continuity"
    stress_text = (
        "recent stress is noticeable" if recent_stress >= 0.5 else "recent stress seems manageable"
    )
    return f"{role_text.title()} relationship with {continuity}; {stress_text}."


def suggest_followup_text(context: str, latest_stress_text: str | None) -> tuple[str, str]:
    """Suggest a contextual follow-up without dumping transcripts."""

    if latest_stress_text:
        topic = latest_stress_text.strip("。.!?？ ")[:18]
        return (
            f"{topic}って言うてたけど、そのあと少しは落ち着いた？",
            "References a same-day stress disclosure without overreaching.",
        )
    if context == "evening_checkin":
        return (
            "今日はだいぶ詰まってそうやったけど、少しは一息つけた？",
            "Uses the active context without inventing details.",
        )
    return (
        "いま気になってること、続きある？",
        "Keeps continuity while staying generic.",
    )
