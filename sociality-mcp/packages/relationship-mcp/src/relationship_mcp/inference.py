"""Deterministic relationship summary helpers."""

from __future__ import annotations

import os

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


# The follow-up is handed to the model as a ready-made line, so once TTS is
# involved its phrasing is spoken in the agent's own voice. It used to be one
# fixed dialect for every deployment (#173, the gap #146 closed for
# system-temperature-mcp). RELATIONSHIP_TONE picks the phrasing and the
# default keeps the original lines. Whatever the tone, each suggestion also
# carries its intent and topic, so a caller can phrase it in its own words.
DEFAULT_TONE = "kansai"

STRESS_CHECK_IN = "check_in_after_stress"
EVENING_CHECK_IN = "evening_check_in"
CONTINUE_OPEN_THREAD = "continue_open_thread"

FOLLOWUP_PHRASES: dict[str, dict[str, str]] = {
    "kansai": {
        STRESS_CHECK_IN: "{topic}って言うてたけど、そのあと少しは落ち着いた？",
        EVENING_CHECK_IN: "今日はだいぶ詰まってそうやったけど、少しは一息つけた？",
        CONTINUE_OPEN_THREAD: "いま気になってること、続きある？",
    },
    "neutral": {
        STRESS_CHECK_IN: "{topic}と言っていたけれど、そのあと少しは落ち着いた？",
        EVENING_CHECK_IN: "今日はだいぶ忙しそうだったけれど、少しは一息つけた？",
        CONTINUE_OPEN_THREAD: "いま気になっていること、続きはある？",
    },
}

FOLLOWUP_REASONS = {
    STRESS_CHECK_IN: "References a same-day stress disclosure without overreaching.",
    EVENING_CHECK_IN: "Uses the active context without inventing details.",
    CONTINUE_OPEN_THREAD: "Keeps continuity while staying generic.",
}


def _tone() -> str:
    """The phrasing tone, read per call; unknown values fall back to neutral."""

    raw = os.environ.get("RELATIONSHIP_TONE", "").strip().lower()
    if not raw:
        return DEFAULT_TONE
    return raw if raw in FOLLOWUP_PHRASES else "neutral"


def suggest_followup_text(
    context: str, latest_stress_text: str | None
) -> tuple[str, str, str, str | None]:
    """Suggest a contextual follow-up without dumping transcripts.

    Returns ``(text, reason, intent, topic)``. Only ``text`` depends on
    RELATIONSHIP_TONE; the rest describe the same suggestion without a voice.
    """

    topic: str | None = None
    if latest_stress_text:
        topic = latest_stress_text.strip("。.!?？ ")[:18]
        intent = STRESS_CHECK_IN
    elif context == "evening_checkin":
        intent = EVENING_CHECK_IN
    else:
        intent = CONTINUE_OPEN_THREAD
    text = FOLLOWUP_PHRASES[_tone()][intent].format(topic=topic)
    return text, FOLLOWUP_REASONS[intent], intent, topic
