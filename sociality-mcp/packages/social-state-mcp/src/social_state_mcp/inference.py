"""Deterministic social state inference rules."""

from __future__ import annotations

from social_core import (
    DEFAULT_POLICY_TIMEZONE,
    clamp01,
    confidence_from_evidence,
    ensure_iso8601,
    in_quiet_hours,
    parse_timestamp,
)
from social_core.models import SocialEvent

from .schemas import (
    AffectGuess,
    RecommendedMove,
    ShouldInterruptResult,
    SocialContextSummary,
    SocialStateResult,
    TurnTakingState,
)

DEFAULT_QUIET_HOURS = ("00:00-07:00",)
QUIET_KEYWORDS = ("静か", "quiet", "集中", "focus", "not now", "later")
WORK_KEYWORDS = ("work", "working", "desk", "meeting", "会議", "作業", "仕事", "集中")
REST_KEYWORDS = ("rest", "resting", "sofa", "寝", "sleep", "眠")
EAT_KEYWORDS = ("eat", "eating", "lunch", "dinner", "breakfast", "ご飯", "食べ")
CHAT_KEYWORDS = ("chat", "talk", "雑談", "話そ")
TIRED_KEYWORDS = ("tired", "疲れ", "眠い", "drained", "low energy")
STRESS_KEYWORDS = ("stressed", "stress", "しんど", "大変", "overwhelmed")
QUESTION_MARKERS = ("?", "？")


def get_social_state_result(
    events: list[SocialEvent],
    *,
    person_id: str | None = None,
    include_evidence: bool = True,
    reference_ts: str | None = None,
    quiet_hours_windows: list[str] | None = None,
    policy_timezone: str = DEFAULT_POLICY_TIMEZONE,
) -> SocialStateResult:
    """Infer a compact social state from recent events."""

    ordered = sorted(events, key=lambda item: (item.ts, item.event_seq or 0), reverse=True)
    timestamp = ensure_iso8601(
        reference_ts or (ordered[0].ts if ordered else ensure_iso8601("2026-01-01T12:00:00+00:00"))
    )
    ref_dt = parse_timestamp(timestamp)
    evidence: list[str] = []

    latest_scene = _latest(ordered, "scene_parse")
    latest_human = _latest(ordered, "human_utterance")
    latest_agent = _latest(ordered, "agent_utterance")
    latest_health = _latest(ordered, "health_summary")
    recent_question = _recent_human_question(ordered, ref_dt)
    quiet_request = _recent_quiet_request(ordered, ref_dt)
    recent_nudges = _count_recent_nudges(ordered, ref_dt)
    windows = list(quiet_hours_windows) if quiet_hours_windows is not None else list(
        DEFAULT_QUIET_HOURS
    )
    quiet_active, _quiet_until = in_quiet_hours(timestamp, windows, policy_timezone)
    quiet_hours = bool(ordered or reference_ts) and quiet_active

    presence = "absent"
    if latest_human and _age_seconds(latest_human, ref_dt) <= 120:
        presence = "speaking"
        evidence.append("recent human utterance")
    elif latest_scene and _age_seconds(latest_scene, ref_dt) <= 900:
        presence = "present"
        evidence.append("recent scene parse shows presence")
    elif ordered:
        presence = "possible"
        evidence.append("recent social events exist")

    activity = _infer_activity(latest_scene, latest_human, ref_dt, quiet_hours, evidence)
    energy = _infer_energy(latest_health, latest_human, evidence)
    affect_guess = _infer_affect(latest_health, latest_human, evidence)

    interaction_phase = "idle"
    if recent_question:
        interaction_phase = "awaiting_reply"
        evidence.append("recent direct human question")
    elif quiet_request:
        interaction_phase = "quiet_focus"
        evidence.append("recent quiet request")
    elif (
        latest_agent
        and latest_human
        and _age_seconds(latest_agent, ref_dt) <= 300
        and _age_seconds(latest_human, ref_dt) <= 300
    ):
        interaction_phase = "ongoing"
    elif recent_nudges >= 2:
        interaction_phase = "cooling_down"
        evidence.append("multiple recent nudges")

    interrupt_cost = _interrupt_cost(
        presence=presence,
        activity=activity,
        energy=energy,
        quiet_request=quiet_request,
        quiet_hours=quiet_hours,
        recent_nudges=recent_nudges,
        awaiting_reply=bool(recent_question),
    )
    availability = _availability_from_cost(interrupt_cost)
    if recent_question:
        availability = "interruptible"
    elif quiet_hours and not recent_question:
        availability = "do_not_interrupt"
        if "local time is within quiet hours" not in evidence:
            evidence.append("local time is within quiet hours")

    recommended_moves = _recommended_moves(availability, interaction_phase, energy)
    summary = _build_summary(
        presence=presence,
        activity=activity,
        availability=availability,
        energy=energy,
        affect=affect_guess.label,
    )
    return SocialStateResult(
        timestamp=timestamp,
        person_id=person_id,
        presence=presence,
        activity=activity,
        availability=availability,
        interaction_phase=interaction_phase,
        energy=energy,
        affect_guess=affect_guess,
        interrupt_cost=interrupt_cost,
        recommended_moves=recommended_moves,
        summary_for_prompt=summary,
        evidence=evidence if include_evidence else [],
    )


def should_interrupt_result(
    state: SocialStateResult,
    *,
    candidate_action: str,
    urgency: str,
    message_preview: str = "",
) -> ShouldInterruptResult:
    """Decide whether an interruption is worth the social cost."""

    urgency_weight = {
        "low": 0.2,
        "medium": 0.5,
        "high": 0.8,
        "critical": 1.0,
    }.get(urgency, 0.2)
    deny = state.interaction_phase != "awaiting_reply" and urgency_weight < state.interrupt_cost
    if candidate_action == "say" and state.availability == "do_not_interrupt" and urgency != "high":
        deny = True
    confidence = confidence_from_evidence([abs(state.interrupt_cost - urgency_weight), 0.55], 0.05)
    reason = (
        "Current context suggests focused work and a recent request for quiet."
        if deny and state.interaction_phase == "quiet_focus"
        else "Recent context leaves room for a response."
    )
    if state.interaction_phase == "awaiting_reply":
        reason = "A recent human question makes a reply socially appropriate."
    if message_preview and deny and "?" in message_preview:
        reason = "The message looks optional while current context signals low interruptibility."
    cooldown = 0 if not deny else max(300, int(state.interrupt_cost * 1800))
    return ShouldInterruptResult(
        decision="no" if deny else "yes",
        confidence=confidence,
        reason=reason,
        cooldown_seconds=cooldown,
    )


def turn_taking_state(
    events: list[SocialEvent], *, reference_ts: str | None = None
) -> TurnTakingState:
    """Infer whether the model should respond or hold."""

    ordered = sorted(events, key=lambda item: (item.ts, item.event_seq or 0), reverse=True)
    timestamp = parse_timestamp(
        reference_ts or (ordered[0].ts if ordered else ensure_iso8601("2026-01-01T12:00:00+00:00"))
    )
    latest_human = _latest(ordered, "human_utterance")
    latest_agent = _latest(ordered, "agent_utterance")
    if (
        latest_human
        and _looks_like_question(latest_human.payload_json.get("text", ""))
        and _age_seconds(latest_human, timestamp) <= 600
    ):
        return TurnTakingState(
            state="respond",
            confidence=0.84,
            reason="Recent human question still looks like an invitation for a reply.",
        )
    if latest_agent and (latest_human is None or latest_agent.ts >= latest_human.ts):
        return TurnTakingState(
            state="hold",
            confidence=0.78,
            reason="The agent spoke most recently, so holding gives the human room to answer.",
        )
    return TurnTakingState(
        state="hold",
        confidence=0.72,
        reason="No direct address; recent pause is compatible with thinking, not invitation.",
    )


def summarize_social_context(state: SocialStateResult, max_chars: int) -> SocialContextSummary:
    """Return a prompt-friendly summary clipped to a character budget."""

    summary = state.summary_for_prompt
    if len(summary) > max_chars:
        summary = summary[: max_chars - 1].rstrip() + "…"
    return SocialContextSummary(summary=summary)


def _latest(events: list[SocialEvent], kind: str) -> SocialEvent | None:
    return next((event for event in events if event.kind == kind), None)


def _age_seconds(event: SocialEvent, reference_dt) -> float:
    return max(0.0, (reference_dt - parse_timestamp(event.ts)).total_seconds())


def _recent_human_question(events: list[SocialEvent], reference_dt) -> SocialEvent | None:
    for event in events:
        if event.kind != "human_utterance":
            continue
        if _age_seconds(event, reference_dt) > 1200:
            continue
        if _looks_like_question(event.payload_json.get("text", "")) or event.payload_json.get(
            "expects_reply"
        ):
            return event
    return None


def _recent_quiet_request(events: list[SocialEvent], reference_dt) -> SocialEvent | None:
    for event in events:
        if event.kind != "human_utterance":
            continue
        if _age_seconds(event, reference_dt) > 14_400:
            continue
        text = _payload_text(event).lower()
        if any(keyword in text for keyword in QUIET_KEYWORDS):
            return event
    return None


def _count_recent_nudges(events: list[SocialEvent], reference_dt) -> int:
    count = 0
    for event in events:
        if _age_seconds(event, reference_dt) > 3600:
            continue
        if event.kind not in {"touchpoint", "agent_utterance"}:
            continue
        action = str(event.payload_json.get("action", "")).lower()
        style = str(event.payload_json.get("style", "")).lower()
        if "nudge" in action or "reminder" in action or "nudge" in style:
            count += 1
    return count


def _infer_activity(
    latest_scene: SocialEvent | None,
    latest_human: SocialEvent | None,
    reference_dt,
    quiet_hours: bool,
    evidence: list[str],
) -> str:
    if latest_scene:
        payload = latest_scene.payload_json
        explicit = str(payload.get("activity", "")).strip().lower()
        if explicit in {"working", "commuting", "eating", "resting", "sleeping", "chatting"}:
            evidence.append(f"scene parse suggests {explicit}")
            return explicit
        scene_text = _payload_text(latest_scene).lower()
        for label, keywords in {
            "working": WORK_KEYWORDS,
            "resting": REST_KEYWORDS,
            "eating": EAT_KEYWORDS,
            "chatting": CHAT_KEYWORDS,
        }.items():
            if any(keyword in scene_text for keyword in keywords):
                evidence.append(f"scene summary suggests {label}")
                return label
    if latest_human:
        text = _payload_text(latest_human).lower()
        if any(keyword in text for keyword in WORK_KEYWORDS):
            evidence.append("recent utterance suggests work context")
            return "working"
        if any(keyword in text for keyword in REST_KEYWORDS):
            return "resting"
        if any(keyword in text for keyword in EAT_KEYWORDS):
            return "eating"
    if quiet_hours:
        return "sleeping"
    return "unknown"


def _infer_energy(
    latest_health: SocialEvent | None, latest_human: SocialEvent | None, evidence: list[str]
) -> str:
    if latest_health:
        payload = latest_health.payload_json
        body_battery = payload.get("body_battery")
        if isinstance(body_battery, (int, float)):
            if body_battery < 35:
                evidence.append("body battery low")
                return "low"
            if body_battery < 70:
                return "medium"
            return "high"
        explicit = str(payload.get("energy", "")).lower()
        if explicit in {"high", "medium", "low"}:
            return explicit
    if latest_human and any(
        keyword in _payload_text(latest_human).lower() for keyword in TIRED_KEYWORDS
    ):
        evidence.append("recent utterance suggests tiredness")
        return "low"
    return "unknown"


def _infer_affect(
    latest_health: SocialEvent | None, latest_human: SocialEvent | None, evidence: list[str]
) -> AffectGuess:
    weights: list[float] = []
    label = "uncertain"
    if latest_health and isinstance(latest_health.payload_json.get("body_battery"), (int, float)):
        body_battery = float(latest_health.payload_json["body_battery"])
        if body_battery < 35:
            label = "tired"
            weights.append(0.65)
    if latest_human:
        text = _payload_text(latest_human).lower()
        if any(keyword in text for keyword in TIRED_KEYWORDS):
            label = "tired"
            weights.append(0.7)
        elif any(keyword in text for keyword in STRESS_KEYWORDS):
            label = "stressed"
            weights.append(0.62)
    if not weights:
        return AffectGuess(label="uncertain", confidence=0.18)
    evidence.append(f"affect guess leans {label}")
    return AffectGuess(label=label, confidence=confidence_from_evidence(weights, 0.2))


def _interrupt_cost(
    *,
    presence: str,
    activity: str,
    energy: str,
    quiet_request: SocialEvent | None,
    quiet_hours: bool,
    recent_nudges: int,
    awaiting_reply: bool,
) -> float:
    cost = {
        "absent": 0.55,
        "possible": 0.45,
        "present": 0.38,
        "speaking": 0.42,
    }[presence]
    if activity == "working":
        cost += 0.18
    if activity == "sleeping":
        cost += 0.32
    if energy == "low":
        cost += 0.15
    if quiet_request:
        cost += 0.22
    if quiet_hours:
        cost += 0.2
    cost += min(0.2, recent_nudges * 0.07)
    if awaiting_reply:
        cost -= 0.42
    return clamp01(cost)


def _availability_from_cost(interrupt_cost: float) -> str:
    if interrupt_cost >= 0.72:
        return "do_not_interrupt"
    if interrupt_cost >= 0.38:
        return "maybe_interruptible"
    return "interruptible"


def _recommended_moves(
    availability: str, interaction_phase: str, energy: str
) -> list[RecommendedMove]:
    if interaction_phase == "awaiting_reply":
        return [
            RecommendedMove(action="answer_briefly", confidence=0.86, style="clear"),
            RecommendedMove(action="ask_clarifying_question", confidence=0.42, style="brief"),
        ]
    if availability == "do_not_interrupt":
        return [
            RecommendedMove(action="stay_silent", confidence=0.84),
            RecommendedMove(action="check_later", confidence=0.46, style="brief_gentle"),
        ]
    if energy == "low":
        return [
            RecommendedMove(action="nudge_human", confidence=0.34, style="brief_gentle"),
            RecommendedMove(action="stay_nearby", confidence=0.51),
        ]
    return [
        RecommendedMove(action="stay_available", confidence=0.61),
        RecommendedMove(action="nudge_human", confidence=0.28, style="brief_gentle"),
    ]


def _build_summary(
    *, presence: str, activity: str, availability: str, energy: str, affect: str
) -> str:
    affect_text = "with uncertain affect" if affect == "uncertain" else f"somewhat {affect}"
    return (
        f"The person seems {presence}, {activity}, {affect_text}, and "
        f"{availability.replace('_', ' ')} right now."
    )


def _payload_text(event: SocialEvent) -> str:
    payload = event.payload_json
    for key in ("text", "scene_summary", "summary"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""


def _looks_like_question(text: str) -> bool:
    clean = text.strip().lower()
    if not clean:
        return False
    if clean.endswith(QUESTION_MARKERS):
        return True
    return any(token in clean for token in ("can you", "would you", "どう", "かな", "?", "？"))
