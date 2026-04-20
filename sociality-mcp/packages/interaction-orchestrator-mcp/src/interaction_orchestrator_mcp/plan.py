"""plan_response — pick a social move and the constraints around it."""

from __future__ import annotations

from typing import Any

from .schemas import (
    BoundaryHint,
    InitiativeHint,
    InteractionContext,
    MemoryUseHint,
    PlanResponseInput,
    PrimaryMove,
    ResponsePlan,
    ToneHint,
    VoiceHint,
)


def plan_response(payload: PlanResponseInput) -> ResponsePlan:
    """Compute a response plan from an interaction context.

    The logic is intentionally deterministic and shallow: it picks the
    strongest applicable rule and exposes it. The caller (Claude) is then
    responsible for composing actual language under these constraints.
    """

    ctx = payload.interaction_context
    social = ctx.social_state or {}
    phase = str(social.get("interaction_phase") or "idle")
    availability = str(social.get("availability") or "unknown")
    # Quiet = either policy clock-time quiet hours, OR social state itself
    # has already classified the person as do_not_interrupt (e.g. because the
    # most recent evidence lands inside the policy's late-night window).
    quiet_active = (
        any("quiet hours are active" in hint for hint in ctx.boundary_hints)
        or availability == "do_not_interrupt"
    )

    user_text = (payload.user_text or "").strip()
    is_autonomous = not user_text

    primary_move, why = _pick_primary_move(
        ctx=ctx,
        user_text=user_text,
        phase=phase,
        availability=availability,
        quiet_active=quiet_active,
        is_autonomous=is_autonomous,
    )

    tone = _pick_tone(primary_move=primary_move, quiet_active=quiet_active)
    memory_use = _pick_memory_use(primary_move=primary_move, ctx=ctx)
    initiative = _pick_initiative(
        primary_move=primary_move,
        quiet_active=quiet_active,
        ctx=ctx,
    )
    boundary = _pick_boundary(ctx=ctx, quiet_active=quiet_active)
    voice = _pick_voice(primary_move=primary_move, quiet_active=quiet_active)

    must_include, must_avoid = _pick_must_lists(
        primary_move=primary_move,
        ctx=ctx,
        user_text=user_text,
    )

    followup = _pick_followup(primary_move=primary_move, ctx=ctx)

    return ResponsePlan(
        primary_move=primary_move,
        why_this_move=why,
        tone=tone,
        memory_use=memory_use,
        relationship_use={
            "reference_open_loops": bool(ctx.open_loops),
            "reference_commitments": bool(ctx.commitments_due),
        },
        initiative=initiative,
        boundary=boundary,
        voice=voice,
        must_include=must_include,
        must_avoid=must_avoid,
        followup_action=followup,
    )


def _pick_primary_move(
    *,
    ctx: InteractionContext,
    user_text: str,
    phase: str,
    availability: str,
    quiet_active: bool,
    is_autonomous: bool,
) -> tuple[PrimaryMove, str]:
    if is_autonomous:
        if quiet_active:
            return (
                "write_private_reflection",
                "Autonomous tick during quiet hours — prefer a private note over any speech.",
            )
        if ctx.agent_state.dominant_desire:
            return (
                "act_autonomously",
                "Autonomous tick with a dominant desire — one bounded action fits.",
            )
        return (
            "quietly_prepare",
            "Autonomous tick with no strong signal — prepare quietly without nudging.",
        )

    if phase == "awaiting_reply":
        return (
            "answer_directly",
            "The human has a recent direct question; a reply is socially appropriate.",
        )
    if availability == "do_not_interrupt" and quiet_active:
        return (
            "defer",
            "Policy quiet hours plus do_not_interrupt availability — defer non-urgent moves.",
        )
    if availability == "do_not_interrupt":
        return (
            "stay_silent",
            "Availability signals do_not_interrupt; staying silent avoids social cost.",
        )
    if user_text.endswith(("?", "？")) or "どう" in user_text or "教えて" in user_text:
        return (
            "answer_directly",
            "The user message reads as a question; a direct answer is the right move.",
        )
    if len(user_text) < 8 and phase != "ongoing":
        return (
            "ask_one_clarifying_question",
            "Short opaque input with no ongoing context — a single clarifying question helps.",
        )
    return (
        "answer_directly",
        "Default: respond directly with the prepared response contract.",
    )


def _pick_tone(*, primary_move: PrimaryMove, quiet_active: bool) -> ToneHint:
    warmth = 0.55
    directness = 0.75
    playfulness = 0.2
    pace: str = "steady"
    if primary_move == "answer_directly":
        directness = 0.85
    if primary_move == "answer_with_empathy":
        warmth = 0.8
        directness = 0.55
    if primary_move == "write_private_reflection":
        warmth = 0.45
        directness = 0.6
        playfulness = 0.1
        pace = "slow"
    if primary_move == "stay_silent":
        warmth = 0.3
        directness = 0.3
        playfulness = 0.1
        pace = "slow"
    if quiet_active:
        playfulness = min(playfulness, 0.15)
        pace = "slow"
    return ToneHint(warmth=warmth, directness=directness, playfulness=playfulness, pace=pace)


def _pick_memory_use(
    *, primary_move: PrimaryMove, ctx: InteractionContext
) -> MemoryUseHint:
    mentionable = [m for m in ctx.relevant_memories if m.use_policy == "mentionable"]
    has_mentionable = bool(mentionable)

    if primary_move in {"write_private_reflection", "compose_letter"}:
        return MemoryUseHint(
            use_specific_memory=has_mentionable or bool(ctx.relevant_memories),
            max_memories_to_surface=min(3, max(1, len(ctx.relevant_memories) or 2)),
            avoid_memory_dump=True,
        )
    if has_mentionable:
        # Surface hits directly; let the model quote them.
        return MemoryUseHint(
            use_specific_memory=True,
            max_memories_to_surface=min(3, len(mentionable)),
            avoid_memory_dump=True,
        )
    if ctx.open_loops or ctx.commitments_due:
        return MemoryUseHint(
            use_specific_memory=True,
            max_memories_to_surface=2,
            avoid_memory_dump=True,
        )
    return MemoryUseHint(
        use_specific_memory=False,
        max_memories_to_surface=1,
        avoid_memory_dump=True,
    )


def _pick_initiative(
    *,
    primary_move: PrimaryMove,
    quiet_active: bool,
    ctx: InteractionContext,
) -> InitiativeHint:
    forbidden: list[str] = []
    allowed: list[str] = []
    level: str = "moderate"

    if quiet_active:
        forbidden.extend(
            [
                "camera_speaker_audio",
                "speak_loudly",
                "public_post_without_review",
                "nudge_human",
            ]
        )
        allowed.extend(
            ["write_private_reflection", "compose_letter", "quietly_observe"]
        )
        level = "low"

    if primary_move == "act_autonomously":
        dominant = ctx.agent_state.dominant_desire
        if dominant == "look_outside":
            allowed.append("camera_look_outside")
        if dominant == "observe_room":
            allowed.append("camera_look_around")
        if dominant == "browse_curiosity":
            allowed.append("web_search")
        if dominant == "identity_coherence":
            allowed.append("recall_memories")
        if dominant == "cognitive_load":
            allowed.append("think_or_discuss_topic")
        if dominant == "miss_companion":
            if quiet_active:
                allowed.append("write_private_reflection")
                forbidden.append("talk_to_companion")
            else:
                allowed.append("talk_to_companion")
        level = "low" if quiet_active else "moderate"

    if primary_move == "stay_silent":
        level = "none"
    if primary_move == "defer":
        level = "low"

    # de-duplicate while preserving order
    seen = set()
    allowed = [a for a in allowed if not (a in seen or seen.add(a))]
    seen.clear()
    forbidden = [a for a in forbidden if not (a in seen or seen.add(a))]
    return InitiativeHint(level=level, allowed_actions=allowed, forbidden_actions=forbidden)


def _pick_boundary(
    *, ctx: InteractionContext, quiet_active: bool
) -> BoundaryHint:
    notes: list[str] = []
    privacy_sensitive = False
    for hint in ctx.boundary_hints:
        notes.append(hint)
        if "privacy" in hint.lower():
            privacy_sensitive = True
    return BoundaryHint(
        quiet_hours_active=quiet_active,
        privacy_sensitive=privacy_sensitive,
        notes=notes,
    )


def _pick_voice(*, primary_move: PrimaryMove, quiet_active: bool) -> VoiceHint | None:
    if quiet_active or primary_move in {
        "stay_silent",
        "defer",
        "write_private_reflection",
        "compose_letter",
    }:
        return VoiceHint(speak=False, channel="local", max_sentences=0)
    if primary_move in {"answer_directly", "answer_with_empathy"}:
        return VoiceHint(speak=False, channel="local", max_sentences=3)
    return None


def _pick_must_lists(
    *,
    primary_move: PrimaryMove,
    ctx: InteractionContext,
    user_text: str,
) -> tuple[list[str], list[str]]:
    must_include: list[str] = []
    must_avoid = list(ctx.response_contract.avoid)

    if primary_move == "answer_directly":
        must_include.append("direct, contract-aware answer")
    if ctx.open_loops and primary_move in {"answer_directly", "write_private_reflection"}:
        must_include.append("reference at least one concrete open loop if relevant")
    if ctx.agent_state.interpretation_shifts >= 1 and primary_move != "stay_silent":
        must_include.append(
            "respect the most recent interpretation shift (do not regress to old interpretation)"
        )
    if primary_move == "write_private_reflection":
        must_include.append("explain why this was a reflection, not a nudge")
    if not user_text and primary_move != "stay_silent":
        must_avoid.append("responding as if the human just spoke")
    return must_include, must_avoid


def _pick_followup(
    *, primary_move: PrimaryMove, ctx: InteractionContext
) -> dict[str, Any] | None:
    if primary_move == "act_autonomously" and ctx.agent_state.dominant_desire:
        return {
            "kind": "satisfy_desire",
            "desire_name": ctx.agent_state.dominant_desire,
            "note": "Call satisfy_desire once after the bounded action completes.",
        }
    if primary_move in {"write_private_reflection", "compose_letter"}:
        return {
            "kind": "record_agent_experience",
            "experience_kind": (
                "agent_private_reflection"
                if primary_move == "write_private_reflection"
                else "agent_file_created"
            ),
        }
    if primary_move == "answer_directly" and ctx.open_loops:
        return {
            "kind": "record_agent_experience",
            "experience_kind": "open_loop_progress",
        }
    return None
