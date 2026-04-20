"""compose_interaction_context — assemble a prompt-ready interaction frame."""

from __future__ import annotations

from typing import Any

from boundary_mcp.store import BoundaryStore
from joint_attention_mcp.store import JointAttentionStore
from relationship_mcp.store import RelationshipStore
from self_narrative_mcp.store import SelfNarrativeStore
from social_core import DEFAULT_POLICY_TIMEZONE, local_view, utc_now
from social_state_mcp.store import SocialStateStore

from .desire_source import load_desire_snapshot
from .schemas import (
    AgentStateSummary,
    CommitmentSummary,
    ComposeInteractionContextInput,
    FollowupSuggestion,
    InteractionContext,
    OpenLoopSummary,
    ResponseContract,
)
from .store import InteractionOrchestratorStore


def compose_interaction_context(
    payload: ComposeInteractionContextInput,
    *,
    social_state_store: SocialStateStore,
    relationship_store: RelationshipStore,
    joint_attention_store: JointAttentionStore,
    boundary_store: BoundaryStore,
    self_narrative_store: SelfNarrativeStore,
    orchestrator_store: InteractionOrchestratorStore,
    policy_timezone: str = DEFAULT_POLICY_TIMEZONE,
) -> InteractionContext:
    """Gather everything the next move needs into a single snapshot."""

    ts = utc_now()
    local = local_view(ts, policy_timezone)
    local_time_text = local.isoformat(timespec="seconds")

    social_state = _call(
        social_state_store.get_social_state,
        window_seconds=900,
        person_id=payload.person_id,
        include_evidence=payload.include_private,
    )
    turn_taking_data = _infer_turn_taking(social_state_store, payload.person_id)

    person_model = None
    person_name = None
    open_loops: list[OpenLoopSummary] = []
    commitments: list[CommitmentSummary] = []
    followups: list[FollowupSuggestion] = []
    if payload.person_id:
        person_model = _optional_dict(
            relationship_store, "get_person_model", person_id=payload.person_id
        )
        if person_model:
            person_name = person_model.get("canonical_name") or person_model.get("person_id")
        open_loops = [
            OpenLoopSummary(
                loop_id=item.get("loop_id", ""),
                topic=item.get("topic", ""),
                status=item.get("status", ""),
                updated_at=item.get("updated_at"),
            )
            for item in _optional_list(
                relationship_store, "list_open_loops", person_id=payload.person_id
            )
        ]
        commitments = [
            CommitmentSummary(
                commitment_id=item.get("commitment_id", ""),
                text=item.get("text", ""),
                due_at=item.get("due_at"),
                status=item.get("status", ""),
            )
            for item in (person_model or {}).get("commitments", [])
        ]
        followups = _collect_followups(relationship_store, payload.person_id, payload.user_text)

    self_summary_dict = _optional_dict(self_narrative_store, "get_self_summary")
    active_arcs = _optional_list(self_narrative_store, "list_active_arcs")
    latest_daybook = (self_summary_dict or {}).get("latest_daybook") if self_summary_dict else None

    desire_data = load_desire_snapshot() or {}
    dominant_desire = desire_data.get("dominant") if desire_data else None
    desires = desire_data.get("desires") or {}
    discomforts = desire_data.get("discomforts") or {}

    recent_experiences = orchestrator_store.recent_agent_experiences(
        person_id=payload.person_id, limit=5, include_private=payload.include_private
    )

    joint_focus = _optional_dict(
        joint_attention_store,
        "get_current_joint_focus",
        person_id=payload.person_id,
    )

    boundary_hints = _derive_boundary_hints(boundary_store, social_state, ts)

    quiet_mode = _call(
        boundary_store.get_quiet_mode_state, ts=ts
    )
    quiet_active = bool(getattr(quiet_mode, "active", False)) if quiet_mode else False

    response_contract = _pick_contract(
        person_id=payload.person_id,
        channel=payload.channel,
        quiet_active=quiet_active,
    )

    agent_state = AgentStateSummary(
        ts=ts,
        desires={str(k): float(v) for k, v in desires.items()},
        discomforts={str(k): float(v) for k, v in discomforts.items()},
        dominant_desire=dominant_desire,
        recent_experiences=recent_experiences,
        active_arcs=list(active_arcs),
        private_reflections=orchestrator_store.count_private_reflections(
            person_id=payload.person_id
        ),
        interpretation_shifts=orchestrator_store.count_interpretation_shifts(),
    )

    prompt_summary = _build_prompt_summary(
        local_time=local_time_text,
        person_id=payload.person_id,
        person_name=person_name,
        social_state=_to_plain(social_state),
        agent_state=agent_state,
        quiet_active=quiet_active,
        dominant_desire=dominant_desire,
        desires=desires,
        open_loops=open_loops,
    )

    compact_prompt_block = _compact_block(
        prompt_summary=prompt_summary,
        response_contract=response_contract,
        max_chars=payload.max_chars,
    )

    return InteractionContext(
        ts=ts,
        local_time=local_time_text,
        timezone=policy_timezone,
        person_id=payload.person_id,
        person_name=person_name,
        social_state=_to_plain(social_state) or {},
        turn_taking=turn_taking_data,
        boundary_hints=boundary_hints,
        person_model=person_model,
        open_loops=open_loops,
        commitments_due=commitments,
        suggested_followups=followups,
        self_summary=self_summary_dict,
        active_arcs=list(active_arcs),
        latest_daybook=latest_daybook,
        desire_state=desire_data or None,
        agent_state=agent_state,
        relevant_memories=[],
        recent_experiences=recent_experiences,
        joint_focus=joint_focus,
        current_scene_summary=(
            (joint_focus or {}).get("current_scene_summary") if joint_focus else None
        ),
        response_contract=response_contract,
        prompt_summary=prompt_summary,
        compact_prompt_block=compact_prompt_block,
    )


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------


def _call(fn, **kwargs):
    try:
        return fn(**kwargs) if kwargs else fn()
    except Exception:
        return None


def _optional_dict(store: Any, method: str, **kwargs) -> dict[str, Any] | None:
    fn = getattr(store, method, None)
    if fn is None:
        return None
    try:
        result = fn(**kwargs) if kwargs else fn()
    except Exception:
        return None
    return _to_plain(result)


def _optional_list(store: Any, method: str, **kwargs) -> list[dict[str, Any]]:
    fn = getattr(store, method, None)
    if fn is None:
        return []
    try:
        result = fn(**kwargs) if kwargs else fn()
    except Exception:
        return []
    if result is None:
        return []
    plain = _to_plain(result)
    if isinstance(plain, list):
        return [item if isinstance(item, dict) else {"value": item} for item in plain]
    if isinstance(plain, dict):
        # Some stores return {"items": [...]} shapes.
        items = plain.get("items") or plain.get("open_loops") or plain.get("results")
        if isinstance(items, list):
            return [item if isinstance(item, dict) else {"value": item} for item in items]
    return []


def _to_plain(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    return value


def _infer_turn_taking(
    social_state_store: SocialStateStore, person_id: str | None
) -> dict[str, Any]:
    """Call get_turn_taking_state if the store exposes it, else return idle."""

    fn = getattr(social_state_store, "get_turn_taking_state", None)
    if fn is not None:
        try:
            result = fn(person_id=person_id) if person_id else fn()
        except TypeError:
            try:
                result = fn()
            except Exception:
                return {"state": "hold", "confidence": 0.5}
        except Exception:
            return {"state": "hold", "confidence": 0.5}
        return _to_plain(result) or {"state": "hold", "confidence": 0.5}
    return {"state": "hold", "confidence": 0.5}


def _collect_followups(
    relationship_store: RelationshipStore, person_id: str, user_text: str | None
) -> list[FollowupSuggestion]:
    fn = getattr(relationship_store, "suggest_followup", None)
    if fn is None:
        return []
    try:
        raw = fn(person_id=person_id, context=user_text)
    except TypeError:
        try:
            raw = fn(person_id=person_id)
        except Exception:
            return []
    except Exception:
        return []
    plain = _to_plain(raw)
    items: list[dict[str, Any]] = []
    if isinstance(plain, list):
        items = [item for item in plain if isinstance(item, dict)]
    elif isinstance(plain, dict):
        seq = plain.get("suggestions") or plain.get("items") or []
        if isinstance(seq, list):
            items = [item for item in seq if isinstance(item, dict)]
    return [
        FollowupSuggestion(
            text=str(item.get("text") or item.get("suggestion") or ""),
            reason=item.get("reason"),
        )
        for item in items
        if item.get("text") or item.get("suggestion")
    ]


def _derive_boundary_hints(
    boundary_store: BoundaryStore, social_state: Any, ts: str
) -> list[str]:
    hints: list[str] = []
    plain = _to_plain(social_state) or {}
    availability = plain.get("availability")
    if availability == "do_not_interrupt":
        hints.append("current context signals do_not_interrupt")
    if availability == "maybe_interruptible":
        hints.append("availability is ambivalent; prefer bounded replies")
    if plain.get("interaction_phase") == "quiet_focus":
        hints.append("recent quiet request — avoid nudges")
    quiet_mode = _call(boundary_store.get_quiet_mode_state, ts=ts)
    if quiet_mode and getattr(quiet_mode, "active", False):
        hints.append("policy quiet hours are active — prefer private/non-voice actions")
    return hints


def _pick_contract(
    *, person_id: str | None, channel: str, quiet_active: bool
) -> ResponseContract:
    base = ResponseContract()
    if person_id == "kouta":
        base = ResponseContract(
            treat_user_as="high-context technical partner",
            avoid=[
                "generic reassurance",
                "beginner tutorial unless explicitly requested",
                "pretending to remember unsupported facts",
                "over-explaining already-established premises",
            ],
            prefer=[
                "direct technical framing",
                "relationship-aware specificity",
                "one clear implementation path",
                "explicit uncertainty when memory or evidence is weak",
            ],
            initiative_policy="bounded",
            max_clarifying_questions=1,
        )
    if channel == "autonomous":
        base = base.model_copy(
            update={
                "initiative_policy": "bounded",
                "avoid": list(base.avoid)
                + ["public posting without review", "loud voice during quiet hours"],
                "max_clarifying_questions": 0,
            }
        )
    if quiet_active:
        base = base.model_copy(
            update={
                "prefer": list(base.prefer)
                + [
                    "private written reflection over spoken output",
                    "actions the human can read later, not interrupt on",
                ],
            }
        )
    return base


def _build_prompt_summary(
    *,
    local_time: str,
    person_id: str | None,
    person_name: str | None,
    social_state: dict[str, Any] | None,
    agent_state: AgentStateSummary,
    quiet_active: bool,
    dominant_desire: str | None,
    desires: dict[str, Any],
    open_loops: list[OpenLoopSummary],
) -> str:
    social = social_state or {}
    availability = social.get("availability", "unknown")
    activity = social.get("activity", "unknown")
    phase = social.get("interaction_phase", "idle")
    who = person_name or person_id or "the current person"
    desire_text = ""
    if dominant_desire:
        lvl = desires.get(dominant_desire)
        if isinstance(lvl, (int, float)):
            desire_text = f"Dominant pull: {dominant_desire} ({lvl:.2f})."
        else:
            desire_text = f"Dominant pull: {dominant_desire}."
    open_loop_text = (
        f"Open loops: {', '.join(loop.topic for loop in open_loops[:3])}."
        if open_loops
        else "No tracked open loops."
    )
    quiet_text = (
        "Policy quiet hours are ACTIVE — prefer private, non-voice, non-interruptive moves."
        if quiet_active
        else "Policy quiet hours are NOT active."
    )
    return (
        f"Now {local_time} • {who} seems {availability}, {activity}, phase={phase}. "
        f"{desire_text} {open_loop_text} {quiet_text} "
        f"Recent agent experiences: {len(agent_state.recent_experiences)}; "
        f"interpretation_shifts so far: {agent_state.interpretation_shifts}."
    ).strip()


def _compact_block(
    *, prompt_summary: str, response_contract: ResponseContract, max_chars: int
) -> str:
    contract_lines = [f"treat_user_as: {response_contract.treat_user_as}"]
    if response_contract.avoid:
        contract_lines.append("avoid: " + "; ".join(response_contract.avoid[:4]))
    if response_contract.prefer:
        contract_lines.append("prefer: " + "; ".join(response_contract.prefer[:4]))
    contract_lines.append(
        f"initiative: {response_contract.initiative_policy}, "
        f"max_clarifying={response_contract.max_clarifying_questions}"
    )
    block = "\n".join(
        [
            "[interaction_context]",
            prompt_summary,
            "[response_contract]",
            *contract_lines,
        ]
    )
    if len(block) > max_chars:
        block = block[: max_chars - 1].rstrip() + "…"
    return block
