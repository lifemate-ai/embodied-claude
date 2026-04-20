"""FastMCP server that exposes the full sociality tool surface through one process."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from boundary_mcp.policy import load_policy
from boundary_mcp.store import BoundaryStore
from interaction_orchestrator_mcp.compose import compose_interaction_context
from interaction_orchestrator_mcp.plan import plan_response
from interaction_orchestrator_mcp.schemas import (
    AppendPrivateReflectionInput,
    ComposeInteractionContextInput,
    ComposePrivateLetterInput,
    PlanResponseInput,
    RecordAgentExperienceInput,
    RecordInterpretationShiftInput,
)
from interaction_orchestrator_mcp.store import InteractionOrchestratorStore
from joint_attention_mcp.store import JointAttentionStore
from mcp.server.fastmcp import FastMCP
from relationship_mcp.store import RelationshipStore
from self_narrative_mcp.store import SelfNarrativeStore
from social_core import SocialDB
from social_state_mcp.inference import should_interrupt_result, turn_taking_state
from social_state_mcp.inference import (
    summarize_social_context as build_social_context_summary,
)
from social_state_mcp.store import SocialStateStore

mcp = FastMCP("sociality-mcp")


@dataclass(slots=True)
class SocialityStores:
    """Shared store bundle backed by a single social DB connection."""

    db: SocialDB
    social_state: SocialStateStore
    relationship: RelationshipStore
    joint_attention: JointAttentionStore
    boundary: BoundaryStore
    self_narrative: SelfNarrativeStore
    orchestrator: InteractionOrchestratorStore
    policy_timezone: str


@lru_cache(maxsize=1)
def _stores() -> SocialityStores:
    db = SocialDB()
    policy = load_policy()
    return SocialityStores(
        db=db,
        social_state=SocialStateStore(
            db=db,
            quiet_hours_windows=list(policy.global_policy.quiet_hours),
            policy_timezone=policy.global_policy.timezone,
        ),
        relationship=RelationshipStore(db=db),
        joint_attention=JointAttentionStore(db=db),
        boundary=BoundaryStore(db=db),
        self_narrative=SelfNarrativeStore(db=db),
        orchestrator=InteractionOrchestratorStore(db=db),
        policy_timezone=policy.global_policy.timezone,
    )


def reset_store_cache() -> None:
    """Clear cached stores so tests or env changes get a fresh shared DB."""

    if _stores.cache_info().currsize:
        _stores().db.close()
        _stores.cache_clear()


@mcp.tool()
def ingest_social_event(event: dict[str, Any]) -> dict[str, str]:
    """Validate and append a social event into the shared store."""

    return _stores().social_state.ingest_social_event(event)


@mcp.tool()
def get_social_state(
    window_seconds: int = 900,
    person_id: str | None = None,
    include_evidence: bool = True,
) -> dict[str, Any]:
    """Infer compact recent social state from the append-only event stream."""

    return (
        _stores()
        .social_state.get_social_state(
            window_seconds=window_seconds,
            person_id=person_id,
            include_evidence=include_evidence,
        )
        .model_dump(mode="json")
    )


@mcp.tool()
def should_interrupt(
    candidate_action: str,
    urgency: str = "low",
    person_id: str | None = None,
    message_preview: str = "",
) -> dict[str, Any]:
    """Decide whether the candidate interruption is socially appropriate."""

    state = _stores().social_state.get_social_state(
        window_seconds=900,
        person_id=person_id,
        include_evidence=True,
    )
    return (
        should_interrupt_result(
            state,
            candidate_action=candidate_action,
            urgency=urgency,
            message_preview=message_preview,
        )
        .model_dump(mode="json")
    )


@mcp.tool()
def get_turn_taking_state(person_id: str | None = None) -> dict[str, Any]:
    """Infer whether the current conversational turn belongs to the model or the human."""

    reference_ts = _stores().social_state.events.get_latest_timestamp(person_id=person_id)
    events = _stores().social_state.events.fetch_events(person_id=person_id, limit=100)
    return turn_taking_state(events, reference_ts=reference_ts).model_dump(mode="json")


@mcp.tool()
def summarize_social_context(person_id: str | None = None, max_chars: int = 180) -> dict[str, Any]:
    """Return a compact summary for prompt injection."""

    state = _stores().social_state.get_social_state(
        window_seconds=900,
        person_id=person_id,
        include_evidence=False,
    )
    return build_social_context_summary(state, max_chars=max_chars).model_dump(mode="json")


@mcp.tool()
def upsert_person(
    person_id: str,
    canonical_name: str,
    aliases: list[str] | None = None,
    role: str | None = None,
) -> dict[str, str]:
    """Create or update a compact person record."""

    return _stores().relationship.upsert_person(
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

    return _stores().relationship.ingest_interaction(
        person_id=person_id,
        channel=channel,
        direction=direction,
        text=text,
        ts=ts,
    )


@mcp.tool()
def get_person_model(person_id: str) -> dict[str, Any]:
    """Return a compact relationship abstraction for one person."""

    return _stores().relationship.get_person_model(person_id=person_id).model_dump(mode="json")


@mcp.tool()
def create_commitment(
    person_id: str,
    text: str,
    due_at: str | None = None,
    source: str = "conversation",
) -> dict[str, str]:
    """Create a reminder or promise that should persist across restarts."""

    return _stores().relationship.create_commitment(
        person_id=person_id,
        text=text,
        due_at=due_at,
        source=source,
    )


@mcp.tool()
def complete_commitment(commitment_id: str) -> dict[str, str]:
    """Mark a commitment complete."""

    return _stores().relationship.complete_commitment(commitment_id)


@mcp.tool()
def list_open_loops(person_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """List currently open loops for a person."""

    return [
        loop.model_dump(mode="json")
        for loop in _stores().relationship.list_open_loops(person_id=person_id, limit=limit)
    ]


@mcp.tool()
def suggest_followup(person_id: str, context: str) -> dict[str, Any]:
    """Suggest a context-aware follow-up."""

    suggestions = _stores().relationship.suggest_followup(person_id=person_id, context=context)
    return {"suggestions": [item.model_dump(mode="json") for item in suggestions]}


@mcp.tool()
def record_boundary(person_id: str, kind: str, rule: str, source_text: str) -> dict[str, str]:
    """Record a person-specific communication boundary."""

    return _stores().relationship.record_boundary(
        person_id=person_id,
        kind=kind,
        rule=rule,
        source_text=source_text,
    )


@mcp.tool()
def ingest_scene_parse(scene: dict[str, Any]) -> dict[str, str]:
    """Store a structured scene parse from an adapter or orchestrator."""

    return _stores().joint_attention.ingest_scene_parse(scene)


@mcp.tool()
def resolve_reference(
    expression: str,
    person_id: str | None = None,
    lookback_frames: int = 5,
) -> dict[str, Any]:
    """Resolve a deictic or descriptive expression against recent scene objects."""

    return (
        _stores()
        .joint_attention.resolve_reference(
            expression=expression,
            person_id=person_id,
            lookback_frames=lookback_frames,
        )
        .model_dump(mode="json")
    )


@mcp.tool()
def get_current_joint_focus(person_id: str | None = None) -> dict[str, Any]:
    """Infer the current joint focus target."""

    return _stores().joint_attention.get_current_joint_focus(person_id=person_id)


@mcp.tool()
def set_joint_focus(person_id: str | None, target_id: str, initiator: str) -> dict[str, str]:
    """Record an explicit joint focus target."""

    return _stores().joint_attention.set_joint_focus(
        person_id=person_id,
        target_id=target_id,
        initiator=initiator,
    )


@mcp.tool()
def compare_recent_scenes(person_id: str | None = None, window_minutes: int = 30) -> dict[str, Any]:
    """Return compact changes across recent scene parses."""

    return _stores().joint_attention.compare_recent_scenes(
        person_id=person_id,
        window_minutes=window_minutes,
    )


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
        _stores()
        .boundary.evaluate_action(
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

    stores = _stores()
    result = stores.boundary.review_social_post(
        channel=channel,
        text=text,
        scene_contains_face=scene_contains_face,
        person_mentions=person_mentions,
    )
    out = result.model_dump(mode="json")

    # Enhanced: check recorded boundaries for mentioned persons
    person_mentions = person_mentions or []
    for pid in person_mentions:
        rows = stores.db.connect().execute(
            "SELECT rule FROM person_boundaries WHERE person_id = ? AND kind = 'privacy'",
            (pid,),
        ).fetchall()
        for row in rows:
            rule_lower = (row[0] if isinstance(row, tuple) else row["rule"]).lower()
            # Location identification patterns
            if "住所" in rule_lower or "特定" in rule_lower:
                import re
                location_signals = re.findall(
                    r'(\d+階|マンション|ベランダ|看板|教育セミナー|郵便局|薬局|橋|タワー)',
                    text,
                )
                if len(location_signals) >= 2:
                    out["risk_level"] = "high"
                    out["issues"] = out.get("issues", []) + [
                        f"location identification risk: {len(location_signals)} signals found ({', '.join(location_signals)})"
                    ]
                    out["recommendation"] = "deny"
            # Window/balcony photo rule
            if ("窓" in rule_lower or "ベランダ" in rule_lower) and ("写真" in rule_lower or "投稿" in rule_lower):
                if "ベランダ" in text or "窓から" in text:
                    if scene_contains_face or "写真" in text or "画像" in text:
                        out["risk_level"] = "high"
                        out["issues"] = out.get("issues", []) + [
                            "window/balcony photo violates recorded boundary"
                        ]
                        out["recommendation"] = "deny"

    return out


@mcp.tool()
def record_consent(person_id: str, consent_type: str, value: bool, source: str) -> dict[str, str]:
    """Record consent or refusal for a boundary-sensitive action."""

    return _stores().boundary.record_consent(
        person_id=person_id,
        consent_type=consent_type,
        value=value,
        source=source,
    )


@mcp.tool()
def get_quiet_mode_state(ts: str) -> dict[str, Any]:
    """Return whether quiet mode is active at the supplied timestamp."""

    return _stores().boundary.get_quiet_mode_state(ts=ts).model_dump(mode="json")


@mcp.tool()
def append_daybook(day: str | None = None) -> dict[str, Any]:
    """Create or refresh a compact daybook entry from the shared event store."""

    return _stores().self_narrative.append_daybook(day=day).model_dump(mode="json")


@mcp.tool()
def get_self_summary() -> dict[str, Any]:
    """Return a compact self summary for prompt injection."""

    return _stores().self_narrative.get_self_summary().model_dump(mode="json")


@mcp.tool()
def list_active_arcs() -> list[dict[str, Any]]:
    """List currently active narrative arcs."""

    return [arc.model_dump(mode="json") for arc in _stores().self_narrative.list_active_arcs()]


@mcp.tool()
def reflect_on_change(horizon_days: int = 7) -> dict[str, Any]:
    """Summarize change across a recent horizon."""

    return _stores().self_narrative.reflect_on_change(horizon_days=horizon_days).model_dump(
        mode="json"
    )


# ---------------------------------------------------------------------------
# Human Response Orchestrator tools (v0.3)
# ---------------------------------------------------------------------------


@mcp.tool()
def compose_interaction_context_tool(
    person_id: str | None = "kouta",
    channel: str = "chat",
    user_text: str | None = None,
    autonomous_trigger: str | None = None,
    include_private: bool = True,
    max_chars: int = 3000,
) -> dict[str, Any]:
    """Assemble a compact, prompt-ready interaction context before responding.

    Call this BEFORE generating a response (or taking an autonomous action).
    The returned object contains social state, turn-taking, relationship model,
    open loops, commitments, self-summary, active arcs, desire state, a
    response contract with dos/don'ts, and a compact_prompt_block ready for
    prompt injection. Use the contract and boundary_hints to shape tone and
    initiative, and pass this object straight into plan_response.
    """

    stores = _stores()
    ctx = compose_interaction_context(
        ComposeInteractionContextInput(
            person_id=person_id,
            channel=channel,
            user_text=user_text,
            autonomous_trigger=autonomous_trigger,
            include_private=include_private,
            max_chars=max_chars,
        ),
        social_state_store=stores.social_state,
        relationship_store=stores.relationship,
        joint_attention_store=stores.joint_attention,
        boundary_store=stores.boundary,
        self_narrative_store=stores.self_narrative,
        orchestrator_store=stores.orchestrator,
        policy_timezone=stores.policy_timezone,
    )
    return ctx.model_dump(mode="json")


@mcp.tool()
def plan_response_tool(
    interaction_context: dict[str, Any],
    user_text: str | None = None,
    candidate_goal: str | None = None,
) -> dict[str, Any]:
    """Pick a bounded social move for the current context.

    Takes the interaction_context produced by compose_interaction_context_tool
    and returns a ResponsePlan: primary_move (answer_directly / stay_silent /
    write_private_reflection / act_autonomously / …), tone, memory_use,
    initiative (allowed_actions, forbidden_actions), boundary, voice hint,
    must_include / must_avoid, and an optional followup_action. Use this to
    shape the actual prose; do NOT let the model override a 'stay_silent'
    plan into speech.
    """

    payload = PlanResponseInput(
        interaction_context=interaction_context,
        user_text=user_text,
        candidate_goal=candidate_goal,
    )
    return plan_response(payload).model_dump(mode="json")


@mcp.tool()
def record_agent_experience(payload: dict[str, Any]) -> dict[str, str]:
    """Persist a thing the agent just did as an experience, not a log line.

    Call this AFTER each significant action — reply, private reflection,
    autonomous move, boundary respect, desire satisfaction, user correction,
    or interpretation shift. Stores the summary, felt state, desires before
    and after, and related event/memory IDs so compose_interaction_context
    sees the agent's own recent history on the next turn.
    """

    stored = _stores().orchestrator.record_agent_experience(
        RecordAgentExperienceInput.model_validate(payload)
    )
    return {"experience_id": stored.experience_id, "ts": stored.ts}


@mcp.tool()
def record_interpretation_shift(payload: dict[str, Any]) -> dict[str, str]:
    """Remember a moment where the agent changed how it interprets a rule.

    Use this when the agent notices it had misread a convention, a policy,
    a relationship signal, or its own self-model — and updates. Stored
    shifts are surfaced as a counter in agent_state and constrain future
    plans via the response contract's 'do not regress' guidance.
    """

    stored = _stores().orchestrator.record_interpretation_shift(
        RecordInterpretationShiftInput.model_validate(payload)
    )
    return {"shift_id": stored.experience_id, "ts": stored.ts}


@mcp.tool()
def append_private_reflection(payload: dict[str, Any]) -> dict[str, str]:
    """Write a private reflection without nudging anyone.

    Private reflections belong to moments that want to be thought through,
    not spoken. They do not create events or contact the human. They can
    later be surfaced in compose_interaction_context via the agent_state.
    """

    stored = _stores().orchestrator.append_private_reflection(
        AppendPrivateReflectionInput.model_validate(payload)
    )
    return {"reflection_id": stored.experience_id, "ts": stored.ts}


@mcp.tool()
def compose_private_letter(payload: dict[str, Any]) -> dict[str, str]:
    """Store a composed letter that may later be shared.

    This tool does NOT write prose — Claude composes the body; the tool
    persists it together with metadata (intended_time, visibility,
    related_open_loops). Useful for the morning-letter / end-of-day pattern.
    """

    stored = _stores().orchestrator.compose_private_letter(
        ComposePrivateLetterInput.model_validate(payload)
    )
    return {"letter_id": stored.experience_id, "ts": stored.ts}


@mcp.tool()
def get_agent_state(person_id: str | None = None) -> dict[str, Any]:
    """Return a short self-state summary: desires, recent experiences, arcs.

    Lightweight alternative to compose_interaction_context when the caller
    only wants the agent's own recent continuity, not a full interaction
    frame. Good for introspection prompts and for autonomous ticks that do
    not yet need social state.
    """

    stores = _stores()
    ctx = compose_interaction_context(
        ComposeInteractionContextInput(
            person_id=person_id, channel="system", user_text=None, include_private=True
        ),
        social_state_store=stores.social_state,
        relationship_store=stores.relationship,
        joint_attention_store=stores.joint_attention,
        boundary_store=stores.boundary,
        self_narrative_store=stores.self_narrative,
        orchestrator_store=stores.orchestrator,
        policy_timezone=stores.policy_timezone,
    )
    return ctx.agent_state.model_dump(mode="json")


async def _handle_http(reader: __import__("asyncio").StreamReader, writer: __import__("asyncio").StreamWriter) -> None:
    """Lightweight HTTP endpoints for hook integration."""
    import asyncio
    import json
    import urllib.parse

    try:
        request_line = await asyncio.wait_for(reader.readline(), timeout=5)
        headers: dict[str, str] = {}
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=5)
            if line in (b"\r\n", b"\n", b""):
                break
            text_line = line.decode("utf-8", errors="replace").rstrip("\r\n")
            if ":" in text_line:
                key, _, value = text_line.partition(":")
                headers[key.strip().lower()] = value.strip()

        req = request_line.decode("utf-8", errors="replace")
        path_str = req.split(" ")[1] if " " in req else "/"
        parsed = urllib.parse.urlparse(path_str)
        params = urllib.parse.parse_qs(parsed.query)

        request_body_bytes = b""
        try:
            content_length = int(headers.get("content-length", "0"))
        except ValueError:
            content_length = 0
        if content_length > 0:
            request_body_bytes = await asyncio.wait_for(
                reader.readexactly(content_length), timeout=5
            )

        def _json_body() -> dict[str, Any]:
            if not request_body_bytes:
                return {}
            try:
                parsed_body = json.loads(request_body_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return {}
            return parsed_body if isinstance(parsed_body, dict) else {}

        body = '{"error":"unknown endpoint"}'
        status = "404 Not Found"

        if "GET /ingest" in req:
            # Ingest a human utterance: /ingest?person_id=kouta&text=hello&kind=human_utterance
            from datetime import datetime, timezone
            person_id = params.get("person_id", [None])[0]
            text = params.get("text", [""])[0]
            kind = params.get("kind", ["human_utterance"])[0]
            source = params.get("source", ["hook"])[0]

            if text:
                stores = _stores()
                from social_core.events import EventStore
                event_store = EventStore(db=stores.db)
                stored = event_store.ingest({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "source": source,
                    "kind": kind,
                    "person_id": person_id,
                    "confidence": 1.0,
                    "payload": {"text": text},
                })
                body = json.dumps({"event_id": stored.event_id}, ensure_ascii=False)
            else:
                body = '{"error":"text required"}'
            status = "200 OK"

        elif "GET /review_post" in req:
            # Review a tweet draft: /review_post?text=...&channel=x
            text = params.get("text", [""])[0]
            channel = params.get("channel", ["x"])[0]
            face = params.get("face", ["false"])[0] == "true"
            mentions_raw = params.get("mentions", [""])[0]
            mentions = [m for m in mentions_raw.split(",") if m] if mentions_raw else None

            if text:
                stores = _stores()
                result = stores.boundary.review_social_post(
                    channel=channel, text=text,
                    scene_contains_face=face,
                    person_mentions=mentions,
                )
                body = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
            else:
                body = '{"error":"text required"}'
            status = "200 OK"

        elif "GET /social_state" in req:
            person_id = params.get("person_id", [None])[0]
            stores = _stores()
            window = int(params.get("window", ["900"])[0])
            result = stores.social_state.get_social_state(person_id=person_id, window_seconds=window)
            body = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
            status = "200 OK"

        elif "GET /interaction_context" in req:
            stores = _stores()
            ctx = compose_interaction_context(
                ComposeInteractionContextInput(
                    person_id=params.get("person_id", ["kouta"])[0],
                    channel=params.get("channel", ["chat"])[0],
                    user_text=params.get("text", [None])[0],
                    autonomous_trigger=params.get("trigger", [None])[0],
                    include_private=params.get("include_private", ["true"])[0] == "true",
                    max_chars=int(params.get("max_chars", ["3000"])[0]),
                ),
                social_state_store=stores.social_state,
                relationship_store=stores.relationship,
                joint_attention_store=stores.joint_attention,
                boundary_store=stores.boundary,
                self_narrative_store=stores.self_narrative,
                orchestrator_store=stores.orchestrator,
                policy_timezone=stores.policy_timezone,
            )
            body = json.dumps(ctx.model_dump(mode="json"), ensure_ascii=False)
            status = "200 OK"

        elif "POST /record_agent_experience" in req:
            stored = _stores().orchestrator.record_agent_experience(
                RecordAgentExperienceInput.model_validate(_json_body())
            )
            body = json.dumps(
                {"experience_id": stored.experience_id, "ts": stored.ts},
                ensure_ascii=False,
            )
            status = "200 OK"

        elif "POST /private_reflection" in req:
            stored = _stores().orchestrator.append_private_reflection(
                AppendPrivateReflectionInput.model_validate(_json_body())
            )
            body = json.dumps(
                {"reflection_id": stored.experience_id, "ts": stored.ts},
                ensure_ascii=False,
            )
            status = "200 OK"

        elif "POST /interpretation_shift" in req:
            stored = _stores().orchestrator.record_interpretation_shift(
                RecordInterpretationShiftInput.model_validate(_json_body())
            )
            body = json.dumps(
                {"shift_id": stored.experience_id, "ts": stored.ts},
                ensure_ascii=False,
            )
            status = "200 OK"

        response = f"HTTP/1.1 {status}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {len(body.encode())}\r\nConnection: close\r\n\r\n{body}"
        writer.write(response.encode("utf-8"))
        await writer.drain()
    except Exception as e:
        import traceback
        err_body = json.dumps({"error": str(e), "trace": traceback.format_exc()})
        err_resp = f"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: {len(err_body.encode())}\r\nConnection: close\r\n\r\n{err_body}"
        try:
            writer.write(err_resp.encode("utf-8"))
            await writer.drain()
        except Exception:
            pass
    finally:
        writer.close()


def main() -> None:
    import asyncio
    import os
    import threading

    http_port = int(os.environ.get("SOCIALITY_HTTP_PORT", "18901"))

    def _run_http() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _serve() -> None:
            server = await asyncio.start_server(_handle_http, "127.0.0.1", http_port)
            import logging
            logging.getLogger("sociality-mcp").info(f"HTTP endpoint on 127.0.0.1:{http_port}")
            async with server:
                await server.serve_forever()

        loop.run_until_complete(_serve())

    # Start HTTP server in a background thread
    http_thread = threading.Thread(target=_run_http, daemon=True)
    http_thread.start()

    # Run MCP server in main thread (it owns the event loop)
    mcp.run()


if __name__ == "__main__":
    main()
