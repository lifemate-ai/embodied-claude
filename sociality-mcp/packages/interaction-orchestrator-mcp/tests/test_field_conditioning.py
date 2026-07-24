"""Causal field conditioning tests for the interaction orchestrator."""

from __future__ import annotations

import json

import pytest

from interaction_orchestrator_mcp.compose import compose_interaction_context
from interaction_orchestrator_mcp.memory_adapter import RecallHit
from interaction_orchestrator_mcp.plan import plan_response
from interaction_orchestrator_mcp.schemas import (
    ComposeInteractionContextInput,
    PlanResponseInput,
)


class _FixedMemoryAdapter:
    def recall_for_response(self, **_kwargs) -> list[RecallHit]:
        return [
            RecallHit(
                memory_id="mem_garden",
                content="garden watering schedule",
                timestamp="2026-07-24T00:00:00Z",
                category="daily",
                emotion="neutral",
                importance=3,
                relevance=0.5,
                use_policy="mentionable",
                reason="fixture",
            ),
            RecallHit(
                memory_id="mem_camera",
                content="camera room calibration",
                timestamp="2026-07-24T00:00:00Z",
                category="daily",
                emotion="neutral",
                importance=3,
                relevance=0.5,
                use_policy="mentionable",
                reason="fixture",
            ),
        ]


def _compose(stores, payload: ComposeInteractionContextInput):
    return compose_interaction_context(
        payload,
        social_state_store=stores["social_state"],
        relationship_store=stores["relationship"],
        joint_attention_store=stores["joint_attention"],
        boundary_store=stores["boundary"],
        self_narrative_store=stores["self_narrative"],
        orchestrator_store=stores["orchestrator"],
        policy_timezone="Asia/Tokyo",
        memory_adapter=_FixedMemoryAdapter(),
    )


def _seed_committed_field(stores) -> None:
    db = stores["orchestrator"].db
    ts = "2026-07-24T00:00:00Z"
    trace = json.dumps({"focal_summary": "camera room focus"})
    with db.transaction() as connection:
        connection.execute(
            """
            INSERT INTO tick_frames (
                tick_id, ts, ignited, conflicted, reportability, created_at
            ) VALUES (?, ?, 1, 0, 'mentionable', ?)
            """,
            ("tick_field_conditioning", ts, ts),
        )
        connection.execute(
            """
            INSERT INTO enacted_fields (
                field_id, owner_id, tick_id, continuity_token, status,
                trigger_kind, started_at, committed_at, self_origin_json,
                reality_mode, reality_score, peripheral_content_refs_json,
                retention_refs_json, protention_json, interoception_json,
                precision_json, affordance_refs_json, agency_state_json,
                hor_refs_json, phenomenal_surface, epistemic_trace_json,
                created_at, updated_at, focal_content_ref
            ) VALUES (
                ?, 'self', ?, 'continuity-fixture', 'COMMITTED',
                'user_prompt', ?, ?, '{}',
                'inferred', 0.6, '[]',
                '[]', '{}', '{}',
                '{}', '[]', '{}',
                '[]', '<current_field>camera room focus</current_field>', ?,
                ?, ?, 'percept:camera-room'
            )
            """,
            ("field_conditioning", "tick_field_conditioning", ts, ts, trace, ts, ts),
        )
        connection.execute(
            """
            INSERT INTO field_runtime_state (
                owner_id, continuity_token, current_field_id, open_tick_id,
                state, last_trigger_kind, updated_at
            ) VALUES (
                'self', 'continuity-fixture', 'field_conditioning',
                'tick_field_conditioning', 'ACTIVE', 'user_prompt', ?
            )
            """,
            (ts,),
        )


def test_strict_composition_requires_committed_field(stores):
    with pytest.raises(RuntimeError, match="requires a committed field"):
        _compose(
            stores,
            ComposeInteractionContextInput(
                user_text="same raw input",
                require_committed_field=True,
            ),
        )


def test_committed_focus_changes_memory_selection_and_plan_provenance(stores):
    _seed_committed_field(stores)

    context = _compose(
        stores,
        ComposeInteractionContextInput(
            user_text="same raw input",
            require_committed_field=True,
        ),
    )

    assert context.current_field_id == "field_conditioning"
    assert context.relevant_memories[0].memory_id == "mem_camera"
    assert context.relevant_memories[0].relevance > context.relevant_memories[1].relevance
    assert "conditioned by committed field" in (context.relevant_memories[0].reason or "")

    plan = plan_response(
        PlanResponseInput(interaction_context=context, user_text="same raw input")
    )
    assert plan.source_field_id == "field_conditioning"
