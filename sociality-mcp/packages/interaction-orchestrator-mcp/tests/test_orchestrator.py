"""Tests for the interaction orchestrator."""

from __future__ import annotations

import json

from boundary_mcp.schemas import QuietModeState

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


def _compose(stores, *, user_text=None, channel="chat", person_id="kouta", memory_adapter=None):
    return compose_interaction_context(
        ComposeInteractionContextInput(
            person_id=person_id, channel=channel, user_text=user_text
        ),
        social_state_store=stores["social_state"],
        relationship_store=stores["relationship"],
        joint_attention_store=stores["joint_attention"],
        boundary_store=stores["boundary"],
        self_narrative_store=stores["self_narrative"],
        orchestrator_store=stores["orchestrator"],
        policy_timezone="Asia/Tokyo",
        memory_adapter=memory_adapter or stores.get("memory_adapter"),
    )


def _seed_dominant_desire(monkeypatch, tmp_path):
    """Point the orchestrator at a snapshot with one clear dominant desire."""
    fake_desires = tmp_path / "desires.json"
    fake_desires.write_text(
        json.dumps(
            {
                "updated_at": "2026-04-19T10:00:00+00:00",
                "desires": {"browse_curiosity": 0.9, "observe_room": 0.2},
                "discomforts": {"browse_curiosity": 0.6, "observe_room": 0.0},
                "dominant": "browse_curiosity",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DESIRES_PATH", str(fake_desires))


def _pin_quiet_hours(monkeypatch, stores, *, active: bool) -> None:
    """Fix the quiet-hours verdict.

    The policy resolves quiet hours against the wall clock in Asia/Tokyo, so a
    planning test that leaves it unset asserts a different branch depending on
    the hour it runs at. Each caller states which regime it is testing.
    """
    state = QuietModeState(active=active, confidence=1.0, reasons=["pinned by test"])
    monkeypatch.setattr(
        stores["boundary"], "get_quiet_mode_state", lambda **_kwargs: state
    )


class TestCompose:
    def test_returns_context_with_contract_and_prompt_block(self, stores):
        ctx = _compose(stores, user_text="テスト")
        assert ctx.response_contract.treat_user_as.startswith("high-context")
        assert "[response_contract]" in ctx.compact_prompt_block
        assert ctx.compact_prompt_block.startswith("[interaction_context]")
        assert ctx.timezone == "Asia/Tokyo"

    def test_primary_contract_follows_companion_id(self, stores, monkeypatch):
        """The primary-companion contract is for whoever COMPANION_ID names (#135)."""
        monkeypatch.delenv("COMPANION_ID", raising=False)
        default_ctx = _compose(stores, user_text="テスト", person_id="companion")
        assert "generic reassurance" in default_ctx.response_contract.avoid

        other = _compose(stores, user_text="テスト", person_id="kouta")
        assert "generic reassurance" not in other.response_contract.avoid

        monkeypatch.setenv("COMPANION_ID", "kouta")
        named = _compose(stores, user_text="テスト", person_id="kouta")
        assert "generic reassurance" in named.response_contract.avoid

    def test_person_id_defaults_to_companion_id(self, monkeypatch):
        monkeypatch.delenv("COMPANION_ID", raising=False)
        assert ComposeInteractionContextInput().person_id == "companion"
        monkeypatch.setenv("COMPANION_ID", "kouta")
        assert ComposeInteractionContextInput().person_id == "kouta"

    def test_autonomous_channel_tightens_contract(self, stores):
        ctx = _compose(stores, user_text=None, channel="autonomous")
        joined = " ".join(ctx.response_contract.avoid)
        assert "public posting without review" in joined

    def test_missing_person_still_works(self, stores):
        ctx = _compose(stores, user_text="hello", person_id=None)
        assert ctx.person_id is None
        assert ctx.agent_state is not None

    def test_agent_state_includes_counts(self, stores):
        ctx = _compose(stores)
        assert ctx.agent_state.private_reflections == 0
        assert ctx.agent_state.interpretation_shifts == 0
        assert ctx.agent_state.recent_experiences == []


class TestRecord:
    def test_record_agent_experience_is_visible_next_compose(self, stores):
        stores["orchestrator"].record_agent_experience(
            RecordAgentExperienceInput(
                person_id="kouta",
                kind="agent_response",
                summary="Wrote v0.3 spec baseline",
                importance=4,
                privacy_level="private",
            )
        )
        ctx = _compose(stores)
        assert len(ctx.agent_state.recent_experiences) == 1
        assert ctx.agent_state.recent_experiences[0].summary.startswith("Wrote v0.3")

    def test_record_interpretation_shift_counts_up(self, stores):
        stores["orchestrator"].record_interpretation_shift(
            RecordInterpretationShiftInput(
                person_id="kouta",
                topic="late-night behavior",
                old_interpretation="sample wording is a hard rule",
                new_interpretation="policy purpose (protect sleep) is the rule",
                trigger="Kouta pointed out the confusion",
                confidence=0.92,
                implications=["Check policy purpose before suppressing action."],
            )
        )
        ctx = _compose(stores)
        assert ctx.agent_state.interpretation_shifts == 1

    def test_append_private_reflection_counts_up(self, stores):
        stores["orchestrator"].append_private_reflection(
            AppendPrivateReflectionInput(
                person_id="kouta",
                title="morning notes",
                body="Quiet rebuild of morning routine.",
                importance=3,
            )
        )
        ctx = _compose(stores)
        assert ctx.agent_state.private_reflections == 1

    def test_compose_private_letter_persists(self, stores):
        stored = stores["orchestrator"].compose_private_letter(
            ComposePrivateLetterInput(
                person_id="kouta",
                title="朝のお手紙",
                body="深夜のループを振り返って...",
                visibility="private",
            )
        )
        assert stored.experience_id.startswith("ltr_")


class TestPlan:
    def test_direct_question_produces_answer_directly(self, stores):
        ctx = _compose(stores, user_text="このPRどう見る？")
        plan = plan_response(
            PlanResponseInput(interaction_context=ctx, user_text="このPRどう見る？")
        )
        assert plan.primary_move == "answer_directly"
        assert plan.initiative.level != "none"

    def test_autonomous_quiet_night_prefers_private_reflection(self, stores):
        # Force quiet hours by ingesting a late-night event in JST
        stores["social_state"].ingest_social_event(
            {
                "ts": "2026-04-18T16:30:00Z",  # 01:30 JST
                "source": "camera",
                "kind": "scene_parse",
                "person_id": "kouta",
                "confidence": 0.8,
                "payload": {"scene_summary": "Dim room, no speech."},
            }
        )
        ctx = _compose(stores, user_text=None, channel="autonomous")
        plan = plan_response(
            PlanResponseInput(interaction_context=ctx, user_text=None)
        )
        # Either a private reflection or deferring silently — never loud voice.
        assert plan.primary_move in {
            "write_private_reflection",
            "stay_silent",
            "quietly_prepare",
        }
        assert plan.voice is not None
        assert plan.voice.speak is False
        assert "camera_speaker_audio" in plan.initiative.forbidden_actions

    def test_plan_must_avoid_includes_contract_avoid(self, stores):
        ctx = _compose(stores, user_text="please help")
        plan = plan_response(
            PlanResponseInput(interaction_context=ctx, user_text="please help")
        )
        joined = " ".join(plan.must_avoid)
        assert "generic reassurance" in joined

    def test_ambiguous_short_input_asks_clarifying_question(self, stores):
        ctx = _compose(stores, user_text="ね")
        plan = plan_response(PlanResponseInput(interaction_context=ctx, user_text="ね"))
        assert plan.primary_move == "ask_one_clarifying_question"

    def test_relevant_memories_flow_into_memory_use(self, stores, tmp_path):
        """With a seeded memory-db, recall_for_response populates relevant_memories
        and plan.memory_use flips to use_specific_memory=True."""
        import sqlite3

        from interaction_orchestrator_mcp.memory_adapter import SQLiteMemoryAdapter

        db = tmp_path / "memory.db"
        conn = sqlite3.connect(str(db))
        try:
            conn.executescript(
                """
                CREATE TABLE memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    normalized_content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    emotion TEXT NOT NULL DEFAULT 'neutral',
                    importance INTEGER NOT NULL DEFAULT 3,
                    category TEXT NOT NULL DEFAULT 'daily',
                    tags TEXT NOT NULL DEFAULT ''
                );
                """
            )
            conn.execute(
                "INSERT INTO memories(id, content, normalized_content, timestamp, "
                "emotion, importance, category) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "mem1",
                    "kokone.one の DNS が落ちてた。NSレコードを設定し直して復旧した。",
                    "kokone.one dns",
                    "2026-04-19T20:00:00+00:00",
                    "neutral",
                    4,
                    "technical",
                ),
            )
            conn.commit()
        finally:
            conn.close()

        adapter = SQLiteMemoryAdapter(db)
        ctx = _compose(
            stores,
            user_text="kokone.one の DNS また見ておく？",
            memory_adapter=adapter,
        )
        assert len(ctx.relevant_memories) >= 1
        assert ctx.relevant_memories[0].use_policy == "mentionable"
        assert "[relevant_memories]" in ctx.compact_prompt_block

        plan = plan_response(
            PlanResponseInput(
                interaction_context=ctx,
                user_text="kokone.one の DNS また見ておく？",
            )
        )
        assert plan.memory_use.use_specific_memory is True
        assert plan.memory_use.max_memories_to_surface >= 1

    def test_autonomous_with_dominant_desire_is_bounded(self, stores, monkeypatch, tmp_path):
        _seed_dominant_desire(monkeypatch, tmp_path)
        _pin_quiet_hours(monkeypatch, stores, active=False)
        ctx = _compose(stores, user_text=None, channel="autonomous")
        plan = plan_response(
            PlanResponseInput(interaction_context=ctx, user_text=None)
        )
        assert plan.primary_move == "act_autonomously"
        assert "web_search" in plan.initiative.allowed_actions
        assert plan.followup_action is not None
        assert plan.followup_action["kind"] == "satisfy_desire"

    def test_autonomous_during_quiet_hours_stays_private(
        self, stores, monkeypatch, tmp_path
    ):
        _seed_dominant_desire(monkeypatch, tmp_path)
        _pin_quiet_hours(monkeypatch, stores, active=True)
        ctx = _compose(stores, user_text=None, channel="autonomous")
        plan = plan_response(
            PlanResponseInput(interaction_context=ctx, user_text=None)
        )
        assert plan.primary_move == "write_private_reflection"
        assert plan.voice is not None
        assert plan.voice.speak is False
