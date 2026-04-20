"""Tests for the interaction orchestrator."""

from __future__ import annotations

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


def _compose(stores, *, user_text=None, channel="chat", person_id="kouta"):
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
    )


class TestCompose:
    def test_returns_context_with_contract_and_prompt_block(self, stores):
        ctx = _compose(stores, user_text="テスト")
        assert ctx.response_contract.treat_user_as.startswith("high-context")
        assert "[response_contract]" in ctx.compact_prompt_block
        assert ctx.compact_prompt_block.startswith("[interaction_context]")
        assert ctx.timezone == "Asia/Tokyo"

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

    def test_autonomous_with_dominant_desire_is_bounded(self, stores, monkeypatch, tmp_path):
        # Seed desires.json so the orchestrator sees a dominant desire.
        import json as _json

        fake_desires = tmp_path / "desires.json"
        fake_desires.write_text(
            _json.dumps(
                {
                    "updated_at": "2026-04-19T10:00:00+00:00",
                    "desires": {
                        "browse_curiosity": 0.9,
                        "observe_room": 0.2,
                    },
                    "discomforts": {
                        "browse_curiosity": 0.6,
                        "observe_room": 0.0,
                    },
                    "dominant": "browse_curiosity",
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("DESIRES_PATH", str(fake_desires))
        ctx = _compose(stores, user_text=None, channel="autonomous")
        plan = plan_response(
            PlanResponseInput(interaction_context=ctx, user_text=None)
        )
        assert plan.primary_move == "act_autonomously"
        assert "web_search" in plan.initiative.allowed_actions
        assert plan.followup_action is not None
        assert plan.followup_action["kind"] == "satisfy_desire"
