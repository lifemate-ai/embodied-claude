"""Acceptance fixture: field -> rollout -> intention -> action -> error -> update."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from social_core.db import SocialDB

from individual_kernel_mcp.agency import ActionProposal, PredictedEffects
from individual_kernel_mcp.enacted_field import TriggerKind
from individual_kernel_mcp.experienced_transition import ExperiencedTransitionStore
from individual_kernel_mcp.generative_model import CountBasedGenerativeFieldModel
from individual_kernel_mcp.tick import FieldRuntime, TickProducer
from individual_kernel_mcp.trajectory import TrajectoryStatus, TrajectoryStore
from individual_kernel_mcp.workspace import (
    CandidateKind,
    CandidateSource,
    SourceMode,
    WorkspaceCandidate,
)

_NEW_TABLES = (
    "protention_distributions",
    "imagined_trajectories",
    "experienced_transitions",
    "generative_transition_stats",
)


@pytest.fixture
def flag_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    toml = tmp_path / "behavior-on.toml"
    toml.write_text(
        "[individual-kernel]\n"
        "generative_field_model = true\n"
        "generative_rollout_horizon = 2\n"
    )
    monkeypatch.setenv("MCP_BEHAVIOR_TOML", str(toml))
    return toml


def _flag_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toml = tmp_path / "behavior-off.toml"
    toml.write_text("[individual-kernel]\ngenerative_field_model = false\n")
    monkeypatch.setenv("MCP_BEHAVIOR_TOML", str(toml))


def _producer(social_db: SocialDB, state_dir: Path) -> TickProducer:
    state_dir.mkdir(parents=True, exist_ok=True)
    interoception = state_dir / "interoception.json"
    interoception.write_text(json.dumps({"now": {"arousal": 50.0}}))
    desires = state_dir / "desires.json"
    desires.write_text(
        json.dumps(
            {
                "desires": {"identity_coherence": 0.9},
                "discomforts": {"identity_coherence": 0.5},
                "dominant": "identity_coherence",
            }
        )
    )
    return TickProducer(
        social_db,
        interoception_path=interoception,
        desires_path=desires,
    )


def _commit(producer: TickProducer, focus_ref: str = "desire:identity_coherence"):
    opened = producer.begin_tick(TriggerKind.USER_PROMPT)
    producer.workspace.add_candidate(
        WorkspaceCandidate(
            tick_id=opened.tick_id,
            kind=CandidateKind.GOAL,
            content_ref=focus_ref,
            content_summary=f"focus {focus_ref}",
            source=CandidateSource.EXPLICIT,
            source_mode=SourceMode.INFERRED,
            precision=1.0,
            prediction_error=1.0,
            need_relevance=1.0,
            goal_relevance=1.0,
            expected_information_gain=1.0,
            continuity_with_previous=1.0,
            controllability=1.0,
            social_relevance=1.0,
        )
    )
    return producer.compete_and_commit(opened.tick_id).field


_TOOL_INPUT = {"file_path": "/tmp/prediction-loop-fixture.txt", "content": "x"}


def _proposal(field, confidence: float = 0.6) -> ActionProposal:
    return ActionProposal(
        field_id=field.field_id,
        tool_name="Write",
        tool_input=dict(_TOOL_INPUT),
        predicted_effects=PredictedEffects(),
        goal="write the fixture file",
        confidence=confidence,
    )


class TestImagineAtProposeTime:
    def test_distribution_shape_and_modal_intended(
        self, social_db, tmp_path, flag_on
    ) -> None:
        producer = _producer(social_db, tmp_path / "state")
        runtime = FieldRuntime(social_db, producer=producer)
        field = _commit(producer)
        intention = runtime.propose_action(_proposal(field))

        refreshed = producer.fields.get(field.field_id)
        assert refreshed is not None
        ref = refreshed.epistemic_trace.get("protention_distribution_ref")
        assert ref
        store = TrajectoryStore(social_db)
        distribution = store.get_distribution(ref)
        assert distribution is not None
        assert len(distribution.trajectories) >= 2
        no_action = [t for t in distribution.trajectories if t.action_kind == "no_action"]
        assert len(no_action) == 1
        assert sum(t.probability for t in distribution.trajectories) == pytest.approx(1.0)

        linked = store.find_active_for_action(intention.action_id)
        assert linked is not None
        assert linked.status is TrajectoryStatus.INTENDED
        others = [
            t
            for t in distribution.trajectories
            if t.trajectory_id != linked.trajectory_id
        ]
        assert all(
            store.get_trajectory(t.trajectory_id).status is TrajectoryStatus.IMAGINED
            for t in others
        )


class TestActRecordLearnCycle:
    def test_full_cycle_records_transition_and_improves_prediction(
        self, social_db, tmp_path, flag_on
    ) -> None:
        producer = _producer(social_db, tmp_path / "state")
        runtime = FieldRuntime(social_db, producer=producer)
        field = _commit(producer)
        intention = runtime.propose_action(_proposal(field))

        decision = runtime.gate_tool(tool_name="Write", tool_input=dict(_TOOL_INPUT))
        assert decision.allow is True
        store = TrajectoryStore(social_db)
        assert (
            store.find_active_for_action(intention.action_id).status
            is TrajectoryStatus.ENACTED
        )

        outcome, assessment, committed = runtime.close_tool(
            tool_name="Write",
            tool_input=dict(_TOOL_INPUT),
            actual_result_summary="fixture file written successfully",
            success=True,
            latency_ms=50,
        )
        assert outcome is not None
        assert committed is not None

        transitions = ExperiencedTransitionStore(social_db)
        transition = transitions.get_by_next_field(committed.field_id)
        assert transition is not None
        assert transition.action_ref == intention.action_id
        assert transition.action_kind == "tool:Write"
        assert transition.intended_trajectory_id is not None
        assert transition.applied_at is not None
        for key in (
            "exteroceptive",
            "interoceptive",
            "social",
            "mnemonic",
            "latency",
            "focus_kind_error",
            "valence_delta_error",
            "probability_of_observed",
        ):
            assert key in transition.prediction_errors

        settled = store.get_trajectory(transition.intended_trajectory_id)
        assert settled is not None
        assert settled.status in {
            TrajectoryStatus.OBSERVED,
            TrajectoryStatus.PARTIALLY_OBSERVED,
        }

        model = CountBasedGenerativeFieldModel(social_db)
        signature = model.infer(field, action_kind="tool:Write")
        before = transition.prediction_errors["probability_of_observed"]
        after, basis, uncertainty = model.probability_of(
            signature, transition.outcome_bucket
        )
        assert after > before
        assert uncertainty < 1.0
        assert basis == "count"

    def test_failed_action_contradicts_trajectory(
        self, social_db, tmp_path, flag_on
    ) -> None:
        producer = _producer(social_db, tmp_path / "state")
        runtime = FieldRuntime(social_db, producer=producer)
        field = _commit(producer)
        intention = runtime.propose_action(_proposal(field, confidence=0.9))
        assert runtime.gate_tool(
            tool_name="Write", tool_input=dict(_TOOL_INPUT)
        ).allow
        runtime.close_tool(
            tool_name="Write",
            tool_input=dict(_TOOL_INPUT),
            actual_result_summary="permission denied while writing fixture",
            success=False,
            latency_ms=20,
        )
        store = TrajectoryStore(social_db)
        assert (
            store.find_active_for_action(intention.action_id) is None
            or store.find_active_for_action(intention.action_id).status
            is not TrajectoryStatus.ENACTED
        )
        settled = TrajectoryStore(social_db).get_trajectory(
            ExperiencedTransitionStore(social_db)
            .query(action_kind="tool:Write")[0]
            .intended_trajectory_id
        )
        assert settled.status is TrajectoryStatus.CONTRADICTED


class TestNoActionAndExpiry:
    def test_plain_commits_record_no_action_transition_once(
        self, social_db, tmp_path, flag_on
    ) -> None:
        producer = _producer(social_db, tmp_path / "state")
        first = _commit(producer)
        second = _commit(producer)
        transitions = ExperiencedTransitionStore(social_db)
        transition = transitions.get_by_next_field(second.field_id)
        assert transition is not None
        assert transition.action_ref is None
        assert transition.action_kind == "no_action"
        assert transition.previous_field_id == first.field_id
        row = social_db.fetchone(
            "SELECT COUNT(*) AS n FROM experienced_transitions WHERE next_field_id = ?",
            (second.field_id,),
        )
        assert row["n"] == 1

    def test_unexecuted_intention_returns_trajectory_to_imagined(
        self, social_db, tmp_path, flag_on
    ) -> None:
        producer = _producer(social_db, tmp_path / "state")
        runtime = FieldRuntime(social_db, producer=producer)
        field = _commit(producer)
        intention = runtime.propose_action(_proposal(field))
        _commit(producer)  # recovers the dangling intention, no execution
        store = TrajectoryStore(social_db)
        rows = store.query(action_ref=intention.action_id)
        assert rows
        assert all(t.status is TrajectoryStatus.IMAGINED for t in rows)
        assert any(
            entry["reason"] == "intention expired without execution"
            for t in rows
            for entry in t.status_history
        )


class TestBoundaryIndependenceAndFlag:
    def test_gate_decision_identical_with_flag_on_and_off(
        self, tmp_path, monkeypatch
    ) -> None:
        def run(db_name: str, state: str) -> tuple:
            db = SocialDB(tmp_path / db_name)
            db.connect()
            try:
                producer = _producer(db, tmp_path / state)
                runtime = FieldRuntime(db, producer=producer)
                field = _commit(producer)
                runtime.propose_action(_proposal(field))
                decision = runtime.gate_tool(
                    tool_name="Write", tool_input=dict(_TOOL_INPUT)
                )
                return (
                    decision.allow,
                    decision.reason,
                    decision.external,
                    decision.deferred,
                )
            finally:
                db.close()

        toml_on = tmp_path / "on.toml"
        toml_on.write_text("[individual-kernel]\ngenerative_field_model = true\n")
        monkeypatch.setenv("MCP_BEHAVIOR_TOML", str(toml_on))
        with_flag = run("a.db", "state-a")

        _flag_off(tmp_path, monkeypatch)
        without_flag = run("b.db", "state-b")

        assert with_flag == without_flag

    def test_flag_off_writes_nothing(self, tmp_path, monkeypatch) -> None:
        _flag_off(tmp_path, monkeypatch)
        db = SocialDB(tmp_path / "off.db")
        db.connect()
        try:
            producer = _producer(db, tmp_path / "state-off")
            runtime = FieldRuntime(db, producer=producer)
            field = _commit(producer)
            runtime.propose_action(_proposal(field))
            assert runtime.gate_tool(
                tool_name="Write", tool_input=dict(_TOOL_INPUT)
            ).allow
            runtime.close_tool(
                tool_name="Write",
                tool_input=dict(_TOOL_INPUT),
                actual_result_summary="fixture file written successfully",
                success=True,
                latency_ms=50,
            )
            for table in _NEW_TABLES:
                row = db.fetchone(f"SELECT COUNT(*) AS n FROM {table}")
                assert row["n"] == 0, table
        finally:
            db.close()

    def test_boundary_denial_returns_trajectory(
        self, social_db, tmp_path, flag_on
    ) -> None:
        producer = _producer(social_db, tmp_path / "state")
        runtime = FieldRuntime(
            social_db,
            producer=producer,
            boundary_evaluator=lambda name, tool_input, field: (False, "fixture deny"),
        )
        field = _commit(producer)
        intention = runtime.propose_action(_proposal(field))
        decision = runtime.gate_tool(tool_name="Write", tool_input=dict(_TOOL_INPUT))
        assert decision.allow is False
        store = TrajectoryStore(social_db)
        rows = store.query(action_ref=intention.action_id)
        assert rows
        assert all(t.status is TrajectoryStatus.IMAGINED for t in rows)
        assert any(
            entry["reason"] == "boundary denied"
            for t in rows
            for entry in t.status_history
        )
