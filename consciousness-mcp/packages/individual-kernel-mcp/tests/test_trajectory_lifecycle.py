"""Imagined trajectories: status lifecycle, immutability, distribution invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from social_core.db import SocialDB

from individual_kernel_mcp.enacted_field import TriggerKind
from individual_kernel_mcp.generative_model import FieldBelief, TrajectoryStep
from individual_kernel_mcp.tick import TickProducer
from individual_kernel_mcp.trajectory import (
    ImaginedTrajectory,
    ProtentionDistribution,
    TrajectoryStatus,
    TrajectoryStore,
    normalized_entropy,
)
from individual_kernel_mcp.workspace import (
    CandidateKind,
    CandidateSource,
    SourceMode,
    WorkspaceCandidate,
)


def _producer(social_db: SocialDB, tmp_path: Path) -> TickProducer:
    interoception = tmp_path / "interoception.json"
    interoception.write_text(json.dumps({"now": {"arousal": 50.0}}))
    desires = tmp_path / "desires.json"
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


def _step(index: int = 1) -> TrajectoryStep:
    return TrajectoryStep(
        step_index=index,
        predicted_belief=FieldBelief(source_mode=SourceMode.IMAGINED),
        outcome_bucket="ok/short/tool_result/=",
        conditional_probability=0.5,
        uncertainty=0.5,
        support_observations=0.0,
        basis="prior",
    )


def _trajectory(
    field,
    distribution_id: str,
    *,
    probability: float,
    action_kind: str = "tool:Bash",
) -> ImaginedTrajectory:
    return ImaginedTrajectory(
        distribution_id=distribution_id,
        field_id=field.field_id,
        tick_id=field.tick_id,
        action_kind=action_kind,
        context_signature="desire|user_prompt|identity_coherence|neu|mid|" + action_kind,
        horizon=1,
        steps=[_step()],
        probability=probability,
        uncertainty=0.5,
    )


def _distribution(field) -> ProtentionDistribution:
    distribution_id = "prot_test_000000"
    trajectories = [
        _trajectory(field, distribution_id, probability=0.6),
        _trajectory(field, distribution_id, probability=0.4, action_kind="no_action"),
    ]
    return ProtentionDistribution(
        distribution_id=distribution_id,
        field_id=field.field_id,
        tick_id=field.tick_id,
        trajectories=trajectories,
        entropy=normalized_entropy([0.6, 0.4]),
    )


class TestDistributionInvariants:
    def test_probabilities_must_sum_to_one(self, social_db, tmp_path) -> None:
        producer = _producer(social_db, tmp_path)
        field = _commit(producer)
        with pytest.raises(ValidationError):
            ProtentionDistribution(
                field_id=field.field_id,
                tick_id=field.tick_id,
                trajectories=[
                    _trajectory(field, "prot_x", probability=0.6),
                    _trajectory(field, "prot_x", probability=0.6),
                ],
                entropy=0.5,
            )

    def test_trajectory_source_mode_must_be_imagined(self, social_db, tmp_path) -> None:
        producer = _producer(social_db, tmp_path)
        field = _commit(producer)
        with pytest.raises(ValidationError):
            ImaginedTrajectory(
                distribution_id="prot_x",
                field_id=field.field_id,
                tick_id=field.tick_id,
                action_kind="no_action",
                context_signature="a|b|c|d|e|no_action",
                horizon=1,
                steps=[_step()],
                probability=1.0,
                uncertainty=0.5,
                source_mode=SourceMode.LIVE,
            )

    def test_steps_must_match_horizon(self, social_db, tmp_path) -> None:
        producer = _producer(social_db, tmp_path)
        field = _commit(producer)
        with pytest.raises(ValidationError):
            ImaginedTrajectory(
                distribution_id="prot_x",
                field_id=field.field_id,
                tick_id=field.tick_id,
                action_kind="no_action",
                context_signature="a|b|c|d|e|no_action",
                horizon=2,
                steps=[_step()],
                probability=1.0,
                uncertainty=0.5,
            )

    def test_entropy_of_uniform_pair_is_one(self) -> None:
        assert normalized_entropy([0.5, 0.5]) == pytest.approx(1.0)
        assert normalized_entropy([1.0]) == 0.0


class TestStoreRoundTrip:
    def test_distribution_round_trips(self, social_db, tmp_path) -> None:
        producer = _producer(social_db, tmp_path)
        field = _commit(producer)
        store = TrajectoryStore(social_db)
        created = store.create_distribution(_distribution(field))
        loaded = store.get_distribution(created.distribution_id)
        assert loaded is not None
        key = lambda t: t.trajectory_id  # noqa: E731
        assert sorted(loaded.trajectories, key=key) == sorted(created.trajectories, key=key)


class TestStatusLifecycle:
    def _stored_trajectory(self, social_db, tmp_path):
        producer = _producer(social_db, tmp_path)
        field = _commit(producer)
        store = TrajectoryStore(social_db)
        distribution = store.create_distribution(_distribution(field))
        return store, distribution.trajectories[0].trajectory_id

    def test_full_legal_chain(self, social_db, tmp_path) -> None:
        store, trajectory_id = self._stored_trajectory(social_db, tmp_path)
        store.transition(trajectory_id, TrajectoryStatus.INTENDED, reason="linked")
        store.transition(trajectory_id, TrajectoryStatus.ENACTED, reason="gate allowed")
        final = store.transition(
            trajectory_id, TrajectoryStatus.OBSERVED, reason="low prediction error"
        )
        assert final.status is TrajectoryStatus.OBSERVED
        assert [entry["status"] for entry in final.status_history] == [
            "intended",
            "enacted",
            "observed",
        ]

    def test_boundary_denial_returns_to_imagined(self, social_db, tmp_path) -> None:
        store, trajectory_id = self._stored_trajectory(social_db, tmp_path)
        store.transition(trajectory_id, TrajectoryStatus.INTENDED, reason="linked")
        back = store.transition(
            trajectory_id, TrajectoryStatus.IMAGINED, reason="boundary denied"
        )
        assert back.status is TrajectoryStatus.IMAGINED

    def test_enacted_can_contradict_or_partially_observe(self, social_db, tmp_path) -> None:
        store, first = self._stored_trajectory(social_db, tmp_path)
        store.transition(first, TrajectoryStatus.INTENDED, reason="linked")
        store.transition(first, TrajectoryStatus.ENACTED, reason="gate allowed")
        assert (
            store.transition(first, TrajectoryStatus.CONTRADICTED, reason="mismatch").status
            is TrajectoryStatus.CONTRADICTED
        )

    def test_imagined_never_auto_promotes_to_enacted(self, social_db, tmp_path) -> None:
        store, trajectory_id = self._stored_trajectory(social_db, tmp_path)
        with pytest.raises(ValueError):
            store.transition(trajectory_id, TrajectoryStatus.ENACTED, reason="skip")

    def test_terminal_states_reject_transitions(self, social_db, tmp_path) -> None:
        store, trajectory_id = self._stored_trajectory(social_db, tmp_path)
        store.transition(trajectory_id, TrajectoryStatus.INTENDED, reason="linked")
        store.transition(trajectory_id, TrajectoryStatus.ENACTED, reason="gate allowed")
        store.transition(trajectory_id, TrajectoryStatus.OBSERVED, reason="done")
        for target in (
            TrajectoryStatus.INTENDED,
            TrajectoryStatus.ENACTED,
            TrajectoryStatus.IMAGINED,
            TrajectoryStatus.CONTRADICTED,
        ):
            with pytest.raises(ValueError):
                store.transition(trajectory_id, target, reason="illegal")

    def test_unknown_trajectory_raises(self, social_db) -> None:
        store = TrajectoryStore(social_db)
        with pytest.raises(ValueError):
            store.transition("traj_missing", TrajectoryStatus.INTENDED, reason="x")
