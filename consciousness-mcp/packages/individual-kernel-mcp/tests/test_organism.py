"""Internal time has to do something, not merely exist.

The claim under test is narrow and checkable: with no input at all, recorded
state changes, and under the right internal conditions a tick opens that nobody
asked for.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from social_core.db import SocialDB

from individual_kernel_mcp.enacted_field import TriggerKind
from individual_kernel_mcp.generative_model import FieldBelief, TrajectoryStep
from individual_kernel_mcp.organism import (
    MIN_OBSERVATION_COUNT,
    STATS_HALF_LIFE_HOURS,
    OrganismDaemon,
    decay_factor,
    expiry_deadline,
    ignition_score,
    should_ignite,
)
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

WEEK_SECONDS = STATS_HALF_LIFE_HOURS * 3600.0


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def _producer(social_db: SocialDB, tmp_path: Path, *, unmet: bool = True) -> TickProducer:
    """A producer whose desire snapshot is either strained or settled.

    `look_outside` sits well above its set point when unmet, which is the only
    thing that makes the need term of the ignition score nonzero.
    """
    interoception = tmp_path / "interoception.json"
    interoception.write_text(json.dumps({"now": {"arousal": 50.0}}), encoding="utf-8")
    desires = tmp_path / "desires.json"
    desires.write_text(
        json.dumps(
            {
                "desires": {"look_outside": 1.0 if unmet else 0.3},
                "dominant": "look_outside",
            }
        ),
        encoding="utf-8",
    )
    return TickProducer(
        social_db, interoception_path=interoception, desires_path=desires
    )


def _commit(producer: TickProducer):
    opened = producer.begin_tick(TriggerKind.USER_PROMPT)
    producer.workspace.add_candidate(
        WorkspaceCandidate(
            tick_id=opened.tick_id,
            kind=CandidateKind.GOAL,
            content_ref="desire:look_outside",
            content_summary="need look_outside",
            source=CandidateSource.DESIRE,
            source_mode=SourceMode.INFERRED,
            precision=1.0,
            need_relevance=1.0,
            goal_relevance=1.0,
            continuity_with_previous=1.0,
            controllability=1.0,
        )
    )
    return producer.compete_and_commit(opened.tick_id).field


def _seed_stat(db: SocialDB, count: float) -> None:
    db.execute(
        "INSERT INTO generative_transition_stats("
        "owner_id, focus_kind, trigger_kind, dominant_desire, valence_bucket, "
        "arousal_bucket, action_kind, outcome_bucket, observation_count, "
        "model_version, first_observed_at, updated_at"
        ") VALUES ('self', 'desire', 'user_prompt', 'look_outside', 'neu', "
        "'mid', 'tool:look_left', 'ok/short/percept/=', ?, 'v1', ?, ?)",
        (count, "2026-07-01T00:00:00Z", "2026-07-01T00:00:00Z"),
    )


def _observations(db: SocialDB) -> float:
    row = db.fetchone(
        "SELECT COALESCE(SUM(observation_count), 0.0) AS total "
        "FROM generative_transition_stats",
        (),
    )
    return float(row["total"]) if row else 0.0


class TestDecay:
    def test_a_half_life_halves(self) -> None:
        assert decay_factor(WEEK_SECONDS, STATS_HALF_LIFE_HOURS) == pytest.approx(0.5)

    def test_no_elapsed_time_retains_everything(self) -> None:
        assert decay_factor(0.0, STATS_HALF_LIFE_HOURS) == 1.0

    def test_two_steps_equal_one_long_step(self) -> None:
        # This is what lets a scheduler run at any cadence: the state reached
        # depends on total elapsed time, not on how it was divided up.
        split = decay_factor(3600.0, 24.0) * decay_factor(7200.0, 24.0)
        assert split == pytest.approx(decay_factor(10800.0, 24.0))


class TestIgnitionScore:
    def test_need_without_silence_is_not_enough_on_its_own(self) -> None:
        score = ignition_score(max_discomfort=1.0, seconds_since_last_field=0.0)
        ignite, reason = should_ignite(score, seconds_since_last_field=0.0)
        assert not ignite
        assert "floor" in reason

    def test_silence_without_need_stays_below_the_threshold(self) -> None:
        score = ignition_score(max_discomfort=0.0, seconds_since_last_field=1e6)
        ignite, reason = should_ignite(score, seconds_since_last_field=1e6)
        assert not ignite
        assert "threshold" in reason

    def test_both_together_ignite(self) -> None:
        score = ignition_score(max_discomfort=1.0, seconds_since_last_field=1e6)
        ignite, _ = should_ignite(score, seconds_since_last_field=1e6)
        assert ignite

    def test_the_score_rises_with_each_term(self) -> None:
        low = ignition_score(max_discomfort=0.2, seconds_since_last_field=600.0)
        more_need = ignition_score(max_discomfort=0.8, seconds_since_last_field=600.0)
        more_silence = ignition_score(max_discomfort=0.2, seconds_since_last_field=6000.0)
        assert more_need > low
        assert more_silence > low


class TestExpiryDeadline:
    def test_an_explicit_deadline_wins(self) -> None:
        deadline = expiry_deadline(
            created_at="2026-07-01T00:00:00Z",
            expires_at="2026-07-01T12:00:00Z",
            ttl_seconds=60.0,
        )
        assert deadline == datetime(2026, 7, 1, 12, tzinfo=timezone.utc)

    def test_creation_plus_ttl_is_the_fallback(self) -> None:
        deadline = expiry_deadline(
            created_at="2026-07-01T00:00:00Z", expires_at=None, ttl_seconds=1800.0
        )
        assert deadline == datetime(2026, 7, 1, 0, 30, tzinfo=timezone.utc)

    def test_an_unreadable_row_has_no_deadline(self) -> None:
        assert expiry_deadline(created_at=None, expires_at=None, ttl_seconds=60.0) is None


def _trajectory(
    field,
    distribution_id: str,
    *,
    probability: float,
    created_at: str,
    action_kind: str = "tool:look_left",
) -> ImaginedTrajectory:
    return ImaginedTrajectory(
        distribution_id=distribution_id,
        field_id=field.field_id,
        tick_id=field.tick_id,
        action_kind=action_kind,
        context_signature="desire|user_prompt|look_outside|neu|mid|" + action_kind,
        horizon=1,
        steps=[
            TrajectoryStep(
                step_index=1,
                predicted_belief=FieldBelief(source_mode=SourceMode.IMAGINED),
                outcome_bucket="ok/short/percept/=",
                conditional_probability=0.5,
                uncertainty=0.5,
                support_observations=0.0,
                basis="prior",
            )
        ],
        probability=probability,
        uncertainty=0.5,
        created_at=created_at,
    )


class TestStep:
    def test_a_first_step_records_a_run(self, social_db, tmp_path) -> None:
        daemon = OrganismDaemon(
            social_db, _producer(social_db, tmp_path), ignition_threshold=2.0
        )
        step = daemon.step()
        assert step.elapsed_seconds == 0.0
        assert daemon.runs.recent()[0]["run_id"] == step.run_id

    def test_elapsed_comes_from_the_previous_run(self, social_db, tmp_path) -> None:
        # The daemon has no process to time itself by, so this is the only
        # thing that makes its clock a clock.
        daemon = OrganismDaemon(
            social_db, _producer(social_db, tmp_path), ignition_threshold=2.0
        )
        start = datetime(2026, 7, 1, 3, tzinfo=timezone.utc)
        daemon.step(now=start)
        second = daemon.step(now=start + timedelta(hours=6))
        assert second.elapsed_seconds == pytest.approx(6 * 3600.0)

    def test_counts_decay_by_elapsed_time(self, social_db, tmp_path) -> None:
        _seed_stat(social_db, 4.0)
        daemon = OrganismDaemon(
            social_db, _producer(social_db, tmp_path), ignition_threshold=2.0
        )
        start = datetime(2026, 7, 1, 3, tzinfo=timezone.utc)
        daemon.step(now=start)
        assert _observations(social_db) == pytest.approx(4.0)
        step = daemon.step(now=start + timedelta(seconds=WEEK_SECONDS))
        assert step.stats_decayed == 1
        assert _observations(social_db) == pytest.approx(2.0)

    def test_counts_stop_at_the_floor_rather_than_vanishing(
        self, social_db, tmp_path
    ) -> None:
        _seed_stat(social_db, MIN_OBSERVATION_COUNT)
        daemon = OrganismDaemon(
            social_db, _producer(social_db, tmp_path), ignition_threshold=2.0
        )
        start = datetime(2026, 7, 1, 3, tzinfo=timezone.utc)
        daemon.step(now=start)
        step = daemon.step(now=start + timedelta(days=365))
        assert step.stats_decayed == 0
        assert _observations(social_db) == pytest.approx(MIN_OBSERVATION_COUNT)

    def test_a_dry_run_changes_nothing(self, social_db, tmp_path) -> None:
        _seed_stat(social_db, 4.0)
        daemon = OrganismDaemon(
            social_db, _producer(social_db, tmp_path), ignition_threshold=2.0
        )
        start = datetime(2026, 7, 1, 3, tzinfo=timezone.utc)
        daemon.step(now=start)
        daemon.step(now=start + timedelta(seconds=WEEK_SECONDS), dry_run=True)
        assert _observations(social_db) == pytest.approx(4.0)
        assert len(daemon.runs.recent()) == 1


class TestProtentionExpiry:
    def test_an_untaken_imagining_expires_and_a_fresh_one_does_not(
        self, social_db, tmp_path
    ) -> None:
        producer = _producer(social_db, tmp_path)
        field = _commit(producer)
        moment = datetime.now(timezone.utc)
        stale = _trajectory(
            field,
            "prot_expiry_test",
            probability=0.6,
            created_at=_iso(moment - timedelta(hours=2)),
        )
        fresh = _trajectory(
            field,
            "prot_expiry_test",
            probability=0.4,
            created_at=_iso(moment),
            action_kind="no_action",
        )
        store = TrajectoryStore(social_db)
        store.create_distribution(
            ProtentionDistribution(
                distribution_id="prot_expiry_test",
                field_id=field.field_id,
                tick_id=field.tick_id,
                trajectories=[stale, fresh],
                entropy=normalized_entropy([0.6, 0.4]),
            )
        )
        daemon = OrganismDaemon(social_db, producer, ignition_threshold=2.0)
        step = daemon.step(now=moment)
        assert step.protentions_expired == 1
        assert store.get_trajectory(stale.trajectory_id).status is (
            TrajectoryStatus.EXPIRED
        )
        assert store.get_trajectory(fresh.trajectory_id).status is (
            TrajectoryStatus.IMAGINED
        )


class TestIgnition:
    def test_need_and_silence_open_a_tick_nobody_asked_for(
        self, social_db, tmp_path
    ) -> None:
        producer = _producer(social_db, tmp_path)
        _commit(producer)
        daemon = OrganismDaemon(social_db, producer)
        step = daemon.step(now=datetime.now(timezone.utc) + timedelta(hours=3))
        assert step.ignited
        assert step.tick_id is not None
        row = social_db.fetchone(
            "SELECT trigger_kind FROM enacted_fields WHERE tick_id = ?",
            (step.tick_id,),
        )
        assert row["trigger_kind"] == TriggerKind.AUTONOMOUS.value

    def test_a_recent_field_holds_ignition_off(self, social_db, tmp_path) -> None:
        producer = _producer(social_db, tmp_path)
        _commit(producer)
        daemon = OrganismDaemon(social_db, producer)
        step = daemon.step()
        assert not step.ignited
        assert step.tick_id is None
        assert "floor" in step.reason

    def test_a_settled_need_holds_ignition_off_through_any_silence(
        self, social_db, tmp_path
    ) -> None:
        producer = _producer(social_db, tmp_path, unmet=False)
        _commit(producer)
        daemon = OrganismDaemon(social_db, producer)
        step = daemon.step(now=datetime.now(timezone.utc) + timedelta(hours=3))
        assert not step.ignited
        assert "threshold" in step.reason
