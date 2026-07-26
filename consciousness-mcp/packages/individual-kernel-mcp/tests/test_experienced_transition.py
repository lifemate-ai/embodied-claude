"""Experienced transitions: idempotent recording and exactly-once learning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from social_core.db import SocialDB

from individual_kernel_mcp.enacted_field import TriggerKind
from individual_kernel_mcp.experienced_transition import (
    ExperiencedTransition,
    ExperiencedTransitionStore,
)
from individual_kernel_mcp.generative_model import CountBasedGenerativeFieldModel
from individual_kernel_mcp.tick import TickProducer
from individual_kernel_mcp.workspace import (
    CandidateKind,
    CandidateSource,
    SourceMode,
    WorkspaceCandidate,
)

_SIGNATURE = "desire|user_prompt|identity_coherence|neu|mid|no_action"


@pytest.fixture(autouse=True)
def _generative_flag_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate store/model unit tests from the automatic commit-time loop."""
    toml = tmp_path / "behavior-off.toml"
    toml.write_text("[individual-kernel]\ngenerative_field_model = false\n")
    monkeypatch.setenv("MCP_BEHAVIOR_TOML", str(toml))


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


def _transition(previous_field, next_field, **overrides) -> ExperiencedTransition:
    data: dict = {
        "owner_id": "self",
        "previous_field_id": previous_field.field_id if previous_field else None,
        "next_field_id": next_field.field_id,
        "previous_tick_id": previous_field.tick_id if previous_field else None,
        "next_tick_id": next_field.tick_id,
        "context_signature": _SIGNATURE,
        "action_kind": "no_action",
        "outcome_bucket": "na/na/desire/=",
        "prediction_errors": {"probability_of_observed": 0.5},
        "mean_prediction_error": 0.5,
        "valence_before": -0.1,
        "valence_after": -0.1,
        "valence_change": 0.0,
        "arousal_before": 0.5,
        "arousal_after": 0.5,
        "agency_confidence": 0.5,
        "ownership_confidence": 0.5,
        "source_mode": SourceMode.INFERRED,
    }
    data.update(overrides)
    return ExperiencedTransition(**data)


class TestSourceDiscipline:
    def test_imagined_and_remembered_sources_are_rejected(
        self, social_db, tmp_path
    ) -> None:
        producer = _producer(social_db, tmp_path)
        field = _commit(producer)
        for mode in (SourceMode.IMAGINED, SourceMode.REMEMBERED):
            with pytest.raises(ValidationError):
                _transition(None, field, source_mode=mode)

    def test_knowledge_source_is_experienced_only_in_v1(
        self, social_db, tmp_path
    ) -> None:
        producer = _producer(social_db, tmp_path)
        field = _commit(producer)
        for value in ("told", "imagined", "replayed"):
            with pytest.raises(ValidationError):
                _transition(None, field, knowledge_source=value)


class TestIdempotentRecording:
    def test_round_trip(self, social_db, tmp_path) -> None:
        producer = _producer(social_db, tmp_path)
        previous = _commit(producer)
        following = _commit(producer)
        store = ExperiencedTransitionStore(social_db)
        transition, created = store.record(_transition(previous, following))
        assert created is True
        loaded = store.get(transition.transition_id)
        assert loaded == transition

    def test_double_record_for_same_next_field_returns_existing(
        self, social_db, tmp_path
    ) -> None:
        producer = _producer(social_db, tmp_path)
        previous = _commit(producer)
        following = _commit(producer)
        store = ExperiencedTransitionStore(social_db)
        first, created_first = store.record(_transition(previous, following))
        second, created_second = store.record(_transition(previous, following))
        assert created_first is True
        assert created_second is False
        assert second.transition_id == first.transition_id
        count = social_db.fetchone(
            "SELECT COUNT(*) AS n FROM experienced_transitions WHERE next_field_id = ?",
            (following.field_id,),
        )
        assert count["n"] == 1

    def test_trajectory_link_is_set_once(self, social_db, tmp_path) -> None:
        producer = _producer(social_db, tmp_path)
        previous = _commit(producer)
        following = _commit(producer)
        store = ExperiencedTransitionStore(social_db)
        transition, _ = store.record(_transition(previous, following))
        assert (
            store.link_trajectory(
                transition.transition_id,
                intended_trajectory_id=None,
                distribution_id="prot_a",
            )
            is True
        )
        assert (
            store.link_trajectory(
                transition.transition_id,
                intended_trajectory_id=None,
                distribution_id="prot_b",
            )
            is False
        )
        loaded = store.get(transition.transition_id)
        assert loaded is not None
        assert loaded.distribution_id == "prot_a"


class TestExactlyOnceLearning:
    def test_update_applies_once_within_one_connection(
        self, social_db, tmp_path
    ) -> None:
        producer = _producer(social_db, tmp_path)
        previous = _commit(producer)
        following = _commit(producer)
        store = ExperiencedTransitionStore(social_db)
        transition, _ = store.record(_transition(previous, following))
        model = CountBasedGenerativeFieldModel(social_db)
        assert model.update(transition) is True
        assert model.update(transition) is False
        row = social_db.fetchone(
            "SELECT observation_count FROM generative_transition_stats "
            "WHERE action_kind = 'no_action' AND outcome_bucket = 'na/na/desire/='",
        )
        assert row is not None
        assert row["observation_count"] == pytest.approx(1.0)

    def test_update_applies_once_across_connections(
        self, temp_db_path: Path, tmp_path: Path
    ) -> None:
        first = SocialDB(temp_db_path)
        first.connect()
        producer = _producer(first, tmp_path)
        previous = _commit(producer)
        following = _commit(producer)
        store = ExperiencedTransitionStore(first)
        transition, _ = store.record(_transition(previous, following))
        assert CountBasedGenerativeFieldModel(first).update(transition) is True
        first.close()

        second = SocialDB(temp_db_path)
        second.connect()
        try:
            loaded = ExperiencedTransitionStore(second).get(transition.transition_id)
            assert loaded is not None
            assert CountBasedGenerativeFieldModel(second).update(loaded) is False
            row = second.fetchone(
                "SELECT observation_count FROM generative_transition_stats "
                "WHERE action_kind = 'no_action' AND outcome_bucket = 'na/na/desire/='",
            )
            assert row is not None
            assert row["observation_count"] == pytest.approx(1.0)
        finally:
            second.close()
