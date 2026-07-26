"""Count-based generative field model: signatures, smoothing, backoff, rollout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from social_core.db import SocialDB

from individual_kernel_mcp.enacted_field import TriggerKind
from individual_kernel_mcp.generative_model import (
    DIRICHLET_ALPHA,
    NO_ACTION,
    ContextSignature,
    CountBasedGenerativeFieldModel,
    arousal_bucket_of,
    dominant_desire_of,
    focus_kind_of,
    latency_bucket_of,
    valence_bucket_of,
    valence_delta_facet_of,
)
from individual_kernel_mcp.tick import TickProducer
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


def _seed_stats(
    db: SocialDB,
    *,
    signature: ContextSignature,
    bucket: str,
    count: float,
    sum_valence_delta: float = 0.0,
) -> None:
    db.execute(
        """
        INSERT INTO generative_transition_stats(
            owner_id, focus_kind, trigger_kind, dominant_desire, valence_bucket,
            arousal_bucket, action_kind, outcome_bucket, observation_count,
            sum_valence_delta, sum_latency_ms, sum_prediction_error,
            last_transition_id, model_version, first_observed_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.0, 0.0, NULL, 'count_v1',
                  '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')
        """,
        (
            "self",
            signature.focus_kind,
            signature.trigger_kind,
            signature.dominant_desire,
            signature.valence_bucket,
            signature.arousal_bucket,
            signature.action_kind,
            bucket,
            count,
            sum_valence_delta,
        ),
    )


_SIG = ContextSignature(
    focus_kind="desire",
    trigger_kind="user_prompt",
    dominant_desire="identity_coherence",
    valence_bucket="neu",
    arousal_bucket="mid",
    action_kind="tool:Bash",
)


class TestBucketBoundaries:
    def test_valence_deadband_edges(self) -> None:
        assert valence_bucket_of(-0.150) == "neu"
        assert valence_bucket_of(-0.151) == "neg"
        assert valence_bucket_of(0.150) == "neu"
        assert valence_bucket_of(0.151) == "pos"

    def test_arousal_edges(self) -> None:
        assert arousal_bucket_of(0.329) == "low"
        assert arousal_bucket_of(0.33) == "mid"
        assert arousal_bucket_of(0.66) == "mid"
        assert arousal_bucket_of(0.661) == "high"

    def test_latency_edges(self) -> None:
        assert latency_bucket_of(None) == "na"
        assert latency_bucket_of(999) == "short"
        assert latency_bucket_of(1000) == "mid"
        assert latency_bucket_of(9999) == "mid"
        assert latency_bucket_of(10000) == "long"

    def test_valence_delta_facets(self) -> None:
        assert valence_delta_facet_of(-0.051) == "-"
        assert valence_delta_facet_of(0.0) == "="
        assert valence_delta_facet_of(0.051) == "+"

    def test_focus_kind_prefix(self) -> None:
        assert focus_kind_of("desire:identity_coherence") == "desire"
        assert focus_kind_of(None) == "none"

    def test_dominant_desire_lexical_tiebreak(self) -> None:
        assert dominant_desire_of({"b": 0.5, "a": 0.5}) == "a"
        assert dominant_desire_of({}) == ""


class TestContextSignature:
    def test_from_committed_field_is_deterministic(self, social_db, tmp_path) -> None:
        producer = _producer(social_db, tmp_path)
        field = _commit(producer)
        model = CountBasedGenerativeFieldModel(social_db)
        first = model.infer(field, action_kind="tool:Bash")
        second = model.infer(field, action_kind="tool:Bash")
        assert first == second
        assert first.focus_kind == "desire"
        assert first.trigger_kind == "user_prompt"
        assert first.dominant_desire == "identity_coherence"
        assert first.valence_bucket == "neu"
        assert first.arousal_bucket == "mid"

    def test_key_round_trips(self) -> None:
        assert ContextSignature.from_key(_SIG.key()) == _SIG


class TestDirichletSmoothing:
    def test_hand_computed_probabilities(self, social_db) -> None:
        _seed_stats(social_db, signature=_SIG, bucket="ok/short/tool_result/=", count=3)
        _seed_stats(social_db, signature=_SIG, bucket="fail/short/tool_result/=", count=1)
        model = CountBasedGenerativeFieldModel(social_db)
        forecast = model.forecast(_SIG)
        # K = 2 observed + 1 novel = 3; N = 4; alpha = 0.5
        assert forecast.basis == "count"
        assert forecast.predictions[0].bucket == "ok/short/tool_result/="
        assert forecast.predictions[0].probability == pytest.approx(3.5 / 5.5)
        assert forecast.predictions[1].probability == pytest.approx(1.5 / 5.5)
        assert forecast.novel_probability == pytest.approx(0.5 / 5.5)
        assert forecast.uncertainty == pytest.approx(1.5 / 5.5)

    def test_probability_mass_sums_to_one(self, social_db) -> None:
        _seed_stats(social_db, signature=_SIG, bucket="ok/short/tool_result/=", count=3)
        _seed_stats(social_db, signature=_SIG, bucket="fail/short/tool_result/=", count=1)
        _seed_stats(social_db, signature=_SIG, bucket="ok/mid/tool_result/+", count=2)
        model = CountBasedGenerativeFieldModel(social_db)
        forecast = model.forecast(_SIG)
        total = sum(p.probability for p in forecast.predictions) + forecast.novel_probability
        assert total == pytest.approx(1.0)

    def test_probability_of_unseen_bucket_equals_novel_mass(self, social_db) -> None:
        _seed_stats(social_db, signature=_SIG, bucket="ok/short/tool_result/=", count=3)
        model = CountBasedGenerativeFieldModel(social_db)
        probability, basis, uncertainty = model.probability_of(_SIG, "fail/long/none/-")
        # K = 1 observed + 1 novel = 2; N = 3
        assert probability == pytest.approx(0.5 / 4.0)
        assert basis == "count"
        assert uncertainty == pytest.approx(1.0 / 4.0)

    def test_uncertainty_decreases_with_observations(self, social_db) -> None:
        model = CountBasedGenerativeFieldModel(social_db)
        empty = model.forecast(_SIG)
        assert empty.uncertainty == 1.0
        _seed_stats(social_db, signature=_SIG, bucket="ok/short/tool_result/=", count=1)
        one = model.forecast(_SIG)
        _seed_stats(social_db, signature=_SIG, bucket="ok/mid/tool_result/=", count=9)
        many = model.forecast(_SIG)
        assert 0.0 < many.uncertainty < one.uncertainty < 1.0


class TestBackoffChain:
    def test_backoff_levels_in_declared_order(self, social_db) -> None:
        seeded = _SIG.model_copy(update={"dominant_desire": "browse_curiosity"})
        _seed_stats(social_db, signature=seeded, bucket="ok/short/tool_result/=", count=2)
        model = CountBasedGenerativeFieldModel(social_db)
        exact_missing = model.forecast(_SIG)
        assert exact_missing.basis == "backoff:1"

        far = _SIG.model_copy(
            update={"dominant_desire": "x", "arousal_bucket": "high", "valence_bucket": "neg"}
        )
        assert model.forecast(far).basis == "backoff:3"

        other_action = _SIG.model_copy(update={"action_kind": "tool:Write"})
        assert model.forecast(other_action).basis == "prior"

    def test_action_only_backoff_level(self, social_db) -> None:
        seeded = _SIG.model_copy(
            update={
                "focus_kind": "event",
                "trigger_kind": "tool_result",
                "dominant_desire": "x",
                "valence_bucket": "neg",
                "arousal_bucket": "high",
            }
        )
        _seed_stats(social_db, signature=seeded, bucket="ok/short/tool_result/=", count=2)
        model = CountBasedGenerativeFieldModel(social_db)
        assert model.forecast(_SIG).basis == "backoff:4"


class TestRollout:
    def test_rejects_out_of_range_horizon(self, social_db, tmp_path) -> None:
        producer = _producer(social_db, tmp_path)
        field = _commit(producer)
        model = CountBasedGenerativeFieldModel(social_db)
        with pytest.raises(ValueError):
            model.rollout(field, action_kind="tool:Bash", horizon=0)
        with pytest.raises(ValueError):
            model.rollout(field, action_kind="tool:Bash", horizon=6)

    def test_horizon_chain_produces_imagined_steps(self, social_db, tmp_path) -> None:
        producer = _producer(social_db, tmp_path)
        field = _commit(producer)
        model = CountBasedGenerativeFieldModel(social_db)
        steps = model.rollout(field, action_kind="tool:Bash", horizon=3)
        assert [step.step_index for step in steps] == [1, 2, 3]
        for step in steps:
            assert step.predicted_belief.source_mode is SourceMode.IMAGINED
            assert 0.0 <= step.conditional_probability <= 1.0
            assert 0.0 <= step.uncertainty <= 1.0

    def test_first_bucket_override_selects_branch(self, social_db, tmp_path) -> None:
        producer = _producer(social_db, tmp_path)
        field = _commit(producer)
        model = CountBasedGenerativeFieldModel(social_db)
        signature = model.infer(field, action_kind="tool:Bash")
        _seed_stats(social_db, signature=signature, bucket="ok/short/tool_result/=", count=3)
        _seed_stats(social_db, signature=signature, bucket="fail/short/tool_result/-", count=1)
        steps = model.rollout(
            field,
            action_kind="tool:Bash",
            horizon=1,
            first_bucket="fail/short/tool_result/-",
        )
        assert steps[0].outcome_bucket == "fail/short/tool_result/-"

    def test_predictions_persist_across_reconnect(self, temp_db_path: Path) -> None:
        first = SocialDB(temp_db_path)
        first.connect()
        _seed_stats(first, signature=_SIG, bucket="ok/short/tool_result/=", count=3)
        before = CountBasedGenerativeFieldModel(first).forecast(_SIG)
        first.close()

        second = SocialDB(temp_db_path)
        second.connect()
        try:
            after = CountBasedGenerativeFieldModel(second).forecast(_SIG)
            assert after == before
        finally:
            second.close()

    def test_no_action_constant_exported(self) -> None:
        assert NO_ACTION == "no_action"
        assert DIRICHLET_ALPHA == 0.5
