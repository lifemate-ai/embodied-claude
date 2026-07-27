"""The model's bucket-weighted estimate of how affect will move.

This is what turns learned transition counts into an appetitive signal: given a
context, how much is valence expected to change, averaging over the outcomes
that context actually produced.
"""

from __future__ import annotations

import pytest
from social_core.db import SocialDB

from individual_kernel_mcp.generative_model import (
    ContextSignature,
    CountBasedGenerativeFieldModel,
)

SIGNATURE = ContextSignature(
    focus_kind="desire",
    trigger_kind="tool_result",
    dominant_desire="identity_coherence",
    valence_bucket="neg",
    arousal_bucket="low",
    action_kind="tool:Write",
)


def _seed(
    db: SocialDB,
    bucket: str,
    observations: float,
    sum_valence_delta: float,
) -> None:
    db.execute(
        "INSERT INTO generative_transition_stats ("
        "owner_id, focus_kind, trigger_kind, dominant_desire, valence_bucket, "
        "arousal_bucket, action_kind, outcome_bucket, observation_count, "
        "sum_valence_delta, sum_latency_ms, sum_prediction_error, "
        "model_version, first_observed_at, updated_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 'count_v1', ?, ?)",
        (
            "self",
            SIGNATURE.focus_kind,
            SIGNATURE.trigger_kind,
            SIGNATURE.dominant_desire,
            SIGNATURE.valence_bucket,
            SIGNATURE.arousal_bucket,
            SIGNATURE.action_kind,
            bucket,
            observations,
            sum_valence_delta,
            "2026-07-27T00:00:00+00:00",
            "2026-07-27T00:00:00+00:00",
        ),
    )


def _model(db: SocialDB) -> CountBasedGenerativeFieldModel:
    return CountBasedGenerativeFieldModel(db)


class TestExpectedValenceDelta:
    def test_no_history_is_neutral(self, social_db: SocialDB) -> None:
        assert _model(social_db).expected_valence_delta(SIGNATURE) == pytest.approx(
            0.0, abs=0.15
        )

    def test_improving_context_is_positive(self, social_db: SocialDB) -> None:
        _seed(social_db, "ok/short/desire/+", observations=20.0, sum_valence_delta=4.0)
        assert _model(social_db).expected_valence_delta(SIGNATURE) > 0.0

    def test_worsening_context_is_negative(self, social_db: SocialDB) -> None:
        _seed(social_db, "ok/short/desire/-", observations=20.0, sum_valence_delta=-4.0)
        assert _model(social_db).expected_valence_delta(SIGNATURE) < 0.0

    def test_mixed_history_is_weighted_by_outcome_probability(
        self, social_db: SocialDB
    ) -> None:
        # Improvement is rare, worsening is common: the estimate leans negative.
        _seed(social_db, "ok/short/desire/+", observations=1.0, sum_valence_delta=0.5)
        _seed(social_db, "ok/short/desire/-", observations=19.0, sum_valence_delta=-3.8)
        assert _model(social_db).expected_valence_delta(SIGNATURE) < 0.0

    def test_estimate_stays_within_unit_range(self, social_db: SocialDB) -> None:
        _seed(social_db, "ok/short/desire/+", observations=2.0, sum_valence_delta=99.0)
        assert _model(social_db).expected_valence_delta(SIGNATURE) <= 1.0
