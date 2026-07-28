"""Staying silent about a channel must not be cheaper than predicting it.

A channel with no declared prediction was scored anyway: 0.25 when the action
succeeded, 0.75 when it failed. A channel that did carry a prediction was
floored at 0.8 on failure. So writing a prediction could only lower the score,
and three of the four channels were a constant that had nothing to do with
anything predicted -- yet they made up three quarters of the mean that feeds
ownership.

That is why an agent using this interface fills one channel and leaves the rest
empty. It is not laziness; it is the scoring rule.

A channel nobody predicted is now simply absent from the vector, and coverage
becomes an explicit term: you cannot claim to have caused an outcome you never
said anything about.
"""

from __future__ import annotations

from social_core.db import SocialDB

from individual_kernel_mcp.agency import (
    AgencyStore,
    ExpectedEffect,
    IntentionRecord,
    IntentionStatus,
    PredictedEffects,
)


def _intention(**summaries: str) -> IntentionRecord:
    return IntentionRecord(
        action_id="act_test",
        owner_id="self",
        field_id="field_test",
        tick_id="tick_test",
        tool_name="Bash",
        tool_input_hash="hash",
        normalized_tool_input={},
        intended_goal="check the scoring rule",
        predicted_effects=PredictedEffects(
            **{
                channel: ExpectedEffect(summary=text)
                for channel, text in summaries.items()
            }
        ),
        expected_latency_ms=100,
        confidence=0.8,
        status=IntentionStatus.PENDING,
        created_at="2026-07-28T00:00:00+00:00",
    )


class TestAnUnpredictedChannelIsNotScored:
    def test_only_predicted_channels_appear(self) -> None:
        vector = AgencyStore._mismatch_vector(
            intention=_intention(exteroceptive="the file is written"),
            actual_result_summary="the file is written",
            success=True,
            latency_ms=100,
        )

        assert "exteroceptive" in vector
        assert "interoceptive" not in vector
        assert "social" not in vector
        assert "mnemonic" not in vector

    def test_latency_is_scored_without_being_declared(self) -> None:
        # Latency is measured either way, so it needs no declared prediction.
        vector = AgencyStore._mismatch_vector(
            intention=_intention(),
            actual_result_summary="anything",
            success=True,
            latency_ms=100,
        )

        assert set(vector) == {"latency"}

    def test_a_failure_invents_no_verdict_for_a_silent_channel(self) -> None:
        # This is the 0.75 constant that used to fill three channels.
        vector = AgencyStore._mismatch_vector(
            intention=_intention(exteroceptive="the file is written"),
            actual_result_summary="permission denied",
            success=False,
            latency_ms=100,
        )

        assert set(vector) == {"exteroceptive", "latency"}


class TestSilenceCannotBuyAgency:
    def test_an_accurate_prediction_beats_saying_nothing(
        self, social_db: SocialDB
    ) -> None:
        store = AgencyStore(social_db)
        common = dict(
            registered_intention=True,
            latency_ms=100,
            expected_latency_ms=100,
            exclusive_causal_fit=1.0,
        )

        silent = store.assess(
            mismatch_vector={"latency": 0.0}, prediction_coverage=0.0, **common
        )
        accurate = store.assess(
            mismatch_vector={"exteroceptive": 0.0, "latency": 0.0},
            prediction_coverage=0.25,
            **common,
        )

        assert accurate.ownership_score > silent.ownership_score

    def test_predicting_every_channel_beats_predicting_one(
        self, social_db: SocialDB
    ) -> None:
        store = AgencyStore(social_db)
        common = dict(
            registered_intention=True,
            mismatch_vector={"exteroceptive": 0.0, "latency": 0.0},
            latency_ms=100,
            expected_latency_ms=100,
            exclusive_causal_fit=1.0,
        )

        narrow = store.assess(prediction_coverage=0.25, **common)
        broad = store.assess(prediction_coverage=1.0, **common)

        assert broad.ownership_score > narrow.ownership_score

    def test_the_coverage_is_reported(self, social_db: SocialDB) -> None:
        # Whoever reads the assessment can see how much of it was actually
        # claimed in advance, rather than inferring it from a missing key.
        assessment = AgencyStore(social_db).assess(
            registered_intention=True,
            mismatch_vector={"exteroceptive": 0.2, "latency": 0.0},
            prediction_coverage=0.5,
            latency_ms=100,
            expected_latency_ms=100,
            exclusive_causal_fit=1.0,
        )

        assert assessment.prediction_coverage == 0.5
