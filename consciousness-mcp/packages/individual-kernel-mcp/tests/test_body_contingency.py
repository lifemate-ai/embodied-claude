"""Reafference scoring: the commanded body change against the observed one."""

from __future__ import annotations

import pytest

from individual_kernel_mcp.body_contingency import (
    BodyObservation,
    BodyPose,
    ContingencyVerdict,
    commanded_delta_from_tool,
    evaluate_reafference,
)

EXPECTED_LATENCY_MS = 1000


def _observation(
    *,
    pan_before: float = 0.0,
    tilt_before: float = 0.0,
    pan_after: float = 0.0,
    tilt_after: float = 0.0,
    latency_ms: int | None = EXPECTED_LATENCY_MS,
) -> BodyObservation:
    return BodyObservation(
        before=BodyPose(pan=pan_before, tilt=tilt_before),
        after=BodyPose(pan=pan_after, tilt=tilt_after),
        observed_latency_ms=latency_ms,
    )


class TestMatchedMovement:
    def test_commanded_and_observed_agree_scores_high(self):
        verdict = evaluate_reafference(
            commanded_delta={"pan": 30.0, "tilt": 0.0},
            observation=_observation(pan_after=30.0),
            expected_latency_ms=EXPECTED_LATENCY_MS,
        )
        assert verdict.verdict is ContingencyVerdict.SELF_CAUSED
        assert verdict.score > 0.9
        assert verdict.magnitude_ratio == pytest.approx(1.0)

    def test_slightly_short_movement_still_self_caused(self):
        verdict = evaluate_reafference(
            commanded_delta={"pan": 30.0, "tilt": 0.0},
            observation=_observation(pan_after=24.0),
            expected_latency_ms=EXPECTED_LATENCY_MS,
        )
        assert verdict.verdict is ContingencyVerdict.SELF_CAUSED
        # 20% short is inside tolerance, so magnitude is not penalised at all.
        assert verdict.magnitude_score == pytest.approx(1.0)

    def test_overshoot_beyond_tolerance_lowers_the_score(self):
        tight = evaluate_reafference(
            commanded_delta={"pan": 30.0, "tilt": 0.0},
            observation=_observation(pan_after=30.0),
            expected_latency_ms=EXPECTED_LATENCY_MS,
        )
        loose = evaluate_reafference(
            commanded_delta={"pan": 30.0, "tilt": 0.0},
            observation=_observation(pan_after=55.0),
            expected_latency_ms=EXPECTED_LATENCY_MS,
        )
        assert loose.verdict is ContingencyVerdict.SELF_CAUSED
        assert loose.score < tight.score


class TestDelayedMovement:
    def test_late_arrival_lowers_the_score_but_keeps_the_verdict(self):
        prompt = evaluate_reafference(
            commanded_delta={"pan": 30.0, "tilt": 0.0},
            observation=_observation(pan_after=30.0, latency_ms=1000),
            expected_latency_ms=EXPECTED_LATENCY_MS,
        )
        late = evaluate_reafference(
            commanded_delta={"pan": 30.0, "tilt": 0.0},
            observation=_observation(pan_after=30.0, latency_ms=20000),
            expected_latency_ms=EXPECTED_LATENCY_MS,
        )
        assert late.verdict is ContingencyVerdict.SELF_CAUSED
        assert late.timing_score < prompt.timing_score
        assert late.score < prompt.score


class TestFalseMovement:
    def test_movement_with_nothing_commanded_is_externally_caused(self):
        verdict = evaluate_reafference(
            commanded_delta={"pan": 0.0, "tilt": 0.0},
            observation=_observation(pan_after=25.0),
            expected_latency_ms=EXPECTED_LATENCY_MS,
        )
        assert verdict.verdict is ContingencyVerdict.EXTERNALLY_CAUSED
        assert verdict.score <= 0.1

    def test_command_with_no_movement_is_unresponsive(self):
        verdict = evaluate_reafference(
            commanded_delta={"pan": 30.0, "tilt": 0.0},
            observation=_observation(pan_after=0.2),
            expected_latency_ms=EXPECTED_LATENCY_MS,
        )
        assert verdict.verdict is ContingencyVerdict.UNRESPONSIVE
        assert verdict.score <= 0.15

    def test_still_body_with_no_command_carries_no_evidence(self):
        verdict = evaluate_reafference(
            commanded_delta={"pan": 0.0, "tilt": 0.0},
            observation=_observation(),
            expected_latency_ms=EXPECTED_LATENCY_MS,
        )
        assert verdict.verdict is ContingencyVerdict.NO_CHANGE
        assert verdict.score == pytest.approx(0.5)


class TestInvertedMovement:
    def test_opposite_direction_scores_at_the_floor(self):
        verdict = evaluate_reafference(
            commanded_delta={"pan": 30.0, "tilt": 0.0},
            observation=_observation(pan_after=-30.0),
            expected_latency_ms=EXPECTED_LATENCY_MS,
        )
        assert verdict.verdict is ContingencyVerdict.INVERTED
        assert verdict.score <= 0.1

    def test_inverted_scores_below_a_perfectly_matched_move(self):
        matched = evaluate_reafference(
            commanded_delta={"pan": 30.0, "tilt": 0.0},
            observation=_observation(pan_after=30.0),
            expected_latency_ms=EXPECTED_LATENCY_MS,
        )
        inverted = evaluate_reafference(
            commanded_delta={"pan": 30.0, "tilt": 0.0},
            observation=_observation(pan_after=-30.0),
            expected_latency_ms=EXPECTED_LATENCY_MS,
        )
        assert inverted.score < matched.score


class TestUnverified:
    def test_missing_observation_is_unverified(self):
        verdict = evaluate_reafference(
            commanded_delta={"pan": 30.0, "tilt": 0.0},
            observation=None,
            expected_latency_ms=EXPECTED_LATENCY_MS,
        )
        assert verdict.verdict is ContingencyVerdict.UNVERIFIED

    def test_missing_command_is_unverified(self):
        verdict = evaluate_reafference(
            commanded_delta=None,
            observation=_observation(pan_after=30.0),
            expected_latency_ms=EXPECTED_LATENCY_MS,
        )
        assert verdict.verdict is ContingencyVerdict.UNVERIFIED


class TestCommandedDeltaFromTool:
    def test_reads_direction_and_amount(self):
        assert commanded_delta_from_tool("look_left", {"degrees": 45}) == {
            "pan": -45.0,
            "tilt": 0.0,
        }
        assert commanded_delta_from_tool("look_up", {"degrees": 20}) == {
            "pan": 0.0,
            "tilt": 20.0,
        }

    def test_omitted_degrees_falls_back_to_the_tool_default(self):
        assert commanded_delta_from_tool("look_right", {}) == {"pan": 30.0, "tilt": 0.0}
        assert commanded_delta_from_tool("look_down", {}) == {"pan": 0.0, "tilt": -20.0}

    def test_non_body_tool_returns_none(self):
        assert commanded_delta_from_tool("Bash", {"command": "ls"}) is None
        assert commanded_delta_from_tool("see", {}) is None
