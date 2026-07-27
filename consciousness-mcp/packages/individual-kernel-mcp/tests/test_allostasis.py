"""Desire projection under a stalled writer, and valence composition.

The desire file is produced by an external updater that can stop running. These
tests pin the contract that the kernel re-derives current levels from the
recorded snapshot plus elapsed time, so a stalled writer degrades gracefully
instead of freezing the body state.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from individual_kernel_mcp.allostasis import (
    DEFAULT_DESIRE_SPECS,
    allostatic_set_point,
    extrapolate_desire_level,
    project_desires,
)

UTC = timezone.utc
T0 = datetime(2026, 7, 27, 3, 0, tzinfo=UTC)


class TestExtrapolateDesireLevel:
    def test_level_grows_with_elapsed_time(self) -> None:
        level = extrapolate_desire_level(0.25, T0, 4.0, T0 + timedelta(hours=2))
        assert level == pytest.approx(0.75)

    def test_saturated_level_stays_saturated(self) -> None:
        level = extrapolate_desire_level(1.0, T0, 2.0, T0 + timedelta(days=30))
        assert level == 1.0

    def test_long_stall_saturates_instead_of_overflowing(self) -> None:
        level = extrapolate_desire_level(0.4, T0, 1.0, T0 + timedelta(days=70))
        assert level == 1.0

    def test_clock_skew_backwards_does_not_reduce_below_recorded(self) -> None:
        level = extrapolate_desire_level(0.4, T0, 1.0, T0 - timedelta(hours=5))
        assert level == pytest.approx(0.4)

    def test_zero_cycle_is_treated_as_saturated(self) -> None:
        assert extrapolate_desire_level(0.4, T0, 0.0, T0) == 1.0


class TestAllostaticSetPoint:
    def test_identity_coherence_is_time_invariant(self) -> None:
        spec = DEFAULT_DESIRE_SPECS["identity_coherence"]
        night = datetime(2026, 7, 27, 17, 0, tzinfo=UTC)  # 02:00 JST
        noon = datetime(2026, 7, 27, 3, 0, tzinfo=UTC)  # 12:00 JST
        assert allostatic_set_point("identity_coherence", spec, night) == spec.set_point
        assert allostatic_set_point("identity_coherence", spec, noon) == spec.set_point

    def test_companion_set_point_drops_at_night(self) -> None:
        spec = DEFAULT_DESIRE_SPECS["miss_companion"]
        night = datetime(2026, 7, 27, 17, 0, tzinfo=UTC)  # 02:00 JST
        noon = datetime(2026, 7, 27, 3, 0, tzinfo=UTC)  # 12:00 JST
        assert allostatic_set_point("miss_companion", spec, night) < allostatic_set_point(
            "miss_companion", spec, noon
        )

    def test_unknown_desire_falls_back_to_its_own_set_point(self) -> None:
        spec = DEFAULT_DESIRE_SPECS["browse_curiosity"]
        assert allostatic_set_point("not_a_real_desire", spec, T0) == spec.set_point


class TestProjectDesires:
    def test_stalled_snapshot_still_moves(self) -> None:
        """The live failure mode: the writer stopped months ago."""
        snapshot = {
            "updated_at": "2026-05-18T02:04:32+00:00",
            "desires": {"browse_curiosity": 0.2, "identity_coherence": 0.4},
            "discomforts": {"identity_coherence": 0.5},
        }
        projected = project_desires(snapshot, now=T0)
        assert projected.desires["browse_curiosity"] > 0.2
        assert projected.stale_seconds > 30 * 24 * 3600

    def test_dominant_is_the_largest_discomfort(self) -> None:
        snapshot = {
            "updated_at": T0.isoformat(),
            "desires": {"browse_curiosity": 0.3, "identity_coherence": 0.1},
        }
        projected = project_desires(snapshot, now=T0)
        # identity_coherence sits far below its 0.9 set point, so it dominates.
        assert projected.dominant == "identity_coherence"

    def test_discomfort_is_distance_from_set_point(self) -> None:
        snapshot = {
            "updated_at": T0.isoformat(),
            "desires": {"identity_coherence": 0.9},
        }
        projected = project_desires(snapshot, now=T0)
        assert projected.discomforts["identity_coherence"] == pytest.approx(0.0)

    def test_missing_file_yields_empty_projection(self) -> None:
        projected = project_desires({}, now=T0)
        assert projected.desires == {}
        assert projected.dominant is None

    def test_unparsable_timestamp_is_treated_as_no_elapsed_time(self) -> None:
        snapshot = {"updated_at": "not-a-date", "desires": {"browse_curiosity": 0.2}}
        projected = project_desires(snapshot, now=T0)
        assert projected.desires["browse_curiosity"] == pytest.approx(0.2)
        assert projected.stale_seconds == 0.0
