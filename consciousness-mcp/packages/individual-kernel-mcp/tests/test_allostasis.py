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


class TestQuietHoursAreConfigurable:
    """Night is read from the same env desire-system uses (#135).

    With JST and 0-5 hardcoded, a deployment elsewhere ran its night in the
    local afternoon and nothing said so.
    """

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch) -> None:
        for key in ("DESIRE_TIMEZONE", "DESIRE_NIGHT_START", "DESIRE_NIGHT_END", "DESIRE_DAWN_END"):
            monkeypatch.delenv(key, raising=False)

    def test_default_is_jst_zero_to_five(self) -> None:
        spec = DEFAULT_DESIRE_SPECS["miss_companion"]
        night = datetime(2026, 7, 27, 17, 0, tzinfo=UTC)  # 02:00 JST
        assert allostatic_set_point("miss_companion", spec, night) == pytest.approx(
            spec.set_point - 0.15
        )
        assert DEFAULT_DESIRE_SPECS.keys() >= {"miss_companion"}
        assert "miss_kouta" not in DEFAULT_DESIRE_SPECS

    def test_other_timezone_shifts_the_band(self, monkeypatch) -> None:
        spec = DEFAULT_DESIRE_SPECS["miss_companion"]
        instant = datetime(2026, 7, 27, 8, 0, tzinfo=UTC)  # 17:00 JST, 04:00 New York
        assert allostatic_set_point("miss_companion", spec, instant) == spec.set_point

        monkeypatch.setenv("DESIRE_TIMEZONE", "America/New_York")
        assert allostatic_set_point("miss_companion", spec, instant) == pytest.approx(
            spec.set_point - 0.15
        )

    def test_fixed_offset_and_unknown_name(self, monkeypatch) -> None:
        from individual_kernel_mcp.allostasis import resolve_timezone

        monkeypatch.setenv("DESIRE_TIMEZONE", "-05:00")
        probe = datetime(2026, 7, 27, 0, 0, tzinfo=UTC)
        assert probe.astimezone(resolve_timezone()).utcoffset() == timedelta(hours=-5)

        monkeypatch.setenv("DESIRE_TIMEZONE", "Not/AZone")
        assert probe.astimezone(resolve_timezone()).utcoffset() == timedelta(hours=9)

    def test_wrap_around_night_band(self, monkeypatch) -> None:
        monkeypatch.setenv("DESIRE_TIMEZONE", "+09:00")
        monkeypatch.setenv("DESIRE_NIGHT_START", "22")
        monkeypatch.setenv("DESIRE_NIGHT_END", "5")
        spec = DEFAULT_DESIRE_SPECS["miss_companion"]
        jst = timezone(timedelta(hours=9))

        at_23 = datetime(2026, 7, 27, 23, 0, tzinfo=jst)
        at_03 = datetime(2026, 7, 27, 3, 0, tzinfo=jst)
        at_06 = datetime(2026, 7, 27, 6, 0, tzinfo=jst)
        at_12 = datetime(2026, 7, 27, 12, 0, tzinfo=jst)

        assert allostatic_set_point("miss_companion", spec, at_23) == pytest.approx(
            spec.set_point - 0.15
        )
        assert allostatic_set_point("miss_companion", spec, at_03) == pytest.approx(
            spec.set_point - 0.15
        )
        assert allostatic_set_point("miss_companion", spec, at_06) == pytest.approx(
            spec.set_point - 0.05
        )
        assert allostatic_set_point("miss_companion", spec, at_12) == spec.set_point


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
