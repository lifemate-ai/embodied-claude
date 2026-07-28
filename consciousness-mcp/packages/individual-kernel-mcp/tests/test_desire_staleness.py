"""A snapshot old enough to say nothing must not be read as saying everything.

Desire levels grow linearly toward 1.0 over `satisfaction_hours`, the longest of
which is three. Extrapolating a snapshot therefore saturates every need within
hours, and keeps returning the saturated answer forever. That is correct for a
few hours of genuine neglect and wrong for a writer that stopped: the live
snapshot went unwritten for 71 days and every field committed in that time
carried a maximal need vector, which is indistinguishable from an agent that
really had not looked outside since May.

Past a bound, the snapshot is an artifact rather than a record, and the honest
reading is the resting state plus a note that the input was unusable.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from individual_kernel_mcp.allostasis import (
    DEFAULT_DESIRE_SPECS,
    MAX_SNAPSHOT_AGE_HOURS,
    allostatic_set_point,
    project_desires,
)

NOW = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)


def _snapshot(age: timedelta) -> dict:
    return {
        "updated_at": (NOW - age).isoformat(),
        "desires": {
            "look_outside": 0.018,
            "browse_curiosity": 1.0,
            "miss_kouta": 1.0,
            "observe_room": 1.0,
            "identity_coherence": 0.4,
        },
    }


class TestAFreshSnapshotIsUnaffected:
    def test_recent_levels_still_extrapolate(self) -> None:
        projection = project_desires(_snapshot(timedelta(minutes=30)), now=NOW)

        # 0.018 + 0.5h / 1.0h
        assert projection.desires["look_outside"] == 0.518
        assert projection.usable is True

    def test_hours_of_real_neglect_still_saturate(self) -> None:
        # Being genuinely unattended for a few hours is what saturation is for.
        projection = project_desires(_snapshot(timedelta(hours=6)), now=NOW)

        assert projection.desires["look_outside"] == 1.0
        assert projection.usable is True
        assert projection.dominant == "observe_room"


class TestAnAbandonedSnapshotIsRefused:
    def test_a_snapshot_older_than_the_bound_falls_back_to_rest(self) -> None:
        projection = project_desires(_snapshot(timedelta(days=71)), now=NOW)

        assert projection.usable is False
        for name, spec in DEFAULT_DESIRE_SPECS.items():
            if name not in projection.desires:
                continue
            assert projection.desires[name] == allostatic_set_point(name, spec, NOW)

    def test_an_unusable_snapshot_reports_no_discomfort(self) -> None:
        # The daemon fires on unmet need. A snapshot that cannot say what is unmet
        # must not be able to answer that question either.
        projection = project_desires(_snapshot(timedelta(days=71)), now=NOW)

        assert projection.mean_discomfort == 0.0
        assert projection.dominant is None

    def test_the_age_is_still_reported(self) -> None:
        # Refusing to extrapolate is not the same as pretending it is current.
        projection = project_desires(_snapshot(timedelta(days=71)), now=NOW)

        assert projection.stale_seconds == 71 * 24 * 3600


class TestTheBoundary:
    def test_exactly_at_the_bound_is_still_usable(self) -> None:
        projection = project_desires(
            _snapshot(timedelta(hours=MAX_SNAPSHOT_AGE_HOURS)), now=NOW
        )

        assert projection.usable is True

    def test_just_past_the_bound_is_not(self) -> None:
        projection = project_desires(
            _snapshot(timedelta(hours=MAX_SNAPSHOT_AGE_HOURS, seconds=1)), now=NOW
        )

        assert projection.usable is False


class TestASnapshotWithoutATimestamp:
    def test_it_is_treated_as_just_written(self) -> None:
        # Unchanged behaviour: an absent timestamp already meant "use as-is"
        # rather than "extrapolate from a guessed instant".
        projection = project_desires({"desires": {"look_outside": 0.4}}, now=NOW)

        assert projection.desires["look_outside"] == 0.4
        assert projection.usable is True
