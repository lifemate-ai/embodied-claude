"""Tests for timezone-aware policy time helpers."""

from social_core.time import in_quiet_hours, local_view


def test_local_view_converts_utc_to_jst():
    local = local_view("2026-04-18T16:30:00Z", "Asia/Tokyo")
    assert local.hour == 1
    assert local.minute == 30
    # JST calendar day rolls forward.
    assert local.day == 19


def test_local_view_falls_back_to_utc_on_bad_zone():
    local = local_view("2026-04-18T16:30:00Z", "Not/A_Zone")
    assert local.hour == 16
    assert local.utcoffset().total_seconds() == 0


def test_in_quiet_hours_tokyo_acceptance_within_window():
    # Spec §8.2: 2026-04-18T16:30:00Z -> 2026-04-19 01:30 JST -> quiet
    active, until = in_quiet_hours(
        "2026-04-18T16:30:00Z", ["00:00-07:00"], "Asia/Tokyo"
    )
    assert active is True
    assert until is not None
    assert until.endswith("+09:00") or "07:00:00" in until


def test_in_quiet_hours_tokyo_acceptance_outside_window():
    # Spec §8.2: 2026-04-18T23:30:00Z -> 2026-04-19 08:30 JST -> not quiet
    active, until = in_quiet_hours(
        "2026-04-18T23:30:00Z", ["00:00-07:00"], "Asia/Tokyo"
    )
    assert active is False
    assert until is None


def test_in_quiet_hours_crosses_midnight_window():
    # 22:00 JST falls inside a 23:00-06:00 window (wraps midnight) when it is 00:30 JST.
    # 2026-04-19 15:30 UTC -> 2026-04-20 00:30 JST -> quiet under 23:00-06:00.
    active, _ = in_quiet_hours(
        "2026-04-19T15:30:00Z", ["23:00-06:00"], "Asia/Tokyo"
    )
    assert active is True


def test_in_quiet_hours_empty_windows():
    active, until = in_quiet_hours("2026-04-18T16:30:00Z", [], "Asia/Tokyo")
    assert active is False
    assert until is None


def test_in_quiet_hours_defaults_to_utc():
    # Without an explicit tz the helper treats the timestamp as-is (UTC-equivalent),
    # so 03:00 UTC sits inside 00:00-07:00.
    active, _ = in_quiet_hours("2026-04-18T03:00:00Z", ["00:00-07:00"])
    assert active is True
