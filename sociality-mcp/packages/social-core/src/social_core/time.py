"""Time helpers with deterministic test support."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_POLICY_TIMEZONE = "UTC"


def parse_timestamp(value: str | datetime) -> datetime:
    """Parse an ISO8601 timestamp and normalize it to an aware datetime."""

    if isinstance(value, datetime):
        dt = value
    else:
        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def ensure_iso8601(value: str | datetime) -> str:
    """Normalize timestamps to a stable ISO8601 string."""

    return parse_timestamp(value).isoformat(timespec="seconds")


def utc_now() -> str:
    """Return the current UTC time as ISO8601."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def local_view(value: str | datetime, tz_name: str = DEFAULT_POLICY_TIMEZONE) -> datetime:
    """Return the timestamp converted to the given IANA timezone.

    Falls back to UTC silently if the zone name cannot be resolved, so a bad
    policy file never crashes the gate.
    """

    dt = parse_timestamp(value)
    try:
        tz = ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        tz = ZoneInfo("UTC")
    return dt.astimezone(tz)


def in_quiet_hours(
    ts: str | datetime,
    windows: list[str],
    tz_name: str = DEFAULT_POLICY_TIMEZONE,
) -> tuple[bool, str | None]:
    """Check whether ``ts`` falls in any ``HH:MM-HH:MM`` window in ``tz_name``.

    Returns ``(active, until_iso)`` where ``until_iso`` is the end-of-window
    instant in the policy timezone. Windows that cross midnight (e.g.
    ``"23:00-06:00"``) are supported.
    """

    if not windows:
        return False, None
    local = local_view(ts, tz_name)
    current_minutes = local.hour * 60 + local.minute
    for window in windows:
        start_text, end_text = window.split("-", 1)
        start = _to_minutes(start_text)
        end = _to_minutes(end_text)
        active = False
        if start <= end and start <= current_minutes < end:
            active = True
        elif start > end and (current_minutes >= start or current_minutes < end):
            active = True
        if active:
            end_dt = _window_end_local(local, end)
            return True, end_dt.isoformat(timespec="seconds")
    return False, None


def _to_minutes(value: str) -> int:
    hours, minutes = value.split(":", 1)
    return int(hours) * 60 + int(minutes)


def _window_end_local(reference: datetime, end_minutes: int) -> datetime:
    current = reference.hour * 60 + reference.minute
    day_offset = 1 if end_minutes <= current else 0
    midnight = reference.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight + timedelta(days=day_offset, minutes=end_minutes)


@dataclass(slots=True)
class FixedClock:
    """Deterministic clock for tests and replay."""

    value: str

    def now(self) -> str:
        return ensure_iso8601(self.value)
