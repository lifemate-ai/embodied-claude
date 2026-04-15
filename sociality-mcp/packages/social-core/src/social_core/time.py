"""Time helpers with deterministic test support."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


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


@dataclass(slots=True)
class FixedClock:
    """Deterministic clock for tests and replay."""

    value: str

    def now(self) -> str:
        return ensure_iso8601(self.value)
