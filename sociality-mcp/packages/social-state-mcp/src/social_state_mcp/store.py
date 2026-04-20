"""Persistence helpers for social-state-mcp."""

from __future__ import annotations

import json
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

from social_core import (
    DEFAULT_POLICY_TIMEZONE,
    EventStore,
    SocialDB,
    SocialEventCreate,
    ensure_iso8601,
    parse_timestamp,
)

from .inference import get_social_state_result
from .schemas import SocialStateResult


class SocialStateStore:
    """Store wrapper for social state tools."""

    def __init__(
        self,
        path: str | Path | None = None,
        db: SocialDB | None = None,
        *,
        quiet_hours_windows: list[str] | None = None,
        policy_timezone: str = DEFAULT_POLICY_TIMEZONE,
    ) -> None:
        self.db = db or SocialDB(path)
        self.events = EventStore(self.db)
        self.quiet_hours_windows = quiet_hours_windows
        self.policy_timezone = policy_timezone

    def close(self) -> None:
        self.db.close()

    def ingest_social_event(self, event: dict[str, Any]) -> dict[str, str]:
        stored = self.events.ingest(SocialEventCreate.model_validate(event))
        return {"event_id": stored.event_id}

    def get_social_state(
        self,
        *,
        window_seconds: int,
        person_id: str | None = None,
        include_evidence: bool = True,
    ) -> SocialStateResult:
        reference_ts = self.events.get_latest_timestamp(person_id=person_id)
        since = None
        if reference_ts is not None:
            start = parse_timestamp(reference_ts) - timedelta(seconds=window_seconds)
            since = ensure_iso8601(start)
        events = self.events.fetch_events(
            person_id=person_id,
            since=since,
            limit=400,
        )
        state = get_social_state_result(
            events,
            person_id=person_id,
            include_evidence=include_evidence,
            reference_ts=reference_ts,
            quiet_hours_windows=self.quiet_hours_windows,
            policy_timezone=self.policy_timezone,
        )
        self._save_snapshot(state)
        return state

    def _save_snapshot(self, state: SocialStateResult) -> None:
        payload = state.model_dump(mode="json")
        with self.db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO social_state_snapshots(
                    snapshot_id,
                    ts,
                    person_id,
                    state_json,
                    summary_for_prompt
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    f"snapshot_{uuid.uuid4().hex[:12]}",
                    state.timestamp,
                    state.person_id,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    state.summary_for_prompt,
                ),
            )
