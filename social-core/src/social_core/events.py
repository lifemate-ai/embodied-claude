"""Append-only social event storage helpers."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from hashlib import sha1
from pathlib import Path
from typing import Any

from .db import SocialDB
from .models import SocialEvent, SocialEventCreate
from .time import parse_timestamp, utc_now


def build_event_id(event: SocialEventCreate) -> str:
    """Build a deterministic event identifier from stable event fields."""

    payload = {
        "ts": event.ts,
        "source": event.source,
        "kind": event.kind,
        "person_id": event.person_id,
        "session_id": event.session_id,
        "correlation_id": event.correlation_id,
        "confidence": event.confidence,
        "payload_json": event.payload_json,
    }
    digest = sha1(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"evt_{digest[:16]}"


class EventStore:
    """Shared append-only event store."""

    def __init__(self, db: SocialDB | None = None, path: str | Path | None = None) -> None:
        self.db = db or SocialDB(path)

    def ingest(self, event: SocialEventCreate | dict[str, Any]) -> SocialEvent:
        payload = (
            event
            if isinstance(event, SocialEventCreate)
            else SocialEventCreate.model_validate(event)
        )
        event_id = build_event_id(payload)
        now = utc_now()
        with self.db.transaction() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO events(
                        event_id, ts, source, kind, person_id, session_id, correlation_id,
                        confidence, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        payload.ts,
                        payload.source,
                        payload.kind,
                        payload.person_id,
                        payload.session_id,
                        payload.correlation_id,
                        payload.confidence,
                        json.dumps(payload.payload_json, ensure_ascii=False, sort_keys=True),
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                if payload.correlation_id:
                    existing = connection.execute(
                        "SELECT * FROM events WHERE source = ? AND correlation_id = ?",
                        (payload.source, payload.correlation_id),
                    ).fetchone()
                    if existing is not None:
                        return self._row_to_event(existing)
                existing = connection.execute(
                    "SELECT * FROM events WHERE event_id = ?",
                    (event_id,),
                ).fetchone()
                if existing is not None:
                    return self._row_to_event(existing)
                raise RuntimeError(f"Failed to ingest event: {exc}") from exc
            row = connection.execute(
                "SELECT * FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("Inserted event could not be reloaded")
            return self._row_to_event(row)

    def fetch_events(
        self,
        *,
        person_id: str | None = None,
        kinds: Iterable[str] | None = None,
        limit: int = 200,
        since: str | None = None,
        include_global: bool = True,
    ) -> list[SocialEvent]:
        clauses = []
        params: list[Any] = []
        if person_id:
            if include_global:
                clauses.append("(person_id = ? OR person_id IS NULL)")
            else:
                clauses.append("person_id = ?")
            params.append(person_id)
        if kinds:
            kind_list = list(kinds)
            placeholders = ", ".join("?" for _ in kind_list)
            clauses.append(f"kind IN ({placeholders})")
            params.extend(kind_list)
        if since:
            clauses.append("ts >= ?")
            params.append(since)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.db.fetchall(
            f"""
            SELECT * FROM events
            {where}
            ORDER BY ts DESC, event_seq DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        return [self._row_to_event(row) for row in rows]

    def get_latest_timestamp(self, person_id: str | None = None) -> str | None:
        if person_id:
            row = self.db.fetchone(
                """
                SELECT ts
                FROM events
                WHERE person_id = ? OR person_id IS NULL
                ORDER BY ts DESC, event_seq DESC
                LIMIT 1
                """,
                (person_id,),
            )
        else:
            row = self.db.fetchone("SELECT ts FROM events ORDER BY ts DESC, event_seq DESC LIMIT 1")
        return None if row is None else str(row["ts"])

    def replay(self, events: Iterable[SocialEventCreate | dict[str, Any]]) -> list[SocialEvent]:
        ordered = sorted(
            (
                event
                if isinstance(event, SocialEventCreate)
                else SocialEventCreate.model_validate(event)
                for event in events
            ),
            key=lambda item: parse_timestamp(item.ts),
        )
        return [self.ingest(event) for event in ordered]

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> SocialEvent:
        return SocialEvent(
            event_id=row["event_id"],
            event_seq=row["event_seq"],
            ts=row["ts"],
            source=row["source"],
            kind=row["kind"],
            person_id=row["person_id"],
            session_id=row["session_id"],
            correlation_id=row["correlation_id"],
            confidence=float(row["confidence"]),
            payload_json=json.loads(row["payload_json"]),
        )
