"""Persistence helpers for self-narrative-mcp."""

from __future__ import annotations

import json
import uuid
from datetime import timedelta
from pathlib import Path

from social_core import EventStore, SocialDB, parse_timestamp

from .schemas import ArcRecord, DaybookRecord, SelfSummary
from .summarizer import build_day_summary, build_self_summary, infer_arcs, summarize_change


class SelfNarrativeStore:
    """Compact autobiographical store."""

    def __init__(self, path: str | Path | None = None, db: SocialDB | None = None) -> None:
        self.db = db or SocialDB(path)
        self.events = EventStore(self.db)

    def close(self) -> None:
        self.db.close()

    def append_daybook(self, *, day: str | None = None) -> DaybookRecord:
        latest_ts = self.events.get_latest_timestamp()
        if day is None:
            reference_ts = latest_ts or "2026-01-01T12:00:00+00:00"
            day = parse_timestamp(reference_ts).date().isoformat()
        since = f"{day}T00:00:00+00:00"
        until = f"{day}T23:59:59+00:00"
        events = [
            event for event in self.events.fetch_events(limit=400) if since <= event.ts <= until
        ]
        event_kinds = [event.kind for event in events]
        person_ids = [event.person_id for event in events if event.person_id]
        summary = build_day_summary(day, event_kinds, person_ids)
        with self.db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO narrative_daybooks(daybook_id, day, ts, summary, evidence_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(day) DO UPDATE SET
                    ts = excluded.ts,
                    summary = excluded.summary,
                    evidence_json = excluded.evidence_json
                """,
                (
                    f"daybook_{uuid.uuid4().hex[:10]}",
                    day,
                    latest_ts or f"{day}T12:00:00+00:00",
                    summary,
                    json.dumps(
                        {"event_kinds": event_kinds, "person_ids": person_ids}, ensure_ascii=False
                    ),
                ),
            )
        self._refresh_facets_and_arcs(event_kinds, person_ids, latest_ts or f"{day}T12:00:00+00:00")
        return DaybookRecord(day=day, summary=summary)

    def get_self_summary(self) -> SelfSummary:
        latest_daybook = self.db.fetchone(
            "SELECT summary FROM narrative_daybooks ORDER BY day DESC LIMIT 1"
        )
        facets = [
            str(row["summary"])
            for row in self.db.fetchall(
                "SELECT summary FROM identity_facets ORDER BY updated_at DESC LIMIT 3"
            )
        ]
        arcs = [
            str(row["title"])
            for row in self.db.fetchall(
                """
                SELECT title
                FROM narrative_arcs
                WHERE status = 'active'
                ORDER BY importance DESC
                LIMIT 3
                """
            )
        ]
        return SelfSummary(
            summary=build_self_summary(
                None if latest_daybook is None else str(latest_daybook["summary"]),
                arcs,
                facets,
            )
        )

    def list_active_arcs(self) -> list[ArcRecord]:
        rows = self.db.fetchall(
            """
            SELECT title, status, importance, summary
            FROM narrative_arcs
            WHERE status = 'active'
            ORDER BY importance DESC, updated_at DESC
            """
        )
        return [
            ArcRecord(
                title=row["title"],
                status=row["status"],
                importance=float(row["importance"]),
                summary=row["summary"],
            )
            for row in rows
        ]

    def reflect_on_change(self, *, horizon_days: int = 7) -> SelfSummary:
        latest_ts = self.events.get_latest_timestamp() or "2026-01-01T12:00:00+00:00"
        latest_day = parse_timestamp(latest_ts).date().isoformat()
        earliest_day = (
            (parse_timestamp(latest_ts) - timedelta(days=horizon_days)).date().isoformat()
        )
        earlier = self.db.fetchone(
            """
            SELECT summary FROM narrative_daybooks
            WHERE day >= ?
            ORDER BY day ASC
            LIMIT 1
            """,
            (earliest_day,),
        )
        later = self.db.fetchone(
            """
            SELECT summary FROM narrative_daybooks
            WHERE day <= ?
            ORDER BY day DESC
            LIMIT 1
            """,
            (latest_day,),
        )
        return SelfSummary(
            summary=summarize_change(
                None if earlier is None else str(earlier["summary"]),
                None if later is None else str(later["summary"]),
            )
        )

    def _refresh_facets_and_arcs(
        self, event_kinds: list[str], person_ids: list[str], ts: str
    ) -> None:
        facets = [
            ("social_style", "tries to stay gentle and context-aware", 0.82),
            ("continuity", "values continuity across days and interactions", 0.79),
        ]
        arcs = infer_arcs(event_kinds, person_ids)
        with self.db.transaction() as connection:
            for key, summary, confidence in facets:
                connection.execute(
                    """
                    INSERT INTO identity_facets(
                        facet_id,
                        facet_key,
                        summary,
                        confidence,
                        updated_at,
                        evidence_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(facet_key) DO UPDATE SET
                        summary = excluded.summary,
                        confidence = excluded.confidence,
                        updated_at = excluded.updated_at,
                        evidence_json = excluded.evidence_json
                    """,
                    (
                        f"facet_{uuid.uuid4().hex[:10]}",
                        key,
                        summary,
                        confidence,
                        ts,
                        json.dumps({"heuristic": True}, ensure_ascii=False),
                    ),
                )
            for title, status, importance in arcs:
                connection.execute(
                    """
                    INSERT INTO narrative_arcs(
                        arc_id,
                        title,
                        status,
                        importance,
                        summary,
                        updated_at,
                        notes_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(title) DO UPDATE SET
                        status = excluded.status,
                        importance = excluded.importance,
                        summary = excluded.summary,
                        updated_at = excluded.updated_at,
                        notes_json = excluded.notes_json
                    """,
                    (
                        f"arc_{uuid.uuid4().hex[:10]}",
                        title,
                        status,
                        importance,
                        title,
                        ts,
                        json.dumps({"heuristic": True}, ensure_ascii=False),
                    ),
                )
