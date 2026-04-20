"""Persistence helpers for self-narrative-mcp."""

from __future__ import annotations

import json
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

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

        concrete_events = self._fetch_concrete_events(since, until)
        noticed_changes = self._fetch_noticed_changes(since, until)
        relationship_moments = self._fetch_relationship_moments(since, until)
        open_loops = self._fetch_open_loops()
        boundaries_respected = self._fetch_boundary_events(events)
        private_reflections = self._fetch_private_reflections(since, until)
        evidence_event_ids = [event.event_id for event in events[:20]]
        next_gentle_actions = self._suggest_next_actions(
            open_loops=open_loops,
            noticed_changes=noticed_changes,
        )

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
                        {
                            "event_kinds": event_kinds,
                            "person_ids": person_ids,
                            "concrete_events": concrete_events,
                            "noticed_changes": noticed_changes,
                            "relationship_moments": relationship_moments,
                            "open_loops": open_loops,
                            "boundaries_respected": boundaries_respected,
                            "private_reflections": private_reflections,
                            "evidence_event_ids": evidence_event_ids,
                            "next_gentle_actions": next_gentle_actions,
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
        self._refresh_facets_and_arcs(event_kinds, person_ids, latest_ts or f"{day}T12:00:00+00:00")
        return DaybookRecord(
            day=day,
            summary=summary,
            concrete_events=concrete_events,
            noticed_changes=noticed_changes,
            relationship_moments=relationship_moments,
            open_loops=open_loops,
            boundaries_respected=boundaries_respected,
            private_reflections=private_reflections,
            next_gentle_actions=next_gentle_actions,
            evidence_event_ids=evidence_event_ids,
        )

    def get_self_summary(self) -> SelfSummary:
        latest_daybook_row = self.db.fetchone(
            "SELECT summary FROM narrative_daybooks ORDER BY day DESC LIMIT 1"
        )
        latest_daybook = (
            None if latest_daybook_row is None else str(latest_daybook_row["summary"])
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
        recent_events = [
            f"{row['kind']}: {row['summary']}"
            for row in self._fetchall_safely(
                """
                SELECT kind, summary
                FROM agent_experiences
                ORDER BY ts DESC
                LIMIT 3
                """
            )
        ]
        recent_shifts = [
            f"{row['topic']}: {row['new_interpretation']}"
            for row in self._fetchall_safely(
                """
                SELECT topic, new_interpretation
                FROM interpretation_shifts
                ORDER BY ts DESC
                LIMIT 2
                """
            )
        ]
        return SelfSummary(
            summary=build_self_summary(
                latest_daybook,
                arcs,
                facets,
                recent_events=recent_events,
                recent_shifts=recent_shifts,
            ),
            recent_concrete_events=recent_events,
            recent_interpretation_shifts=recent_shifts,
            latest_daybook=latest_daybook,
        )

    def _fetchall_safely(self, sql: str, args: tuple | None = None) -> list[dict[str, Any]]:
        """Return rows as dicts; tolerate missing orchestrator tables gracefully."""

        try:
            rows = self.db.fetchall(sql, args or ())
        except Exception:
            return []
        return [dict(row) for row in rows]

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

    # ------------------------------------------------------------------
    # helpers that enrich the daybook
    # ------------------------------------------------------------------

    def _fetch_concrete_events(self, since: str, until: str) -> list[str]:
        rows = self._fetchall_safely(
            """
            SELECT kind, summary
            FROM agent_experiences
            WHERE ts BETWEEN ? AND ?
            ORDER BY ts ASC
            LIMIT 12
            """,
            (since, until),
        )
        return [f"{row['kind']}: {row['summary']}" for row in rows]

    def _fetch_noticed_changes(self, since: str, until: str) -> list[str]:
        rows = self._fetchall_safely(
            """
            SELECT topic, old_interpretation, new_interpretation
            FROM interpretation_shifts
            WHERE ts BETWEEN ? AND ?
            ORDER BY ts ASC
            """,
            (since, until),
        )
        return [
            f"{row['topic']}: from '{row['old_interpretation']}' → '{row['new_interpretation']}'"
            for row in rows
        ]

    def _fetch_relationship_moments(self, since: str, until: str) -> list[str]:
        rows = self._fetchall_safely(
            """
            SELECT kind, person_id, summary
            FROM agent_experiences
            WHERE ts BETWEEN ? AND ?
              AND person_id IS NOT NULL
              AND kind IN (
                'agent_response', 'user_correction', 'agent_social_post',
                'open_loop_progress', 'boundary_respected'
              )
            ORDER BY ts ASC
            LIMIT 8
            """,
            (since, until),
        )
        return [f"{row['person_id']} · {row['summary']}" for row in rows]

    def _fetch_open_loops(self) -> list[str]:
        rows = self._fetchall_safely(
            """
            SELECT topic, person_id
            FROM open_loops
            WHERE status != 'closed'
            ORDER BY updated_at DESC
            LIMIT 5
            """,
        )
        return [
            (f"{row['person_id']} · {row['topic']}" if row.get("person_id") else str(row["topic"]))
            for row in rows
        ]

    def _fetch_boundary_events(self, events: list[Any]) -> list[str]:
        boundaries: list[str] = []
        for event in events:
            if event.kind == "boundary_updated":
                rule = ""
                payload = getattr(event, "payload_json", {}) or {}
                if isinstance(payload, dict):
                    rule = str(payload.get("rule") or payload.get("kind") or "")
                boundaries.append(rule or event.event_id)
        return boundaries[:5]

    def _fetch_private_reflections(self, since: str, until: str) -> list[str]:
        rows = self._fetchall_safely(
            """
            SELECT title FROM private_reflections
            WHERE ts BETWEEN ? AND ?
            ORDER BY ts ASC
            LIMIT 5
            """,
            (since, until),
        )
        return [str(row["title"]) for row in rows]

    def _suggest_next_actions(
        self, *, open_loops: list[str], noticed_changes: list[str]
    ) -> list[str]:
        """Tiny heuristic: suggest gentle follow-ups for open loops and recent shifts."""

        actions: list[str] = []
        for loop in open_loops[:2]:
            actions.append(f"Follow up gently on: {loop}")
        for change in noticed_changes[:1]:
            actions.append(f"Honor the interpretation shift: {change}")
        return actions

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
