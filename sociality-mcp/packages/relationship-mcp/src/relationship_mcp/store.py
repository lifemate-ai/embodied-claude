"""Persistence helpers for relationship-mcp."""

from __future__ import annotations

import json
import re
import uuid
from datetime import timedelta
from pathlib import Path

from social_core import (
    EventStore,
    SocialDB,
    SocialEventCreate,
    ensure_iso8601,
    parse_timestamp,
)

from .inference import (
    FUTURE_MARKERS,
    STRESS_KEYWORDS,
    compute_snapshot_metrics,
    suggest_followup_text,
    summarize_relationship,
)
from .schemas import CommitmentRecord, OpenLoopRecord, PersonModel, RitualRecord, SuggestionRecord


class RelationshipStore:
    """Compact relationship storage built on the shared social DB."""

    def __init__(self, path: str | Path | None = None, db: SocialDB | None = None) -> None:
        self.db = db or SocialDB(path)
        self.events = EventStore(self.db)

    def close(self) -> None:
        self.db.close()

    def upsert_person(
        self,
        *,
        person_id: str,
        canonical_name: str,
        aliases: list[str] | None = None,
        role: str | None = None,
    ) -> dict[str, str]:
        aliases = aliases or []
        now = ensure_iso8601(self.events.get_latest_timestamp() or "2026-01-01T12:00:00+00:00")
        with self.db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO persons(
                    person_id,
                    canonical_name,
                    role,
                    profile_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, '{}', ?, ?)
                ON CONFLICT(person_id) DO UPDATE SET
                    canonical_name = excluded.canonical_name,
                    role = excluded.role,
                    updated_at = excluded.updated_at
                """,
                (person_id, canonical_name, role, now, now),
            )
            connection.execute("DELETE FROM person_aliases WHERE person_id = ?", (person_id,))
            for alias in dict.fromkeys([canonical_name, *aliases]):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO person_aliases(alias_id, person_id, alias)
                    VALUES (?, ?, ?)
                    """,
                    (f"alias_{uuid.uuid4().hex[:12]}", person_id, alias),
                )
        return {"person_id": person_id}

    def resolve_person_id(self, name_or_id: str) -> str | None:
        row = self.db.fetchone("SELECT person_id FROM persons WHERE person_id = ?", (name_or_id,))
        if row is not None:
            return str(row["person_id"])
        row = self.db.fetchone(
            "SELECT person_id FROM person_aliases WHERE alias = ?", (name_or_id,)
        )
        return None if row is None else str(row["person_id"])

    def ingest_interaction(
        self,
        *,
        person_id: str,
        channel: str,
        direction: str,
        text: str,
        ts: str,
    ) -> dict[str, str]:
        self._ensure_person(person_id)
        kind = "human_utterance" if direction == "human_to_ai" else "agent_utterance"
        event = self.events.ingest(
            SocialEventCreate(
                ts=ts,
                source=channel,
                kind=kind,
                person_id=person_id,
                confidence=0.92,
                payload={"text": text, "direction": direction},
            )
        )
        self._update_open_loops(
            person_id=person_id, text=text, source_event_id=event.event_id, ts=ts
        )
        self.refresh_snapshot(person_id)
        return {"event_id": event.event_id}

    def create_commitment(
        self,
        *,
        person_id: str,
        text: str,
        due_at: str | None,
        source: str,
    ) -> dict[str, str]:
        self._ensure_person(person_id)
        commitment_id = f"commit_{uuid.uuid4().hex[:10]}"
        created_at = ensure_iso8601(
            self.events.get_latest_timestamp(person_id=person_id) or "2026-01-01T12:00:00+00:00"
        )
        with self.db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO commitments(
                    commitment_id,
                    person_id,
                    text,
                    due_at,
                    source,
                    status,
                    created_at,
                    completed_at,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, 'active', ?, NULL, '{}')
                """,
                (commitment_id, person_id, text, due_at, source, created_at),
            )
        self.events.ingest(
            SocialEventCreate(
                ts=created_at,
                source="relationship_mcp",
                kind="commitment_created",
                person_id=person_id,
                correlation_id=commitment_id,
                confidence=0.95,
                payload={"text": text, "due_at": due_at, "source": source},
            )
        )
        return {"commitment_id": commitment_id}

    def complete_commitment(self, commitment_id: str) -> dict[str, str]:
        row = self.db.fetchone(
            "SELECT person_id, text FROM commitments WHERE commitment_id = ?", (commitment_id,)
        )
        if row is None:
            raise ValueError(f"Unknown commitment_id: {commitment_id}")
        completed_at = ensure_iso8601(
            self.events.get_latest_timestamp(person_id=row["person_id"])
            or "2026-01-01T12:00:00+00:00"
        )
        with self.db.transaction() as connection:
            connection.execute(
                """
                UPDATE commitments
                SET status = 'completed', completed_at = ?
                WHERE commitment_id = ?
                """,
                (completed_at, commitment_id),
            )
        self.events.ingest(
            SocialEventCreate(
                ts=completed_at,
                source="relationship_mcp",
                kind="commitment_completed",
                person_id=row["person_id"],
                correlation_id=commitment_id,
                confidence=0.95,
                payload={"text": row["text"]},
            )
        )
        return {"commitment_id": commitment_id}

    def list_open_loops(self, *, person_id: str, limit: int = 10) -> list[OpenLoopRecord]:
        rows = self.db.fetchall(
            """
            SELECT loop_id, topic, status
            FROM open_loops
            WHERE person_id = ? AND status = 'open'
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (person_id, limit),
        )
        return [
            OpenLoopRecord(id=row["loop_id"], topic=row["topic"], status=row["status"])
            for row in rows
        ]

    def record_boundary(
        self,
        *,
        person_id: str,
        kind: str,
        rule: str,
        source_text: str,
    ) -> dict[str, str]:
        self._ensure_person(person_id)
        boundary_id = f"boundary_{uuid.uuid4().hex[:10]}"
        created_at = ensure_iso8601(
            self.events.get_latest_timestamp(person_id=person_id) or "2026-01-01T12:00:00+00:00"
        )
        with self.db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO person_boundaries(
                    boundary_id,
                    person_id,
                    kind,
                    rule,
                    source_text,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (boundary_id, person_id, kind, rule, source_text, created_at),
            )
        self.events.ingest(
            SocialEventCreate(
                ts=created_at,
                source="relationship_mcp",
                kind="boundary_updated",
                person_id=person_id,
                correlation_id=boundary_id,
                confidence=0.95,
                payload={"kind": kind, "rule": rule, "source_text": source_text},
            )
        )
        return {"boundary_id": boundary_id}

    def get_person_model(self, *, person_id: str) -> PersonModel:
        self.refresh_snapshot(person_id)
        person = self.db.fetchone(
            "SELECT canonical_name, role, updated_at FROM persons WHERE person_id = ?",
            (person_id,),
        )
        if person is None:
            raise ValueError(f"Unknown person_id: {person_id}")
        aliases = [
            str(row["alias"])
            for row in self.db.fetchall(
                "SELECT alias FROM person_aliases WHERE person_id = ? ORDER BY alias",
                (person_id,),
            )
            if row["alias"] != person["canonical_name"]
        ]
        snapshot = self.db.fetchone(
            """
            SELECT relationship_summary, ts
            FROM relationship_snapshots
            WHERE person_id = ?
            ORDER BY ts DESC
            LIMIT 1
            """,
            (person_id,),
        )
        commitments = [
            CommitmentRecord(
                id=row["commitment_id"],
                text=row["text"],
                due_at=row["due_at"],
                source=row["source"],
            )
            for row in self.db.fetchall(
                """
                SELECT commitment_id, text, due_at, source
                FROM commitments
                WHERE person_id = ? AND status = 'active'
                ORDER BY COALESCE(due_at, created_at)
                """,
                (person_id,),
            )
        ]
        loops = self.list_open_loops(person_id=person_id)
        rituals = [
            RitualRecord(id=row["ritual_id"], kind=row["kind"], cadence=row["cadence"])
            for row in self.db.fetchall(
                """
                SELECT ritual_id, kind, cadence
                FROM rituals
                WHERE person_id = ?
                ORDER BY updated_at DESC
                """,
                (person_id,),
            )
        ]
        boundaries = [
            str(row["rule"])
            for row in self.db.fetchall(
                "SELECT rule FROM person_boundaries WHERE person_id = ? ORDER BY created_at DESC",
                (person_id,),
            )
        ]
        preferences = self._salient_preferences(person_id, boundaries)
        summary = (
            snapshot["relationship_summary"]
            if snapshot
            else "Relationship summary not available yet."
        )
        return PersonModel(
            person_id=person_id,
            canonical_name=str(person["canonical_name"]),
            aliases=aliases,
            role=person["role"],
            salient_preferences=preferences,
            open_loops=loops,
            active_commitments=commitments,
            rituals=rituals,
            boundaries=boundaries,
            relationship_summary=summary,
            last_updated=str(snapshot["ts"] if snapshot else person["updated_at"]),
        )

    def suggest_followup(self, *, person_id: str, context: str) -> list[SuggestionRecord]:
        latest_ts = self.events.get_latest_timestamp(person_id=person_id)
        since = None
        if latest_ts:
            since = ensure_iso8601(parse_timestamp(latest_ts) - timedelta(days=1))
        events = self.events.fetch_events(
            person_id=person_id, since=since, limit=80, include_global=False
        )
        latest_stress_text = None
        for event in events:
            if event.kind != "human_utterance":
                continue
            text = str(event.payload_json.get("text", ""))
            if any(keyword in text.lower() for keyword in STRESS_KEYWORDS):
                latest_stress_text = text
                break
        text, reason = suggest_followup_text(context, latest_stress_text)
        return [SuggestionRecord(text=text, reason=reason)]

    def refresh_snapshot(self, person_id: str) -> None:
        latest_ts = self.events.get_latest_timestamp(person_id=person_id)
        since = None
        if latest_ts:
            since = ensure_iso8601(parse_timestamp(latest_ts) - timedelta(days=30))
        events = self.events.fetch_events(
            person_id=person_id, since=since, limit=200, include_global=False
        )
        human_messages = [
            str(event.payload_json.get("text", ""))
            for event in events
            if event.kind == "human_utterance"
        ]
        agent_messages = [
            str(event.payload_json.get("text", ""))
            for event in events
            if event.kind == "agent_utterance"
        ]
        metrics = compute_snapshot_metrics(
            interaction_count=len(events),
            human_messages=human_messages,
            agent_messages=agent_messages,
        )
        role_row = self.db.fetchone("SELECT role FROM persons WHERE person_id = ?", (person_id,))
        open_loop_count = len(self.list_open_loops(person_id=person_id, limit=20))
        summary = summarize_relationship(
            role=None if role_row is None else role_row["role"],
            recent_stress=metrics["recent_stress"],
            open_loop_count=open_loop_count,
        )
        ts = latest_ts or ensure_iso8601("2026-01-01T12:00:00+00:00")
        with self.db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO relationship_snapshots(
                    snapshot_id, person_id, ts, warmth, trust, fragility, expected_response_latency,
                    recent_stress, reciprocity_balance, relationship_summary, notes_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"relsnap_{uuid.uuid4().hex[:12]}",
                    person_id,
                    ts,
                    metrics["warmth"],
                    metrics["trust"],
                    metrics["fragility"],
                    metrics["expected_response_latency"],
                    metrics["recent_stress"],
                    metrics["reciprocity_balance"],
                    summary,
                    json.dumps({"heuristic": True}, ensure_ascii=False),
                ),
            )

    def _ensure_person(self, person_id: str) -> None:
        if (
            self.db.fetchone("SELECT person_id FROM persons WHERE person_id = ?", (person_id,))
            is not None
        ):
            return
        self.upsert_person(person_id=person_id, canonical_name=person_id, aliases=[], role=None)

    def _update_open_loops(
        self, *, person_id: str, text: str, source_event_id: str, ts: str
    ) -> None:
        topic = self._extract_topic(text)
        if topic is None:
            return
        existing = self.db.fetchone(
            """
            SELECT loop_id FROM open_loops
            WHERE person_id = ? AND topic = ? AND status = 'open'
            """,
            (person_id, topic),
        )
        with self.db.transaction() as connection:
            if existing is not None:
                connection.execute(
                    """
                    UPDATE open_loops
                    SET updated_at = ?, source_event_id = ?
                    WHERE loop_id = ?
                    """,
                    (ts, source_event_id, existing["loop_id"]),
                )
                return
            connection.execute(
                """
                INSERT INTO open_loops(
                    loop_id,
                    person_id,
                    topic,
                    status,
                    source_event_id,
                    updated_at,
                    detail_json
                )
                VALUES (?, ?, ?, 'open', ?, ?, ?)
                """,
                (
                    f"loop_{uuid.uuid4().hex[:10]}",
                    person_id,
                    topic,
                    source_event_id,
                    ts,
                    json.dumps({"kind": "future_task_or_question"}, ensure_ascii=False),
                ),
            )

    def _extract_topic(self, text: str) -> str | None:
        lowered = text.lower()
        if any(marker in lowered for marker in FUTURE_MARKERS):
            if "dentist" in lowered or "歯医者" in text:
                return "dentist"
            if "pr" in lowered and "review" in lowered:
                return "pr review"
            return _normalize_topic(text)
        if "?" in text or "？" in text:
            return _normalize_topic(text)
        return None

    def _salient_preferences(self, person_id: str, boundaries: list[str]) -> list[str]:
        preferences = ["likes contextual continuity more than generic encouragement"]
        if any("quiet" in boundary or "midnight" in boundary for boundary in boundaries):
            preferences.insert(0, "prefers gentle brief nudges while working")
        elif self.db.fetchone(
            """
            SELECT 1
            FROM events
            WHERE person_id = ? AND kind = 'human_utterance' AND payload_json LIKE '%静か%'
            LIMIT 1
            """,
            (person_id,),
        ):
            preferences.insert(0, "prefers quieter interaction when focused")
        return preferences[:3]


def _normalize_topic(text: str) -> str:
    compact = re.sub(r"\s+", " ", text.strip("。.!?？ "))
    return compact[:48]
