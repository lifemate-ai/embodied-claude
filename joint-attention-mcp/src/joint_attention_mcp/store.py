"""Persistence helpers for joint-attention-mcp."""

from __future__ import annotations

import json
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

from social_core import EventStore, SocialDB, SocialEventCreate, ensure_iso8601, parse_timestamp

from .resolver import compare_scenes, infer_joint_focus, resolve_reference
from .schemas import ReferenceResolution, SceneParse


class JointAttentionStore:
    """Scene and joint focus storage."""

    def __init__(self, path: str | Path | None = None, db: SocialDB | None = None) -> None:
        self.db = db or SocialDB(path)
        self.events = EventStore(self.db)

    def close(self) -> None:
        self.db.close()

    def ingest_scene_parse(self, scene: dict[str, Any]) -> dict[str, str]:
        payload = SceneParse.model_validate(scene)
        frame_id = f"frame_{uuid.uuid4().hex[:12]}"
        primary_person = payload.people[0].person_id if payload.people else None
        with self.db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO scene_frames(
                    frame_id,
                    ts,
                    person_id,
                    session_id,
                    camera_pose_json,
                    scene_summary,
                    raw_json
                )
                VALUES (?, ?, ?, NULL, ?, ?, ?)
                """,
                (
                    frame_id,
                    payload.ts,
                    primary_person,
                    json.dumps(payload.camera_pose.model_dump(mode="json"), ensure_ascii=False),
                    payload.scene_summary,
                    payload.model_dump_json(),
                ),
            )
            for person in payload.people:
                connection.execute(
                    """
                    INSERT INTO scene_people(
                        frame_person_id,
                        frame_id,
                        person_id,
                        display_name,
                        relative_position,
                        distance,
                        gaze_target,
                        confidence,
                        raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"fperson_{uuid.uuid4().hex[:12]}",
                        frame_id,
                        person.person_id,
                        person.display_name,
                        person.relative_position,
                        person.distance,
                        person.gaze_target,
                        person.confidence,
                        person.model_dump_json(),
                    ),
                )
            for obj in payload.objects:
                connection.execute(
                    """
                    INSERT INTO scene_objects(
                        frame_object_id,
                        frame_id,
                        object_id,
                        label,
                        attributes_json,
                        relations_json,
                        salience,
                        raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"fobject_{uuid.uuid4().hex[:12]}",
                        frame_id,
                        obj.object_id,
                        obj.label,
                        json.dumps(obj.attributes, ensure_ascii=False, sort_keys=True),
                        json.dumps(obj.relative_position, ensure_ascii=False),
                        obj.salience,
                        obj.model_dump_json(),
                    ),
                )
        self.events.ingest(
            SocialEventCreate(
                ts=payload.ts,
                source="camera",
                kind="scene_parse",
                person_id=primary_person,
                confidence=0.9,
                payload={"scene_summary": payload.scene_summary, "activity": None},
            )
        )
        return {"frame_id": frame_id}

    def resolve_reference(
        self,
        *,
        expression: str,
        person_id: str | None = None,
        lookback_frames: int = 5,
    ) -> ReferenceResolution:
        frames = self._recent_frames(person_id=person_id, limit=lookback_frames)
        objects_by_frame = [frame["objects"] for frame in frames]
        prior_focus = self._latest_joint_focus(person_id)
        matches = resolve_reference(
            expression,
            objects_by_frame,
            prior_focus=None if prior_focus is None else str(prior_focus["target_id"]),
        )
        return ReferenceResolution(matches=matches)

    def get_current_joint_focus(self, *, person_id: str | None = None) -> dict[str, Any]:
        frames = self._recent_frames(person_id=person_id, limit=1)
        if not frames:
            return {"focus_target": None, "confidence": 0.12, "based_on": ["no recent scene"]}
        explicit = self._latest_joint_focus(person_id)
        return infer_joint_focus(frames[0]["objects"], frames[0]["people"], explicit)

    def set_joint_focus(
        self,
        *,
        person_id: str | None,
        target_id: str,
        initiator: str,
    ) -> dict[str, str]:
        focus_id = f"focus_{uuid.uuid4().hex[:10]}"
        ts = self.events.get_latest_timestamp(person_id=person_id) or ensure_iso8601(
            "2026-01-01T12:00:00+00:00"
        )
        with self.db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO joint_focus(
                    focus_id,
                    ts,
                    person_id,
                    target_id,
                    initiator,
                    confidence,
                    based_on_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    focus_id,
                    ts,
                    person_id,
                    target_id,
                    initiator,
                    0.9 if initiator == "human" else 0.75,
                    json.dumps(["explicit set_joint_focus"], ensure_ascii=False),
                ),
            )
        return {"focus_id": focus_id}

    def compare_recent_scenes(
        self,
        *,
        person_id: str | None = None,
        window_minutes: int = 30,
    ) -> dict[str, list[str]]:
        latest_ts = self.events.get_latest_timestamp(person_id=person_id)
        since = None
        if latest_ts:
            since = ensure_iso8601(parse_timestamp(latest_ts) - timedelta(minutes=window_minutes))
        frames = self._recent_frames(person_id=person_id, limit=10, since=since)
        if len(frames) < 2:
            return {"changes": []}
        changes = compare_scenes(frames[-1]["objects"], frames[0]["objects"])
        return {"changes": changes}

    def _recent_frames(
        self,
        *,
        person_id: str | None,
        limit: int,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if person_id:
            clauses.append("(person_id = ? OR person_id IS NULL)")
            params.append(person_id)
        if since:
            clauses.append("ts >= ?")
            params.append(since)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        frames = self.db.fetchall(
            f"""
            SELECT frame_id, ts, scene_summary
            FROM scene_frames
            {where}
            ORDER BY ts DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        results: list[dict[str, Any]] = []
        for frame in frames:
            objects = [
                {
                    "object_id": row["object_id"],
                    "label": row["label"],
                    "attributes": json.loads(row["attributes_json"]),
                    "relative_position": json.loads(row["relations_json"]),
                    "salience": float(row["salience"]),
                }
                for row in self.db.fetchall(
                    """
                    SELECT object_id, label, attributes_json, relations_json, salience
                    FROM scene_objects
                    WHERE frame_id = ?
                    """,
                    (frame["frame_id"],),
                )
            ]
            people = [
                {
                    "person_id": row["person_id"],
                    "display_name": row["display_name"],
                    "relative_position": row["relative_position"],
                    "distance": row["distance"],
                    "gaze_target": row["gaze_target"],
                    "confidence": float(row["confidence"]),
                }
                for row in self.db.fetchall(
                    """
                    SELECT
                        person_id,
                        display_name,
                        relative_position,
                        distance,
                        gaze_target,
                        confidence
                    FROM scene_people
                    WHERE frame_id = ?
                    """,
                    (frame["frame_id"],),
                )
            ]
            results.append(
                {
                    "frame_id": frame["frame_id"],
                    "ts": frame["ts"],
                    "scene_summary": frame["scene_summary"],
                    "objects": objects,
                    "people": people,
                }
            )
        return results

    def _latest_joint_focus(self, person_id: str | None) -> dict[str, Any] | None:
        if person_id:
            row = self.db.fetchone(
                """
                SELECT target_id, confidence, based_on_json
                FROM joint_focus
                WHERE person_id = ?
                ORDER BY ts DESC
                LIMIT 1
                """,
                (person_id,),
            )
        else:
            row = self.db.fetchone(
                """
                SELECT target_id, confidence, based_on_json
                FROM joint_focus
                ORDER BY ts DESC
                LIMIT 1
                """
            )
        if row is None:
            return None
        return {
            "target_id": row["target_id"],
            "confidence": float(row["confidence"]),
            "based_on": json.loads(row["based_on_json"]),
        }
