"""AttentionSchema + AttentionSchemaTracker (Phase 2.4).

Implements Attention Schema Theory (AST / Theory B R4): a low-dimensional
model of the pattern of winning, distinct from the workspace itself.
workspace.py picks WHICH memories win; the attention schema MODELS the
shape of attention (focal target, modality, intensity, dwell), with a
control_handle so the agent can intervene rather than only observe.

In-memory ring buffer (default capacity 60) holds the recent trajectory.
flush() persists to attention_schemas table on tick boundary or on demand.

Phase 2.5 will surface this via introspect() and a HOR layer where each
schema_snapshot binds to higher-order claims.

See consciousness-mcp/AGENTS.md for vocabulary discipline.
"""

from __future__ import annotations

import secrets
from collections import deque
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from social_core.db import SocialDB
from social_core.time import utc_now

from individual_kernel_mcp.frame import ConsciousFrame

Modality = Literal["visual", "auditory", "internal", "social"]


class AttentionSchema(BaseModel):
    """Low-dimensional model of attention for a single owner (self or other)."""

    model_config = ConfigDict(extra="ignore")

    schema_id: str
    ts: str
    owner_id: str
    focal_target_ref: str | None = None
    modality: Modality
    intensity: float = Field(ge=0.0, le=1.0)
    dwell_seconds: float = 0.0
    predicted_next_focus: str | None = None
    control_handle: str | None = None
    source_tick_id: str | None = None


class AttentionSchemaTracker:
    """In-memory ring buffer + SQLite persistence at flush boundary.

    Per-tick update is cheap (Python only). Persistence is via flush(),
    which writes all buffered schemas and clears the buffer. Suitable
    for cron tick + UserPromptSubmit cadence — schemas accumulate during
    short bursts and persist when the kernel decides.
    """

    def __init__(
        self,
        *,
        db: SocialDB,
        owner_id: str = "self",
        capacity: int = 60,
    ) -> None:
        self._db = db
        self._owner_id = owner_id
        self._capacity = capacity
        self._buffer: deque[AttentionSchema] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def owner_id(self) -> str:
        return self._owner_id

    @property
    def db(self) -> SocialDB:
        return self._db

    @property
    def buffer(self) -> tuple[AttentionSchema, ...]:
        return tuple(self._buffer)

    def record(
        self,
        *,
        focal_target_ref: str | None,
        modality: Modality,
        intensity: float,
        dwell_seconds: float = 0.0,
        predicted_next_focus: str | None = None,
        control_handle: str | None = None,
        source_tick_id: str | None = None,
        ts: str | None = None,
    ) -> AttentionSchema:
        schema = AttentionSchema(
            schema_id=f"sch_{secrets.token_urlsafe(12)}",
            ts=ts or utc_now(),
            owner_id=self._owner_id,
            focal_target_ref=focal_target_ref,
            modality=modality,
            intensity=intensity,
            dwell_seconds=dwell_seconds,
            predicted_next_focus=predicted_next_focus,
            control_handle=control_handle,
            source_tick_id=source_tick_id,
        )
        self._buffer.append(schema)
        return schema

    def update_dwell_from_ts(
        self,
        *,
        focal_target_ref: str | None,
        modality: Modality,
        intensity: float,
        ts: str,
        predicted_next_focus: str | None = None,
        control_handle: str | None = None,
        source_tick_id: str | None = None,
    ) -> AttentionSchema:
        """Record a new schema whose dwell accumulates only if focal is unchanged.

        If the previous buffered schema's focal_target_ref matches, dwell
        accumulates by (ts - prev.ts) seconds. Different focal resets to 0.
        """
        dwell = self._compute_dwell(focal_target_ref, ts)
        return self.record(
            focal_target_ref=focal_target_ref,
            modality=modality,
            intensity=intensity,
            dwell_seconds=dwell,
            predicted_next_focus=predicted_next_focus,
            control_handle=control_handle,
            source_tick_id=source_tick_id,
            ts=ts,
        )

    def update_from_frame(self, frame: ConsciousFrame) -> AttentionSchema:
        """Build a schema entry from a ConsciousFrame's attention slice.

        intensity is a coarse proxy: ignited→0.8, subliminal→0.3. dwell
        accumulates only if the focal target matches the previous buffered
        schema.
        """
        focal = frame.attention_target_ref
        modality = self._infer_modality(focal)
        intensity = 0.8 if frame.ignited else 0.3
        dwell = self._compute_dwell(focal, frame.ts)
        return self.record(
            focal_target_ref=focal,
            modality=modality,
            intensity=intensity,
            dwell_seconds=dwell,
            source_tick_id=frame.tick_id,
            ts=frame.ts,
        )

    def flush(self) -> int:
        """Persist all buffered schemas to SQLite. Returns count, clears buffer."""
        count = 0
        for schema in self._buffer:
            self._db.execute(
                """
                INSERT INTO attention_schemas(
                    schema_id, ts, owner_id, focal_target_ref, modality,
                    intensity, dwell_seconds, predicted_next_focus,
                    control_handle, source_tick_id, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    schema.schema_id,
                    schema.ts,
                    schema.owner_id,
                    schema.focal_target_ref,
                    schema.modality,
                    schema.intensity,
                    schema.dwell_seconds,
                    schema.predicted_next_focus,
                    schema.control_handle,
                    schema.source_tick_id,
                    utc_now(),
                ),
            )
            count += 1
        self._buffer.clear()
        return count

    @staticmethod
    def _infer_modality(focal_target_ref: str | None) -> Modality:
        if not focal_target_ref:
            return "internal"
        f = focal_target_ref.lower()
        if "listen" in f or "audio" in f or "auditory" in f or "ear" in f:
            return "auditory"
        if "cam" in f or "see" in f or "visual" in f or "eye" in f:
            return "visual"
        if "person:" in f or "social" in f:
            return "social"
        return "internal"

    def _compute_dwell(self, focal: str | None, ts: str) -> float:
        if not self._buffer:
            return 0.0
        last = self._buffer[-1]
        if last.focal_target_ref != focal:
            return 0.0
        return last.dwell_seconds + self._ts_delta_seconds(last.ts, ts)

    @staticmethod
    def _ts_delta_seconds(ts_a: str, ts_b: str) -> float:
        a = datetime.fromisoformat(ts_a.replace("Z", "+00:00"))
        b = datetime.fromisoformat(ts_b.replace("Z", "+00:00"))
        return max(0.0, (b - a).total_seconds())
