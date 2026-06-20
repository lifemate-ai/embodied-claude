"""HORRecord + HORStore — Higher-Order Representations (Phase 2.5).

Phase 1 integration is load-bearing: HORRecord is a SPECIALIZATION of
EpistemicClaim where evidence_type='inferred' and asserted_content
paraphrases "I am [mode] [target]". to_epistemic_claim() projects each
record back into the Phase 1 primitive type, so HOR storage extends the
existing epistemic discipline rather than parallelling it.

Backs Phase 2.5 of the consciousness architecture from the plan-of-record
at ~/.claude/plans/jazzy-wishing-starfish.md. The HOR layer is the
substrate for the existing '抑圧バイアス' (suppression-bias) self-check
formalized — Theory B R9-R10.

See consciousness-mcp/AGENTS.md for vocabulary discipline. asserted_mode
values (seeing/wanting/intending/remembering/feeling/attending) are
behavioral predicates, not phenomenal claims.
"""

from __future__ import annotations

import secrets
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from social_core.db import SocialDB
from social_core.models import EpistemicClaim
from social_core.time import utc_now

ASSERTED_MODES = (
    "seeing",
    "wanting",
    "intending",
    "remembering",
    "feeling",
    "attending",
)

AssertedMode = Literal[
    "seeing",
    "wanting",
    "intending",
    "remembering",
    "feeling",
    "attending",
]

TARGET_KINDS = (
    "memory",
    "desire",
    "action",
    "frame",
    "none",
)

TargetKind = Literal[
    "memory",
    "desire",
    "action",
    "frame",
    "none",
]

HOR_SOURCES = (
    "schema_readout",
    "reflection",
    "post_hoc",
    "audit_subagent",
)

HORSource = Literal[
    "schema_readout",
    "reflection",
    "post_hoc",
    "audit_subagent",
]


class HORInput(BaseModel):
    """Payload for recording a higher-order representation.

    target_kind names the kind of first-order ref the HOR is about
    (memory/desire/action/frame, or 'none' for ungrounded). target_ref
    is the actual FK (memory_id, desire_name, action_ref, frame_id) —
    free-form by design so any of the source tables can be pointed to.
    """

    model_config = ConfigDict(extra="forbid")

    ts: str | None = None
    owner_id: str = "self"
    target_kind: TargetKind
    target_ref: str | None = None
    asserted_mode: AssertedMode
    asserted_content: str = Field(min_length=1)
    schema_snapshot_id: str | None = None
    source_tick_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.6)
    source: HORSource


class HORRecord(BaseModel):
    """Persisted higher-order representation, returned by HORStore."""

    model_config = ConfigDict(extra="ignore")

    hor_id: str
    ts: str
    owner_id: str
    target_kind: TargetKind
    target_ref: str | None = None
    asserted_mode: AssertedMode
    asserted_content: str
    schema_snapshot_id: str | None = None
    source_tick_id: str | None = None
    confidence: float
    source: HORSource
    created_at: str

    def to_epistemic_claim(self) -> EpistemicClaim:
        """Project HORRecord as an EpistemicClaim (Phase 1 type).

        evidence_type is always 'inferred' (HORs are inferred from
        first-order states, not directly observed). source carries the
        HOR-source channel (schema_readout / reflection / post_hoc /
        audit_subagent). derived_from lists the target_ref when present
        so downstream graph traversal can follow back to the first-order
        substrate.
        """
        derived: list[str] = []
        if self.target_ref:
            derived.append(self.target_ref)
        if self.schema_snapshot_id:
            derived.append(self.schema_snapshot_id)
        content = (
            f"HOR[{self.asserted_mode}/{self.target_kind}]: {self.asserted_content}"
        )
        return EpistemicClaim(
            content=content,
            evidence_type="inferred",
            source=self.source,
            confidence=self.confidence,
            derived_from=derived,
        )


class HORStore:
    """SQLite-backed store for HORRecord rows over hor_records table."""

    def __init__(self, db: SocialDB) -> None:
        self._db = db

    def record(self, payload: HORInput) -> HORRecord:
        hor_id = f"hor_{secrets.token_urlsafe(12)}"
        ts = payload.ts or utc_now()
        created_at = utc_now()
        self._db.execute(
            """
            INSERT INTO hor_records(
                hor_id, ts, owner_id, target_kind, target_ref,
                asserted_mode, asserted_content, schema_snapshot_id,
                source_tick_id, confidence, source, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                hor_id,
                ts,
                payload.owner_id,
                payload.target_kind,
                payload.target_ref,
                payload.asserted_mode,
                payload.asserted_content,
                payload.schema_snapshot_id,
                payload.source_tick_id,
                payload.confidence,
                payload.source,
                created_at,
            ),
        )
        return HORRecord(
            hor_id=hor_id,
            ts=ts,
            owner_id=payload.owner_id,
            target_kind=payload.target_kind,
            target_ref=payload.target_ref,
            asserted_mode=payload.asserted_mode,
            asserted_content=payload.asserted_content,
            schema_snapshot_id=payload.schema_snapshot_id,
            source_tick_id=payload.source_tick_id,
            confidence=payload.confidence,
            source=payload.source,
            created_at=created_at,
        )

    def get_by_hor_id(self, hor_id: str) -> HORRecord | None:
        row = self._db.fetchone(
            "SELECT * FROM hor_records WHERE hor_id = ?",
            (hor_id,),
        )
        if row is None:
            return None
        return self._row_to_record(row)

    def query(
        self,
        *,
        since: str | None = None,
        owner_id: str | None = None,
        asserted_mode: AssertedMode | None = None,
        target_kind: TargetKind | None = None,
        source_tick_id: str | None = None,
        limit: int = 50,
    ) -> list[HORRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if since is not None:
            clauses.append("ts >= ?")
            params.append(since)
        if owner_id is not None:
            clauses.append("owner_id = ?")
            params.append(owner_id)
        if asserted_mode is not None:
            clauses.append("asserted_mode = ?")
            params.append(asserted_mode)
        if target_kind is not None:
            clauses.append("target_kind = ?")
            params.append(target_kind)
        if source_tick_id is not None:
            clauses.append("source_tick_id = ?")
            params.append(source_tick_id)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)

        rows = self._db.fetchall(
            f"""
            SELECT * FROM hor_records
            {where}
            ORDER BY ts DESC, created_at DESC
            LIMIT ?
            """,
            tuple(params),
        )
        return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _row_to_record(row) -> HORRecord:
        return HORRecord(
            hor_id=row["hor_id"],
            ts=row["ts"],
            owner_id=row["owner_id"],
            target_kind=row["target_kind"],
            target_ref=row["target_ref"],
            asserted_mode=row["asserted_mode"],
            asserted_content=row["asserted_content"],
            schema_snapshot_id=row["schema_snapshot_id"],
            source_tick_id=row["source_tick_id"],
            confidence=row["confidence"],
            source=row["source"],
            created_at=row["created_at"],
        )
