"""Inspectable local transition geometry for focal content."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import sqlite3
import struct
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from social_core.db import SocialDB
from social_core.time import utc_now

from individual_kernel_mcp.workspace import SourceMode

EmbeddingProvider = Callable[[str], list[float]]


class StoredMemoryEmbeddingProvider:
    """Read an existing memory-mcp E5 vector without loading its model."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path).expanduser()

    def __call__(self, content_ref: str) -> list[float]:
        if ":" not in content_ref:
            raise LookupError("content ref has no embedding namespace")
        namespace, item_id = content_ref.split(":", 1)
        relation = {
            "memory": ("embeddings", "memory_id"),
            "episode": ("episode_embeddings", "episode_id"),
        }.get(namespace)
        if relation is None or not item_id:
            raise LookupError(f"unsupported embedding ref {content_ref!r}")
        table, id_column = relation
        uri = self._db_path.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            row = connection.execute(
                f"SELECT vector FROM {table} WHERE {id_column} = ?",
                (item_id,),
            ).fetchone()
        if row is None:
            raise LookupError(f"no stored embedding for {content_ref!r}")
        blob = bytes(row[0])
        if not blob or len(blob) % 4:
            raise ValueError(f"invalid stored embedding for {content_ref!r}")
        return [value[0] for value in struct.iter_unpack("=f", blob)]


class QualityNeighbor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_ref: str
    distance: float = Field(ge=0.0)
    modality: str
    source_mode: SourceMode


class ActionConditionedTransition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_ref: str | None = None
    to_content_ref: str
    probability: float = Field(ge=0.0, le=1.0)
    observations: int = Field(ge=1)


class QualitySignature(BaseModel):
    """Local, auditable structure around one content state."""

    model_config = ConfigDict(extra="forbid")

    signature_id: str
    content_ref: str
    modality: str
    source_mode: SourceMode
    feature_vector: list[float]
    neighbors: list[QualityNeighbor] = Field(default_factory=list)
    predicted_transitions: list[ActionConditionedTransition] = Field(default_factory=list)
    valence_associations: dict[str, float] = Field(default_factory=dict)
    discriminability: float = Field(default=0.0, ge=0.0)
    adaptation: dict[str, float] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class QualityGeometry:
    """SQLite transition graph with optional reuse of an existing embedder."""

    def __init__(
        self,
        db: SocialDB,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        fallback_dimensions: int = 24,
    ) -> None:
        self._db = db
        self._embedding_provider = embedding_provider or _stored_memory_provider()
        self._fallback_dimensions = max(8, fallback_dimensions)

    def feature_vector(self, content_ref: str) -> list[float]:
        if self._embedding_provider is not None:
            try:
                vector = [float(value) for value in self._embedding_provider(content_ref)]
                if vector:
                    return _normalize(vector)
            except Exception:
                pass
        return _deterministic_vector(content_ref, self._fallback_dimensions)

    def distance(self, a: str, b: str) -> float:
        return _cosine_distance(self.feature_vector(a), self.feature_vector(b))

    def neighbors(self, content_ref: str, *, limit: int = 5) -> list[QualityNeighbor]:
        target = self.feature_vector(content_ref)
        rows = self._db.fetchall(
            """
            SELECT content_ref, modality, source_mode, feature_vector_json
            FROM quality_signatures
            WHERE content_ref != ?
            ORDER BY updated_at DESC
            LIMIT 500
            """,
            (content_ref,),
        )
        nearest = []
        seen: set[str] = set()
        for row in rows:
            ref = str(row["content_ref"])
            if ref in seen:
                continue
            seen.add(ref)
            vector = _load_vector(row["feature_vector_json"])
            nearest.append(
                QualityNeighbor(
                    content_ref=ref,
                    distance=_cosine_distance(target, vector),
                    modality=row["modality"],
                    source_mode=row["source_mode"],
                )
            )
        return sorted(nearest, key=lambda item: (item.distance, item.content_ref))[
            : max(0, limit)
        ]

    def predict_transition(
        self,
        content_ref: str,
        action_ref: str | None = None,
        *,
        limit: int = 5,
    ) -> list[ActionConditionedTransition]:
        clauses = ["from_content_ref = ?", "to_content_ref IS NOT NULL"]
        params: list[object] = [content_ref]
        if action_ref is not None:
            clauses.append("action_id = ?")
            params.append(action_ref)
        where = " AND ".join(clauses)
        rows = self._db.fetchall(
            f"""
            SELECT action_id, to_content_ref, COUNT(*) AS n
            FROM field_transitions
            WHERE {where}
            GROUP BY action_id, to_content_ref
            ORDER BY n DESC, to_content_ref ASC
            LIMIT ?
            """,
            (*params, max(1, min(limit, 50))),
        )
        total = sum(int(row["n"]) for row in rows)
        if total == 0:
            return []
        return [
            ActionConditionedTransition(
                action_ref=row["action_id"],
                to_content_ref=row["to_content_ref"],
                probability=int(row["n"]) / total,
                observations=int(row["n"]),
            )
            for row in rows
        ]

    def local_signature(
        self,
        content_ref: str,
        *,
        modality: str = "internal",
        source_mode: SourceMode = SourceMode.INFERRED,
        recent_valence: float | None = None,
    ) -> QualitySignature:
        vector = self.feature_vector(content_ref)
        nearest = self.neighbors(content_ref)
        transitions = self.predict_transition(content_ref)
        if nearest:
            discriminability = sum(item.distance for item in nearest) / len(nearest)
        else:
            discriminability = 1.0
        exposure_row = self._db.fetchone(
            """
            SELECT COUNT(*) AS n FROM enacted_fields
            WHERE focal_content_ref = ? AND status IN ('COMMITTED', 'CLOSED', 'INVALIDATED')
            """,
            (content_ref,),
        )
        exposures = int(exposure_row["n"]) if exposure_row else 0
        now = utc_now()
        signature = QualitySignature(
            signature_id=f"qual_{secrets.token_urlsafe(12)}",
            content_ref=content_ref,
            modality=modality,
            source_mode=source_mode,
            feature_vector=vector,
            neighbors=nearest,
            predicted_transitions=transitions,
            valence_associations=(
                {"recent": max(-1.0, min(1.0, recent_valence))}
                if recent_valence is not None
                else {}
            ),
            discriminability=discriminability,
            adaptation={
                "recent_exposures": float(exposures),
                "novelty": 1.0 / (1.0 + exposures),
            },
            created_at=now,
            updated_at=now,
        )
        self._store(signature)
        return signature

    def get(self, signature_id: str) -> QualitySignature | None:
        row = self._db.fetchone(
            "SELECT * FROM quality_signatures WHERE signature_id = ?", (signature_id,)
        )
        if row is None:
            return None
        return QualitySignature(
            signature_id=row["signature_id"],
            content_ref=row["content_ref"],
            modality=row["modality"],
            source_mode=row["source_mode"],
            feature_vector=_load_vector(row["feature_vector_json"]),
            neighbors=[
                QualityNeighbor.model_validate(item)
                for item in json.loads(row["neighbors_json"] or "[]")
            ],
            predicted_transitions=[
                ActionConditionedTransition.model_validate(item)
                for item in json.loads(row["predicted_transitions_json"] or "[]")
            ],
            valence_associations=json.loads(row["valence_associations_json"] or "{}"),
            discriminability=float(row["discriminability"]),
            adaptation=json.loads(row["adaptation_json"] or "{}"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def record_transition(
        self,
        *,
        owner_id: str,
        from_field_id: str | None,
        to_field_id: str | None,
        from_content_ref: str | None,
        to_content_ref: str | None,
        action_id: str | None = None,
        continuity_score: float = 0.0,
        prediction_match: float = 0.0,
        temporal_order: int = 0,
        metadata: dict | None = None,
    ) -> str:
        transition_id = f"trans_{secrets.token_urlsafe(12)}"
        self._db.execute(
            """
            INSERT INTO field_transitions(
                transition_id, owner_id, from_field_id, to_field_id, action_id,
                from_content_ref, to_content_ref, continuity_score,
                prediction_match, temporal_order, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transition_id,
                owner_id,
                from_field_id,
                to_field_id,
                action_id,
                from_content_ref,
                to_content_ref,
                _clamp01(continuity_score),
                _clamp01(prediction_match),
                temporal_order,
                json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                utc_now(),
            ),
        )
        return transition_id

    def _store(self, signature: QualitySignature) -> None:
        self._db.execute(
            """
            INSERT INTO quality_signatures(
                signature_id, content_ref, modality, source_mode,
                feature_vector_json, neighbors_json, predicted_transitions_json,
                valence_associations_json, discriminability, adaptation_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signature.signature_id,
                signature.content_ref,
                signature.modality,
                signature.source_mode.value,
                json.dumps(signature.feature_vector),
                json.dumps(
                    [item.model_dump(mode="json") for item in signature.neighbors],
                    sort_keys=True,
                ),
                json.dumps(
                    [
                        item.model_dump(mode="json")
                        for item in signature.predicted_transitions
                    ],
                    sort_keys=True,
                ),
                json.dumps(signature.valence_associations, sort_keys=True),
                signature.discriminability,
                json.dumps(signature.adaptation, sort_keys=True),
                signature.created_at,
                signature.updated_at,
            ),
        )


def _deterministic_vector(value: str, dimensions: int) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < dimensions:
        digest = hashlib.sha256(f"{counter}:{value}".encode("utf-8")).digest()
        values.extend((byte / 127.5) - 1.0 for byte in digest)
        counter += 1
    return _normalize(values[:dimensions])


def _stored_memory_provider() -> EmbeddingProvider | None:
    db_path = Path(
        os.getenv(
            "MEMORY_DB_PATH",
            str(Path.home() / ".claude" / "memories" / "memory.db"),
        )
    ).expanduser()
    return StoredMemoryEmbeddingProvider(db_path) if db_path.is_file() else None


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def _cosine_distance(a: list[float], b: list[float]) -> float:
    width = min(len(a), len(b))
    if width == 0:
        return 1.0
    left = _normalize(a[:width])
    right = _normalize(b[:width])
    similarity = sum(x * y for x, y in zip(left, right))
    return max(0.0, min(2.0, 1.0 - similarity))


def _load_vector(raw: str) -> list[float]:
    try:
        return [float(value) for value in json.loads(raw or "[]")]
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
