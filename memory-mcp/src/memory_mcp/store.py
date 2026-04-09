"""SQLite + numpy backed memory storage (Phase 11: ChromaDB → SQLite+numpy)."""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import uuid
from datetime import datetime
from typing import Any

import numpy as np

from .association import (
    AssociationDiagnostics,
    AssociationEngine,
    adaptive_search_params,
)
from .bm25 import BM25Index
from .config import MemoryConfig
from .consolidation import ConsolidationEngine
from .embedding import E5EmbeddingFunction
from .hopfield import HopfieldRecallResult, ModernHopfieldNetwork
from .normalizer import get_reading, normalize_japanese
from .predictive import (
    PredictiveDiagnostics,
    calculate_context_relevance,
    calculate_novelty_score,
    calculate_prediction_error,
)
from .types import (
    CameraPosition,
    Episode,
    Memory,
    MemoryLink,
    MemorySearchResult,
    MemoryStats,
    ScoredMemory,
    SensoryData,
)
from .vector import cosine_similarity, decode_vector, encode_vector
from .working_memory import WorkingMemoryBuffer
from .workspace import (
    WorkspaceCandidate,
    diversity_score,
    select_workspace_candidates,
)

# ──────────────────────────────────────────────
# DDL
# ──────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    normalized_content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    emotion TEXT NOT NULL DEFAULT 'neutral',
    importance INTEGER NOT NULL DEFAULT 3,
    category TEXT NOT NULL DEFAULT 'daily',
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT NOT NULL DEFAULT '',
    linked_ids TEXT NOT NULL DEFAULT '',
    episode_id TEXT,
    sensory_data TEXT NOT NULL DEFAULT '',
    camera_position TEXT,
    tags TEXT NOT NULL DEFAULT '',
    links TEXT NOT NULL DEFAULT '',
    novelty_score REAL NOT NULL DEFAULT 0.0,
    prediction_error REAL NOT NULL DEFAULT 0.0,
    activation_count INTEGER NOT NULL DEFAULT 0,
    last_activated TEXT NOT NULL DEFAULT '',
    reading TEXT,
    pan_angle REAL,
    tilt_angle REAL
);
CREATE INDEX IF NOT EXISTS idx_memories_emotion    ON memories(emotion);
CREATE INDEX IF NOT EXISTS idx_memories_category   ON memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_timestamp  ON memories(timestamp);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
CREATE INDEX IF NOT EXISTS idx_memories_pan_angle  ON memories(pan_angle);
CREATE INDEX IF NOT EXISTS idx_memories_tilt_angle ON memories(tilt_angle);

CREATE TABLE IF NOT EXISTS embeddings (
    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS coactivation (
    source_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    weight REAL NOT NULL CHECK(weight >= 0.0 AND weight <= 1.0),
    PRIMARY KEY (source_id, target_id)
);
CREATE INDEX IF NOT EXISTS idx_coactivation_source ON coactivation(source_id);
CREATE INDEX IF NOT EXISTS idx_coactivation_target ON coactivation(target_id);

CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT,
    memory_ids TEXT NOT NULL DEFAULT '',
    participants TEXT NOT NULL DEFAULT '',
    location_context TEXT,
    summary TEXT NOT NULL DEFAULT '',
    emotion TEXT NOT NULL DEFAULT 'neutral',
    importance INTEGER NOT NULL DEFAULT 3
);

CREATE TABLE IF NOT EXISTS episode_embeddings (
    episode_id TEXT PRIMARY KEY REFERENCES episodes(id) ON DELETE CASCADE,
    vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    link_type TEXT NOT NULL DEFAULT 'similar',
    note TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(source_id, target_id, link_type)
);
CREATE INDEX IF NOT EXISTS idx_ml_source ON memory_links(source_id);
CREATE INDEX IF NOT EXISTS idx_ml_target ON memory_links(target_id);
"""

# ──────────────────────────────────────────────
# Score helpers (shared with memory.py callers)
# ──────────────────────────────────────────────

EMOTION_BOOST_MAP: dict[str, float] = {
    "excited": 0.4,
    "surprised": 0.35,
    "moved": 0.3,
    "sad": 0.25,
    "happy": 0.2,
    "nostalgic": 0.15,
    "curious": 0.1,
    "neutral": 0.0,
}


def calculate_time_decay(
    timestamp: str,
    now: datetime | None = None,
    half_life_days: float = 30.0,
) -> float:
    if now is None:
        now = datetime.now()
    try:
        memory_time = datetime.fromisoformat(timestamp)
    except ValueError:
        return 1.0
    age_seconds = (now - memory_time).total_seconds()
    if age_seconds < 0:
        return 1.0
    age_days = age_seconds / 86400
    decay = math.pow(2, -age_days / half_life_days)
    return max(0.0, min(1.0, decay))


def calculate_emotion_boost(emotion: str) -> float:
    return EMOTION_BOOST_MAP.get(emotion, 0.0)


def calculate_importance_boost(importance: int) -> float:
    clamped = max(1, min(5, importance))
    return (clamped - 1) / 10


def calculate_final_score(
    semantic_distance: float,
    time_decay: float,
    emotion_boost: float,
    importance_boost: float,
    category: str = "daily",
    semantic_weight: float = 1.0,
    decay_weight: float = 0.3,
    emotion_weight: float = 0.2,
    importance_weight: float = 0.2,
) -> float:
    decay_penalty = (1.0 - time_decay) * decay_weight
    total_boost = emotion_boost * emotion_weight + importance_boost * importance_weight
    if category == "core":
        total_boost += 0.3  # Core memories always surface first
    final = semantic_distance * semantic_weight + decay_penalty - total_boost
    return max(0.0, final)


# ──────────────────────────────────────────────
# Row → Memory helpers
# ──────────────────────────────────────────────


def _parse_linked_ids(linked_ids_str: str) -> tuple[str, ...]:
    if not linked_ids_str:
        return ()
    return tuple(id.strip() for id in linked_ids_str.split(",") if id.strip())


def _parse_sensory_data(sensory_data_json: str) -> tuple[SensoryData, ...]:
    if not sensory_data_json:
        return ()
    try:
        data_list = json.loads(sensory_data_json)
        return tuple(SensoryData.from_dict(d) for d in data_list)
    except (json.JSONDecodeError, KeyError, TypeError):
        return ()


def _parse_camera_position(camera_position_json: str | None) -> CameraPosition | None:
    if not camera_position_json:
        return None
    try:
        data = json.loads(camera_position_json)
        return CameraPosition.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _parse_tags(tags_str: str) -> tuple[str, ...]:
    if not tags_str:
        return ()
    return tuple(tag.strip() for tag in tags_str.split(",") if tag.strip())


def _parse_links(links_json: str) -> tuple[MemoryLink, ...]:
    if not links_json:
        return ()
    try:
        data_list = json.loads(links_json)
        return tuple(MemoryLink.from_dict(d) for d in data_list)
    except (json.JSONDecodeError, KeyError, TypeError):
        return ()


def _row_to_memory(row: sqlite3.Row, coactivation: tuple[tuple[str, float], ...] = ()) -> Memory:
    """Convert a SQLite Row from the memories table to a Memory object."""
    episode_id_raw = row["episode_id"]
    episode_id = episode_id_raw if episode_id_raw else None

    return Memory(
        id=row["id"],
        content=row["content"],
        timestamp=row["timestamp"],
        emotion=row["emotion"],
        importance=row["importance"],
        category=row["category"],
        access_count=row["access_count"],
        last_accessed=row["last_accessed"] or "",
        linked_ids=_parse_linked_ids(row["linked_ids"] or ""),
        episode_id=episode_id,
        sensory_data=_parse_sensory_data(row["sensory_data"] or ""),
        camera_position=_parse_camera_position(row["camera_position"]),
        tags=_parse_tags(row["tags"] or ""),
        links=_parse_links(row["links"] or ""),
        novelty_score=float(row["novelty_score"] or 0.0),
        prediction_error=float(row["prediction_error"] or 0.0),
        activation_count=int(row["activation_count"] or 0),
        last_activated=row["last_activated"] or "",
        coactivation_weights=coactivation,
    )


def _row_to_episode(row: sqlite3.Row) -> Episode:
    """Convert a SQLite Row from the episodes table to an Episode object."""
    memory_ids_raw = row["memory_ids"] or ""
    participants_raw = row["participants"] or ""
    return Episode(
        id=row["id"],
        title=row["title"],
        start_time=row["start_time"],
        end_time=row["end_time"] or None,
        memory_ids=tuple(memory_ids_raw.split(",") if memory_ids_raw else []),
        participants=tuple(participants_raw.split(",") if participants_raw else []),
        location_context=row["location_context"] or None,
        summary=row["summary"] or "",
        emotion=row["emotion"],
        importance=int(row["importance"]),
    )


# ──────────────────────────────────────────────
# MemoryStore
# ──────────────────────────────────────────────


class MemoryStore:
    """SQLite + numpy memory storage (Phase 11)."""

    def __init__(self, config: MemoryConfig):
        self._config = config
        self._db: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._working_memory = WorkingMemoryBuffer(capacity=20)
        self._association_engine = AssociationEngine()
        self._consolidation_engine = ConsolidationEngine()
        self._hopfield = ModernHopfieldNetwork(beta=4.0, n_iters=3)
        self._embedding_fn = E5EmbeddingFunction(config.embedding_model)
        self._bm25_index = BM25Index()

    # ── Connection ──────────────────────────────

    async def connect(self) -> None:
        """Open SQLite database and create tables."""
        async with self._lock:
            if self._db is None:
                db_path = self._config.db_path

                def _open() -> sqlite3.Connection:
                    conn = sqlite3.connect(db_path, check_same_thread=False)
                    conn.row_factory = sqlite3.Row
                    conn.execute("PRAGMA foreign_keys = ON")
                    conn.execute("PRAGMA journal_mode = WAL")
                    for stmt in _DDL.strip().split(";"):
                        stmt = stmt.strip()
                        if stmt:
                            try:
                                conn.execute(stmt)
                            except Exception:
                                pass  # Ignore DDL errors (e.g., index on column not yet migrated)
                    conn.commit()
                    # Migration: add pan_angle/tilt_angle if missing
                    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
                    if "pan_angle" not in existing_cols:
                        conn.execute("ALTER TABLE memories ADD COLUMN pan_angle REAL")
                        conn.execute("ALTER TABLE memories ADD COLUMN tilt_angle REAL")
                        rows = conn.execute(
                            "SELECT id, camera_position FROM memories WHERE camera_position IS NOT NULL"
                        ).fetchall()
                        for row in rows:
                            try:
                                data = json.loads(row[1])
                                conn.execute(
                                    "UPDATE memories SET pan_angle=?, tilt_angle=? WHERE id=?",
                                    (data.get("pan_angle"), data.get("tilt_angle"), row[0]),
                                )
                            except Exception:
                                pass
                        conn.commit()

                    # Migration: populate memory_links from legacy linked_ids / links columns
                    count = conn.execute("SELECT COUNT(*) FROM memory_links").fetchone()[0]
                    if count == 0:
                        now = datetime.now().isoformat()
                        rows = conn.execute("SELECT id, linked_ids, links FROM memories").fetchall()
                        for row in rows:
                            mem_id = row["id"]
                            # linked_ids (CSV) → type='similar', bidirectional
                            linked_ids_str = row["linked_ids"] or ""
                            if linked_ids_str:
                                for lid in linked_ids_str.split(","):
                                    lid = lid.strip()
                                    if lid:
                                        conn.execute(
                                            """INSERT OR IGNORE INTO memory_links
                                               (source_id, target_id, link_type, created_at)
                                               VALUES (?, ?, 'similar', ?)""",
                                            (mem_id, lid, now),
                                        )
                            # links (JSON) → structured links with type
                            links_json = row["links"] or ""
                            if links_json:
                                try:
                                    links_data = json.loads(links_json)
                                    for link in links_data:
                                        conn.execute(
                                            """INSERT OR IGNORE INTO memory_links
                                               (source_id, target_id, link_type, note, created_at)
                                               VALUES (?, ?, ?, ?, ?)""",
                                            (
                                                mem_id,
                                                link["target_id"],
                                                link["link_type"],
                                                link.get("note"),
                                                link.get("created_at", now),
                                            ),
                                        )
                                except (json.JSONDecodeError, KeyError):
                                    pass
                        conn.commit()

                    return conn

                self._db = await asyncio.to_thread(_open)

    async def disconnect(self) -> None:
        """Close the SQLite connection."""
        async with self._lock:
            if self._db is not None:
                await asyncio.to_thread(self._db.close)
                self._db = None

    def _ensure_connected(self) -> sqlite3.Connection:
        if self._db is None:
            raise RuntimeError("MemoryStore not connected. Call connect() first.")
        return self._db

    # ── Embedding helpers ───────────────────────

    async def _encode_document(self, text: str) -> list[float]:
        return (await asyncio.to_thread(self._embedding_fn, [text]))[0]

    async def _encode_query(self, text: str) -> list[float]:
        return (await asyncio.to_thread(self._embedding_fn.encode_query, [text]))[0]

    # ── Coactivation helpers ────────────────────

    def _get_coactivation(self, db: sqlite3.Connection, memory_id: str) -> tuple[tuple[str, float], ...]:
        rows = db.execute(
            "SELECT target_id, weight FROM coactivation WHERE source_id = ?",
            (memory_id,),
        ).fetchall()
        return tuple((row["target_id"], float(row["weight"])) for row in rows)

    def _fetch_coactivation_batch(
        self, db: sqlite3.Connection, memory_ids: list[str]
    ) -> dict[str, tuple[tuple[str, float], ...]]:
        """Fetch coactivation for multiple memories in one query."""
        if not memory_ids:
            return {}
        placeholders = ",".join("?" * len(memory_ids))
        rows = db.execute(
            f"SELECT source_id, target_id, weight FROM coactivation WHERE source_id IN ({placeholders})",
            memory_ids,
        ).fetchall()
        result: dict[str, list[tuple[str, float]]] = {mid: [] for mid in memory_ids}
        for row in rows:
            result[row["source_id"]].append((row["target_id"], float(row["weight"])))
        return {k: tuple(v) for k, v in result.items()}

    # ── Fetch helpers ───────────────────────────

    def _fetch_memory_by_id(self, db: sqlite3.Connection, memory_id: str) -> Memory | None:
        row = db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        coactivation = self._get_coactivation(db, memory_id)
        return _row_to_memory(row, coactivation)

    def _fetch_memories_by_ids_sync(self, db: sqlite3.Connection, memory_ids: list[str]) -> list[Memory]:
        if not memory_ids:
            return []
        placeholders = ",".join("?" * len(memory_ids))
        rows = db.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})", memory_ids
        ).fetchall()
        # Batch fetch coactivation to avoid N+1
        coactivation_map = self._fetch_coactivation_batch(db, memory_ids)
        memories: list[Memory] = []
        for row in rows:
            coactivation = coactivation_map.get(row["id"], ())
            memories.append(_row_to_memory(row, coactivation))
        return memories

    # ── Save ────────────────────────────────────

    async def save(
        self,
        content: str,
        emotion: str = "neutral",
        importance: int = 3,
        category: str = "daily",
        episode_id: str | None = None,
        sensory_data: tuple[SensoryData, ...] = (),
        camera_position: CameraPosition | None = None,
        tags: tuple[str, ...] = (),
        auto_link: bool = True,
        link_threshold: float = 0.8,
        max_links: int = 5,
    ) -> Memory:
        """Save a new memory."""
        db = self._ensure_connected()
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        importance = max(1, min(5, importance))

        memory = Memory(
            id=memory_id,
            content=content,
            timestamp=timestamp,
            emotion=emotion,
            importance=importance,
            category=category,
            episode_id=episode_id,
            sensory_data=sensory_data,
            camera_position=camera_position,
            tags=tags,
        )

        normalized_content = normalize_japanese(content)
        reading = get_reading(content)

        embedding = await self._encode_document(normalized_content)
        vector_blob = encode_vector(embedding)

        pan = camera_position.pan_angle if camera_position else None
        tilt = camera_position.tilt_angle if camera_position else None

        def _insert() -> None:
            meta = memory.to_metadata()
            db.execute(
                """INSERT INTO memories (
                    id, content, normalized_content, timestamp,
                    emotion, importance, category, access_count, last_accessed,
                    linked_ids, episode_id, sensory_data, camera_position,
                    tags, links, novelty_score, prediction_error,
                    activation_count, last_activated, reading,
                    pan_angle, tilt_angle
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    memory_id, content, normalized_content, timestamp,
                    emotion, importance, category,
                    meta.get("access_count", 0), meta.get("last_accessed", ""),
                    meta.get("linked_ids", ""), episode_id or None,
                    meta.get("sensory_data", ""),
                    meta.get("camera_position") or None,
                    meta.get("tags", ""), meta.get("links", ""),
                    0.0, 0.0, 0, "", reading,
                    pan, tilt,
                ),
            )
            db.execute(
                "INSERT INTO embeddings (memory_id, vector) VALUES (?,?)",
                (memory_id, vector_blob),
            )
            db.commit()

        await asyncio.to_thread(_insert)
        self._bm25_index.mark_dirty()
        await self._working_memory.add(memory)

        # Auto-link to similar existing memories
        if auto_link:
            similar = await self.search(query=content, n_results=max_links)
            for result in similar:
                if result.distance <= link_threshold and result.memory.id != memory_id:
                    await self.add_link(
                        source_id=memory_id,
                        target_id=result.memory.id,
                        link_type="similar",
                        bidirectional=True,
                    )

        return memory

    # ── Vector search helpers ───────────────────

    async def _vector_search(
        self,
        query: str,
        n_results: int,
        emotion_filter: str | None = None,
        category_filter: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[tuple[Memory, float]]:
        """Return (memory, cosine_distance) pairs, sorted ascending by distance."""
        db = self._ensure_connected()
        normalized_query = normalize_japanese(query)
        query_emb = await self._encode_query(normalized_query)
        query_vec = np.array(query_emb, dtype=np.float32)

        # Build WHERE clause for filters
        conditions: list[str] = []
        params: list[Any] = []
        if emotion_filter:
            conditions.append("m.emotion = ?")
            params.append(emotion_filter)
        if category_filter:
            conditions.append("m.category = ?")
            params.append(category_filter)
        if date_from:
            conditions.append("m.timestamp >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("m.timestamp <= ?")
            params.append(date_to)

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"SELECT m.*, e.vector FROM memories m JOIN embeddings e ON m.id = e.memory_id {where_clause}"

        def _query() -> list[tuple[sqlite3.Row, bytes]]:
            rows = db.execute(sql, params).fetchall()
            return [(row, bytes(row["vector"])) for row in rows]

        rows_with_vecs = await asyncio.to_thread(_query)
        if not rows_with_vecs:
            return []

        # Stack vectors for batch cosine similarity
        vecs = np.stack([decode_vector(blob) for _, blob in rows_with_vecs])
        scores = cosine_similarity(query_vec, vecs)  # higher = more similar

        # Convert similarity to distance (like ChromaDB cosine distance)
        # cosine distance = 1 - similarity
        indexed = list(enumerate(rows_with_vecs))
        ranked = sorted(indexed, key=lambda t: scores[t[0]], reverse=True)
        ranked = ranked[:n_results]

        results: list[tuple[Memory, float]] = []
        for idx, (row, _) in ranked:
            memory_id = row["id"]
            coactivation = await asyncio.to_thread(self._get_coactivation, db, memory_id)
            memory = _row_to_memory(row, coactivation)
            distance = float(1.0 - scores[idx])
            results.append((memory, distance))

        return results

    # ── search ──────────────────────────────────

    async def search(
        self,
        query: str,
        n_results: int = 5,
        emotion_filter: str | None = None,
        category_filter: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[MemorySearchResult]:
        """Search memories by semantic similarity."""
        pairs = await self._vector_search(
            query=query,
            n_results=n_results,
            emotion_filter=emotion_filter,
            category_filter=category_filter,
            date_from=date_from,
            date_to=date_to,
        )
        return [MemorySearchResult(memory=m, distance=d) for m, d in pairs]

    # ── search_with_scoring ─────────────────────

    async def search_with_scoring(
        self,
        query: str,
        n_results: int = 5,
        use_time_decay: bool = True,
        use_emotion_boost: bool = True,
        decay_half_life_days: float = 30.0,
        emotion_filter: str | None = None,
        category_filter: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[ScoredMemory]:
        """Search with time decay + emotion boost scoring."""
        fetch_count = min(n_results * 3, 50)
        pairs = await self._vector_search(
            query=query,
            n_results=fetch_count,
            emotion_filter=emotion_filter,
            category_filter=category_filter,
            date_from=date_from,
            date_to=date_to,
        )

        scored_results: list[ScoredMemory] = []
        now = datetime.now()

        for memory, semantic_distance in pairs:
            time_decay = (
                calculate_time_decay(memory.timestamp, now, decay_half_life_days)
                if use_time_decay
                else 1.0
            )
            emotion_boost = calculate_emotion_boost(memory.emotion) if use_emotion_boost else 0.0
            importance_boost = calculate_importance_boost(memory.importance)
            final_score = calculate_final_score(
                semantic_distance=semantic_distance,
                time_decay=time_decay,
                emotion_boost=emotion_boost,
                importance_boost=importance_boost,
                category=memory.category,
            )
            scored_results.append(
                ScoredMemory(
                    memory=memory,
                    semantic_distance=semantic_distance,
                    time_decay_factor=time_decay,
                    emotion_boost=emotion_boost,
                    importance_boost=importance_boost,
                    final_score=final_score,
                )
            )

        # Phase 9: BM25 hybrid re-ranking
        if self._config.enable_bm25 and scored_results:
            if self._bm25_index.is_dirty:
                all_memories = await self.get_all()
                await asyncio.to_thread(
                    self._bm25_index.build,
                    [(m.id, m.content) for m in all_memories],
                )
            result_ids = [sr.memory.id for sr in scored_results]
            bm25_scores = self._bm25_index.scores(query, result_ids)
            query_reading = get_reading(query)
            bm25_weight = 0.2
            reading_weight = 0.15
            reranked: list[ScoredMemory] = []
            for sr in scored_results:
                boost = bm25_scores.get(sr.memory.id, 0.0) * bm25_weight
                if query_reading:
                    doc_reading = get_reading(sr.memory.content) or ""
                    if doc_reading and query_reading == doc_reading:
                        boost += reading_weight
                reranked.append(
                    ScoredMemory(
                        memory=sr.memory,
                        semantic_distance=sr.semantic_distance,
                        time_decay_factor=sr.time_decay_factor,
                        emotion_boost=sr.emotion_boost,
                        importance_boost=sr.importance_boost,
                        final_score=sr.final_score - boost,
                    )
                )
            scored_results = reranked

        scored_results.sort(key=lambda x: x.final_score)
        return scored_results[:n_results]

    # ── recall ──────────────────────────────────

    async def recall(self, context: str, n_results: int = 3) -> list[MemorySearchResult]:
        """Recall using hybrid semantic + Hopfield scoring."""
        pool_size = min(n_results * 3, 20)
        scored_results = await self.search_with_scoring(
            query=context, n_results=pool_size, use_time_decay=True, use_emotion_boost=True
        )
        if not scored_results:
            return []

        try:
            hopfield_results = await self.hopfield_recall(query=context, n_results=pool_size, auto_load=True)
            hopfield_scores: dict[str, float] = {r.memory_id: r.hopfield_score for r in hopfield_results}
        except Exception:
            hopfield_scores = {}

        hopfield_weight = 0.15
        blended: list[tuple[ScoredMemory, float]] = []
        for sr in scored_results:
            h_score = hopfield_scores.get(sr.memory.id, 0.0)
            h_boost = max(0.0, h_score) * hopfield_weight
            blended.append((sr, sr.final_score - h_boost))

        blended.sort(key=lambda x: x[1])
        return [
            MemorySearchResult(memory=sr.memory, distance=blended_score)
            for sr, blended_score in blended[:n_results]
        ]

    # ── list_recent ─────────────────────────────

    async def list_recent(self, limit: int = 10, category_filter: str | None = None) -> list[Memory]:
        """List recent memories sorted by timestamp descending."""
        db = self._ensure_connected()

        def _fetch() -> list[sqlite3.Row]:
            if category_filter:
                return db.execute(
                    "SELECT * FROM memories WHERE category = ? ORDER BY timestamp DESC LIMIT ?",
                    (category_filter, limit),
                ).fetchall()
            return db.execute(
                "SELECT * FROM memories ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()

        rows = await asyncio.to_thread(_fetch)
        memories: list[Memory] = []
        for row in rows:
            coactivation = await asyncio.to_thread(self._get_coactivation, db, row["id"])
            memories.append(_row_to_memory(row, coactivation))
        return memories

    # ── get_stats ───────────────────────────────

    async def get_stats(self) -> MemoryStats:
        """Get statistics about stored memories."""
        db = self._ensure_connected()

        def _fetch() -> tuple[list[sqlite3.Row], str | None, str | None]:
            rows = db.execute("SELECT emotion, category, timestamp FROM memories").fetchall()
            oldest = db.execute("SELECT MIN(timestamp) FROM memories").fetchone()[0]
            newest = db.execute("SELECT MAX(timestamp) FROM memories").fetchone()[0]
            return rows, oldest, newest

        rows, oldest, newest = await asyncio.to_thread(_fetch)

        by_category: dict[str, int] = {}
        by_emotion: dict[str, int] = {}
        for row in rows:
            cat = row["category"] or "daily"
            emo = row["emotion"] or "neutral"
            by_category[cat] = by_category.get(cat, 0) + 1
            by_emotion[emo] = by_emotion.get(emo, 0) + 1

        return MemoryStats(
            total_count=len(rows),
            by_category=by_category,
            by_emotion=by_emotion,
            oldest_timestamp=oldest,
            newest_timestamp=newest,
        )

    # ── get_by_id / get_by_ids ──────────────────

    async def get_by_id(self, memory_id: str) -> Memory | None:
        db = self._ensure_connected()

        def _fetch() -> Memory | None:
            return self._fetch_memory_by_id(db, memory_id)

        return await asyncio.to_thread(_fetch)

    async def get_by_ids(self, memory_ids: list[str]) -> list[Memory]:
        if not memory_ids:
            return []
        db = self._ensure_connected()

        def _fetch() -> list[Memory]:
            return self._fetch_memories_by_ids_sync(db, memory_ids)

        return await asyncio.to_thread(_fetch)

    # ── get_all ─────────────────────────────────

    async def get_all(self) -> list[Memory]:
        """Return all memories."""
        db = self._ensure_connected()

        def _fetch() -> list[Memory]:
            rows = db.execute("SELECT * FROM memories").fetchall()
            memories: list[Memory] = []
            for row in rows:
                coactivation = self._get_coactivation(db, row["id"])
                memories.append(_row_to_memory(row, coactivation))
            return memories

        return await asyncio.to_thread(_fetch)

    # ── update_access ───────────────────────────

    async def update_access(self, memory_id: str) -> None:
        db = self._ensure_connected()

        def _update() -> None:
            db.execute(
                """UPDATE memories
                   SET access_count = access_count + 1,
                       last_accessed = ?
                   WHERE id = ?""",
                (datetime.now().isoformat(), memory_id),
            )
            db.commit()

        await asyncio.to_thread(_update)

    # ── update_episode_id ───────────────────────

    async def update_episode_id(self, memory_id: str, episode_id: str) -> None:
        db = self._ensure_connected()
        ep_val = episode_id if episode_id else None

        def _update() -> None:
            result = db.execute(
                "UPDATE memories SET episode_id = ? WHERE id = ?",
                (ep_val, memory_id),
            )
            if result.rowcount == 0:
                raise ValueError(f"Memory not found: {memory_id}")
            db.commit()

        await asyncio.to_thread(_update)

    # ── update_memory_fields ────────────────────

    async def update_memory_fields(self, memory_id: str, **fields: Any) -> bool:
        """Update arbitrary fields on a memory row."""
        if not fields:
            return True
        db = self._ensure_connected()

        # Map field names to column names (most are 1:1)
        valid_cols = {
            "access_count", "last_accessed", "linked_ids", "episode_id",
            "sensory_data", "camera_position", "tags", "links",
            "novelty_score", "prediction_error", "activation_count",
            "last_activated", "reading",
        }
        valid = {k: v for k, v in fields.items() if k in valid_cols}
        if not valid:
            return True

        set_clause = ", ".join(f"{k} = ?" for k in valid)
        values = list(valid.values()) + [memory_id]

        def _update() -> bool:
            result = db.execute(f"UPDATE memories SET {set_clause} WHERE id = ?", values)
            db.commit()
            return result.rowcount > 0

        return await asyncio.to_thread(_update)

    # ── record_activation ───────────────────────

    async def record_activation(
        self,
        memory_id: str,
        prediction_error: float | None = None,
    ) -> bool:
        memory = await self.get_by_id(memory_id)
        if memory is None:
            return False
        payload: dict[str, Any] = {
            "activation_count": memory.activation_count + 1,
            "last_activated": datetime.now().isoformat(),
        }
        if prediction_error is not None:
            payload["prediction_error"] = max(0.0, min(1.0, prediction_error))
        return await self.update_memory_fields(memory_id, **payload)

    # ── bump_coactivation ───────────────────────

    async def bump_coactivation(
        self,
        source_id: str,
        target_id: str,
        delta: float = 0.1,
    ) -> bool:
        """Increment coactivation weights symmetrically (no N+1 existence check)."""
        db = self._ensure_connected()
        delta = max(0.0, min(1.0, delta))

        def _bump() -> bool:
            # Lightweight existence check: one COUNT query instead of two get_by_id
            count = db.execute(
                "SELECT COUNT(*) FROM memories WHERE id IN (?, ?)", (source_id, target_id)
            ).fetchone()[0]
            if count < 2:
                return False
            for s_id, t_id in [(source_id, target_id), (target_id, source_id)]:
                db.execute(
                    """INSERT INTO coactivation (source_id, target_id, weight)
                       VALUES (?, ?, ?)
                       ON CONFLICT(source_id, target_id)
                       DO UPDATE SET weight = MIN(1.0, coactivation.weight + excluded.weight)""",
                    (s_id, t_id, delta),
                )
            db.commit()
            return True

        return await asyncio.to_thread(_bump)

    async def decay_coactivation(self, factor: float = 0.95) -> int:
        """Multiply all coactivation weights by factor; prune near-zero entries."""
        db = self._ensure_connected()

        def _decay() -> int:
            db.execute("UPDATE coactivation SET weight = weight * ?", (factor,))
            result = db.execute("DELETE FROM coactivation WHERE weight < 0.01")
            db.commit()
            return result.rowcount

        return await asyncio.to_thread(_decay)

    # ── maybe_add_related_link ──────────────────

    async def maybe_add_related_link(
        self,
        source_id: str,
        target_id: str,
        threshold: float = 0.6,
    ) -> bool:
        db = self._ensure_connected()

        def _check() -> float | None:
            row = db.execute(
                "SELECT weight FROM coactivation WHERE source_id = ? AND target_id = ?",
                (source_id, target_id),
            ).fetchone()
            return float(row["weight"]) if row else None

        weight = await asyncio.to_thread(_check)
        if weight is None or weight < threshold:
            return False
        await self.add_link(
            source_id=source_id,
            target_id=target_id,
            link_type="related",
            note="auto-linked by consolidation replay",
        )
        return True

    # ── delete ──────────────────────────────────

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory and clean up all references."""
        db = self._ensure_connected()

        def _delete() -> bool:
            # Check existence
            row = db.execute("SELECT id, linked_ids, episode_id FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if row is None:
                return False

            # Clean up legacy linked_ids CSV on related memories
            linked_ids_str = row["linked_ids"] or ""
            if linked_ids_str:
                linked = [lid.strip() for lid in linked_ids_str.split(",") if lid.strip()]
                for lid in linked:
                    other = db.execute("SELECT id, linked_ids FROM memories WHERE id = ?", (lid,)).fetchone()
                    if other:
                        other_links = [x.strip() for x in (other["linked_ids"] or "").split(",") if x.strip()]
                        other_links = [x for x in other_links if x != memory_id]
                        db.execute(
                            "UPDATE memories SET linked_ids = ? WHERE id = ?",
                            (",".join(other_links), lid),
                        )
            # memory_links rows cascade via ON DELETE CASCADE (foreign key)

            # Remove from episode memory_ids
            episode_id = row["episode_id"]
            if episode_id:
                ep_row = db.execute("SELECT id, memory_ids FROM episodes WHERE id = ?", (episode_id,)).fetchone()
                if ep_row:
                    ep_mids = [m.strip() for m in (ep_row["memory_ids"] or "").split(",") if m.strip()]
                    ep_mids = [m for m in ep_mids if m != memory_id]
                    db.execute(
                        "UPDATE episodes SET memory_ids = ? WHERE id = ?",
                        (",".join(ep_mids), episode_id),
                    )

            # Delete memory (embeddings + coactivation cascade)
            db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            db.commit()
            return True

        result = await asyncio.to_thread(_delete)
        if result:
            self._bm25_index.mark_dirty()
            self._working_memory.remove(memory_id)
        return result

    # ── update ──────────────────────────────────

    async def update(
        self,
        memory_id: str,
        content: str | None = None,
        emotion: str | None = None,
        importance: int | None = None,
        category: str | None = None,
    ) -> bool:
        """Update a memory's user-facing fields. Re-embeds if content changes."""
        db = self._ensure_connected()

        existing = await self.get_by_id(memory_id)
        if existing is None:
            return False

        fields: dict[str, Any] = {}
        if emotion is not None:
            fields["emotion"] = emotion
        if importance is not None:
            fields["importance"] = max(1, min(5, importance))
        if category is not None:
            fields["category"] = category

        new_embedding: list[float] | None = None

        if content is not None and content != existing.content:
            normalized = normalize_japanese(content)
            reading = get_reading(content)
            new_embedding = await self._encode_document(normalized)
            fields["content"] = content
            fields["normalized_content"] = normalized
            fields["reading"] = reading

        if not fields:
            return True

        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [memory_id]

        def _update() -> None:
            db.execute(f"UPDATE memories SET {set_clause} WHERE id = ?", values)
            if new_embedding is not None:
                vector_blob = encode_vector(new_embedding)
                db.execute(
                    "INSERT OR REPLACE INTO embeddings (memory_id, vector) VALUES (?,?)",
                    (memory_id, vector_blob),
                )
            db.commit()

        await asyncio.to_thread(_update)
        if content is not None:
            self._bm25_index.mark_dirty()
        return True

    # ── save_with_auto_link ─────────────────────

    async def save_with_auto_link(
        self,
        content: str,
        emotion: str = "neutral",
        importance: int = 3,
        category: str = "daily",
        link_threshold: float = 0.8,
        max_links: int = 5,
    ) -> Memory:
        similar_memories = await self.search(query=content, n_results=max_links)
        memories_to_link = [r.memory for r in similar_memories if r.distance <= link_threshold]
        linked_ids = tuple(m.id for m in memories_to_link)

        db = self._ensure_connected()
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        importance = max(1, min(5, importance))

        memory = Memory(
            id=memory_id,
            content=content,
            timestamp=timestamp,
            emotion=emotion,
            importance=importance,
            category=category,
            linked_ids=linked_ids,
        )

        normalized_content = normalize_japanese(content)
        reading = get_reading(content)
        embedding = await self._encode_document(normalized_content)
        vector_blob = encode_vector(embedding)

        def _insert() -> None:
            meta = memory.to_metadata()
            db.execute(
                """INSERT INTO memories (
                    id, content, normalized_content, timestamp,
                    emotion, importance, category, access_count, last_accessed,
                    linked_ids, episode_id, sensory_data, camera_position,
                    tags, links, novelty_score, prediction_error,
                    activation_count, last_activated, reading
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    memory_id, content, normalized_content, timestamp,
                    emotion, importance, category,
                    0, "",
                    meta.get("linked_ids", ""), None,
                    "", None, "", "", 0.0, 0.0, 0, "", reading,
                ),
            )
            db.execute(
                "INSERT INTO embeddings (memory_id, vector) VALUES (?,?)",
                (memory_id, vector_blob),
            )
            db.commit()

        await asyncio.to_thread(_insert)
        self._bm25_index.mark_dirty()

        for target_id in linked_ids:
            await self._add_bidirectional_link(memory_id, target_id)

        return memory

    # ── _add_bidirectional_link ─────────────────

    async def _add_bidirectional_link(self, source_id: str, target_id: str) -> None:
        db = self._ensure_connected()

        def _link() -> None:
            for mem_id, other_id in [(source_id, target_id), (target_id, source_id)]:
                row = db.execute("SELECT linked_ids FROM memories WHERE id = ?", (mem_id,)).fetchone()
                if row is None:
                    continue
                current = _parse_linked_ids(row["linked_ids"] or "")
                if other_id not in current:
                    new_linked = ",".join(current + (other_id,))
                    db.execute("UPDATE memories SET linked_ids = ? WHERE id = ?", (new_linked, mem_id))
            db.commit()

        await asyncio.to_thread(_link)

    # ── memory_links table operations ───────────

    async def add_link(
        self,
        source_id: str,
        target_id: str,
        link_type: str = "similar",
        note: str | None = None,
        bidirectional: bool = False,
    ) -> None:
        """Add a link to the memory_links table."""
        db = self._ensure_connected()
        now = datetime.now().isoformat()

        def _insert() -> None:
            db.execute(
                """INSERT OR IGNORE INTO memory_links
                   (source_id, target_id, link_type, note, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (source_id, target_id, link_type, note, now),
            )
            if bidirectional:
                db.execute(
                    """INSERT OR IGNORE INTO memory_links
                       (source_id, target_id, link_type, note, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (target_id, source_id, link_type, note, now),
                )
            db.commit()

        await asyncio.to_thread(_insert)

    async def remove_link(
        self,
        source_id: str,
        target_id: str,
        link_type: str | None = None,
    ) -> bool:
        """Remove link(s) between two memories. Returns True if any removed."""
        db = self._ensure_connected()

        def _delete() -> int:
            if link_type is not None:
                r = db.execute(
                    "DELETE FROM memory_links WHERE source_id=? AND target_id=? AND link_type=?",
                    (source_id, target_id, link_type),
                )
            else:
                r = db.execute(
                    "DELETE FROM memory_links WHERE source_id=? AND target_id=?",
                    (source_id, target_id),
                )
            db.commit()
            return r.rowcount

        return await asyncio.to_thread(_delete) > 0

    async def get_links_from(
        self,
        memory_id: str,
        link_type: str | None = None,
    ) -> list[dict]:
        """Get all outgoing links from a memory."""
        db = self._ensure_connected()

        def _fetch() -> list[dict]:
            if link_type is not None:
                rows = db.execute(
                    "SELECT * FROM memory_links WHERE source_id=? AND link_type=? ORDER BY created_at",
                    (memory_id, link_type),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM memory_links WHERE source_id=? ORDER BY created_at",
                    (memory_id,),
                ).fetchall()
            return [dict(row) for row in rows]

        return await asyncio.to_thread(_fetch)

    async def get_links_to(
        self,
        memory_id: str,
        link_type: str | None = None,
    ) -> list[dict]:
        """Get all incoming links to a memory (reverse query)."""
        db = self._ensure_connected()

        def _fetch() -> list[dict]:
            if link_type is not None:
                rows = db.execute(
                    "SELECT * FROM memory_links WHERE target_id=? AND link_type=? ORDER BY created_at",
                    (memory_id, link_type),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM memory_links WHERE target_id=? ORDER BY created_at",
                    (memory_id,),
                ).fetchall()
            return [dict(row) for row in rows]

        return await asyncio.to_thread(_fetch)

    async def _get_links_batch(self, memory_ids: list[str]) -> list[dict]:
        """Fetch all links where any of memory_ids is source OR target (one query)."""
        if not memory_ids:
            return []
        db = self._ensure_connected()
        placeholders = ",".join("?" * len(memory_ids))

        def _fetch() -> list[dict]:
            rows = db.execute(
                f"""SELECT * FROM memory_links
                    WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})""",
                memory_ids + memory_ids,
            ).fetchall()
            return [dict(row) for row in rows]

        return await asyncio.to_thread(_fetch)

    # ── get_linked_memories ─────────────────────

    async def get_linked_memories(self, memory_id: str, depth: int = 1) -> list[Memory]:
        """BFS over memory_links table (all link types, bidirectional)."""
        depth = max(1, min(5, depth))
        visited: set[str] = {memory_id}
        result: list[Memory] = []
        current_ids = [memory_id]

        for _ in range(depth):
            if not current_ids:
                break
            # Batch fetch links for current level
            links = await self._get_links_batch(current_ids)
            next_ids: list[str] = []
            for link in links:
                # traverse both directions (bidirectional)
                for tid in (link["target_id"], link["source_id"]):
                    if tid not in visited and tid != memory_id:
                        visited.add(tid)
                        next_ids.append(tid)
            if next_ids:
                fetched = await asyncio.to_thread(
                    self._fetch_memories_by_ids_sync, self._ensure_connected(), next_ids
                )
                result.extend(fetched)
            current_ids = next_ids

        return result

    # ── recall_with_chain ───────────────────────

    async def recall_with_chain(
        self,
        context: str,
        n_results: int = 3,
        chain_depth: int = 1,
    ) -> list[MemorySearchResult]:
        main_results = await self.recall(context=context, n_results=n_results)
        seen_ids: set[str] = {r.memory.id for r in main_results}
        linked_results: list[MemorySearchResult] = []
        for result in main_results:
            linked = await self.get_linked_memories(memory_id=result.memory.id, depth=chain_depth)
            for mem in linked:
                if mem.id not in seen_ids:
                    seen_ids.add(mem.id)
                    # Use parent distance + small penalty instead of sentinel 999.0
                    linked_results.append(
                        MemorySearchResult(memory=mem, distance=result.distance + 0.1)
                    )
        return main_results + linked_results

    # ── add_causal_link ─────────────────────────

    async def add_causal_link(
        self,
        source_id: str,
        target_id: str,
        link_type: str = "caused_by",
        note: str | None = None,
    ) -> None:
        """Add a structured causal/related link via the memory_links table."""
        # Validate both exist
        db = self._ensure_connected()

        def _exists(mid: str) -> bool:
            return db.execute("SELECT 1 FROM memories WHERE id = ?", (mid,)).fetchone() is not None

        if not await asyncio.to_thread(_exists, source_id):
            raise ValueError(f"Source memory not found: {source_id}")
        if not await asyncio.to_thread(_exists, target_id):
            raise ValueError(f"Target memory not found: {target_id}")

        # Also write to legacy JSON column for backward compatibility
        source_memory = await self.get_by_id(source_id)
        if source_memory is not None:
            new_link = MemoryLink(
                target_id=target_id,
                link_type=link_type,
                created_at=datetime.now().isoformat(),
                note=note,
            )
            existing_links = list(source_memory.links)
            if not any(lk.target_id == target_id and lk.link_type == link_type for lk in existing_links):
                updated_links = tuple(existing_links + [new_link])
                links_json = json.dumps([lk.to_dict() for lk in updated_links])
                await self.update_memory_fields(source_id, links=links_json)

        await self.add_link(source_id, target_id, link_type, note)

    # ── get_causal_chain ────────────────────────

    async def get_causal_chain(
        self,
        memory_id: str,
        direction: str = "backward",
        max_depth: int = 5,
    ) -> list[tuple[Memory, str]]:
        """Follow causal links via memory_links table (batch queries, no N+1)."""
        max_depth = max(1, min(5, max_depth))
        if direction == "backward":
            target_link_types = ("caused_by",)
        elif direction == "forward":
            target_link_types = ("leads_to",)
        else:
            raise ValueError(f"Invalid direction: {direction}")

        db = self._ensure_connected()
        placeholders = ",".join("?" * len(target_link_types))

        visited: set[str] = {memory_id}
        result: list[tuple[Memory, str]] = []
        current_ids = [memory_id]

        for _ in range(max_depth):
            if not current_ids:
                break
            src_ph = ",".join("?" * len(current_ids))

            def _fetch_links(cur_ids: list[str] = current_ids) -> list[dict]:
                rows = db.execute(
                    f"""SELECT source_id, target_id, link_type FROM memory_links
                        WHERE source_id IN ({src_ph}) AND link_type IN ({placeholders})""",
                    cur_ids + list(target_link_types),
                ).fetchall()
                return [dict(row) for row in rows]

            links = await asyncio.to_thread(_fetch_links)
            next_ids = [lk["target_id"] for lk in links if lk["target_id"] not in visited]
            visited.update(next_ids)

            if next_ids:
                fetched = await asyncio.to_thread(
                    self._fetch_memories_by_ids_sync, db, next_ids
                )
                mem_map = {m.id: m for m in fetched}
                for link in links:
                    tid = link["target_id"]
                    if tid in mem_map:
                        result.append((mem_map[tid], link["link_type"]))

            current_ids = next_ids

        return result

    # ── search_important_memories ───────────────

    async def search_important_memories(
        self,
        min_importance: int = 4,
        min_access_count: int = 5,
        since: str | None = None,
        n_results: int = 10,
    ) -> list[Memory]:
        db = self._ensure_connected()

        def _fetch() -> list[Memory]:
            conditions = [
                "importance >= ?",
                "access_count >= ?",
            ]
            params: list[Any] = [min_importance, min_access_count]
            if since:
                conditions.append("last_accessed >= ?")
                params.append(since)
            where = " AND ".join(conditions)
            rows = db.execute(
                f"SELECT * FROM memories WHERE {where} ORDER BY last_accessed DESC LIMIT ?",
                params + [n_results],
            ).fetchall()
            memories: list[Memory] = []
            for row in rows:
                coactivation = self._get_coactivation(db, row["id"])
                memories.append(_row_to_memory(row, coactivation))
            return memories

        return await asyncio.to_thread(_fetch)

    # ── get_memories_by_camera_position ─────────

    async def get_memories_by_camera_position(
        self,
        pan_angle: float,
        tilt_angle: float,
        tolerance: float = 15.0,
    ) -> list[Memory]:
        """Return memories captured near the given camera angle (SQL-indexed)."""
        db = self._ensure_connected()

        def _fetch() -> list[Memory]:
            rows = db.execute(
                """SELECT * FROM memories
                   WHERE pan_angle BETWEEN ? AND ?
                     AND tilt_angle BETWEEN ? AND ?
                   ORDER BY timestamp DESC""",
                (
                    pan_angle - tolerance, pan_angle + tolerance,
                    tilt_angle - tolerance, tilt_angle + tolerance,
                ),
            ).fetchall()
            memories: list[Memory] = []
            for row in rows:
                coactivation = self._get_coactivation(db, row["id"])
                memories.append(_row_to_memory(row, coactivation))
            return memories

        return await asyncio.to_thread(_fetch)

    # ── get_working_memory ──────────────────────

    def get_working_memory(self) -> WorkingMemoryBuffer:
        return self._working_memory

    # ── Episode CRUD ────────────────────────────

    async def save_episode(self, episode: Episode) -> None:
        """Persist an Episode to the episodes table and save its embedding."""
        db = self._ensure_connected()

        def _insert() -> None:
            db.execute(
                """INSERT INTO episodes
                   (id, title, start_time, end_time, memory_ids, participants,
                    location_context, summary, emotion, importance)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    episode.id,
                    episode.title,
                    episode.start_time,
                    episode.end_time or None,
                    ",".join(episode.memory_ids),
                    ",".join(episode.participants),
                    episode.location_context,
                    episode.summary,
                    episode.emotion,
                    episode.importance,
                ),
            )
            db.commit()

        await asyncio.to_thread(_insert)

        # Save embedding for semantic search
        embed_text = f"{episode.title} {episode.summary}".strip()
        if embed_text:
            embedding = await self._encode_document(embed_text)
            vector_blob = encode_vector(embedding)

            def _insert_embedding() -> None:
                db.execute(
                    "INSERT OR REPLACE INTO episode_embeddings (episode_id, vector) VALUES (?,?)",
                    (episode.id, vector_blob),
                )
                db.commit()

            await asyncio.to_thread(_insert_embedding)

    async def get_episode_by_id(self, episode_id: str) -> Episode | None:
        db = self._ensure_connected()

        def _fetch() -> Episode | None:
            row = db.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
            if row is None:
                return None
            return _row_to_episode(row)

        return await asyncio.to_thread(_fetch)

    async def search_episodes(self, query: str, n_results: int = 5) -> list[Episode]:
        """Search episodes by semantic similarity, with LIKE fallback."""
        db = self._ensure_connected()

        # Try semantic search first
        def _fetch_with_embeddings() -> list[tuple[str, bytes]]:
            rows = db.execute(
                "SELECT ee.episode_id, ee.vector FROM episode_embeddings ee"
            ).fetchall()
            return [(row["episode_id"], row["vector"]) for row in rows]

        ep_vectors = await asyncio.to_thread(_fetch_with_embeddings)

        if ep_vectors:
            query_emb = await self._encode_query(normalize_japanese(query))
            query_vec = np.array(query_emb, dtype=np.float32)

            ep_ids = [ep_id for ep_id, _ in ep_vectors]
            vecs = np.stack([decode_vector(blob) for _, blob in ep_vectors])
            scores = cosine_similarity(query_vec, vecs)  # shape (n,), higher = more similar
            order = np.argsort(scores)[::-1]  # descending

            top_ids = [ep_ids[i] for i in order[:n_results]]

            def _fetch_by_ids() -> list[Episode]:
                if not top_ids:
                    return []
                placeholders = ",".join("?" * len(top_ids))
                rows = db.execute(
                    f"SELECT * FROM episodes WHERE id IN ({placeholders})",
                    top_ids,
                ).fetchall()
                id_to_ep = {_row_to_episode(row).id: _row_to_episode(row) for row in rows}
                return [id_to_ep[eid] for eid in top_ids if eid in id_to_ep]

            return await asyncio.to_thread(_fetch_by_ids)

        # Fallback: LIKE search
        pattern = f"%{query}%"

        def _fetch_like() -> list[Episode]:
            rows = db.execute(
                """SELECT * FROM episodes
                   WHERE title LIKE ? OR summary LIKE ?
                   ORDER BY start_time DESC LIMIT ?""",
                (pattern, pattern, n_results),
            ).fetchall()
            return [_row_to_episode(row) for row in rows]

        return await asyncio.to_thread(_fetch_like)

    async def list_all_episodes(self) -> list[Episode]:
        db = self._ensure_connected()

        def _fetch() -> list[Episode]:
            rows = db.execute("SELECT * FROM episodes ORDER BY start_time DESC").fetchall()
            return [_row_to_episode(row) for row in rows]

        return await asyncio.to_thread(_fetch)

    async def delete_episode(self, episode_id: str) -> None:
        db = self._ensure_connected()

        def _delete() -> None:
            db.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
            db.commit()

        await asyncio.to_thread(_delete)

    # ── Divergent recall ─────────────────────────

    async def recall_divergent(
        self,
        context: str,
        n_results: int = 5,
        max_branches: int = 3,
        max_depth: int = 3,
        temperature: float = 0.7,
        include_diagnostics: bool = False,
        record_activation: bool = True,
    ) -> tuple[list[MemorySearchResult], dict[str, Any]]:
        n_results = max(1, min(20, n_results))
        seed_size = max(3, min(25, n_results * 3))
        seeds = await self.search_with_scoring(query=context, n_results=seed_size)
        if not seeds:
            return [], {}

        branch_limit, depth_limit = adaptive_search_params(
            context=context,
            requested_branches=max_branches,
            requested_depth=max_depth,
            seed_count=len(seeds),
        )

        seed_memories = [item.memory for item in seeds]
        expanded, assoc_diag = await self._association_engine.spread(
            seeds=seed_memories,
            fetch_memories_by_ids=self.get_by_ids,
            max_branches=branch_limit,
            max_depth=depth_limit,
        )

        distance_map = {item.memory.id: item.semantic_distance for item in seeds}
        all_candidates: dict[str, Memory] = {}
        for memory in seed_memories + expanded:
            all_candidates[memory.id] = memory

        workspace_candidates: list[WorkspaceCandidate] = []
        prediction_errors: list[float] = []
        novelty_scores: list[float] = []

        for memory in all_candidates.values():
            semantic_distance = distance_map.get(memory.id)
            if semantic_distance is None:
                relevance = calculate_context_relevance(context, memory)
            else:
                relevance = 1.0 / (1.0 + max(0.0, semantic_distance))

            prediction_error = calculate_prediction_error(context, memory)
            novelty = calculate_novelty_score(memory, prediction_error)
            emotion_boost = calculate_emotion_boost(memory.emotion)
            normalized_emotion = max(0.0, min(1.0, emotion_boost / 0.4))

            prediction_errors.append(prediction_error)
            novelty_scores.append(novelty)
            workspace_candidates.append(
                WorkspaceCandidate(
                    memory=memory,
                    relevance=relevance,
                    novelty=novelty,
                    prediction_error=prediction_error,
                    emotion_boost=normalized_emotion,
                )
            )

        selected = select_workspace_candidates(
            candidates=workspace_candidates,
            max_results=n_results,
            temperature=temperature,
        )

        results: list[MemorySearchResult] = []
        selected_memories: list[Memory] = []
        for candidate, utility in selected:
            selected_memories.append(candidate.memory)
            if record_activation:
                await self.record_activation(
                    candidate.memory.id,
                    prediction_error=candidate.prediction_error,
                )
                await self.update_memory_fields(
                    candidate.memory.id,
                    novelty_score=candidate.novelty,
                    prediction_error=candidate.prediction_error,
                )
            score_distance = max(0.0, 1.0 - utility)
            results.append(MemorySearchResult(memory=candidate.memory, distance=score_distance))

        if not include_diagnostics:
            return results, {}

        diagnostics = self._build_divergent_diagnostics(
            context=context,
            association=assoc_diag,
            selected=selected_memories,
            prediction_errors=prediction_errors,
            novelty_scores=novelty_scores,
            branch_limit=branch_limit,
            depth_limit=depth_limit,
        )
        return results, diagnostics

    async def get_association_diagnostics(self, context: str, sample_size: int = 20) -> dict[str, Any]:
        n_results = max(3, min(20, sample_size))
        _, diagnostics = await self.recall_divergent(
            context=context,
            n_results=n_results,
            max_branches=4,
            max_depth=3,
            include_diagnostics=True,
            record_activation=False,
        )
        return diagnostics

    async def consolidate_memories(
        self,
        window_hours: int = 24,
        max_replay_events: int = 200,
        link_update_strength: float = 0.2,
    ) -> dict[str, int]:
        stats = await self._consolidation_engine.run(
            store=self,
            window_hours=window_hours,
            max_replay_events=max_replay_events,
            link_update_strength=link_update_strength,
        )
        return stats.to_dict()

    # ── Hopfield ─────────────────────────────────

    async def hopfield_load(self) -> int:
        """Load all embeddings from SQLite into Hopfield network."""
        db = self._ensure_connected()

        def _fetch() -> list[tuple[str, bytes, str]]:
            sql = (
                "SELECT e.memory_id, e.vector, m.normalized_content"
                " FROM embeddings e JOIN memories m ON m.id = e.memory_id"
            )
            return db.execute(sql).fetchall()

        rows = await asyncio.to_thread(_fetch)
        if not rows:
            self._hopfield.store([], [], [])
            return 0

        ids = [r[0] for r in rows]
        embeddings = [decode_vector(bytes(r[1])).tolist() for r in rows]
        contents = [r[2] for r in rows]
        self._hopfield.store(embeddings, ids, contents)
        return self._hopfield.n_memories

    async def hopfield_recall(
        self,
        query: str,
        n_results: int = 5,
        beta: float | None = None,
        auto_load: bool = True,
    ) -> list[HopfieldRecallResult]:
        if auto_load and not self._hopfield.is_loaded:
            await self.hopfield_load()
        if not self._hopfield.is_loaded:
            return []

        original_beta = self._hopfield.beta
        if beta is not None:
            self._hopfield.beta = beta

        try:
            normalized_query = normalize_japanese(query)
            query_emb = await self._encode_query(normalized_query)
            _, similarities = self._hopfield.retrieve(query_emb)
            results = self._hopfield.recall_results(similarities, k=n_results)
        finally:
            self._hopfield.beta = original_beta

        return results

    # ── Diagnostics helper ───────────────────────

    def _build_divergent_diagnostics(
        self,
        context: str,
        association: AssociationDiagnostics,
        selected: list[Memory],
        prediction_errors: list[float],
        novelty_scores: list[float],
        branch_limit: int,
        depth_limit: int,
    ) -> dict[str, Any]:
        avg_prediction_error = sum(prediction_errors) / len(prediction_errors) if prediction_errors else 0.0
        avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0
        predictive = PredictiveDiagnostics(
            avg_prediction_error=avg_prediction_error,
            avg_novelty=avg_novelty,
        )
        return {
            "context": context,
            "branches_used": association.branches_used,
            "depth_used": association.depth_used,
            "adaptive_branch_limit": branch_limit,
            "adaptive_depth_limit": depth_limit,
            "traversed_edges": association.traversed_edges,
            "expanded_nodes": association.expanded_nodes,
            "avg_branching_factor": association.avg_branching_factor,
            "selected_count": len(selected),
            "diversity_score": diversity_score(selected),
            "avg_prediction_error": predictive.avg_prediction_error,
            "avg_novelty": predictive.avg_novelty,
        }
