"""PostgreSQL-backed memory storage with pgvector + pgroonga."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any

import asyncpg

from .config import MemoryConfig
from .embeddings import EmbeddingAPIProvider, EmbeddingProvider, SentenceTransformerProvider
from .schema import get_all_ddl
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

logger = logging.getLogger(__name__)


def _parse_sensory_data(data: Any) -> tuple[SensoryData, ...]:
    if not data:
        return ()
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return ()
    if not isinstance(data, list):
        return ()
    return tuple(SensoryData.from_dict(d) for d in data)


def _parse_camera_position(data: Any) -> CameraPosition | None:
    if not data:
        return None
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(data, dict):
        return None
    return CameraPosition.from_dict(data)


def _row_to_memory(row: asyncpg.Record) -> Memory:
    """Convert a database row to a Memory object."""
    return Memory(
        id=str(row["id"]),
        content=row["content"],
        timestamp=row["created_at"].isoformat() if row["created_at"] else "",
        emotion=row["emotion"],
        importance=row["importance"],
        category=row["category"],
        access_count=row["access_count"],
        last_accessed=row["last_accessed"].isoformat() if row.get("last_accessed") else "",
        episode_id=str(row["episode_id"]) if row.get("episode_id") else None,
        sensory_data=_parse_sensory_data(row.get("sensory_data")),
        camera_position=_parse_camera_position(row.get("camera_position")),
        tags=tuple(row.get("tags") or []),
        novelty_score=float(row.get("novelty_score", 0.0)),
        prediction_error=float(row.get("prediction_error", 0.0)),
        activation_count=int(row.get("activation_count", 0)),
        last_activated=row["last_activated"].isoformat() if row.get("last_activated") else "",
    )


def _row_to_episode(row: asyncpg.Record) -> Episode:
    """Convert a database row to an Episode object."""
    return Episode(
        id=str(row["id"]),
        title=row["title"],
        start_time=row["start_time"].isoformat() if row["start_time"] else "",
        end_time=row["end_time"].isoformat() if row.get("end_time") else None,
        memory_ids=(),  # Filled via reverse lookup
        participants=tuple(row.get("participants") or []),
        location_context=row.get("location_context"),
        summary=row["summary"],
        emotion=row["emotion"],
        importance=row["importance"],
    )


class PostgresStore:
    """PostgreSQL storage backend for memories."""

    def __init__(self, config: MemoryConfig, embedding_provider: EmbeddingProvider | None = None):
        self._config = config
        self._pool: asyncpg.Pool | None = None
        self._embedding: EmbeddingProvider | None = embedding_provider

    def _get_embedding_provider(self) -> EmbeddingProvider:
        if self._embedding is None:
            if self._config.embedding_api_url:
                self._embedding = EmbeddingAPIProvider(base_url=self._config.embedding_api_url)
            else:
                self._embedding = SentenceTransformerProvider(self._config.embedding_model)
        return self._embedding

    async def connect(self) -> None:
        """Initialize connection pool and create schema."""
        from pgvector.asyncpg import register_vector

        self._pool = await asyncpg.create_pool(
            dsn=self._config.pg_dsn,
            min_size=self._config.pool_min_size,
            max_size=self._config.pool_max_size,
            init=register_vector,
        )

        # Create schema
        dim = self._get_embedding_provider().dimension()
        async with self._pool.acquire() as conn:
            for ddl in get_all_ddl(dim=dim):
                await conn.execute(ddl)

        logger.info("PostgreSQL schema initialized (dim=%d)", dim)

    async def disconnect(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    def _ensure_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgresStore not connected. Call connect() first.")
        return self._pool

    # ── Save ──

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
    ) -> Memory:
        """Save a new memory."""
        pool = self._ensure_pool()
        provider = self._get_embedding_provider()

        memory_id = str(uuid.uuid4())
        importance = max(1, min(5, importance))

        embeddings = await asyncio.to_thread(provider.embed, [content])
        embedding_vec = embeddings[0]

        sensory_json = json.dumps([s.to_dict() for s in sensory_data])
        camera_json = json.dumps(camera_position.to_dict()) if camera_position else None
        ep_id = uuid.UUID(episode_id) if episode_id else None

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO memories (id, content, embedding, emotion, importance, category,
                                      episode_id, sensory_data, camera_position, tags)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb, $10)
                RETURNING *
                """,
                uuid.UUID(memory_id),
                content,
                embedding_vec,
                emotion,
                importance,
                category,
                ep_id,
                sensory_json,
                camera_json,
                list(tags),
            )

        return _row_to_memory(row)

    # ── Get by ID ──

    async def get_by_id(self, memory_id: str) -> Memory | None:
        """Get a single memory by ID."""
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM memories WHERE id = $1",
                uuid.UUID(memory_id),
            )
        if row is None:
            return None
        return _row_to_memory(row)

    async def get_by_ids(self, memory_ids: list[str]) -> list[Memory]:
        """Get multiple memories by IDs."""
        if not memory_ids:
            return []
        pool = self._ensure_pool()
        uuids = [uuid.UUID(mid) for mid in memory_ids]
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM memories WHERE id = ANY($1)",
                uuids,
            )
        return [_row_to_memory(r) for r in rows]

    # ── Search (vector + time decay + emotion/importance boost) ──

    async def search(
        self,
        query: str,
        n_results: int = 5,
        emotion_filter: str | None = None,
        category_filter: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[MemorySearchResult]:
        """Semantic search with optional filters."""
        pool = self._ensure_pool()
        provider = self._get_embedding_provider()
        query_vec = (await asyncio.to_thread(provider.embed_query, [query]))[0]

        conditions = []
        params: list[Any] = [query_vec, n_results]
        idx = 3

        if emotion_filter:
            conditions.append(f"emotion = ${idx}")
            params.append(emotion_filter)
            idx += 1
        if category_filter:
            conditions.append(f"category = ${idx}")
            params.append(category_filter)
            idx += 1
        if date_from:
            conditions.append(f"created_at >= ${idx}::timestamptz")
            params.append(date_from)
            idx += 1
        if date_to:
            conditions.append(f"created_at <= ${idx}::timestamptz")
            params.append(date_to)
            idx += 1

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        sql = f"""
            SELECT *, (embedding <=> $1) AS distance
            FROM memories
            {where_clause}
            ORDER BY embedding <=> $1
            LIMIT $2
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [
            MemorySearchResult(memory=_row_to_memory(r), distance=float(r["distance"]))
            for r in rows
        ]

    async def search_with_scoring(
        self,
        query: str,
        n_results: int = 5,
        use_time_decay: bool = True,
        use_emotion_boost: bool = True,
        decay_half_life_days: float | None = None,
        emotion_filter: str | None = None,
        category_filter: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[ScoredMemory]:
        """Search with time decay + emotion boost scoring in SQL."""
        pool = self._ensure_pool()
        provider = self._get_embedding_provider()
        query_vec = (await asyncio.to_thread(provider.embed_query, [query]))[0]
        now = datetime.now()
        if decay_half_life_days is None:
            decay_half_life_days = self._config.half_life_days

        conditions = []
        params: list[Any] = [query_vec, now, decay_half_life_days]
        idx = 4

        if emotion_filter:
            conditions.append(f"emotion = ${idx}")
            params.append(emotion_filter)
            idx += 1
        if category_filter:
            conditions.append(f"category = ${idx}")
            params.append(category_filter)
            idx += 1
        if date_from:
            conditions.append(f"created_at >= ${idx}::timestamptz")
            params.append(date_from)
            idx += 1
        if date_to:
            conditions.append(f"created_at <= ${idx}::timestamptz")
            params.append(date_to)
            idx += 1

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        # Fetch more candidates, re-score, then limit
        fetch_count = min(n_results * 3, 50)
        params.append(fetch_count)
        limit_idx = idx

        sql = f"""
            SELECT *,
                (embedding <=> $1) AS semantic_distance,
                power(2.0, -EXTRACT(EPOCH FROM ($2::timestamptz - created_at)) / ($3::float8 * 86400.0))
                    AS time_decay,
                CASE emotion
                    WHEN 'excited'   THEN 0.4
                    WHEN 'surprised' THEN 0.35
                    WHEN 'moved'     THEN 0.3
                    WHEN 'sad'       THEN 0.25
                    WHEN 'happy'     THEN 0.2
                    WHEN 'nostalgic' THEN 0.15
                    WHEN 'curious'   THEN 0.1
                    ELSE 0.0
                END AS emotion_boost,
                LEAST((importance - 1) * 0.1, 0.4) AS importance_boost
            FROM memories
            {where_clause}
            ORDER BY embedding <=> $1
            LIMIT ${limit_idx}
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        results: list[ScoredMemory] = []
        for r in rows:
            sd = float(r["semantic_distance"])
            td = float(r["time_decay"]) if use_time_decay else 1.0
            eb = float(r["emotion_boost"]) if use_emotion_boost else 0.0
            ib = float(r["importance_boost"])

            decay_penalty = (1.0 - td) * 0.3
            total_boost = eb * 0.2 + ib * 0.2
            final = max(0.0, sd * 1.0 + decay_penalty - total_boost)

            results.append(
                ScoredMemory(
                    memory=_row_to_memory(r),
                    semantic_distance=sd,
                    time_decay_factor=td,
                    emotion_boost=eb,
                    importance_boost=ib,
                    final_score=final,
                )
            )

        results.sort(key=lambda x: x.final_score)
        return results[:n_results]

    # ── Hybrid search (vector + pgroonga full-text) ──

    async def hybrid_search(
        self,
        query: str,
        n_results: int = 5,
        vector_weight: float | None = None,
        text_weight: float | None = None,
    ) -> list[MemorySearchResult]:
        """Hybrid search combining vector similarity and pgroonga full-text."""
        pool = self._ensure_pool()
        provider = self._get_embedding_provider()
        query_vec = (await asyncio.to_thread(provider.embed_query, [query]))[0]
        if vector_weight is None:
            vector_weight = self._config.vector_weight
        if text_weight is None:
            text_weight = self._config.text_weight

        sql = """
            WITH vector_results AS (
                SELECT id, (embedding <=> $1) AS v_distance
                FROM memories
                ORDER BY embedding <=> $1
                LIMIT $2 * 4
            ),
            text_results AS (
                SELECT id, pgroonga_score(tableoid, ctid) AS t_score
                FROM memories
                WHERE content &@~ $3
                LIMIT $2 * 4
            ),
            merged AS (
                SELECT
                    COALESCE(v.id, t.id) AS id,
                    COALESCE(1.0 - v.v_distance, 0.0) AS v_score,
                    COALESCE(t.t_score, 0.0) AS t_score
                FROM vector_results v
                FULL OUTER JOIN text_results t ON v.id = t.id
            )
            SELECT
                m.*,
                mg.v_score,
                mg.t_score,
                (mg.v_score * $4 + mg.t_score * $5) AS hybrid_score
            FROM merged mg
            JOIN memories m ON m.id = mg.id
            ORDER BY hybrid_score DESC
            LIMIT $2
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, query_vec, n_results, query, vector_weight, text_weight)

        return [
            MemorySearchResult(
                memory=_row_to_memory(r),
                distance=1.0 - float(r["hybrid_score"]) if r["hybrid_score"] else 1.0,
            )
            for r in rows
        ]

    # ── List recent ──

    async def list_recent(
        self,
        limit: int = 10,
        category_filter: str | None = None,
    ) -> list[Memory]:
        """List recent memories sorted by created_at DESC."""
        pool = self._ensure_pool()

        if category_filter:
            sql = "SELECT * FROM memories WHERE category = $1 ORDER BY created_at DESC LIMIT $2"
            params: list[Any] = [category_filter, limit]
        else:
            sql = "SELECT * FROM memories ORDER BY created_at DESC LIMIT $1"
            params = [limit]

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [_row_to_memory(r) for r in rows]

    # ── Stats ──

    async def get_stats(self) -> MemoryStats:
        """Get memory statistics."""
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            total = await conn.fetchval("SELECT count(*) FROM memories")

            cat_rows = await conn.fetch(
                "SELECT category, count(*) AS cnt FROM memories GROUP BY category"
            )
            by_category = {r["category"]: r["cnt"] for r in cat_rows}

            emo_rows = await conn.fetch(
                "SELECT emotion, count(*) AS cnt FROM memories GROUP BY emotion"
            )
            by_emotion = {r["emotion"]: r["cnt"] for r in emo_rows}

            oldest = await conn.fetchval("SELECT min(created_at) FROM memories")
            newest = await conn.fetchval("SELECT max(created_at) FROM memories")

        return MemoryStats(
            total_count=total or 0,
            by_category=by_category,
            by_emotion=by_emotion,
            oldest_timestamp=oldest.isoformat() if oldest else None,
            newest_timestamp=newest.isoformat() if newest else None,
        )

    # ── Access tracking ──

    async def update_access(self, memory_id: str) -> None:
        """Increment access_count and update last_accessed."""
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE memories
                SET access_count = access_count + 1, last_accessed = now()
                WHERE id = $1
                """,
                uuid.UUID(memory_id),
            )

    # ── Field updates ──

    async def update_memory_fields(self, memory_id: str, **fields: Any) -> bool:
        """Partial update of memory fields."""
        if not fields:
            return False

        pool = self._ensure_pool()
        set_parts = []
        params: list[Any] = [uuid.UUID(memory_id)]
        idx = 2

        for key, value in fields.items():
            set_parts.append(f"{key} = ${idx}")
            params.append(value)
            idx += 1

        sql = f"UPDATE memories SET {', '.join(set_parts)} WHERE id = $1"
        async with pool.acquire() as conn:
            result = await conn.execute(sql, *params)

        return result != "UPDATE 0"

    async def record_activation(
        self,
        memory_id: str,
        prediction_error: float | None = None,
    ) -> bool:
        """Update activation info on recall."""
        pool = self._ensure_pool()

        if prediction_error is not None:
            pe = max(0.0, min(1.0, prediction_error))
            sql = """
                UPDATE memories
                SET activation_count = activation_count + 1,
                    last_activated = now(),
                    prediction_error = $2
                WHERE id = $1
            """
            params: list[Any] = [uuid.UUID(memory_id), pe]
        else:
            sql = """
                UPDATE memories
                SET activation_count = activation_count + 1,
                    last_activated = now()
                WHERE id = $1
            """
            params = [uuid.UUID(memory_id)]

        async with pool.acquire() as conn:
            result = await conn.execute(sql, *params)

        return result != "UPDATE 0"

    # ── Important memories (for working memory refresh) ──

    async def search_important_memories(
        self,
        min_importance: int = 4,
        min_access_count: int = 5,
        since: str | None = None,
        n_results: int = 10,
    ) -> list[Memory]:
        """Fetch important, frequently accessed memories."""
        pool = self._ensure_pool()

        if since:
            sql = """
                SELECT * FROM memories
                WHERE importance >= $1
                  AND access_count >= $2
                  AND last_accessed >= $3::timestamptz
                ORDER BY importance DESC, access_count DESC
                LIMIT $4
            """
            params: list[Any] = [min_importance, min_access_count, since, n_results]
        else:
            sql = """
                SELECT * FROM memories
                WHERE importance >= $1
                  AND access_count >= $2
                ORDER BY importance DESC, access_count DESC
                LIMIT $3
            """
            params = [min_importance, min_access_count, n_results]

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [_row_to_memory(r) for r in rows]

    # ── All memories (for camera position search) ──

    async def get_all(self) -> list[Memory]:
        """Get all memories."""
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM memories ORDER BY created_at DESC")
        return [_row_to_memory(r) for r in rows]

    # ── Episode ID ──

    async def update_episode_id(self, memory_id: str, episode_id: str) -> None:
        """Set episode_id on a memory."""
        pool = self._ensure_pool()
        ep_uuid = uuid.UUID(episode_id) if episode_id else None
        async with pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE memories SET episode_id = $2 WHERE id = $1",
                uuid.UUID(memory_id),
                ep_uuid,
            )
        if result == "UPDATE 0":
            raise ValueError(f"Memory not found: {memory_id}")

    # ── Links (memory_links table) ──

    async def add_link(
        self,
        source_id: str,
        target_id: str,
        link_type: str = "related",
        note: str | None = None,
    ) -> None:
        """Add a directional link between memories."""
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memory_links (source_id, target_id, link_type, note)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (source_id, target_id, link_type) DO NOTHING
                """,
                uuid.UUID(source_id),
                uuid.UUID(target_id),
                link_type,
                note,
            )

    async def add_bidirectional_link(
        self,
        source_id: str,
        target_id: str,
        link_type: str = "similar",
    ) -> None:
        """Add bidirectional links (for auto-linking)."""
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memory_links (source_id, target_id, link_type)
                VALUES ($1, $2, $3), ($2, $1, $3)
                ON CONFLICT (source_id, target_id, link_type) DO NOTHING
                """,
                uuid.UUID(source_id),
                uuid.UUID(target_id),
                link_type,
            )

    async def get_links(self, memory_id: str) -> list[MemoryLink]:
        """Get all outgoing links from a memory."""
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM memory_links WHERE source_id = $1",
                uuid.UUID(memory_id),
            )
        return [
            MemoryLink(
                target_id=str(r["target_id"]),
                link_type=r["link_type"],
                created_at=r["created_at"].isoformat(),
                note=r.get("note"),
            )
            for r in rows
        ]

    async def get_linked_memory_ids(self, memory_id: str) -> list[str]:
        """Get IDs of all linked memories (both directions)."""
        pool = self._ensure_pool()
        mid = uuid.UUID(memory_id)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT target_id AS linked_id FROM memory_links WHERE source_id = $1
                UNION
                SELECT DISTINCT source_id AS linked_id FROM memory_links WHERE target_id = $1
                """,
                mid,
            )
        return [str(r["linked_id"]) for r in rows]

    # ── Coactivation weights ──

    async def bump_coactivation(
        self,
        source_id: str,
        target_id: str,
        delta: float = 0.1,
    ) -> bool:
        """Increase coactivation weight between two memories (symmetric)."""
        pool = self._ensure_pool()
        a = uuid.UUID(source_id)
        b = uuid.UUID(target_id)
        # Ensure a < b for the CHECK constraint
        if a > b:
            a, b = b, a

        delta = max(0.0, min(1.0, delta))

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO coactivation_weights (memory_a, memory_b, weight, updated_at)
                VALUES ($1, $2, $3, now())
                ON CONFLICT (memory_a, memory_b) DO UPDATE
                SET weight = LEAST(1.0, coactivation_weights.weight + $3),
                    updated_at = now()
                """,
                a,
                b,
                delta,
            )
        return True

    async def get_coactivation_neighbors(
        self,
        memory_id: str,
        min_weight: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Get coactivation neighbors of a memory."""
        pool = self._ensure_pool()
        mid = uuid.UUID(memory_id)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    CASE WHEN memory_a = $1 THEN memory_b ELSE memory_a END AS neighbor_id,
                    weight
                FROM coactivation_weights
                WHERE (memory_a = $1 OR memory_b = $1)
                  AND weight >= $2
                ORDER BY weight DESC
                """,
                mid,
                min_weight,
            )
        return [(str(r["neighbor_id"]), float(r["weight"])) for r in rows]

    async def maybe_add_related_link(
        self,
        source_id: str,
        target_id: str,
        threshold: float = 0.6,
    ) -> bool:
        """Add 'related' link if coactivation weight exceeds threshold."""
        pool = self._ensure_pool()
        a = uuid.UUID(source_id)
        b = uuid.UUID(target_id)
        if a > b:
            a, b = b, a

        async with pool.acquire() as conn:
            weight = await conn.fetchval(
                "SELECT weight FROM coactivation_weights WHERE memory_a = $1 AND memory_b = $2",
                a,
                b,
            )

        if weight is None or weight < threshold:
            return False

        await self.add_link(source_id, target_id, "related", "auto-consolidated")
        return True

    # ── Association graph traversal (WITH RECURSIVE) ──

    async def spread_associations(
        self,
        seed_ids: list[str],
        max_depth: int = 3,
        max_branches: int = 3,
        min_coactivation: float = 0.3,
    ) -> list[tuple[Memory, int, float]]:
        """Spread activation through association graph using recursive CTE.

        Returns list of (memory, depth, activation_score).
        """
        if not seed_ids:
            return []

        pool = self._ensure_pool()
        seed_uuids = [uuid.UUID(sid) for sid in seed_ids]

        sql = """
            WITH RECURSIVE association_graph AS (
                -- Seeds
                SELECT
                    id AS memory_id,
                    0 AS depth,
                    1.0::real AS activation
                FROM memories
                WHERE id = ANY($1::uuid[])

                UNION

                -- Expansion (no ORDER BY / DISTINCT ON in recursive part)
                SELECT
                    edges.neighbor_id AS memory_id,
                    ag.depth + 1 AS depth,
                    (ag.activation * edges.edge_weight)::real AS activation
                FROM association_graph ag
                CROSS JOIN LATERAL (
                    SELECT * FROM (
                        -- Forward links
                        SELECT
                            ml.target_id AS neighbor_id,
                            CASE ml.link_type
                                WHEN 'similar'   THEN 1.0
                                WHEN 'caused_by' THEN 0.85
                                WHEN 'leads_to'  THEN 0.85
                                WHEN 'related'   THEN 0.8
                                ELSE 0.5
                            END::real AS edge_weight
                        FROM memory_links ml
                        WHERE ml.source_id = ag.memory_id

                        UNION ALL

                        -- Reverse links
                        SELECT
                            ml.source_id AS neighbor_id,
                            CASE ml.link_type
                                WHEN 'similar'   THEN 1.0
                                WHEN 'leads_to'  THEN 0.85
                                WHEN 'caused_by' THEN 0.85
                                WHEN 'related'   THEN 0.8
                                ELSE 0.5
                            END::real AS edge_weight
                        FROM memory_links ml
                        WHERE ml.target_id = ag.memory_id

                        UNION ALL

                        -- Coactivation weights
                        SELECT
                            CASE WHEN cw.memory_a = ag.memory_id THEN cw.memory_b
                                 ELSE cw.memory_a
                            END AS neighbor_id,
                            cw.weight AS edge_weight
                        FROM coactivation_weights cw
                        WHERE (cw.memory_a = ag.memory_id OR cw.memory_b = ag.memory_id)
                          AND cw.weight >= $4::real
                    ) all_edges
                    ORDER BY edge_weight DESC
                    LIMIT $3::int
                ) AS edges
                WHERE ag.depth < $2::int
                  AND ag.activation * edges.edge_weight > 0.1
            ),
            -- Deduplicate: keep highest activation per memory
            best AS (
                SELECT DISTINCT ON (memory_id)
                    memory_id, depth, activation
                FROM association_graph
                ORDER BY memory_id, activation DESC
            )
            SELECT
                b.memory_id,
                b.depth,
                b.activation,
                m.*
            FROM best b
            JOIN memories m ON m.id = b.memory_id
            ORDER BY b.activation DESC
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, seed_uuids, max_depth, max_branches, min_coactivation)

        results = []
        for r in rows:
            memory = _row_to_memory(r)
            results.append((memory, int(r["depth"]), float(r["activation"])))

        return results

    # ── Consolidation batch SQL ──

    async def consolidate_batch(
        self,
        cutoff_timestamp: datetime,
        delta: float = 0.2,
    ) -> dict[str, int]:
        """Run batch consolidation in SQL."""
        pool = self._ensure_pool()

        async with pool.acquire() as conn:
            # Batch coactivation update for adjacent memories
            result = await conn.execute(
                """
                WITH recent_pairs AS (
                    SELECT
                        id AS current_id,
                        LEAD(id) OVER (ORDER BY created_at) AS next_id
                    FROM memories
                    WHERE created_at >= $1
                    ORDER BY created_at
                )
                INSERT INTO coactivation_weights (memory_a, memory_b, weight, updated_at)
                SELECT
                    LEAST(current_id, next_id),
                    GREATEST(current_id, next_id),
                    $2,
                    now()
                FROM recent_pairs
                WHERE next_id IS NOT NULL
                ON CONFLICT (memory_a, memory_b) DO UPDATE
                SET
                    weight = LEAST(1.0, coactivation_weights.weight + EXCLUDED.weight),
                    updated_at = now()
                """,
                cutoff_timestamp,
                delta,
            )
            coactivation_count = int(result.split()[-1]) if result else 0

            # Decay prediction_error
            result2 = await conn.execute(
                """
                UPDATE memories
                SET prediction_error = prediction_error * 0.9
                WHERE created_at >= $1
                  AND prediction_error > 0.001
                """,
                cutoff_timestamp,
            )
            decay_count = int(result2.split()[-1]) if result2 else 0

            # Auto-add related links for high coactivation
            result3 = await conn.execute(
                """
                INSERT INTO memory_links (source_id, target_id, link_type, note)
                SELECT memory_a, memory_b, 'related', 'auto-consolidated'
                FROM coactivation_weights
                WHERE weight >= 0.6
                  AND updated_at >= now() - INTERVAL '1 hour'
                ON CONFLICT (source_id, target_id, link_type) DO NOTHING
                """,
            )
            link_count = int(result3.split()[-1]) if result3 else 0

        return {
            "coactivation_updates": coactivation_count,
            "prediction_error_decays": decay_count,
            "link_updates": link_count,
        }

    # ── Causal chain traversal ──

    async def get_causal_chain(
        self,
        memory_id: str,
        direction: str = "backward",
        max_depth: int = 5,
    ) -> list[tuple[Memory, str]]:
        """Trace causal chain using recursive CTE."""
        pool = self._ensure_pool()
        mid = uuid.UUID(memory_id)

        if direction == "backward":
            link_type = "caused_by"
        elif direction == "forward":
            link_type = "leads_to"
        else:
            raise ValueError(f"Invalid direction: {direction}")

        sql = """
            WITH RECURSIVE chain AS (
                SELECT target_id AS memory_id, link_type, 1 AS depth
                FROM memory_links
                WHERE source_id = $1 AND link_type = $2

                UNION ALL

                SELECT ml.target_id, ml.link_type, c.depth + 1
                FROM chain c
                JOIN memory_links ml ON ml.source_id = c.memory_id AND ml.link_type = $2
                WHERE c.depth < $3
            )
            SELECT c.memory_id, c.link_type, c.depth, m.*
            FROM chain c
            JOIN memories m ON m.id = c.memory_id
            ORDER BY c.depth
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, mid, link_type, max_depth)

        return [(_row_to_memory(r), r["link_type"]) for r in rows]

    # ── Linked memories (BFS via SQL) ──

    async def get_linked_memories(
        self,
        memory_id: str,
        depth: int = 1,
    ) -> list[Memory]:
        """Get linked memories up to a given depth using recursive CTE."""
        pool = self._ensure_pool()
        mid = uuid.UUID(memory_id)
        depth = max(1, min(5, depth))

        sql = """
            WITH RECURSIVE linked AS (
                SELECT target_id AS memory_id, 1 AS depth
                FROM memory_links
                WHERE source_id = $1

                UNION

                SELECT ml.target_id, l.depth + 1
                FROM linked l
                JOIN memory_links ml ON ml.source_id = l.memory_id
                WHERE l.depth < $2
            )
            SELECT DISTINCT m.*
            FROM linked l
            JOIN memories m ON m.id = l.memory_id
            WHERE l.memory_id != $1
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, mid, depth)

        return [_row_to_memory(r) for r in rows]

    # ── Auto-link on save ──

    async def save_with_auto_link(
        self,
        content: str,
        emotion: str = "neutral",
        importance: int = 3,
        category: str = "daily",
        link_threshold: float = 0.8,
        max_links: int = 5,
    ) -> Memory:
        """Save a memory and auto-link to similar existing memories."""
        # Search for similar memories first
        similar = await self.search(query=content, n_results=max_links)
        memories_to_link = [r for r in similar if r.distance <= link_threshold]

        # Save the memory
        memory = await self.save(
            content=content,
            emotion=emotion,
            importance=importance,
            category=category,
        )

        # Add bidirectional links
        for result in memories_to_link:
            await self.add_bidirectional_link(memory.id, result.memory.id, "similar")

        return memory

    # ── Episodes ──

    async def save_episode(self, episode: Episode) -> Episode:
        """Save an episode to the database."""
        pool = self._ensure_pool()
        provider = self._get_embedding_provider()

        # Generate embedding for the summary
        embeddings = await asyncio.to_thread(provider.embed, [episode.summary])
        embedding_vec = embeddings[0]

        ep_id = uuid.UUID(episode.id)
        start_time = datetime.fromisoformat(episode.start_time)
        end_time = datetime.fromisoformat(episode.end_time) if episode.end_time else None

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO episodes (id, title, start_time, end_time, participants,
                                      location_context, summary, embedding, emotion, importance)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (id) DO NOTHING
                """,
                ep_id,
                episode.title,
                start_time,
                end_time,
                list(episode.participants),
                episode.location_context,
                episode.summary,
                embedding_vec,
                episode.emotion,
                episode.importance,
            )

        return episode

    async def search_episodes(self, query: str, n_results: int = 5) -> list[Episode]:
        """Search episodes by summary semantic similarity."""
        pool = self._ensure_pool()
        provider = self._get_embedding_provider()
        query_vec = (await asyncio.to_thread(provider.embed_query, [query]))[0]

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT e.*,
                    array_agg(m.id ORDER BY m.created_at) FILTER (WHERE m.id IS NOT NULL) AS memory_ids
                FROM episodes e
                LEFT JOIN memories m ON m.episode_id = e.id
                GROUP BY e.id
                ORDER BY e.embedding <=> $1
                LIMIT $2
                """,
                query_vec,
                n_results,
            )

        episodes = []
        for r in rows:
            ep = _row_to_episode(r)
            mem_ids = tuple(str(mid) for mid in (r["memory_ids"] or []))
            episodes.append(
                Episode(
                    id=ep.id,
                    title=ep.title,
                    start_time=ep.start_time,
                    end_time=ep.end_time,
                    memory_ids=mem_ids,
                    participants=ep.participants,
                    location_context=ep.location_context,
                    summary=ep.summary,
                    emotion=ep.emotion,
                    importance=ep.importance,
                )
            )

        return episodes

    async def get_episode_by_id(self, episode_id: str) -> Episode | None:
        """Get an episode by ID, including its memory_ids."""
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT e.*,
                    array_agg(m.id ORDER BY m.created_at) FILTER (WHERE m.id IS NOT NULL) AS memory_ids
                FROM episodes e
                LEFT JOIN memories m ON m.episode_id = e.id
                WHERE e.id = $1
                GROUP BY e.id
                """,
                uuid.UUID(episode_id),
            )
        if row is None:
            return None

        ep = _row_to_episode(row)
        mem_ids = tuple(str(mid) for mid in (row["memory_ids"] or []))
        return Episode(
            id=ep.id,
            title=ep.title,
            start_time=ep.start_time,
            end_time=ep.end_time,
            memory_ids=mem_ids,
            participants=ep.participants,
            location_context=ep.location_context,
            summary=ep.summary,
            emotion=ep.emotion,
            importance=ep.importance,
        )

    async def list_all_episodes(self) -> list[Episode]:
        """List all episodes ordered by start_time DESC."""
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT e.*,
                    array_agg(m.id ORDER BY m.created_at) FILTER (WHERE m.id IS NOT NULL) AS memory_ids
                FROM episodes e
                LEFT JOIN memories m ON m.episode_id = e.id
                GROUP BY e.id
                ORDER BY e.start_time DESC
                """
            )

        episodes = []
        for r in rows:
            ep = _row_to_episode(r)
            mem_ids = tuple(str(mid) for mid in (r["memory_ids"] or []))
            episodes.append(
                Episode(
                    id=ep.id,
                    title=ep.title,
                    start_time=ep.start_time,
                    end_time=ep.end_time,
                    memory_ids=mem_ids,
                    participants=ep.participants,
                    location_context=ep.location_context,
                    summary=ep.summary,
                    emotion=ep.emotion,
                    importance=ep.importance,
                )
            )
        return episodes

    async def delete_episode(self, episode_id: str) -> None:
        """Delete an episode (memories keep their data, episode_id is SET NULL by FK)."""
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            # Clear episode_id from memories first
            await conn.execute(
                "UPDATE memories SET episode_id = NULL WHERE episode_id = $1",
                uuid.UUID(episode_id),
            )
            await conn.execute(
                "DELETE FROM episodes WHERE id = $1",
                uuid.UUID(episode_id),
            )
