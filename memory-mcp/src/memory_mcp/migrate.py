"""Migration script: ChromaDB → PostgreSQL."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid as uuid_mod

import asyncpg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def migrate(
    chroma_path: str,
    pg_dsn: str,
    embedding_model: str = "intfloat/multilingual-e5-base",
    batch_size: int = 100,
) -> None:
    """Migrate data from ChromaDB to PostgreSQL."""
    try:
        import chromadb
    except ImportError:
        logger.error("chromadb is required for migration. Install with: uv sync --extra migrate")
        sys.exit(1)

    from .embeddings import SentenceTransformerProvider
    from .schema import get_all_ddl

    # Initialize embedding provider
    logger.info("Loading embedding model: %s", embedding_model)
    provider = SentenceTransformerProvider(embedding_model)
    dim = provider.dimension()
    logger.info("Embedding dimension: %d", dim)

    # Connect to ChromaDB
    logger.info("Connecting to ChromaDB at: %s", chroma_path)
    chroma_client = chromadb.PersistentClient(path=chroma_path)

    # Connect to PostgreSQL
    logger.info("Connecting to PostgreSQL: %s", pg_dsn)
    pool = await asyncpg.create_pool(dsn=pg_dsn, min_size=2, max_size=10)

    # Create schema
    async with pool.acquire() as conn:
        for ddl in get_all_ddl(dim=dim):
            await conn.execute(ddl)
    logger.info("Schema created/verified")

    # Migrate memories
    try:
        collection = chroma_client.get_collection("claude_memories")
    except Exception:
        logger.warning("No 'claude_memories' collection found. Skipping memory migration.")
        collection = None

    if collection:
        await _migrate_memories(collection, pool, provider, batch_size)

    # Migrate episodes
    try:
        episodes_collection = chroma_client.get_collection("episodes")
    except Exception:
        logger.warning("No 'episodes' collection found. Skipping episode migration.")
        episodes_collection = None

    if episodes_collection:
        await _migrate_episodes(episodes_collection, pool, provider, batch_size)

    await pool.close()
    logger.info("Migration complete!")


async def _migrate_memories(
    collection,
    pool: asyncpg.Pool,
    provider,
    batch_size: int,
) -> None:
    """Migrate memories from ChromaDB to PostgreSQL."""
    logger.info("Fetching all memories from ChromaDB...")
    all_data = collection.get(include=["documents", "metadatas", "embeddings"])

    ids = all_data.get("ids", [])
    documents = all_data.get("documents", [])
    metadatas = all_data.get("metadatas", [])

    total = len(ids)
    logger.info("Found %d memories to migrate", total)

    if total == 0:
        return

    # Process in batches
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_ids = ids[start:end]
        batch_docs = documents[start:end]
        batch_metas = metadatas[start:end]

        # Re-embed with new model
        logger.info("Embedding batch %d-%d / %d", start + 1, end, total)
        new_embeddings = provider.embed(batch_docs)

        # Prepare rows
        rows = []
        link_rows = []
        coactivation_rows = []

        for i, memory_id in enumerate(batch_ids):
            content = batch_docs[i]
            metadata = batch_metas[i] if i < len(batch_metas) else {}
            embedding = new_embeddings[i]

            # Parse metadata
            emotion = metadata.get("emotion", "neutral")
            importance = int(metadata.get("importance", 3))
            category = metadata.get("category", "daily")
            timestamp = metadata.get("timestamp", "")
            access_count = int(metadata.get("access_count", 0))
            last_accessed = metadata.get("last_accessed", "")
            episode_id = metadata.get("episode_id", "")
            tags_str = metadata.get("tags", "")
            sensory_data = metadata.get("sensory_data", "[]")
            camera_position = metadata.get("camera_position", "")
            novelty_score = float(metadata.get("novelty_score", 0.0))
            prediction_error = float(metadata.get("prediction_error", 0.0))
            activation_count = int(metadata.get("activation_count", 0))
            last_activated = metadata.get("last_activated", "")

            # Ensure valid UUID
            try:
                mid = uuid_mod.UUID(memory_id)
            except ValueError:
                mid = uuid_mod.uuid4()
                logger.warning("Invalid UUID '%s', generated new: %s", memory_id, mid)

            # Parse tags
            tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []

            # Parse episode_id
            ep_id = None
            if episode_id:
                try:
                    ep_id = uuid_mod.UUID(episode_id)
                except ValueError:
                    pass

            # Parse timestamps
            from datetime import datetime

            try:
                created_at = datetime.fromisoformat(timestamp) if timestamp else datetime.now()
            except ValueError:
                created_at = datetime.now()

            last_accessed_dt = None
            if last_accessed:
                try:
                    last_accessed_dt = datetime.fromisoformat(last_accessed)
                except ValueError:
                    pass

            last_activated_dt = None
            if last_activated:
                try:
                    last_activated_dt = datetime.fromisoformat(last_activated)
                except ValueError:
                    pass

            rows.append((
                mid, content, embedding, created_at, emotion, importance, category,
                access_count, last_accessed_dt, ep_id,
                sensory_data if sensory_data else "[]",
                camera_position if camera_position else None,
                tags, novelty_score, prediction_error, activation_count, last_activated_dt,
            ))

            # Parse linked_ids → memory_links
            linked_ids_str = metadata.get("linked_ids", "")
            if linked_ids_str:
                for lid in linked_ids_str.split(","):
                    lid = lid.strip()
                    if lid:
                        try:
                            target = uuid_mod.UUID(lid)
                            link_rows.append((mid, target, "similar"))
                        except ValueError:
                            pass

            # Parse links JSON → memory_links
            links_json = metadata.get("links", "")
            if links_json:
                try:
                    links_list = json.loads(links_json)
                    for link_data in links_list:
                        try:
                            target = uuid_mod.UUID(link_data["target_id"])
                            link_type = link_data.get("link_type", "related")
                            note = link_data.get("note")
                            link_rows.append((mid, target, link_type, note))
                        except (ValueError, KeyError):
                            pass
                except (json.JSONDecodeError, TypeError):
                    pass

            # Parse coactivation → coactivation_weights
            coactivation_json = metadata.get("coactivation", "")
            if coactivation_json:
                try:
                    if isinstance(coactivation_json, str):
                        coactivation_dict = json.loads(coactivation_json)
                    else:
                        coactivation_dict = coactivation_json
                    if isinstance(coactivation_dict, dict):
                        for other_id, weight in coactivation_dict.items():
                            try:
                                other_uuid = uuid_mod.UUID(other_id)
                                w = max(0.0, min(1.0, float(weight)))
                                a, b = (mid, other_uuid) if mid < other_uuid else (other_uuid, mid)
                                coactivation_rows.append((a, b, w))
                            except (ValueError, TypeError):
                                pass
                except (json.JSONDecodeError, TypeError):
                    pass

        # Bulk insert memories
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO memories (id, content, embedding, created_at, emotion, importance,
                                      category, access_count, last_accessed, episode_id,
                                      sensory_data, camera_position, tags,
                                      novelty_score, prediction_error, activation_count,
                                      last_activated)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11::jsonb, $12::jsonb, $13, $14, $15, $16, $17)
                ON CONFLICT (id) DO NOTHING
                """,
                rows,
            )

        logger.info("Inserted %d memories (%d-%d)", len(rows), start + 1, end)

    # Insert links (after all memories are in)
    if link_rows:
        logger.info("Inserting %d memory links...", len(link_rows))
        async with pool.acquire() as conn:
            # Handle both 3-tuple and 4-tuple link rows
            for lr in link_rows:
                try:
                    if len(lr) == 3:
                        await conn.execute(
                            """
                            INSERT INTO memory_links (source_id, target_id, link_type)
                            VALUES ($1, $2, $3)
                            ON CONFLICT (source_id, target_id, link_type) DO NOTHING
                            """,
                            *lr,
                        )
                    else:
                        await conn.execute(
                            """
                            INSERT INTO memory_links (source_id, target_id, link_type, note)
                            VALUES ($1, $2, $3, $4)
                            ON CONFLICT (source_id, target_id, link_type) DO NOTHING
                            """,
                            *lr,
                        )
                except asyncpg.ForeignKeyViolationError:
                    pass  # Target memory doesn't exist

    # Insert coactivation weights
    if coactivation_rows:
        # Deduplicate
        seen = set()
        deduped = []
        for a, b, w in coactivation_rows:
            key = (a, b)
            if key not in seen:
                seen.add(key)
                deduped.append((a, b, w))

        logger.info("Inserting %d coactivation weights...", len(deduped))
        async with pool.acquire() as conn:
            for a, b, w in deduped:
                try:
                    await conn.execute(
                        """
                        INSERT INTO coactivation_weights (memory_a, memory_b, weight)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (memory_a, memory_b) DO UPDATE
                        SET weight = GREATEST(coactivation_weights.weight, EXCLUDED.weight)
                        """,
                        a, b, w,
                    )
                except asyncpg.ForeignKeyViolationError:
                    pass

    # Verify
    async with pool.acquire() as conn:
        pg_count = await conn.fetchval("SELECT count(*) FROM memories")
        link_count = await conn.fetchval("SELECT count(*) FROM memory_links")
        coact_count = await conn.fetchval("SELECT count(*) FROM coactivation_weights")

    logger.info(
        "Verification: %d memories, %d links, %d coactivation weights in PostgreSQL",
        pg_count, link_count, coact_count,
    )


async def _migrate_episodes(
    collection,
    pool: asyncpg.Pool,
    provider,
    batch_size: int,
) -> None:
    """Migrate episodes from ChromaDB to PostgreSQL."""
    logger.info("Fetching all episodes from ChromaDB...")
    all_data = collection.get(include=["documents", "metadatas"])

    ids = all_data.get("ids", [])
    documents = all_data.get("documents", [])
    metadatas = all_data.get("metadatas", [])

    total = len(ids)
    logger.info("Found %d episodes to migrate", total)

    if total == 0:
        return

    from datetime import datetime

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_ids = ids[start:end]
        batch_docs = documents[start:end]
        batch_metas = metadatas[start:end]

        # Embed episode summaries
        embeddings = provider.embed(batch_docs)

        rows = []
        for i, ep_id in enumerate(batch_ids):
            summary = batch_docs[i]
            metadata = batch_metas[i] if i < len(batch_metas) else {}
            embedding = embeddings[i]

            try:
                eid = uuid_mod.UUID(ep_id)
            except ValueError:
                eid = uuid_mod.uuid4()

            title = metadata.get("title", "Untitled")
            start_time_str = metadata.get("start_time", "")
            end_time_str = metadata.get("end_time", "")
            participants_str = metadata.get("participants", "")
            emotion = metadata.get("emotion", "neutral")
            importance = int(metadata.get("importance", 3))
            location_context = metadata.get("location_context", "") or None

            try:
                start_time = datetime.fromisoformat(start_time_str) if start_time_str else datetime.now()
            except ValueError:
                start_time = datetime.now()

            end_time = None
            if end_time_str:
                try:
                    end_time = datetime.fromisoformat(end_time_str)
                except ValueError:
                    pass

            participants = [p.strip() for p in participants_str.split(",") if p.strip()] if participants_str else []

            rows.append((
                eid, title, start_time, end_time, participants,
                location_context, summary, embedding, emotion, importance,
            ))

        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO episodes (id, title, start_time, end_time, participants,
                                      location_context, summary, embedding, emotion, importance)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (id) DO NOTHING
                """,
                rows,
            )

        logger.info("Inserted %d episodes (%d-%d)", len(rows), start + 1, end)

    async with pool.acquire() as conn:
        ep_count = await conn.fetchval("SELECT count(*) FROM episodes")
    logger.info("Verification: %d episodes in PostgreSQL", ep_count)


def main() -> None:
    """CLI entrypoint for migration."""
    parser = argparse.ArgumentParser(description="Migrate ChromaDB memories to PostgreSQL")
    parser.add_argument(
        "--chroma-path",
        default=None,
        help="Path to ChromaDB data directory (default: ~/.claude/memories/chroma)",
    )
    parser.add_argument(
        "--pg-dsn",
        default=None,
        help="PostgreSQL DSN (default: from MEMORY_PG_DSN env var)",
    )
    parser.add_argument(
        "--embedding-model",
        default="intfloat/multilingual-e5-base",
        help="Embedding model name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing",
    )

    args = parser.parse_args()

    import os
    from pathlib import Path

    chroma_path = args.chroma_path or os.getenv(
        "MEMORY_DB_PATH",
        str(Path.home() / ".claude" / "memories" / "chroma"),
    )
    pg_dsn = args.pg_dsn or os.getenv(
        "MEMORY_PG_DSN",
        "postgresql://memory_mcp:changeme@localhost:5432/embodied_claude",
    )

    asyncio.run(migrate(
        chroma_path=chroma_path,
        pg_dsn=pg_dsn,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
    ))


if __name__ == "__main__":
    main()
