#!/usr/bin/env python3
"""Merge ChromaDB memories from a source database into the local database.

Usage:
    uv run python scripts/merge_memories.py /path/to/downloaded/chroma

This will:
1. Open the source ChromaDB (read-only)
2. Open the local ChromaDB (~/.claude/memories/chroma)
3. Import all memories that don't already exist (by ID)
4. Report what was imported
"""

import sys
from pathlib import Path

import chromadb

DEFAULT_LOCAL_PATH = Path.home() / ".claude" / "memories" / "chroma"
COLLECTIONS_TO_MERGE = ["claude_memories", "episodes"]
BATCH_SIZE = 100


def merge_collection(
    src_col: chromadb.Collection,
    dst_col: chromadb.Collection,
) -> tuple[int, int]:
    """Merge all documents from src_col into dst_col.

    Returns (imported_count, skipped_count).
    """
    total = src_col.count()
    if total == 0:
        return 0, 0

    imported = 0
    skipped = 0

    # Fetch all from source in batches
    for offset in range(0, total, BATCH_SIZE):
        result = src_col.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )

        ids = result["ids"]
        documents = result["documents"] or [None] * len(ids)
        metadatas = result["metadatas"] or [None] * len(ids)
        embeddings = result["embeddings"]

        # Check which IDs already exist in destination
        existing = dst_col.get(ids=ids, include=[])
        existing_ids = set(existing["ids"])

        new_ids = []
        new_docs = []
        new_metas = []
        new_embeds = []

        for i, doc_id in enumerate(ids):
            if doc_id in existing_ids:
                skipped += 1
                continue
            new_ids.append(doc_id)
            new_docs.append(documents[i])
            new_metas.append(metadatas[i])
            if embeddings is not None:
                new_embeds.append(embeddings[i])

        if new_ids:
            add_kwargs = {
                "ids": new_ids,
                "documents": new_docs,
                "metadatas": new_metas,
            }
            if len(new_embeds) > 0:
                add_kwargs["embeddings"] = new_embeds
            dst_col.add(**add_kwargs)
            imported += len(new_ids)

    return imported, skipped


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/merge_memories.py /path/to/source/chroma")
        sys.exit(1)

    src_path = Path(sys.argv[1])
    dst_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_LOCAL_PATH

    if not src_path.exists():
        print(f"Error: Source path does not exist: {src_path}")
        sys.exit(1)

    if not (src_path / "chroma.sqlite3").exists():
        # Maybe user downloaded the parent folder
        candidates = list(src_path.glob("**/chroma.sqlite3"))
        if candidates:
            src_path = candidates[0].parent
            print(f"Found ChromaDB at: {src_path}")
        else:
            print(f"Error: No chroma.sqlite3 found in {src_path}")
            sys.exit(1)

    print(f"Source: {src_path}")
    print(f"Destination: {dst_path}")
    print()

    src_client = chromadb.PersistentClient(path=str(src_path))
    dst_client = chromadb.PersistentClient(path=str(dst_path))

    src_collections = {c.name: c for c in src_client.list_collections()}
    dst_collections = {c.name: c for c in dst_client.list_collections()}

    print("Source collections:")
    for name, col in src_collections.items():
        print(f"  {name}: {col.count()} items")
    print()

    for col_name in COLLECTIONS_TO_MERGE:
        if col_name not in src_collections:
            print(f"[{col_name}] Not found in source, skipping")
            continue

        src_col = src_collections[col_name]
        if src_col.count() == 0:
            print(f"[{col_name}] Empty in source, skipping")
            continue

        # Get or create destination collection
        if col_name in dst_collections:
            dst_col = dst_collections[col_name]
        else:
            dst_col = dst_client.get_or_create_collection(
                name=col_name,
                metadata=src_col.metadata,
            )

        before = dst_col.count()
        imported, skipped = merge_collection(src_col, dst_col)
        after = dst_col.count()

        print(
            f"[{col_name}] imported: {imported}, skipped (duplicate): {skipped}, "
            f"total: {before} -> {after}"
        )

    print()
    print("Done!")


if __name__ == "__main__":
    main()
