"""Migration script for adding memory_type field to existing memories."""

import asyncio
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv

load_dotenv()


async def migrate_memory_type():
    """Migrate existing memories to add memory_type='global' field.

    This is a one-time migration for Phase 7 job isolation feature.
    Existing memories without memory_type field will be updated to have
    memory_type='global'.
    """
    # Get database path from environment or use default
    default_path = str(Path.home() / ".claude" / "memories" / "chroma")
    db_path = os.getenv("MEMORY_DB_PATH", default_path)
    collection_name = os.getenv("MEMORY_COLLECTION_NAME", "claude_memories")

    print(f"Connecting to ChromaDB at: {db_path}")
    print(f"Collection: {collection_name}")

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=db_path)

    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Collection not found: {e}")
        print("No migration needed.")
        return

    # Get all memories
    print("\nFetching all memories...")
    results = collection.get()

    if not results or not results.get("ids"):
        print("No memories found in database.")
        return

    ids = results["ids"]
    metadatas = results.get("metadatas", [])

    print(f"Total memories found: {len(ids)}")

    # Find memories without memory_type field
    memories_to_update = []
    for i, memory_id in enumerate(ids):
        metadata = metadatas[i] if i < len(metadatas) else {}
        if "memory_type" not in metadata:
            memories_to_update.append((memory_id, metadata))

    print(f"Memories needing migration: {len(memories_to_update)}")

    if not memories_to_update:
        print("\n[OK] All memories already have memory_type field.")
        print("Migration not needed.")
        return

    # Update memories
    print(f"\nUpdating {len(memories_to_update)} memories...")
    updated_count = 0

    for memory_id, metadata in memories_to_update:
        # Add memory_type='global' to metadata
        metadata["memory_type"] = "global"

        # Update the memory
        collection.update(ids=[memory_id], metadatas=[metadata])
        updated_count += 1

        if updated_count % 10 == 0:
            print(f"  Progress: {updated_count}/{len(memories_to_update)}")

    print(f"\n[OK] Migration complete!")
    print(f"  Updated {updated_count} memories with memory_type='global'")

    # Verify
    print("\nVerifying migration...")
    results_after = collection.get()
    metadatas_after = results_after.get("metadatas", [])

    missing_count = sum(1 for m in metadatas_after if "memory_type" not in m)

    if missing_count == 0:
        print("[OK] All memories now have memory_type field.")
    else:
        print(f"[WARNING] {missing_count} memories still missing memory_type field.")

    print(f"\nTotal memories in database: {len(results_after.get('ids', []))}")


def main():
    """Run the migration."""
    print("=" * 60)
    print("Memory Type Migration Tool")
    print("=" * 60)
    print()
    print("This script adds 'memory_type=global' to existing memories")
    print("that don't have this field (Phase 7 job isolation feature).")
    print()

    asyncio.run(migrate_memory_type())

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
