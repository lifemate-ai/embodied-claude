"""Memory backends for the desire system.

This module abstracts over the storage used to look up "when was desire X last
satisfied?" and to record new satisfaction events. The project has migrated
from ChromaDB to a SQLite-based memory store; the adapter keeps a legacy
Chroma path available for users who have not completed the migration.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol


def _default_sqlite_path() -> Path:
    return Path(
        os.getenv(
            "MEMORY_DB_FILE",
            str(Path.home() / ".claude" / "memories" / "memory.db"),
        )
    ).expanduser()


def _default_chroma_path() -> Path:
    return Path(
        os.getenv(
            "MEMORY_DB_PATH",
            str(Path.home() / ".claude" / "memories" / "chroma"),
        )
    ).expanduser()


class DesireMemoryAdapter(Protocol):
    """Minimum surface the desire system needs from the memory store."""

    def latest_satisfaction_ts(self, keywords: Iterable[str]) -> datetime | None: ...

    def record_satisfaction(
        self,
        *,
        desire_name: str,
        summary: str,
        ts: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> str: ...


@dataclass(slots=True)
class SQLiteMemoryAdapter:
    """Read-latest / append-minimal adapter against the memory-mcp SQLite store."""

    db_path: Path

    def latest_satisfaction_ts(self, keywords: Iterable[str]) -> datetime | None:
        kw_list = [k for k in keywords if k]
        if not kw_list:
            return None
        if not self.db_path.exists():
            return None
        clauses = " OR ".join(["content LIKE ?"] * len(kw_list))
        params = [f"%{k}%" for k in kw_list]
        # Pull more than one row because memory.db may contain rows with
        # malformed timestamps; we pick the most recent *parseable* one.
        query = (
            "SELECT timestamp FROM memories "
            f"WHERE {clauses} "
            "ORDER BY timestamp DESC LIMIT 50"
        )
        try:
            conn = sqlite3.connect(str(self.db_path))
            try:
                rows = conn.execute(query, params).fetchall()
            finally:
                conn.close()
        except sqlite3.DatabaseError:
            return None
        latest: datetime | None = None
        for (raw,) in rows:
            parsed = _coerce_ts(str(raw)) if raw else None
            if parsed is None:
                continue
            if latest is None or parsed > latest:
                latest = parsed
        return latest

    def record_satisfaction(
        self,
        *,
        desire_name: str,
        summary: str,
        ts: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        memory_id = f"desire_{desire_name}_{uuid.uuid4().hex[:12]}"
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        body = f"[desire:{desire_name}] {summary}"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO memories(
                    id, content, normalized_content, timestamp,
                    emotion, importance, category, tags
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    body,
                    body.lower(),
                    ts.isoformat(),
                    "happy",
                    2,
                    "feeling",
                    f"desire,{desire_name}",
                ),
            )
            conn.commit()
        except sqlite3.OperationalError:
            # Schema mismatch (e.g. older store) — swallow so the adapter stays best-effort.
            pass
        finally:
            conn.close()
        return memory_id


@dataclass(slots=True)
class ChromaMemoryAdapter:
    """Legacy adapter against the ChromaDB store used before the SQLite migration."""

    path: Path
    collection_name: str = "claude_memories"

    def latest_satisfaction_ts(self, keywords: Iterable[str]) -> datetime | None:
        try:
            import chromadb  # type: ignore[import-not-found]
        except ImportError:
            return None
        try:
            client = chromadb.PersistentClient(path=str(self.path))
            collection = client.get_or_create_collection(self.collection_name)
            results = collection.get(limit=500, include=["documents", "metadatas"])
        except Exception:
            return None
        kw_list = [k for k in keywords if k]
        latest: datetime | None = None
        for doc, meta in zip(results.get("documents", []), results.get("metadatas", [])):
            if not any(kw in doc for kw in kw_list):
                continue
            ts_str = (meta or {}).get("timestamp") or ""
            if not ts_str:
                continue
            parsed = _coerce_ts(ts_str)
            if parsed is None:
                continue
            if latest is None or parsed > latest:
                latest = parsed
        return latest

    def record_satisfaction(
        self,
        *,
        desire_name: str,
        summary: str,
        ts: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        # Chroma writes require an embedding path we do not own here; keep this
        # adapter read-only to avoid corrupting the legacy collection.
        return ""


class NullMemoryAdapter:
    """Safe fallback when no backend is reachable. Always returns None / no-op."""

    def latest_satisfaction_ts(self, keywords: Iterable[str]) -> datetime | None:  # noqa: D401
        return None

    def record_satisfaction(
        self,
        *,
        desire_name: str,
        summary: str,
        ts: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return ""


def make_default_adapter() -> DesireMemoryAdapter:
    """Pick an adapter based on DESIRE_MEMORY_BACKEND and filesystem hints."""

    backend = os.getenv("DESIRE_MEMORY_BACKEND", "auto").lower()
    sqlite_path = _default_sqlite_path()
    chroma_path = _default_chroma_path()

    if backend == "sqlite":
        return SQLiteMemoryAdapter(sqlite_path)
    if backend == "chroma":
        return ChromaMemoryAdapter(chroma_path)
    if backend in {"null", "none", "off"}:
        return NullMemoryAdapter()
    # auto
    if sqlite_path.exists():
        return SQLiteMemoryAdapter(sqlite_path)
    if chroma_path.exists():
        return ChromaMemoryAdapter(chroma_path)
    return NullMemoryAdapter()


def _coerce_ts(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
