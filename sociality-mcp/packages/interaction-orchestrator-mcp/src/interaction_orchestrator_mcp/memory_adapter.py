"""Memory recall adapter for the interaction orchestrator.

The orchestrator does not own the memory-mcp store; it just reads from the
same SQLite file the user's ``remember`` calls populate. Keeping this as
a read-only, file-backed adapter means the orchestrator stays testable
without forcing ``memory-mcp`` to run in-process.

Selection is controlled by ``ORCHESTRATOR_MEMORY_BACKEND``:

- ``sqlite`` (default when ``memory.db`` exists)
- ``null`` or ``none``: always return empty
- ``auto``: prefer sqlite, fall back to null

The spec (§11.2) asks for a ``use_policy`` per recall hit. We default to
``mentionable`` for most memories and demote private categories to
``background_only`` when ``include_private=False``.
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Protocol

UsePolicy = Literal["mentionable", "background_only", "do_not_surface"]

_MAX_ROWS_CONSIDERED = 80
_STOP_TOKENS = {
    "の", "を", "に", "が", "は", "で", "と", "も", "な", "か", "ね", "よ", "わ",
    "the", "a", "an", "is", "are", "of", "to", "and", "or", "in", "on",
    "it", "that", "this", "be", "you", "i", "we", "they", "what", "why",
}
_PRIVATE_CATEGORIES = {"philosophical", "feeling"}
_JP_PARTICLE_HEAD = set("のがはをにでとよねわかもやるんー、。っ")
_JP_RUN = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff々]+")


@dataclass(slots=True)
class RecallHit:
    memory_id: str
    content: str
    timestamp: str
    category: str
    emotion: str
    importance: int
    relevance: float
    use_policy: UsePolicy
    reason: str


class OrchestratorMemoryAdapter(Protocol):
    """Read-side access to long-term memory for the orchestrator."""

    def recall_for_response(
        self,
        *,
        user_text: str | None,
        person_id: str | None = None,
        max_results: int = 6,
        include_private: bool = True,
        exclude_categories: Iterable[str] = (),
    ) -> list[RecallHit]:
        ...


def _default_sqlite_path() -> Path:
    return Path(
        os.getenv(
            "MEMORY_DB_FILE",
            str(Path.home() / ".claude" / "memories" / "memory.db"),
        )
    ).expanduser()


def _extract_keywords(text: str, *, max_keywords: int = 6) -> list[str]:
    """Cheap keyword extractor: split on non-word runs, drop stop tokens.

    For long Japanese runs (which have no whitespace between words), also
    emit content-word 2-grams so LIKE-based recall can still fire. This is
    intentionally lightweight — it trades precision for recall, then lets
    the importance/age/match-count scoring rank the hits.
    """

    if not text:
        return []
    tokens = [
        tok for tok in re.split(r"[\s\.,/!?？。、:;\-\(\)\[\]「」『』]+", text) if tok
    ]
    seen: set[str] = set()
    keywords: list[str] = []

    def _push(candidate: str) -> bool:
        """Add a candidate if admissible; return True iff we can keep iterating."""

        low = candidate.lower()
        if len(candidate) >= 2 and low not in _STOP_TOKENS and low not in seen:
            seen.add(low)
            keywords.append(candidate)
        return len(keywords) < max_keywords

    for tok in tokens:
        if not _push(tok):
            return keywords[:max_keywords]

    # Emit Japanese bigrams from content runs (skip leading-particle pairs).
    for match in _JP_RUN.finditer(text):
        run = match.group()
        if len(run) < 3:
            continue
        for i in range(len(run) - 1):
            bigram = run[i : i + 2]
            if bigram[0] in _JP_PARTICLE_HEAD:
                continue
            if not _push(bigram):
                return keywords[:max_keywords]
    return keywords[:max_keywords]


def _use_policy_for(
    *, category: str, importance: int, include_private: bool
) -> tuple[UsePolicy, str]:
    if not include_private and category in _PRIVATE_CATEGORIES:
        return (
            "do_not_surface",
            f"include_private=False and category={category!r} is private",
        )
    if category in _PRIVATE_CATEGORIES and importance < 5:
        return (
            "background_only",
            f"category={category!r} is sensitive; informs the plan but not quotable",
        )
    return ("mentionable", f"category={category!r}, importance={importance}")


def _age_days(ts: str, now: datetime | None = None) -> float:
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return 365.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    now_aware = now or datetime.now(timezone.utc)
    return max(0.0, (now_aware - parsed).total_seconds() / 86400.0)


def _score(
    *, match_count: int, keyword_total: int, importance: int, age_days: float
) -> float:
    if keyword_total <= 0:
        return 0.0
    recall_ratio = match_count / keyword_total
    importance_weight = max(1, importance) / 5.0
    age_decay = 1.0 / (1.0 + age_days / 30.0)
    return max(0.0, min(1.0, recall_ratio * 0.6 + importance_weight * 0.25 + age_decay * 0.15))


@dataclass(slots=True)
class SQLiteMemoryAdapter:
    """Read-only adapter against the memory-mcp SQLite store."""

    db_path: Path

    def recall_for_response(
        self,
        *,
        user_text: str | None,
        person_id: str | None = None,
        max_results: int = 6,
        include_private: bool = True,
        exclude_categories: Iterable[str] = (),
    ) -> list[RecallHit]:
        if not self.db_path.exists():
            return []
        keywords = _extract_keywords(user_text or "")
        if not keywords:
            return []
        exclude = {str(c) for c in exclude_categories}

        clauses = " OR ".join(["content LIKE ?"] * len(keywords))
        params = [f"%{k}%" for k in keywords]
        query = (
            "SELECT id, content, timestamp, emotion, importance, category "
            f"FROM memories WHERE {clauses} "
            "ORDER BY timestamp DESC LIMIT ?"
        )
        params_with_limit = [*params, _MAX_ROWS_CONSIDERED]
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(query, params_with_limit).fetchall()
            finally:
                conn.close()
        except sqlite3.DatabaseError:
            return []

        hits: list[RecallHit] = []
        keyword_total = len(keywords)
        for row in rows:
            category = str(row["category"] or "daily")
            if category in exclude:
                continue
            importance = int(row["importance"] or 3)
            content = str(row["content"] or "")
            match_count = sum(1 for kw in keywords if kw in content)
            if match_count <= 0:
                continue
            relevance = _score(
                match_count=match_count,
                keyword_total=keyword_total,
                importance=importance,
                age_days=_age_days(str(row["timestamp"] or "")),
            )
            policy, reason = _use_policy_for(
                category=category,
                importance=importance,
                include_private=include_private,
            )
            if policy == "do_not_surface":
                continue
            hits.append(
                RecallHit(
                    memory_id=str(row["id"] or ""),
                    content=content,
                    timestamp=str(row["timestamp"] or ""),
                    category=category,
                    emotion=str(row["emotion"] or "neutral"),
                    importance=importance,
                    relevance=relevance,
                    use_policy=policy,
                    reason=reason,
                )
            )
        hits.sort(key=lambda h: h.relevance, reverse=True)
        return hits[:max_results]


class NullMemoryAdapter:
    """Fallback adapter that always returns no hits."""

    def recall_for_response(
        self,
        *,
        user_text: str | None,
        person_id: str | None = None,
        max_results: int = 6,
        include_private: bool = True,
        exclude_categories: Iterable[str] = (),
    ) -> list[RecallHit]:
        return []


def make_default_adapter() -> OrchestratorMemoryAdapter:
    """Choose an adapter based on env + filesystem hints."""

    backend = os.getenv("ORCHESTRATOR_MEMORY_BACKEND", "auto").lower()
    sqlite_path = _default_sqlite_path()
    if backend in {"null", "none", "off"}:
        return NullMemoryAdapter()
    if backend == "sqlite":
        return SQLiteMemoryAdapter(sqlite_path)
    # auto
    if sqlite_path.exists():
        return SQLiteMemoryAdapter(sqlite_path)
    return NullMemoryAdapter()
