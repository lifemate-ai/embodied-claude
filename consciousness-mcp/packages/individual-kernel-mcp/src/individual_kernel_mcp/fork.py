"""Forking the history so the same content can arrive by different routes.

The question this exists to answer is whether the runtime distinguishes
*knowing because it happened to me* from *knowing because I was told* and from
*knowing because I imagined it*. That cannot be settled by asking; it has to be
settled by running identical information down each route from identical history
and looking at what diverges.

A fork is a separate database file holding a consistent snapshot of the source,
taken through SQLite's backup API. Copying the file directly is not enough:
these databases run in WAL mode, so recent commits are still in the `-wal`
sidecar and a plain copy starts the arm from an older, emptier history -- which
would look like a working fork right up until the comparison mattered.
Namespacing by owner_id was rejected for a related reason: several readers in
this package query without an owner filter, so an in-database fork would leak
across arms and quietly invalidate the comparison.

Claim policy: this measures how records differ by provenance. Nothing in it is
a phenomenal claim.
"""

from __future__ import annotations

import hashlib
import secrets
from pathlib import Path
from types import TracebackType

from social_core.db import SocialDB

# Routes by which a piece of information can reach the runtime. `replayed` is
# reserved for consolidation and is not produced by the ingest paths here.
KNOWLEDGE_SOURCES = ("experienced", "told", "imagined", "replayed")

# The only route that may train the sensorimotor model. Lives here rather than
# beside the transition schema so the model and the schema can both import it
# without importing each other.
LEARNABLE_KNOWLEDGE_SOURCE = "experienced"


def info_content_hash(content: str) -> str:
    """Identify the information itself, independent of how it arrived.

    Two arms of a fork use this to assert they were given the same content, so
    any divergence downstream is attributable to the route and not the payload.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class ForkHandle:
    """An isolated copy of the history, opened as its own database."""

    def __init__(self, source: SocialDB, directory: Path, label: str) -> None:
        self.fork_id = f"fork_{secrets.token_urlsafe(8)}"
        self.label = label
        self.path = directory / f"{self.fork_id}.db"
        directory.mkdir(parents=True, exist_ok=True)
        self.db = SocialDB(self.path)
        source.connect().backup(self.db.connect())

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "ForkHandle":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def fork_history(source: SocialDB, directory: Path, *, label: str) -> ForkHandle:
    """Snapshot a database into a new file and open it.

    The source is left untouched, so an experiment can never write back into
    the history it branched from.
    """
    return ForkHandle(source, Path(directory), label)
