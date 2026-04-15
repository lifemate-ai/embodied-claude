"""Shared SQLite access layer."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .migrations import apply_migrations

DEFAULT_SOCIAL_DB_PATH = Path.home() / ".claude" / "sociality" / "social.db"


def get_social_db_path(path: str | Path | None = None) -> Path:
    """Resolve the social DB path with environment override support."""

    if path is not None:
        return Path(path).expanduser()
    env_path = os.environ.get("SOCIAL_DB_PATH")
    if env_path:
        return Path(env_path).expanduser()
    return DEFAULT_SOCIAL_DB_PATH


class SocialDB:
    """Thin SQLite wrapper with shared pragmas and transaction handling."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = get_social_db_path(path)
        self._connection: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=NORMAL")
            apply_migrations(connection)
            self._connection = connection
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    @contextmanager
    def transaction(self) -> Any:
        connection = self.connect()
        outer = connection.in_transaction
        if not outer:
            connection.execute("BEGIN")
        try:
            yield connection
        except Exception:
            if connection.in_transaction and not outer:
                connection.rollback()
            raise
        else:
            if connection.in_transaction and not outer:
                connection.commit()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self.transaction() as connection:
            return connection.execute(sql, params)

    def fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        return self.connect().execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return self.connect().execute(sql, params).fetchall()
