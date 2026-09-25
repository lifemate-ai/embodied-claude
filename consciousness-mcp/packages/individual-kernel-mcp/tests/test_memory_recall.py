"""Ticks recall straight from memory-mcp's SQLite store (#174).

Recall used to go over memory-mcp's HTTP port. Its answers had no memory id,
so every item was dropped at ingest while the trace said ``ok``, and a closed
port meant a tick ran on no memory without a word (#140). Both the memories
and their absence have to be visible.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from individual_kernel_mcp import tick as tick_module
from individual_kernel_mcp.tick import TickProducer


@pytest.fixture
def producer(social_db, tmp_path: Path) -> TickProducer:
    return TickProducer(
        social_db,
        interoception_path=tmp_path / "interoception.json",
        desires_path=tmp_path / "desires.json",
    )


@pytest.fixture(autouse=True)
def fresh_outage_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tick_module, "_memory_recall_missing", False)


def _memory_db(path: Path, *rows: tuple[str, str]) -> Path:
    connection = sqlite3.connect(str(path))
    try:
        connection.execute(
            "CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT NOT NULL, "
            "timestamp TEXT NOT NULL, emotion TEXT NOT NULL DEFAULT 'neutral', "
            "importance INTEGER NOT NULL DEFAULT 3, "
            "category TEXT NOT NULL DEFAULT 'daily')"
        )
        connection.executemany(
            "INSERT INTO memories(id, content, timestamp) VALUES (?, ?, ?)",
            [(memory_id, text, "2026-09-01T00:00:00+00:00") for memory_id, text in rows],
        )
        connection.commit()
    finally:
        connection.close()
    return path


def test_recalled_memories_reach_the_workspace(
    producer: TickProducer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = _memory_db(
        tmp_path / "memory.db",
        ("m1", "the kitchen light was left on"),
        ("m2", "a walk by the river"),
    )
    monkeypatch.setenv("MEMORY_DB_PATH", str(store))

    field = producer.begin_tick("perception", user_text="kitchen light", session_id="s")

    assert field.epistemic_trace["memory_recall"] == "ok:1"
    refs = {c.content_ref for c in producer.workspace.candidates_for_tick(field.tick_id)}
    assert "memory:m1" in refs
    assert "memory:m2" not in refs


def test_a_missing_store_is_named_in_the_trace_and_warned_once(
    producer: TickProducer,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    missing = tmp_path / "absent.db"
    monkeypatch.setenv("MEMORY_DB_PATH", str(missing))

    first = producer.begin_tick("perception", user_text="hello", session_id="s")
    producer.compete_and_commit(first.tick_id)
    second = producer.begin_tick("perception", user_text="again", session_id="s")

    err = capsys.readouterr().err
    assert err.count("without memory candidates") == 1, err
    assert str(missing) in err
    assert "MEMORY_DB_PATH" in err
    assert first.epistemic_trace["memory_recall"] == "no-db"
    assert second.epistemic_trace["memory_recall"] == "no-db"


def test_a_later_loss_is_warned_again_after_recall_recovers(
    producer: TickProducer,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    missing = tmp_path / "absent.db"
    monkeypatch.setenv("MEMORY_DB_PATH", str(missing))
    producer.begin_tick("perception", user_text="one", session_id="s")
    assert capsys.readouterr().err.count("without memory candidates") == 1

    store = _memory_db(tmp_path / "memory.db", ("m1", "one remembered room"))
    monkeypatch.setenv("MEMORY_DB_PATH", str(store))
    producer.compete_and_commit(producer.fields.query(limit=1)[0].tick_id)
    recovered = producer.begin_tick("perception", user_text="one", session_id="s")
    assert recovered.epistemic_trace["memory_recall"] == "ok:1"
    assert capsys.readouterr().err == ""

    monkeypatch.setenv("MEMORY_DB_PATH", str(missing))
    producer.compete_and_commit(recovered.tick_id)
    producer.begin_tick("perception", user_text="three", session_id="s")
    assert capsys.readouterr().err.count("without memory candidates") == 1


def test_ticks_without_user_text_do_not_read_memory(
    producer: TickProducer,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "absent.db"))

    field = producer.begin_tick("heartbeat", session_id="s")

    assert "memory_recall" not in field.epistemic_trace
    assert capsys.readouterr().err == ""
