"""A tick that ran without memory has to say so.

`_fetch_memory_http` asks memory-mcp's HTTP recall port for candidates and
used to return silently when nothing was listening. The field still committed,
`<current_field>` still rendered, and the only thing missing was the memory --
which is exactly the thing nobody could see was missing (#140).
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from contextlib import contextmanager
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
    monkeypatch.setattr(tick_module, "_memory_http_unreachable", False)


def _refusing(*_args, **_kwargs):
    raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))


@contextmanager
def _answering(payload):
    yield io.BytesIO(json.dumps(payload).encode("utf-8"))


def test_outage_is_warned_once_and_named_in_the_trace(
    producer: TickProducer,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", _refusing)

    first = producer.begin_tick("perception", user_text="hello", session_id="s")
    producer.compete_and_commit(first.tick_id)
    second = producer.begin_tick("perception", user_text="again", session_id="s")

    err = capsys.readouterr().err
    assert err.count("memory HTTP recall") == 1, err
    assert "127.0.0.1:18900/recall" in err
    assert "URLError" in err
    assert "MEMORY_HTTP_PORT" in err
    assert first.epistemic_trace["memory_recall"] == "unreachable:URLError"
    assert second.epistemic_trace["memory_recall"] == "unreachable:URLError"


def test_a_later_outage_is_warned_again_after_recall_recovers(
    producer: TickProducer,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", _refusing)
    producer.begin_tick("perception", user_text="one", session_id="s")
    assert capsys.readouterr().err.count("memory HTTP recall") == 1

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_a, **_k: _answering(
            {"memories": [{"id": "m1", "content": "a remembered room"}]}
        ),
    )
    producer.compete_and_commit(producer.fields.query(limit=1)[0].tick_id)
    recovered = producer.begin_tick("perception", user_text="two", session_id="s")
    assert recovered.epistemic_trace["memory_recall"] == "ok:1"
    assert capsys.readouterr().err == ""

    monkeypatch.setattr(urllib.request, "urlopen", _refusing)
    producer.compete_and_commit(recovered.tick_id)
    producer.begin_tick("perception", user_text="three", session_id="s")
    assert capsys.readouterr().err.count("memory HTTP recall") == 1


def test_ticks_without_user_text_do_not_ask_recall(
    producer: TickProducer,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", _refusing)

    field = producer.begin_tick("heartbeat", session_id="s")

    assert "memory_recall" not in field.epistemic_trace
    assert capsys.readouterr().err == ""


def test_port_zero_means_recall_is_off_and_nothing_is_asked(
    producer: TickProducer,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("MEMORY_HTTP_PORT", "0")
    monkeypatch.setattr(urllib.request, "urlopen", _refusing)

    field = producer.begin_tick("perception", user_text="hello", session_id="s")

    assert field.epistemic_trace["memory_recall"] == "disabled"
    assert capsys.readouterr().err == ""


def test_recall_url_follows_memory_http_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_HTTP_PORT", "18901")
    assert tick_module.memory_http_recall_url("q").startswith(
        "http://127.0.0.1:18901/recall?"
    )
