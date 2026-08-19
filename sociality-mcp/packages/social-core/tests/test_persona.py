"""Persona helpers read the environment at call time with neutral defaults."""

from __future__ import annotations

from social_core import companion_id, companion_name, self_name


def test_defaults_are_neutral(monkeypatch) -> None:
    for key in ("COMPANION_ID", "COMPANION_NAME", "SELF_NAME"):
        monkeypatch.delenv(key, raising=False)
    assert companion_id() == "companion"
    assert companion_name() == "あなた"
    assert self_name() == "This agent"


def test_environment_is_read_at_call_time(monkeypatch) -> None:
    monkeypatch.setenv("COMPANION_ID", "kouta")
    monkeypatch.setenv("COMPANION_NAME", "コウタ")
    monkeypatch.setenv("SELF_NAME", "ここね")
    assert companion_id() == "kouta"
    assert companion_name() == "コウタ"
    assert self_name() == "ここね"


def test_blank_values_fall_back_to_defaults(monkeypatch) -> None:
    monkeypatch.setenv("COMPANION_ID", "   ")
    assert companion_id() == "companion"
