from __future__ import annotations

from pathlib import Path

import pytest

from wifi_cam_mcp import config
from wifi_cam_mcp.config import ServerConfig
from wifi_cam_mcp.server import resolve_transcribe


@pytest.mark.parametrize(
    ("value", "expected"),
    (("true", True), ("1", True), ("false", False), ("0", False)),
)
def test_transcription_default_is_configurable(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_DEFAULT", value)

    assert ServerConfig.from_env().transcribe_default is expected


def test_invalid_transcription_default_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_DEFAULT", "sometimes")

    with pytest.raises(ValueError, match="TRANSCRIBE_DEFAULT"):
        ServerConfig.from_env()


def test_capture_directory_uses_platform_temp_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("CAPTURE_DIR", raising=False)
    monkeypatch.setattr(config.tempfile, "gettempdir", lambda: str(tmp_path))

    assert ServerConfig.from_env().capture_dir == str(tmp_path / "wifi-cam-mcp")


def test_direct_server_config_uses_platform_temp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(config.tempfile, "gettempdir", lambda: str(tmp_path))

    assert ServerConfig().capture_dir == str(tmp_path / "wifi-cam-mcp")


@pytest.mark.parametrize(
    ("arguments", "default", "expected"),
    (
        ({}, False, False),
        ({}, True, True),
        ({"transcribe": True}, False, True),
        ({"transcribe": False}, True, False),
    ),
)
def test_listen_resolves_explicit_or_configured_transcription_default(
    arguments: dict[str, bool],
    default: bool,
    expected: bool,
) -> None:
    assert resolve_transcribe(arguments, default=default) is expected
