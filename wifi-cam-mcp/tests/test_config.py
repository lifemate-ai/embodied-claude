from __future__ import annotations

from pathlib import Path

import pytest

from wifi_cam_mcp import config
from wifi_cam_mcp.config import CameraConfig, ServerConfig
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


@pytest.fixture
def right_camera_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A minimal right-camera environment with no PTZ/mount overrides set."""
    monkeypatch.setenv("TAPO_RIGHT_CAMERA_HOST", "192.0.2.2")
    monkeypatch.setenv("TAPO_USERNAME", "user")
    monkeypatch.setenv("TAPO_PASSWORD", "secret")
    for name in (
        "TAPO_PTZ_MODE",
        "TAPO_RIGHT_PTZ_MODE",
        "TAPO_MOUNT_MODE",
        "TAPO_RIGHT_MOUNT_MODE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_right_camera_ptz_mode_defaults_to_auto(right_camera_env: None) -> None:
    right = CameraConfig.right_camera_from_env()

    assert right is not None
    assert right.ptz_mode == "auto"


def test_right_camera_reads_its_own_ptz_mode(
    monkeypatch: pytest.MonkeyPatch, right_camera_env: None
) -> None:
    monkeypatch.setenv("TAPO_RIGHT_PTZ_MODE", "relative")

    right = CameraConfig.right_camera_from_env()

    assert right is not None
    assert right.ptz_mode == "relative"


def test_right_camera_falls_back_to_left_ptz_mode(
    monkeypatch: pytest.MonkeyPatch, right_camera_env: None
) -> None:
    monkeypatch.setenv("TAPO_PTZ_MODE", "continuous")

    right = CameraConfig.right_camera_from_env()

    assert right is not None
    assert right.ptz_mode == "continuous"


def test_right_camera_rejects_invalid_ptz_mode(
    monkeypatch: pytest.MonkeyPatch, right_camera_env: None
) -> None:
    monkeypatch.setenv("TAPO_RIGHT_PTZ_MODE", "sideways")

    with pytest.raises(ValueError, match="PTZ mode"):
        CameraConfig.right_camera_from_env()


def _only(installed: set[str]):
    return lambda name: name in installed


@pytest.mark.parametrize(
    ("installed", "expected"),
    (
        ({"faster_whisper"}, "faster-whisper"),
        ({"whisper"}, "openai-whisper"),
        ({"whisper", "faster_whisper"}, "openai-whisper"),
        (set(), "openai-whisper"),
    ),
)
def test_transcribe_backend_defaults_to_the_installed_one(
    monkeypatch: pytest.MonkeyPatch, installed: set[str], expected: str
) -> None:
    """Installing only transcription-faster must not leave listen pointing at whisper (#151)."""
    monkeypatch.delenv("TRANSCRIBE_BACKEND", raising=False)
    monkeypatch.setattr(config, "_module_available", _only(installed))

    assert config.default_transcribe_backend() == expected
    assert ServerConfig.from_env().transcribe_backend == expected
    assert ServerConfig().transcribe_backend == expected


def test_explicit_transcribe_backend_wins_over_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "faster-whisper")
    monkeypatch.setattr(config, "_module_available", _only({"whisper"}))

    assert ServerConfig.from_env().transcribe_backend == "faster-whisper"


def test_listen_response_keeps_diagnostics_out_of_the_transcript_heading() -> None:
    from wifi_cam_mcp.camera import AudioResult
    from wifi_cam_mcp.server import format_listen_response

    heard = AudioResult("", "/tmp/a.wav", "t", 5.0, transcript="こんにちは")
    missing = AudioResult(
        "",
        "/tmp/a.wav",
        "t",
        5.0,
        transcript=None,
        transcript_error="faster-whisper is not installed",
    )
    silent = AudioResult("", "/tmp/a.wav", "t", 5.0)

    assert "--- Transcript ---\nこんにちは" in format_listen_response(heard)
    assert "--- Transcript ---" not in format_listen_response(missing)
    assert "--- No transcript" in format_listen_response(missing)
    assert "faster-whisper is not installed" in format_listen_response(missing)
    assert "Transcript" not in format_listen_response(silent)
