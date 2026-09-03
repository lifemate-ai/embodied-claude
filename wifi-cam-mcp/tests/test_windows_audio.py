import asyncio
import base64
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from wifi_cam_mcp import camera as camera_module
from wifi_cam_mcp.camera import (
    TapoCamera,
    _find_dshow_audio_device,
    _get_whisper_model,
    _whisper_models,
)
from wifi_cam_mcp.config import CameraConfig, ServerConfig


class _ListingProcess:
    async def communicate(self):
        listing = b'[dshow @ 000001] "Microphone Array" (audio)\n'
        return b"", listing

    def kill(self):
        pass


class _RecordingProcess:
    async def wait(self):
        return 0


@pytest.mark.asyncio
async def test_find_dshow_audio_device_returns_first_audio_device(monkeypatch):
    async def create_subprocess_exec(*args, **kwargs):
        assert args == (
            "ffmpeg",
            "-hide_banner",
            "-list_devices",
            "true",
            "-f",
            "dshow",
            "-i",
            "dummy",
        )
        assert kwargs["stderr"] == asyncio.subprocess.PIPE
        return _ListingProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)

    assert await _find_dshow_audio_device() == "Microphone Array"


@pytest.mark.asyncio
async def test_windows_local_audio_uses_configured_dshow_device(monkeypatch, tmp_path):
    command = []

    async def create_subprocess_exec(*args, **kwargs):
        command.extend(args)
        Path(args[-1]).write_bytes(b"RIFF")
        return _RecordingProcess()

    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)

    camera = TapoCamera(
        CameraConfig(host="127.0.0.1", username="user", password="pass"),
        capture_dir=str(tmp_path),
        mic_device="Microphone Array",
    )
    result = await camera.listen_audio(duration=2.5, mic_source="local")

    assert command[:7] == [
        "ffmpeg",
        "-f",
        "dshow",
        "-audio_buffer_size",
        "50",
        "-i",
        "audio=Microphone Array",
    ]
    assert command[command.index("-t") + 1] == "2.5"
    assert result.audio_base64 == base64.standard_b64encode(b"RIFF").decode("ascii")


def test_server_config_reads_transcription_settings(monkeypatch):
    monkeypatch.setenv("MIC_SOURCE", "local")
    monkeypatch.setenv("MIC_DEVICE", "Microphone Array")
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "faster-whisper")
    monkeypatch.setenv("TRANSCRIBE_MODEL", "small")

    config = ServerConfig.from_env()

    assert config.mic_source == "local"
    assert config.mic_device == "Microphone Array"
    assert config.transcribe_backend == "faster-whisper"
    assert config.transcribe_model == "small"


def test_openai_whisper_model_is_cached_by_backend_and_size(monkeypatch):
    loaded = []
    model = object()

    def load_model(model_size):
        loaded.append(model_size)
        return model

    monkeypatch.setitem(sys.modules, "whisper", SimpleNamespace(load_model=load_model))
    _whisper_models.clear()
    try:
        assert _get_whisper_model("openai-whisper", "base") is model
        assert _get_whisper_model("openai-whisper", "base") is model
        assert loaded == ["base"]
    finally:
        _whisper_models.clear()


@pytest.mark.asyncio
async def test_missing_backend_is_reported_as_error_not_transcript(monkeypatch, tmp_path):
    """The 'not installed' message must not come back as heard speech (#151)."""

    async def create_subprocess_exec(*args, **kwargs):
        Path(args[-1]).write_bytes(b"RIFF")
        return _RecordingProcess()

    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    monkeypatch.setattr(camera_module, "transcribe_backend_available", lambda _backend: False)

    camera = TapoCamera(
        CameraConfig(host="127.0.0.1", username="user", password="pass"),
        capture_dir=str(tmp_path),
        mic_device="Microphone Array",
        transcribe_backend="faster-whisper",
    )
    result = await camera.listen_audio(duration=1, transcribe=True, mic_source="local")

    assert result.transcript is None
    assert result.transcript_error is not None
    assert "faster-whisper is not installed" in result.transcript_error
    assert "transcription-faster" in result.transcript_error
