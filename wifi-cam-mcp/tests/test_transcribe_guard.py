"""listen must not block forever in the transcription stage, and the local
backend's native libraries are imported on the main thread before the event
loop exists (the Windows OpenBLAS worker-thread hang)."""

import asyncio
import importlib.abc
import importlib.util
import logging
import sys
import threading
import types

import pytest

from wifi_cam_mcp import camera as camera_module
from wifi_cam_mcp import server as server_module
from wifi_cam_mcp.camera import AudioResult, TapoCamera, warm_transcribe_backend
from wifi_cam_mcp.config import CameraConfig
from wifi_cam_mcp.server import format_listen_response

BACKEND_MODULES = ("ctranslate2", "faster_whisper")


def _camera(tmp_path):
    return TapoCamera(
        CameraConfig(host="127.0.0.1", username="user", password="pass"),
        capture_dir=str(tmp_path),
        transcribe_backend="faster-whisper",
        transcribe_model="base",
    )


def _audio(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    return str(audio)


@pytest.mark.asyncio
async def test_transcribe_audio_happy_path_joins_segments(tmp_path, monkeypatch):
    class FakeModel:
        def transcribe(self, path, language):
            assert language == "ja"
            segments = [
                types.SimpleNamespace(text="こんにちは"),
                types.SimpleNamespace(text="世界"),
            ]
            return segments, None

    monkeypatch.setattr(camera_module, "transcribe_backend_available", lambda backend: True)
    monkeypatch.setattr(camera_module, "_get_whisper_model", lambda backend, size: FakeModel())

    assert await _camera(tmp_path)._transcribe_audio(_audio(tmp_path)) == ("こんにちは世界", None)


@pytest.mark.asyncio
async def test_model_load_timeout_returns_a_reason_instead_of_hanging(tmp_path, monkeypatch):
    release = threading.Event()

    def blocked_model_load(backend, model_size):
        release.wait(5.0)  # a model load that does not come back in time
        return object()

    monkeypatch.setattr(camera_module, "transcribe_backend_available", lambda backend: True)
    monkeypatch.setattr(camera_module, "_get_whisper_model", blocked_model_load)
    monkeypatch.setattr(camera_module, "TRANSCRIBE_TIMEOUT_S", 0.2)

    try:
        transcript, error = await asyncio.wait_for(
            _camera(tmp_path)._transcribe_audio(_audio(tmp_path)), 3.0
        )
    finally:
        release.set()

    assert transcript is None
    assert "did not finish within" in error
    assert "restart" in error


@pytest.mark.asyncio
async def test_transcribe_stage_timeout_is_reported_too(tmp_path, monkeypatch):
    release = threading.Event()

    def blocked_transcribe(backend, model, path):
        release.wait(5.0)
        return "late"

    monkeypatch.setattr(camera_module, "transcribe_backend_available", lambda backend: True)
    monkeypatch.setattr(camera_module, "_get_whisper_model", lambda backend, size: object())
    monkeypatch.setattr(camera_module, "_transcribe_with_model", blocked_transcribe)
    monkeypatch.setattr(camera_module, "TRANSCRIBE_TIMEOUT_S", 0.2)

    try:
        transcript, error = await asyncio.wait_for(
            _camera(tmp_path)._transcribe_audio(_audio(tmp_path)), 3.0
        )
    finally:
        release.set()

    assert transcript is None
    assert "did not finish within" in error


def test_warm_backend_skips_a_backend_that_is_not_installed(monkeypatch):
    monkeypatch.setattr(camera_module, "transcribe_backend_available", lambda backend: False)
    assert warm_transcribe_backend("faster-whisper") is False


def test_warm_backend_ignores_the_cloud_backend():
    # openai-api has no module to import; the real availability check says so.
    assert warm_transcribe_backend("openai-api") is False


def test_warm_backend_really_imports_the_modules_on_the_calling_thread(monkeypatch):
    seen = []

    class StubLoader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            on_main = threading.current_thread() is threading.main_thread()
            seen.append((module.__name__, on_main))

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name in BACKEND_MODULES:
                return importlib.util.spec_from_loader(name, StubLoader())
            return None

    monkeypatch.setattr(sys, "meta_path", [Finder(), *sys.meta_path])
    for name in BACKEND_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(camera_module, "transcribe_backend_available", lambda backend: True)
    monkeypatch.setattr(camera_module, "_register_cuda_dll_dirs", lambda: None)

    try:
        assert warm_transcribe_backend("faster-whisper") is True
    finally:
        for name in BACKEND_MODULES:  # never leave the stubs behind for other tests
            sys.modules.pop(name, None)

    assert seen == [("ctranslate2", True), ("faster_whisper", True)]


def test_server_warm_transcription_logs_and_does_not_raise(monkeypatch, caplog):
    def boom(backend):
        raise RuntimeError("no dll")

    monkeypatch.setattr(
        server_module.ServerConfig,
        "from_env",
        lambda: types.SimpleNamespace(transcribe_backend="faster-whisper"),
    )
    monkeypatch.setattr(server_module, "warm_transcribe_backend", boom)

    with caplog.at_level(logging.WARNING):
        server_module.CameraMCPServer().warm_transcription()

    assert "Could not pre-import transcription backend faster-whisper" in caplog.text
    assert "no dll" in caplog.text


def test_main_warms_transcription_on_the_main_thread_before_the_event_loop(monkeypatch):
    order = []

    class FakeServer:
        def warm_transcription(self):
            order.append(("warm", threading.current_thread() is threading.main_thread()))

        async def run(self):
            order.append(("run", True))

    monkeypatch.setitem(sys.modules, "jurigged", types.SimpleNamespace(watch=lambda **kw: None))
    monkeypatch.setattr(server_module, "CameraMCPServer", FakeServer)

    server_module.main()

    assert order == [("warm", True), ("run", True)]


def test_empty_transcript_gets_its_own_heading():
    ran_but_silent = AudioResult(audio_base64="", file_path="a.wav", timestamp="t", duration=3,
                                 transcript="", transcript_error=None)
    not_requested = AudioResult(audio_base64="", file_path="a.wav", timestamp="t", duration=3,
                                transcript=None, transcript_error=None)

    assert "Transcript (empty" in format_listen_response(ran_but_silent)
    assert "Transcript" not in format_listen_response(not_requested)
