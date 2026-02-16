"""Tests for ElevenLabs voice fallback behavior."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import elevenlabs_t2s_mcp.server as server_module
from elevenlabs_t2s_mcp.server import ElevenLabsTTSMCP, _is_voice_not_found_error


@dataclass(frozen=True)
class _DummyVoice:
    voice_id: str
    name: str | None = None


class _DummyVoicesClient:
    def __init__(self, voices: list[_DummyVoice]) -> None:
        self._voices = voices

    def get_all(self, *, show_legacy: bool | None = None) -> SimpleNamespace:
        return SimpleNamespace(voices=self._voices)


class _DummyElevenLabs:
    def __init__(self, *, api_key: str, voices: list[_DummyVoice]) -> None:
        self.api_key = api_key
        self.voices = _DummyVoicesClient(voices)
        self.text_to_speech = SimpleNamespace()


def _create_server(
    monkeypatch: pytest.MonkeyPatch,
    *,
    configured_voice_id: str,
    available_voices: list[_DummyVoice],
) -> ElevenLabsTTSMCP:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", configured_voice_id)

    def _factory(*, api_key: str) -> _DummyElevenLabs:
        return _DummyElevenLabs(api_key=api_key, voices=available_voices)

    monkeypatch.setattr(server_module, "ElevenLabs", _factory)
    return ElevenLabsTTSMCP()


def test_is_voice_not_found_error() -> None:
    assert _is_voice_not_found_error(RuntimeError("status=voice_not_found"))
    assert _is_voice_not_found_error(
        RuntimeError("A voice with voice_id 'abc' was not found.")
    )
    assert not _is_voice_not_found_error(RuntimeError("authentication failed"))


@pytest.mark.asyncio
async def test_falls_back_when_requested_voice_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _create_server(
        monkeypatch,
        configured_voice_id="",
        available_voices=[_DummyVoice("voice-valid", "Valid Voice")],
    )

    async def _fake_synthesize_once(**kwargs: object) -> tuple[bytes, str, str]:
        if kwargs["voice_id"] == "voice-missing":
            raise RuntimeError("voice_not_found")
        return b"audio", "/tmp/test.mp3", "played"

    monkeypatch.setattr(server, "_synthesize_once", _fake_synthesize_once)

    _, _, _, resolved_voice_id, note = await server._synthesize_with_voice_fallback(
        text="hello",
        voice_id="voice-missing",
        model_id="eleven_v3",
        output_format="mp3_44100_128",
        play_audio=False,
        use_local=False,
    )

    assert resolved_voice_id == "voice-valid"
    assert note is not None
    assert "voice-missing" in note
    assert "voice-valid" in note


@pytest.mark.asyncio
async def test_uses_available_voice_when_voice_id_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _create_server(
        monkeypatch,
        configured_voice_id="",
        available_voices=[_DummyVoice("voice-default", "Default Voice")],
    )
    called_voice_ids: list[str] = []

    async def _fake_synthesize_once(**kwargs: object) -> tuple[bytes, str, str]:
        called_voice_ids.append(str(kwargs["voice_id"]))
        return b"audio", "/tmp/test.mp3", "played"

    monkeypatch.setattr(server, "_synthesize_once", _fake_synthesize_once)

    _, _, _, resolved_voice_id, note = await server._synthesize_with_voice_fallback(
        text="hello",
        voice_id="",
        model_id="eleven_v3",
        output_format="mp3_44100_128",
        play_audio=False,
        use_local=False,
    )

    assert called_voice_ids == ["voice-default"]
    assert resolved_voice_id == "voice-default"
    assert note is not None
    assert "No voice_id was configured." in note

