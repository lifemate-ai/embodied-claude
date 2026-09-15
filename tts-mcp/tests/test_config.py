"""Tests for TTS MCP config."""

import os
from unittest.mock import patch

from tts_mcp.config import (
    AtlasCloudConfig,
    ElevenLabsConfig,
    PlaybackConfig,
    TTSConfig,
    VoicevoxConfig,
    _parse_bool,
)


class TestAtlasCloudConfig:
    """Tests for Atlas Cloud config."""

    @patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "test-key"}, clear=False)
    def test_from_env(self):
        config = AtlasCloudConfig.from_env()
        assert config is not None
        assert config.api_key == "test-key"
        assert config.model_id == "minimax/speech-2.6-turbo"
        assert config.output_format == "mp3"

    @patch.dict(os.environ, {}, clear=False)
    def test_returns_none_without_api_key(self):
        os.environ.pop("ATLASCLOUD_API_KEY", None)
        assert AtlasCloudConfig.from_env() is None


class TestParseBool:
    """Tests for _parse_bool helper."""

    def test_true_values(self):
        for val in ("1", "true", "True", "TRUE", "yes", "on"):
            assert _parse_bool(val, False) is True

    def test_false_values(self):
        for val in ("0", "false", "False", "no", "off", ""):
            assert _parse_bool(val, True) is False

    def test_none_returns_default(self):
        assert _parse_bool(None, True) is True
        assert _parse_bool(None, False) is False


class TestElevenLabsConfig:
    """Tests for ElevenLabs config."""

    @patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}, clear=False)
    def test_from_env(self):
        os.environ.pop("ELEVENLABS_VOICE_ID", None)
        config = ElevenLabsConfig.from_env()
        assert config is not None
        assert config.api_key == "test-key"
        assert config.voice_id == "uYp2UUDeS74htH10iY2e"

    @patch.dict(os.environ, {}, clear=False)
    def test_returns_none_without_api_key(self):
        os.environ.pop("ELEVENLABS_API_KEY", None)
        config = ElevenLabsConfig.from_env()
        assert config is None


class TestVoicevoxConfig:
    """Tests for VOICEVOX config."""

    @patch.dict(os.environ, {"VOICEVOX_URL": "http://localhost:50021"}, clear=False)
    def test_from_env(self):
        config = VoicevoxConfig.from_env()
        assert config is not None
        assert config.url == "http://localhost:50021"
        assert config.speaker == 3

    @patch.dict(
        os.environ,
        {"VOICEVOX_URL": "http://localhost:50021/", "VOICEVOX_SPEAKER": "8"},
        clear=False,
    )
    def test_trailing_slash_stripped(self):
        config = VoicevoxConfig.from_env()
        assert config is not None
        assert config.url == "http://localhost:50021"
        assert config.speaker == 8

    @patch.dict(os.environ, {}, clear=False)
    def test_returns_none_without_url(self):
        os.environ.pop("VOICEVOX_URL", None)
        config = VoicevoxConfig.from_env()
        assert config is None


class TestPlaybackConfig:
    """Tests for playback/go2rtc config."""

    @patch.dict(os.environ, {}, clear=False)
    def test_go2rtc_defaults(self):
        for key in ("GO2RTC_BIN", "GO2RTC_CONFIG", "GO2RTC_CAMERA_HOST",
                     "GO2RTC_CAMERA_USERNAME", "GO2RTC_CAMERA_PASSWORD"):
            os.environ.pop(key, None)
        config = PlaybackConfig.from_env()
        assert config.go2rtc_bin is None
        assert config.go2rtc_config is None
        assert config.go2rtc_auto_start is True
        assert config.go2rtc_camera_host is None

    @patch.dict(
        os.environ,
        {
            "GO2RTC_BIN": "/opt/go2rtc",
            "GO2RTC_CONFIG": "/etc/go2rtc.yaml",
            "GO2RTC_AUTO_START": "false",
            "GO2RTC_CAMERA_HOST": "10.0.0.1",
            "GO2RTC_CAMERA_USERNAME": "admin",
            "GO2RTC_CAMERA_PASSWORD": "pass123",
        },
        clear=False,
    )
    def test_go2rtc_from_env(self):
        config = PlaybackConfig.from_env()
        assert config.go2rtc_bin == "/opt/go2rtc"
        assert config.go2rtc_config == "/etc/go2rtc.yaml"
        assert config.go2rtc_auto_start is False
        assert config.go2rtc_camera_host == "10.0.0.1"
        assert config.go2rtc_camera_username == "admin"
        assert config.go2rtc_camera_password == "pass123"

    @patch.dict(
        os.environ,
        {
            "TAPO_CAMERA_HOST": "192.168.1.50",
            "TAPO_USERNAME": "tapo_user",
            "TAPO_PASSWORD": "tapo_pass",
        },
        clear=False,
    )
    def test_tapo_env_fallback(self):
        for key in ("GO2RTC_CAMERA_HOST", "GO2RTC_CAMERA_USERNAME", "GO2RTC_CAMERA_PASSWORD"):
            os.environ.pop(key, None)
        config = PlaybackConfig.from_env()
        assert config.go2rtc_camera_host == "192.168.1.50"
        assert config.go2rtc_camera_username == "tapo_user"
        assert config.go2rtc_camera_password == "tapo_pass"

    @patch.dict(
        os.environ,
        {
            "GO2RTC_CAMERA_HOST": "override",
            "TAPO_CAMERA_HOST": "fallback",
        },
        clear=False,
    )
    def test_go2rtc_env_takes_precedence_over_tapo(self):
        config = PlaybackConfig.from_env()
        assert config.go2rtc_camera_host == "override"

    @patch.dict(
        os.environ,
        {"GO2RTC_CAMERA_CLOUD_PASSWORD": "cloud123"},
        clear=False,
    )
    def test_cloud_password_from_go2rtc_env(self):
        config = PlaybackConfig.from_env()
        assert config.go2rtc_camera_cloud_password == "cloud123"

    @patch.dict(
        os.environ,
        {"TAPO_CLOUD_PASSWORD": "tapo_cloud"},
        clear=False,
    )
    def test_cloud_password_fallback_to_tapo_env(self):
        os.environ.pop("GO2RTC_CAMERA_CLOUD_PASSWORD", None)
        config = PlaybackConfig.from_env()
        assert config.go2rtc_camera_cloud_password == "tapo_cloud"

    @patch.dict(os.environ, {}, clear=False)
    def test_cloud_password_defaults_to_none(self):
        os.environ.pop("GO2RTC_CAMERA_CLOUD_PASSWORD", None)
        os.environ.pop("TAPO_CLOUD_PASSWORD", None)
        config = PlaybackConfig.from_env()
        assert config.go2rtc_camera_cloud_password is None


class TestTTSConfig:
    """Tests for top-level TTS config."""

    @patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}, clear=False)
    def test_resolve_elevenlabs_default(self):
        os.environ.pop("TTS_DEFAULT_ENGINE", None)
        os.environ.pop("VOICEVOX_URL", None)
        config = TTSConfig.from_env()
        assert config.resolve_engine() == "elevenlabs"

    @patch.dict(os.environ, {"VOICEVOX_URL": "http://localhost:50021"}, clear=False)
    def test_resolve_voicevox_default(self):
        os.environ.pop("TTS_DEFAULT_ENGINE", None)
        os.environ.pop("ELEVENLABS_API_KEY", None)
        config = TTSConfig.from_env()
        assert config.resolve_engine() == "voicevox"

    @patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "test-key"}, clear=False)
    def test_resolve_atlascloud_default(self):
        os.environ.pop("TTS_DEFAULT_ENGINE", None)
        os.environ.pop("ELEVENLABS_API_KEY", None)
        os.environ.pop("VOICEVOX_URL", None)
        config = TTSConfig.from_env()
        assert config.resolve_engine() == "atlascloud"

    @patch.dict(
        os.environ,
        {"TTS_DEFAULT_ENGINE": "voicevox", "VOICEVOX_URL": "http://localhost:50021",
         "ELEVENLABS_API_KEY": "test-key"},
        clear=False,
    )
    def test_resolve_explicit_engine(self):
        config = TTSConfig.from_env()
        assert config.resolve_engine() == "voicevox"
        assert config.resolve_engine("elevenlabs") == "elevenlabs"

    @patch.dict(os.environ, {}, clear=False)
    def test_resolve_raises_when_no_engine(self):
        os.environ.pop("TTS_DEFAULT_ENGINE", None)
        os.environ.pop("ELEVENLABS_API_KEY", None)
        os.environ.pop("VOICEVOX_URL", None)
        os.environ.pop("ATLASCLOUD_API_KEY", None)
        config = TTSConfig.from_env()
        try:
            config.resolve_engine()
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
