"""Tests for ElevenLabs TTS config."""

import os
from unittest.mock import patch

from elevenlabs_t2s_mcp.config import ElevenLabsConfig, _parse_bool


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


class TestElevenLabsConfigGo2rtc:
    """Tests for go2rtc config fields."""

    @patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}, clear=False)
    def test_go2rtc_defaults(self):
        config = ElevenLabsConfig.from_env()
        assert config.go2rtc_bin is None
        assert config.go2rtc_config is None
        assert config.go2rtc_auto_start is True
        assert config.go2rtc_camera_host is None
        assert config.go2rtc_camera_username is None
        assert config.go2rtc_camera_password is None

    @patch.dict(
        os.environ,
        {
            "ELEVENLABS_API_KEY": "test-key",
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
        config = ElevenLabsConfig.from_env()
        assert config.go2rtc_bin == "/opt/go2rtc"
        assert config.go2rtc_config == "/etc/go2rtc.yaml"
        assert config.go2rtc_auto_start is False
        assert config.go2rtc_camera_host == "10.0.0.1"
        assert config.go2rtc_camera_username == "admin"
        assert config.go2rtc_camera_password == "pass123"

    @patch.dict(
        os.environ,
        {
            "ELEVENLABS_API_KEY": "test-key",
            "TAPO_CAMERA_HOST": "192.168.1.50",
            "TAPO_USERNAME": "tapo_user",
            "TAPO_PASSWORD": "tapo_pass",
        },
        clear=False,
    )
    def test_tapo_env_fallback(self):
        # Clear GO2RTC_ specific vars if present
        for key in ("GO2RTC_CAMERA_HOST", "GO2RTC_CAMERA_USERNAME", "GO2RTC_CAMERA_PASSWORD"):
            os.environ.pop(key, None)
        config = ElevenLabsConfig.from_env()
        assert config.go2rtc_camera_host == "192.168.1.50"
        assert config.go2rtc_camera_username == "tapo_user"
        assert config.go2rtc_camera_password == "tapo_pass"

    @patch.dict(
        os.environ,
        {
            "ELEVENLABS_API_KEY": "test-key",
            "GO2RTC_CAMERA_HOST": "override",
            "TAPO_CAMERA_HOST": "fallback",
        },
        clear=False,
    )
    def test_go2rtc_env_takes_precedence_over_tapo(self):
        config = ElevenLabsConfig.from_env()
        assert config.go2rtc_camera_host == "override"
