"""Tests for configuration module."""

import os
from unittest.mock import patch

import pytest

from mobility_mcp.config import TuyaDeviceConfig


class TestTuyaDeviceConfig:
    def test_from_env_success(self):
        env = {
            "TUYA_DEVICE_ID": "test_device_123",
            "TUYA_IP_ADDRESS": "192.168.1.100",
            "TUYA_LOCAL_KEY": "abc123localkey",
            "TUYA_VERSION": "3.4",
        }
        with patch.dict(os.environ, env, clear=False):
            config = TuyaDeviceConfig.from_env()
            assert config.device_id == "test_device_123"
            assert config.ip_address == "192.168.1.100"
            assert config.local_key == "abc123localkey"
            assert config.version == "3.4"

    def test_from_env_default_version(self):
        env = {
            "TUYA_DEVICE_ID": "test_device",
            "TUYA_IP_ADDRESS": "192.168.1.100",
            "TUYA_LOCAL_KEY": "localkey",
        }
        with patch.dict(os.environ, env, clear=False):
            # Remove TUYA_VERSION if set
            os.environ.pop("TUYA_VERSION", None)
            config = TuyaDeviceConfig.from_env()
            assert config.version == "3.3"

    def test_from_env_missing_device_id(self):
        env = {
            "TUYA_IP_ADDRESS": "192.168.1.100",
            "TUYA_LOCAL_KEY": "localkey",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("TUYA_DEVICE_ID", None)
            with pytest.raises(ValueError, match="TUYA_DEVICE_ID"):
                TuyaDeviceConfig.from_env()

    def test_from_env_missing_ip(self):
        env = {
            "TUYA_DEVICE_ID": "test_device",
            "TUYA_LOCAL_KEY": "localkey",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("TUYA_IP_ADDRESS", None)
            with pytest.raises(ValueError, match="TUYA_IP_ADDRESS"):
                TuyaDeviceConfig.from_env()

    def test_from_env_missing_local_key(self):
        env = {
            "TUYA_DEVICE_ID": "test_device",
            "TUYA_IP_ADDRESS": "192.168.1.100",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("TUYA_LOCAL_KEY", None)
            with pytest.raises(ValueError, match="TUYA_LOCAL_KEY"):
                TuyaDeviceConfig.from_env()

    def test_constructor(self):
        config = TuyaDeviceConfig(
            device_id="dev1",
            ip_address="10.0.0.1",
            local_key="key123",
            version="3.5",
        )
        assert config.device_id == "dev1"
        assert config.ip_address == "10.0.0.1"
        assert config.local_key == "key123"
        assert config.version == "3.5"
