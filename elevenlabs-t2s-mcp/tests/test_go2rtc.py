"""Tests for go2rtc auto-download and process management."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from elevenlabs_t2s_mcp.go2rtc import (
    Go2RTCProcess,
    detect_platform,
    ensure_binary,
    generate_config,
)


class TestDetectPlatform:
    """Tests for platform detection."""

    @patch("elevenlabs_t2s_mcp.go2rtc.platform")
    def test_linux_amd64(self, mock_platform):
        mock_platform.system.return_value = "Linux"
        mock_platform.machine.return_value = "x86_64"
        assert detect_platform() == "go2rtc_linux_amd64"

    @patch("elevenlabs_t2s_mcp.go2rtc.platform")
    def test_linux_arm64(self, mock_platform):
        mock_platform.system.return_value = "Linux"
        mock_platform.machine.return_value = "aarch64"
        assert detect_platform() == "go2rtc_linux_arm64"

    @patch("elevenlabs_t2s_mcp.go2rtc.platform")
    def test_darwin_arm64(self, mock_platform):
        mock_platform.system.return_value = "Darwin"
        mock_platform.machine.return_value = "arm64"
        assert detect_platform() == "go2rtc_mac_arm64"

    @patch("elevenlabs_t2s_mcp.go2rtc.platform")
    def test_unsupported_platform(self, mock_platform):
        mock_platform.system.return_value = "FreeBSD"
        mock_platform.machine.return_value = "sparc64"
        with pytest.raises(RuntimeError, match="Unsupported platform"):
            detect_platform()


class TestEnsureBinary:
    """Tests for binary download."""

    def test_binary_already_exists(self, tmp_path):
        bin_path = tmp_path / "go2rtc"
        bin_path.write_text("fake binary")
        result = ensure_binary(bin_path)
        assert result == bin_path

    @patch("elevenlabs_t2s_mcp.go2rtc._get_download_url")
    @patch("elevenlabs_t2s_mcp.go2rtc._download_file")
    @patch("elevenlabs_t2s_mcp.go2rtc.detect_platform")
    @patch("elevenlabs_t2s_mcp.go2rtc.platform")
    def test_download_linux_binary(
        self, mock_platform, mock_detect, mock_download, mock_url, tmp_path
    ):
        mock_platform.system.return_value = "Linux"
        mock_detect.return_value = "go2rtc_linux_amd64"
        mock_url.return_value = "https://example.com/go2rtc_linux_amd64"

        bin_path = tmp_path / "go2rtc"

        def fake_download(url, dest):
            dest.write_text("fake binary")

        mock_download.side_effect = fake_download

        result = ensure_binary(bin_path)
        assert result == bin_path
        assert bin_path.exists()
        mock_download.assert_called_once()


class TestGenerateConfig:
    """Tests for config file generation."""

    def test_generates_valid_yaml(self, tmp_path):
        config_path = tmp_path / "go2rtc.yaml"
        result = generate_config(
            config_path=config_path,
            stream_name="tapo_cam",
            camera_host="192.168.1.100",
            username="admin",
            password="secret",
            ffmpeg_bin="/usr/bin/ffmpeg",
        )
        assert result == config_path
        content = config_path.read_text()
        assert "tapo_cam:" in content
        assert "rtsp://admin:secret@192.168.1.100:554/stream1" in content
        assert "tapo://secret@192.168.1.100" in content
        assert "/usr/bin/ffmpeg" in content
        assert '":1984"' in content

    def test_creates_parent_directories(self, tmp_path):
        config_path = tmp_path / "sub" / "dir" / "go2rtc.yaml"
        generate_config(
            config_path=config_path,
            stream_name="cam",
            camera_host="10.0.0.1",
            username="u",
            password="p",
        )
        assert config_path.exists()

    def test_overwrites_existing(self, tmp_path):
        config_path = tmp_path / "go2rtc.yaml"
        config_path.write_text("old content")
        generate_config(
            config_path=config_path,
            stream_name="new_cam",
            camera_host="10.0.0.2",
            username="u",
            password="p",
        )
        content = config_path.read_text()
        assert "new_cam:" in content
        assert "old content" not in content


class TestGo2RTCProcess:
    """Tests for process lifecycle."""

    def test_is_running_returns_false_when_unreachable(self):
        proc = Go2RTCProcess(
            Path("/fake/go2rtc"),
            Path("/fake/config.yaml"),
            api_url="http://localhost:19999",
        )
        assert proc.is_running() is False

    @patch("elevenlabs_t2s_mcp.go2rtc.urllib.request.urlopen")
    def test_is_running_returns_true_when_api_responds(self, mock_urlopen):
        mock_urlopen.return_value.__enter__ = MagicMock()
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)
        proc = Go2RTCProcess(
            Path("/fake/go2rtc"),
            Path("/fake/config.yaml"),
        )
        assert proc.is_running() is True

    @pytest.mark.asyncio
    @patch.object(Go2RTCProcess, "is_running", return_value=True)
    async def test_start_skips_when_already_running(self, mock_running):
        proc = Go2RTCProcess(Path("/fake"), Path("/fake"))
        await proc.start()
        # Should not spawn a process
        assert proc._process is None

    def test_stop_when_no_process(self):
        proc = Go2RTCProcess(Path("/fake"), Path("/fake"))
        # Should not raise
        proc.stop()
