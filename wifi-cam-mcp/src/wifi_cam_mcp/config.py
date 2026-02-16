"""Configuration for WiFi Camera MCP Server."""

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _default_capture_dir() -> str:
    """Return a platform-appropriate default capture directory."""
    if sys.platform == "win32":
        local_app_data = os.getenv("LOCALAPPDATA")
        if local_app_data:
            return str(Path(local_app_data) / "wifi-cam-mcp")
        return str(Path(tempfile.gettempdir()) / "wifi-cam-mcp")
    return "/tmp/wifi-cam-mcp"


@dataclass(frozen=True)
class CameraConfig:
    """Camera connection configuration."""

    host: str
    username: str
    password: str
    onvif_port: int = 2020
    stream_url: str | None = None
    max_width: int = 1920
    max_height: int = 1080
    mount_mode: str = "normal"  # "normal" (desktop) or "ceiling" (inverted)
    jpeg_quality: int = 85  # JPEG quality (1-100)
    motor_delay: float = 0.5  # seconds to wait after PTZ move
    preset_delay: float = 1.0  # seconds to wait after go_to_preset

    @classmethod
    def from_env(cls, prefix: str = "TAPO") -> "CameraConfig":
        """Create config from environment variables.

        Args:
            prefix: Environment variable prefix (default: "TAPO")
                    For right camera, use "TAPO_RIGHT"
        """
        host = os.getenv(f"{prefix}_CAMERA_HOST", "") or os.getenv("TAPO_CAMERA_HOST", "")
        username = os.getenv(f"{prefix}_USERNAME", "") or os.getenv("TAPO_USERNAME", "")
        password = os.getenv(f"{prefix}_PASSWORD", "") or os.getenv("TAPO_PASSWORD", "")
        onvif_port = int(
            os.getenv(f"{prefix}_ONVIF_PORT", "") or os.getenv("TAPO_ONVIF_PORT", "") or "2020"
        )
        stream_url = os.getenv(f"{prefix}_STREAM_URL") or os.getenv("TAPO_STREAM_URL")
        mount_mode = (
            os.getenv(f"{prefix}_MOUNT_MODE", "") or os.getenv("TAPO_MOUNT_MODE", "") or "normal"
        ).lower()
        if mount_mode not in ("normal", "ceiling"):
            raise ValueError(f"Invalid mount mode '{mount_mode}'. Must be 'normal' or 'ceiling'.")
        max_width = int(os.getenv("CAPTURE_MAX_WIDTH", "1920"))
        max_height = int(os.getenv("CAPTURE_MAX_HEIGHT", "1080"))
        jpeg_quality = max(
            1,
            min(
                100,
                int(
                    os.getenv(f"{prefix}_JPEG_QUALITY", "")
                    or os.getenv("TAPO_JPEG_QUALITY", "")
                    or "85"
                ),
            ),
        )
        motor_delay = float(
            os.getenv(f"{prefix}_MOTOR_DELAY", "")
            or os.getenv("TAPO_MOTOR_DELAY", "")
            or "0.5"
        )
        preset_delay = float(
            os.getenv(f"{prefix}_PRESET_DELAY", "")
            or os.getenv("TAPO_PRESET_DELAY", "")
            or "1.0"
        )

        if not host:
            raise ValueError(f"{prefix}_CAMERA_HOST environment variable is required")
        if not username:
            raise ValueError(f"{prefix}_USERNAME environment variable is required")
        if not password:
            raise ValueError(f"{prefix}_PASSWORD environment variable is required")

        return cls(
            host=host,
            username=username,
            password=password,
            onvif_port=onvif_port,
            stream_url=stream_url,
            mount_mode=mount_mode,
            max_width=max_width,
            max_height=max_height,
            jpeg_quality=jpeg_quality,
            motor_delay=motor_delay,
            preset_delay=preset_delay,
        )

    @classmethod
    def right_camera_from_env(cls) -> "CameraConfig | None":
        """Create config for right camera if configured.

        Returns:
            CameraConfig for right camera, or None if not configured
        """
        host = os.getenv("TAPO_RIGHT_CAMERA_HOST", "")
        if not host:
            return None

        # Right camera can share username/password with left, or have its own
        username = os.getenv("TAPO_RIGHT_USERNAME", "") or os.getenv("TAPO_USERNAME", "")
        password = os.getenv("TAPO_RIGHT_PASSWORD", "") or os.getenv("TAPO_PASSWORD", "")
        onvif_port = int(
            os.getenv("TAPO_RIGHT_ONVIF_PORT", "") or os.getenv("TAPO_ONVIF_PORT", "") or "2020"
        )
        stream_url = os.getenv("TAPO_RIGHT_STREAM_URL")
        mount_mode = (
            os.getenv("TAPO_RIGHT_MOUNT_MODE", "") or os.getenv("TAPO_MOUNT_MODE", "") or "normal"
        ).lower()
        max_width = int(os.getenv("CAPTURE_MAX_WIDTH", "1920"))
        max_height = int(os.getenv("CAPTURE_MAX_HEIGHT", "1080"))
        jpeg_quality = max(
            1,
            min(
                100,
                int(
                    os.getenv("TAPO_RIGHT_JPEG_QUALITY", "")
                    or os.getenv("TAPO_JPEG_QUALITY", "")
                    or "85"
                ),
            ),
        )
        motor_delay = float(
            os.getenv("TAPO_RIGHT_MOTOR_DELAY", "")
            or os.getenv("TAPO_MOTOR_DELAY", "")
            or "0.5"
        )
        preset_delay = float(
            os.getenv("TAPO_RIGHT_PRESET_DELAY", "")
            or os.getenv("TAPO_PRESET_DELAY", "")
            or "1.0"
        )

        if not username or not password:
            return None

        return cls(
            host=host,
            username=username,
            password=password,
            onvif_port=onvif_port,
            stream_url=stream_url,
            mount_mode=mount_mode,
            max_width=max_width,
            max_height=max_height,
            jpeg_quality=jpeg_quality,
            motor_delay=motor_delay,
            preset_delay=preset_delay,
        )


@dataclass(frozen=True)
class ServerConfig:
    """MCP Server configuration."""

    name: str = "wifi-cam-mcp"
    version: str = "0.1.0"
    capture_dir: str = ""

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create config from environment variables."""
        return cls(
            name=os.getenv("MCP_SERVER_NAME", "wifi-cam-mcp"),
            version=os.getenv("MCP_SERVER_VERSION", "0.1.0"),
            capture_dir=os.getenv("CAPTURE_DIR", "") or _default_capture_dir(),
        )
