"""Configuration for WiFi Camera MCP Server."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class CameraConfig:
    """Camera connection configuration."""

    host: str
    username: str
    password: str
    stream_url: str | None = None
    max_width: int = 1920
    max_height: int = 1080

    @classmethod
    def from_env(cls) -> "CameraConfig":
        """Create config from environment variables."""
        host = os.getenv("TAPO_CAMERA_HOST", "")
        username = os.getenv("TAPO_USERNAME", "")
        password = os.getenv("TAPO_PASSWORD", "")
        stream_url = os.getenv("TAPO_STREAM_URL")
        max_width = int(os.getenv("CAPTURE_MAX_WIDTH", "1920"))
        max_height = int(os.getenv("CAPTURE_MAX_HEIGHT", "1080"))

        if not host:
            raise ValueError("TAPO_CAMERA_HOST environment variable is required")
        if not username:
            raise ValueError("TAPO_USERNAME environment variable is required")
        if not password:
            raise ValueError("TAPO_PASSWORD environment variable is required")

        return cls(
            host=host,
            username=username,
            password=password,
            stream_url=stream_url,
            max_width=max_width,
            max_height=max_height,
        )


@dataclass(frozen=True)
class ServerConfig:
    """MCP Server configuration."""

    name: str = "wifi-cam-mcp"
    version: str = "0.1.0"
    capture_dir: str = "/tmp/wifi-cam-mcp"

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create config from environment variables."""
        return cls(
            name=os.getenv("MCP_SERVER_NAME", "wifi-cam-mcp"),
            version=os.getenv("MCP_SERVER_VERSION", "0.1.0"),
            capture_dir=os.getenv("CAPTURE_DIR", "/tmp/wifi-cam-mcp"),
        )
