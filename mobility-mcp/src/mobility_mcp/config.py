"""Configuration for mobility MCP server."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


class TuyaDeviceConfig:
    """Configuration for a Tuya device connection."""

    def __init__(
        self,
        device_id: str,
        ip_address: str,
        local_key: str,
        version: str = "3.3",
    ):
        self.device_id = device_id
        self.ip_address = ip_address
        self.local_key = local_key
        self.version = version

    @classmethod
    def from_env(cls) -> TuyaDeviceConfig:
        """Create config from environment variables."""
        device_id = os.getenv("TUYA_DEVICE_ID", "")
        ip_address = os.getenv("TUYA_IP_ADDRESS", "")
        local_key = os.getenv("TUYA_LOCAL_KEY", "")
        version = os.getenv("TUYA_VERSION", "3.3")

        if not device_id:
            raise ValueError("TUYA_DEVICE_ID environment variable is required")
        if not ip_address:
            raise ValueError("TUYA_IP_ADDRESS environment variable is required")
        if not local_key:
            raise ValueError("TUYA_LOCAL_KEY environment variable is required")

        return cls(
            device_id=device_id,
            ip_address=ip_address,
            local_key=local_key,
            version=version,
        )


# Direction control Data Point ID (standard Tuya robot vacuum DP)
DIRECTION_DP = int(os.getenv("TUYA_DIRECTION_DP", "4"))

# Duration for timed movements (seconds)
DEFAULT_MOVE_DURATION = float(os.getenv("DEFAULT_MOVE_DURATION", "1.0"))
MAX_MOVE_DURATION = float(os.getenv("MAX_MOVE_DURATION", "10.0"))
