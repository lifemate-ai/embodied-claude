"""Tuya robot vacuum control for mobility."""

from __future__ import annotations

import asyncio
import logging

import tinytuya

from .config import DIRECTION_DP, TuyaDeviceConfig

logger = logging.getLogger(__name__)

# Tuya direction_control standard values
DIRECTION_FORWARD = "forward"
DIRECTION_BACKWARD = "backward"
DIRECTION_LEFT = "turn_left"
DIRECTION_RIGHT = "turn_right"
DIRECTION_STOP = "stop"

VALID_DIRECTIONS = {
    DIRECTION_FORWARD,
    DIRECTION_BACKWARD,
    DIRECTION_LEFT,
    DIRECTION_RIGHT,
    DIRECTION_STOP,
}


class VacuumMobilityController:
    """Controls a Tuya robot vacuum for AI mobility."""

    def __init__(self, config: TuyaDeviceConfig):
        self._config = config
        self._device: tinytuya.Device | None = None

    def _ensure_device(self) -> tinytuya.Device:
        """Get or create device connection."""
        if self._device is None:
            self._device = tinytuya.Device(
                dev_id=self._config.device_id,
                address=self._config.ip_address,
                local_key=self._config.local_key,
                version=self._config.version,
            )
            self._device.set_socketPersistent(True)
            logger.info(
                "Connected to Tuya device %s at %s",
                self._config.device_id,
                self._config.ip_address,
            )
        return self._device

    async def _send_direction(self, direction: str) -> dict:
        """Send a direction command to the vacuum."""
        if direction not in VALID_DIRECTIONS:
            raise ValueError(f"Invalid direction: {direction}. Must be one of {VALID_DIRECTIONS}")

        device = self._ensure_device()
        result = await asyncio.to_thread(device.set_value, DIRECTION_DP, direction)
        logger.info("Sent direction command: %s -> %s", direction, result)
        return result if isinstance(result, dict) else {}

    async def move_forward(self, duration: float | None = None) -> str:
        """Move forward. If duration specified, stop after that many seconds."""
        await self._send_direction(DIRECTION_FORWARD)
        if duration is not None:
            await asyncio.sleep(duration)
            await self._send_direction(DIRECTION_STOP)
            return f"Moved forward for {duration}s and stopped."
        return "Moving forward. Use stop to halt."

    async def move_backward(self, duration: float | None = None) -> str:
        """Move backward. If duration specified, stop after that many seconds."""
        await self._send_direction(DIRECTION_BACKWARD)
        if duration is not None:
            await asyncio.sleep(duration)
            await self._send_direction(DIRECTION_STOP)
            return f"Moved backward for {duration}s and stopped."
        return "Moving backward. Use stop to halt."

    async def turn_left(self, duration: float | None = None) -> str:
        """Turn left. If duration specified, stop after that many seconds."""
        await self._send_direction(DIRECTION_LEFT)
        if duration is not None:
            await asyncio.sleep(duration)
            await self._send_direction(DIRECTION_STOP)
            return f"Turned left for {duration}s and stopped."
        return "Turning left. Use stop to halt."

    async def turn_right(self, duration: float | None = None) -> str:
        """Turn right. If duration specified, stop after that many seconds."""
        await self._send_direction(DIRECTION_RIGHT)
        if duration is not None:
            await asyncio.sleep(duration)
            await self._send_direction(DIRECTION_STOP)
            return f"Turned right for {duration}s and stopped."
        return "Turning right. Use stop to halt."

    async def stop(self) -> str:
        """Stop all movement immediately."""
        await self._send_direction(DIRECTION_STOP)
        return "Stopped."

    async def get_status(self) -> dict:
        """Get current device status."""
        device = self._ensure_device()
        status = await asyncio.to_thread(device.status)
        logger.info("Device status: %s", status)
        return status if isinstance(status, dict) else {}

    def disconnect(self) -> None:
        """Close device connection."""
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None
            logger.info("Disconnected from Tuya device")
