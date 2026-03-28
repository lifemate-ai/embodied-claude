"""Configuration for toio-mcp."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_SCAN_TIMEOUT = 5
DEFAULT_MOVE_SPEED = 50
DEFAULT_ROTATE_SPEED = 60
DEFAULT_MOVE_DURATION = 0.5
DEFAULT_LIGHT_DURATION = 0.5
DEFAULT_NOTE_DURATION = 0.25
DEFAULT_MAX_SPEED = 70
MAX_DURATION_SECONDS = 2.55
MIN_DURATION_SECONDS = 0.05


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class ToioConfig:
    """Runtime configuration for toio-mcp."""

    cube_name: str | None = None
    scan_timeout: int = DEFAULT_SCAN_TIMEOUT
    max_speed: int = DEFAULT_MAX_SPEED
    move_speed: int = DEFAULT_MOVE_SPEED
    rotate_speed: int = DEFAULT_ROTATE_SPEED
    move_duration: float = DEFAULT_MOVE_DURATION
    light_duration: float = DEFAULT_LIGHT_DURATION
    note_duration: float = DEFAULT_NOTE_DURATION
    dry_run: bool = False

    @classmethod
    def from_env(cls) -> "ToioConfig":
        """Build config from environment variables."""
        cube_name = os.getenv("TOIO_CUBE_NAME") or None
        scan_timeout = int(os.getenv("TOIO_SCAN_TIMEOUT", DEFAULT_SCAN_TIMEOUT))
        max_speed = int(os.getenv("TOIO_MAX_SPEED", DEFAULT_MAX_SPEED))
        move_speed = int(os.getenv("TOIO_MOVE_SPEED", DEFAULT_MOVE_SPEED))
        rotate_speed = int(os.getenv("TOIO_ROTATE_SPEED", DEFAULT_ROTATE_SPEED))
        move_duration = float(os.getenv("TOIO_MOVE_DURATION", DEFAULT_MOVE_DURATION))
        light_duration = float(os.getenv("TOIO_LIGHT_DURATION", DEFAULT_LIGHT_DURATION))
        note_duration = float(os.getenv("TOIO_NOTE_DURATION", DEFAULT_NOTE_DURATION))
        dry_run = _env_bool("TOIO_DRY_RUN", False)
        return cls(
            cube_name=cube_name,
            scan_timeout=max(1, scan_timeout),
            max_speed=max(1, min(abs(max_speed), 100)),
            move_speed=max(1, min(abs(move_speed), 100)),
            rotate_speed=max(1, min(abs(rotate_speed), 100)),
            move_duration=max(MIN_DURATION_SECONDS, min(move_duration, MAX_DURATION_SECONDS)),
            light_duration=max(MIN_DURATION_SECONDS, light_duration),
            note_duration=max(MIN_DURATION_SECONDS, min(note_duration, MAX_DURATION_SECONDS)),
            dry_run=dry_run,
        )
