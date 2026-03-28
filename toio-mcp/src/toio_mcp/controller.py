"""Controller layer for a toio Core Cube used as a small hand."""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Protocol

from .config import MAX_DURATION_SECONDS, MIN_DURATION_SECONDS, ToioConfig

logger = logging.getLogger(__name__)


class CubeLike(Protocol):
    """Subset of the toio SimpleCube interface used by this server."""

    def disconnect(self) -> None: ...

    def move(self, speed: int, duration: float) -> None: ...

    def spin(self, speed: int, duration: float) -> None: ...

    def stop_motor(self) -> None: ...

    def move_to(self, speed: int, x: int, y: int) -> bool: ...

    def move_to_the_grid_cell(self, speed: int, cell_x: int, cell_y: int) -> bool: ...

    def set_orientation(self, speed: int, degree: int) -> bool: ...

    def turn_on_cube_lamp(self, r: int, g: int, b: int, duration: float) -> None: ...

    def turn_off_cube_lamp(self) -> None: ...

    def play_sound(self, note: int, duration: float) -> bool: ...

    def stop_sound(self) -> None: ...

    def get_cube_name(self) -> str | None: ...

    def get_battery_level(self) -> int | None: ...

    def get_current_position(self) -> tuple[int, int] | None: ...

    def get_orientation(self) -> int | None: ...

    def get_grid(self) -> tuple[int, int] | None: ...

    def get_posture(self) -> int | None: ...

    def is_button_pressed(self) -> int | None: ...

    def is_magnet_in_contact(self) -> int | None: ...


class DryRunCube:
    """A lightweight cube simulator for design work before hardware arrives."""

    def __init__(self, name: str | None = None, timeout: int = 5):
        self._name = name or "dry-run"
        self._timeout = timeout
        self._x = 0
        self._y = 0
        self._orientation = 0
        self._grid = (0, 0)
        self._battery = 100
        self._posture = 1
        self._button = 0
        self._magnet = 0
        self._lamp = (0, 0, 0)
        self._sound: tuple[int, float] | None = None

    def disconnect(self) -> None:
        return None

    def move(self, speed: int, duration: float) -> None:
        radians = math.radians(self._orientation)
        distance = speed * duration
        self._x += int(round(math.cos(radians) * distance))
        self._y += int(round(math.sin(radians) * distance))

    def spin(self, speed: int, duration: float) -> None:
        self._orientation = int(round((self._orientation + speed * duration) % 360))

    def stop_motor(self) -> None:
        return None

    def move_to(self, speed: int, x: int, y: int) -> bool:
        _ = speed
        self._x = x
        self._y = y
        return True

    def move_to_the_grid_cell(self, speed: int, cell_x: int, cell_y: int) -> bool:
        _ = speed
        self._grid = (cell_x, cell_y)
        self._x = cell_x
        self._y = cell_y
        return True

    def set_orientation(self, speed: int, degree: int) -> bool:
        _ = speed
        self._orientation = degree % 360
        return True

    def turn_on_cube_lamp(self, r: int, g: int, b: int, duration: float) -> None:
        _ = duration
        self._lamp = (r, g, b)

    def turn_off_cube_lamp(self) -> None:
        self._lamp = (0, 0, 0)

    def play_sound(self, note: int, duration: float) -> bool:
        self._sound = (note, duration)
        return True

    def stop_sound(self) -> None:
        self._sound = None

    def get_cube_name(self) -> str | None:
        return self._name

    def get_battery_level(self) -> int | None:
        return self._battery

    def get_current_position(self) -> tuple[int, int] | None:
        return (self._x, self._y)

    def get_orientation(self) -> int | None:
        return self._orientation

    def get_grid(self) -> tuple[int, int] | None:
        return self._grid

    def get_posture(self) -> int | None:
        return self._posture

    def is_button_pressed(self) -> int | None:
        return self._button

    def is_magnet_in_contact(self) -> int | None:
        return self._magnet


class ToioHandController:
    """High-level controller for a toio cube used as a tiny hand."""

    def __init__(self, config: ToioConfig):
        self._config = config
        self._cube: CubeLike | None = None

    def _make_cube(self, name: str | None, timeout: int) -> CubeLike:
        if self._config.dry_run:
            return DryRunCube(name=name, timeout=timeout)
        try:
            from toio.simple import SimpleCube
        except ImportError as exc:  # pragma: no cover - exercised only in broken envs
            raise RuntimeError(
                "toio.py is not installed. Run `uv sync` in toio-mcp first."
            ) from exc
        return SimpleCube(name=name, timeout=timeout)

    def _clamp_speed(self, speed: int | None, default: int) -> int:
        if speed is None:
            speed = default
        return max(1, min(abs(speed), self._config.max_speed))

    def _clamp_duration(self, duration: float | None, default: float) -> float:
        if duration is None:
            duration = default
        return max(MIN_DURATION_SECONDS, min(duration, MAX_DURATION_SECONDS))

    async def connect(self, name: str | None = None, timeout: int | None = None) -> str:
        if self._cube is not None:
            await self.disconnect()
        resolved_name = name or self._config.cube_name
        resolved_timeout = max(1, timeout or self._config.scan_timeout)
        self._cube = await asyncio.to_thread(self._make_cube, resolved_name, resolved_timeout)
        cube_name = await asyncio.to_thread(self._cube.get_cube_name)
        dry_run_suffix = " (dry-run)" if self._config.dry_run else ""
        if cube_name:
            return f"Connected to toio hand '{cube_name}'{dry_run_suffix}."
        return f"Connected to toio hand{dry_run_suffix}."

    async def ensure_connected(self) -> CubeLike:
        if self._cube is None:
            await self.connect()
        assert self._cube is not None
        return self._cube

    async def disconnect(self) -> str:
        if self._cube is None:
            return "toio hand is already disconnected."
        cube = self._cube
        self._cube = None
        await asyncio.to_thread(cube.disconnect)
        return "Disconnected from toio hand."

    async def move_forward(self, speed: int | None = None, duration: float | None = None) -> str:
        cube = await self.ensure_connected()
        actual_speed = self._clamp_speed(speed, self._config.move_speed)
        actual_duration = self._clamp_duration(duration, self._config.move_duration)
        await asyncio.to_thread(cube.move, actual_speed, actual_duration)
        return f"Moved hand forward at speed {actual_speed} for {actual_duration:.2f}s."

    async def move_backward(self, speed: int | None = None, duration: float | None = None) -> str:
        cube = await self.ensure_connected()
        actual_speed = self._clamp_speed(speed, self._config.move_speed)
        actual_duration = self._clamp_duration(duration, self._config.move_duration)
        await asyncio.to_thread(cube.move, -actual_speed, actual_duration)
        return f"Moved hand backward at speed {actual_speed} for {actual_duration:.2f}s."

    async def rotate_left(self, speed: int | None = None, duration: float | None = None) -> str:
        cube = await self.ensure_connected()
        actual_speed = self._clamp_speed(speed, self._config.rotate_speed)
        actual_duration = self._clamp_duration(duration, self._config.move_duration)
        await asyncio.to_thread(cube.spin, actual_speed, actual_duration)
        return f"Rotated hand left at speed {actual_speed} for {actual_duration:.2f}s."

    async def rotate_right(self, speed: int | None = None, duration: float | None = None) -> str:
        cube = await self.ensure_connected()
        actual_speed = self._clamp_speed(speed, self._config.rotate_speed)
        actual_duration = self._clamp_duration(duration, self._config.move_duration)
        await asyncio.to_thread(cube.spin, -actual_speed, actual_duration)
        return f"Rotated hand right at speed {actual_speed} for {actual_duration:.2f}s."

    async def stop(self) -> str:
        cube = await self.ensure_connected()
        await asyncio.to_thread(cube.stop_motor)
        return "Stopped toio hand."

    async def move_to_position(self, x: int, y: int, speed: int | None = None) -> str:
        cube = await self.ensure_connected()
        actual_speed = self._clamp_speed(speed, self._config.move_speed)
        success = await asyncio.to_thread(cube.move_to, actual_speed, x, y)
        if success:
            return f"Moved hand to position ({x}, {y}) at speed {actual_speed}."
        return "Failed to move hand to position. Position ID may be unavailable."

    async def move_to_grid_cell(
        self, cell_x: int, cell_y: int, speed: int | None = None
    ) -> str:
        cube = await self.ensure_connected()
        actual_speed = self._clamp_speed(speed, self._config.move_speed)
        success = await asyncio.to_thread(cube.move_to_the_grid_cell, actual_speed, cell_x, cell_y)
        if success:
            return f"Moved hand to grid cell ({cell_x}, {cell_y}) at speed {actual_speed}."
        return "Failed to move hand to grid cell. A Position ID mat is required."

    async def set_orientation(self, degree: int, speed: int | None = None) -> str:
        cube = await self.ensure_connected()
        actual_speed = self._clamp_speed(speed, self._config.rotate_speed)
        normalized_degree = degree % 360
        success = await asyncio.to_thread(cube.set_orientation, actual_speed, normalized_degree)
        if success:
            return f"Set hand orientation to {normalized_degree} degrees at speed {actual_speed}."
        return "Failed to set hand orientation. A Position ID mat is required."

    async def set_light(self, r: int, g: int, b: int, duration: float | None = None) -> str:
        cube = await self.ensure_connected()
        actual_duration = max(
            MIN_DURATION_SECONDS,
            duration if duration is not None else self._config.light_duration,
        )
        red = max(0, min(r, 255))
        green = max(0, min(g, 255))
        blue = max(0, min(b, 255))
        await asyncio.to_thread(cube.turn_on_cube_lamp, red, green, blue, actual_duration)
        return (
            f"Set hand light to rgb({red}, {green}, {blue}) for {actual_duration:.2f}s."
        )

    async def clear_light(self) -> str:
        cube = await self.ensure_connected()
        await asyncio.to_thread(cube.turn_off_cube_lamp)
        return "Turned off hand light."

    async def play_note(self, note: int, duration: float | None = None) -> str:
        cube = await self.ensure_connected()
        actual_note = max(0, min(note, 127))
        actual_duration = self._clamp_duration(duration, self._config.note_duration)
        success = await asyncio.to_thread(cube.play_sound, actual_note, actual_duration)
        if success:
            return f"Played note {actual_note} for {actual_duration:.2f}s."
        return "Failed to play note on toio hand."

    async def stop_sound(self) -> str:
        cube = await self.ensure_connected()
        await asyncio.to_thread(cube.stop_sound)
        return "Stopped hand sound."

    async def status(self) -> dict:
        if self._cube is None:
            return {"connected": False, "dry_run": self._config.dry_run}

        cube = self._cube
        cube_name = await asyncio.to_thread(cube.get_cube_name)
        battery = await asyncio.to_thread(cube.get_battery_level)
        position = await asyncio.to_thread(cube.get_current_position)
        orientation = await asyncio.to_thread(cube.get_orientation)
        grid = await asyncio.to_thread(cube.get_grid)
        posture = await asyncio.to_thread(cube.get_posture)
        button = await asyncio.to_thread(cube.is_button_pressed)
        magnet = await asyncio.to_thread(cube.is_magnet_in_contact)
        return {
            "connected": True,
            "dry_run": self._config.dry_run,
            "cube_name": cube_name,
            "battery_level": battery,
            "position": (
                {"x": position[0], "y": position[1]} if position is not None else None
            ),
            "orientation": orientation,
            "grid": {"x": grid[0], "y": grid[1]} if grid is not None else None,
            "posture": posture,
            "button_pressed": None if button is None else bool(button),
            "magnet_in_contact": magnet,
        }
