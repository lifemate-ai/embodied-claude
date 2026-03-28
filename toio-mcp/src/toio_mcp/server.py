"""MCP server for a toio Core Cube used as a tiny programmable hand."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .config import MAX_DURATION_SECONDS, MIN_DURATION_SECONDS, ToioConfig
from .controller import ToioHandController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SPEED_SCHEMA = {
    "type": "integer",
    "description": "Motor speed from 1 to 100. Higher is faster.",
    "minimum": 1,
    "maximum": 100,
}

DURATION_SCHEMA = {
    "type": "number",
    "description": (
        f"Duration in seconds. Safe range: {MIN_DURATION_SECONDS} "
        f"to {MAX_DURATION_SECONDS}."
    ),
    "minimum": MIN_DURATION_SECONDS,
    "maximum": MAX_DURATION_SECONDS,
}


class ToioMCPServer:
    """Expose a toio cube as a small hand through MCP tools."""

    def __init__(self):
        self._config = ToioConfig.from_env()
        self._server = Server("toio-mcp")
        self._controller: ToioHandController | None = None
        self._setup_handlers()

    def _ensure_controller(self) -> ToioHandController:
        if self._controller is None:
            self._controller = ToioHandController(self._config)
        return self._controller

    def _setup_handlers(self) -> None:
        @self._server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="connect_hand",
                    description=(
                        "Connect to the toio hand. Optionally specify the cube's unique 3-digit "
                        "name suffix and scan timeout."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Optional 3-digit cube name suffix to target.",
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "BLE scan timeout in seconds.",
                                "minimum": 1,
                                "maximum": 30,
                            },
                        },
                        "required": [],
                    },
                ),
                Tool(
                    name="disconnect_hand",
                    description="Disconnect from the toio hand.",
                    inputSchema={"type": "object", "properties": {}, "required": []},
                ),
                Tool(
                    name="hand_status",
                    description=(
                        "Get battery, position, posture, orientation, "
                        "and other current hand state. "
                        "Position fields are useful only when a Position ID mat is available."
                    ),
                    inputSchema={"type": "object", "properties": {}, "required": []},
                ),
                Tool(
                    name="move_hand_forward",
                    description="Move the hand forward for a short duration.",
                    inputSchema={
                        "type": "object",
                        "properties": {"speed": SPEED_SCHEMA, "duration": DURATION_SCHEMA},
                        "required": [],
                    },
                ),
                Tool(
                    name="move_hand_backward",
                    description="Move the hand backward for a short duration.",
                    inputSchema={
                        "type": "object",
                        "properties": {"speed": SPEED_SCHEMA, "duration": DURATION_SCHEMA},
                        "required": [],
                    },
                ),
                Tool(
                    name="rotate_hand_left",
                    description="Rotate the hand counterclockwise in place.",
                    inputSchema={
                        "type": "object",
                        "properties": {"speed": SPEED_SCHEMA, "duration": DURATION_SCHEMA},
                        "required": [],
                    },
                ),
                Tool(
                    name="rotate_hand_right",
                    description="Rotate the hand clockwise in place.",
                    inputSchema={
                        "type": "object",
                        "properties": {"speed": SPEED_SCHEMA, "duration": DURATION_SCHEMA},
                        "required": [],
                    },
                ),
                Tool(
                    name="stop_hand",
                    description="Stop all motor movement immediately.",
                    inputSchema={"type": "object", "properties": {}, "required": []},
                ),
                Tool(
                    name="move_hand_to_position",
                    description=(
                        "Move the hand to an absolute (x, y) position on a Position ID mat. "
                        "Fails if no mat is visible."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer", "description": "Target x coordinate."},
                            "y": {"type": "integer", "description": "Target y coordinate."},
                            "speed": SPEED_SCHEMA,
                        },
                        "required": ["x", "y"],
                    },
                ),
                Tool(
                    name="move_hand_to_grid_cell",
                    description=(
                        "Move the hand to a grid cell on the bundled simple mat or another "
                        "Position ID mat."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "cell_x": {"type": "integer", "description": "Target grid x cell."},
                            "cell_y": {"type": "integer", "description": "Target grid y cell."},
                            "speed": SPEED_SCHEMA,
                        },
                        "required": ["cell_x", "cell_y"],
                    },
                ),
                Tool(
                    name="set_hand_orientation",
                    description="Set the hand to a specific orientation in degrees on a mat.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "degree": {
                                "type": "integer",
                                "description": "Target orientation in degrees.",
                            },
                            "speed": SPEED_SCHEMA,
                        },
                        "required": ["degree"],
                    },
                ),
                Tool(
                    name="set_hand_light",
                    description="Light the cube's RGB lamp.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "r": {"type": "integer", "minimum": 0, "maximum": 255},
                            "g": {"type": "integer", "minimum": 0, "maximum": 255},
                            "b": {"type": "integer", "minimum": 0, "maximum": 255},
                            "duration": {
                                "type": "number",
                                "minimum": MIN_DURATION_SECONDS,
                                "description": "Lamp duration in seconds.",
                            },
                        },
                        "required": ["r", "g", "b"],
                    },
                ),
                Tool(
                    name="clear_hand_light",
                    description="Turn off the cube's RGB lamp.",
                    inputSchema={"type": "object", "properties": {}, "required": []},
                ),
                Tool(
                    name="play_hand_note",
                    description="Play a short MIDI note through the cube speaker.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "note": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 127,
                                "description": "MIDI note number.",
                            },
                            "duration": DURATION_SCHEMA,
                        },
                        "required": ["note"],
                    },
                ),
                Tool(
                    name="stop_hand_sound",
                    description="Stop cube sound playback immediately.",
                    inputSchema={"type": "object", "properties": {}, "required": []},
                ),
            ]

        @self._server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            controller = self._ensure_controller()

            try:
                if name == "connect_hand":
                    result = await controller.connect(
                        name=arguments.get("name"),
                        timeout=arguments.get("timeout"),
                    )
                elif name == "disconnect_hand":
                    result = await controller.disconnect()
                elif name == "hand_status":
                    result = json.dumps(
                        await controller.status(), ensure_ascii=False, indent=2, sort_keys=True
                    )
                elif name == "move_hand_forward":
                    result = await controller.move_forward(
                        speed=arguments.get("speed"),
                        duration=arguments.get("duration"),
                    )
                elif name == "move_hand_backward":
                    result = await controller.move_backward(
                        speed=arguments.get("speed"),
                        duration=arguments.get("duration"),
                    )
                elif name == "rotate_hand_left":
                    result = await controller.rotate_left(
                        speed=arguments.get("speed"),
                        duration=arguments.get("duration"),
                    )
                elif name == "rotate_hand_right":
                    result = await controller.rotate_right(
                        speed=arguments.get("speed"),
                        duration=arguments.get("duration"),
                    )
                elif name == "stop_hand":
                    result = await controller.stop()
                elif name == "move_hand_to_position":
                    result = await controller.move_to_position(
                        x=int(arguments["x"]),
                        y=int(arguments["y"]),
                        speed=arguments.get("speed"),
                    )
                elif name == "move_hand_to_grid_cell":
                    result = await controller.move_to_grid_cell(
                        cell_x=int(arguments["cell_x"]),
                        cell_y=int(arguments["cell_y"]),
                        speed=arguments.get("speed"),
                    )
                elif name == "set_hand_orientation":
                    result = await controller.set_orientation(
                        degree=int(arguments["degree"]),
                        speed=arguments.get("speed"),
                    )
                elif name == "set_hand_light":
                    result = await controller.set_light(
                        r=int(arguments["r"]),
                        g=int(arguments["g"]),
                        b=int(arguments["b"]),
                        duration=arguments.get("duration"),
                    )
                elif name == "clear_hand_light":
                    result = await controller.clear_light()
                elif name == "play_hand_note":
                    result = await controller.play_note(
                        note=int(arguments["note"]),
                        duration=arguments.get("duration"),
                    )
                elif name == "stop_hand_sound":
                    result = await controller.stop_sound()
                else:
                    result = f"Unknown tool: {name}"
            except Exception as exc:  # pragma: no cover - exercised in integration use
                logger.exception("Tool %s failed", name)
                result = f"Error: {exc}"

            return [TextContent(type="text", text=result)]

    async def run(self) -> None:
        async with stdio_server() as (read_stream, write_stream):
            try:
                await self._server.run(
                    read_stream,
                    write_stream,
                    self._server.create_initialization_options(),
                )
            finally:
                if self._controller is not None:
                    await self._controller.disconnect()


def main() -> None:
    server = ToioMCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
