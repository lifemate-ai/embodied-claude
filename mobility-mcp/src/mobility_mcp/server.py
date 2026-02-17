"""MCP Server for Robot Vacuum Mobility Control - Let AI walk!"""

import asyncio
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .config import MAX_MOVE_DURATION, TuyaDeviceConfig
from .vacuum import VacuumMobilityController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DURATION_SCHEMA = {
    "type": "number",
    "description": (
        "How long to move in seconds (default: continuous until stop). "
        f"Max: {MAX_MOVE_DURATION}s."
    ),
    "minimum": 0.1,
    "maximum": MAX_MOVE_DURATION,
}


class MobilityMCPServer:
    """MCP Server that gives AI legs to move around the room."""

    def __init__(self):
        self._server = Server("mobility-mcp")
        self._controller: VacuumMobilityController | None = None
        self._setup_handlers()

    def _ensure_controller(self) -> VacuumMobilityController:
        """Get or create the vacuum controller."""
        if self._controller is None:
            config = TuyaDeviceConfig.from_env()
            self._controller = VacuumMobilityController(config)
        return self._controller

    def _clamp_duration(self, duration: float | None) -> float | None:
        """Clamp duration to safe range."""
        if duration is None:
            return None
        return max(0.1, min(duration, MAX_MOVE_DURATION))

    def _setup_handlers(self) -> None:
        """Set up MCP tool handlers."""

        @self._server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available mobility tools."""
            return [
                Tool(
                    name="move_forward",
                    description=(
                        "Move your body forward. Use this when you want to go "
                        "toward something you see in front of you. Optionally "
                        "specify duration in seconds."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {"duration": DURATION_SCHEMA},
                        "required": [],
                    },
                ),
                Tool(
                    name="move_backward",
                    description=(
                        "Move your body backward. Use this to back away from "
                        "something. Optionally specify duration in seconds."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {"duration": DURATION_SCHEMA},
                        "required": [],
                    },
                ),
                Tool(
                    name="turn_left",
                    description=(
                        "Turn your body to the left. This rotates your entire "
                        "body (not just your head/camera). Optionally specify "
                        "duration in seconds."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {"duration": DURATION_SCHEMA},
                        "required": [],
                    },
                ),
                Tool(
                    name="turn_right",
                    description=(
                        "Turn your body to the right. This rotates your entire "
                        "body (not just your head/camera). Optionally specify "
                        "duration in seconds."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {"duration": DURATION_SCHEMA},
                        "required": [],
                    },
                ),
                Tool(
                    name="stop_moving",
                    description=(
                        "Stop all body movement immediately. Use this to halt "
                        "when you've reached where you want to be."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                ),
                Tool(
                    name="body_status",
                    description=(
                        "Check the status of your body (robot vacuum). Shows "
                        "battery level and current state."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                ),
            ]

        @self._server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Handle tool calls."""
            controller = self._ensure_controller()

            try:
                if name == "move_forward":
                    duration = self._clamp_duration(arguments.get("duration"))
                    result = await controller.move_forward(duration)

                elif name == "move_backward":
                    duration = self._clamp_duration(arguments.get("duration"))
                    result = await controller.move_backward(duration)

                elif name == "turn_left":
                    duration = self._clamp_duration(arguments.get("duration"))
                    result = await controller.turn_left(duration)

                elif name == "turn_right":
                    duration = self._clamp_duration(arguments.get("duration"))
                    result = await controller.turn_right(duration)

                elif name == "stop_moving":
                    result = await controller.stop()

                elif name == "body_status":
                    status = await controller.get_status()
                    dps = status.get("dps", {})
                    result = f"Device status: {dps}"

                else:
                    result = f"Unknown tool: {name}"

            except Exception as e:
                logger.error("Tool %s failed: %s", name, e)
                result = f"Error: {e}"

            return [TextContent(type="text", text=result)]

    async def run(self) -> None:
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            try:
                await self._server.run(
                    read_stream,
                    write_stream,
                    self._server.create_initialization_options(),
                )
            finally:
                if self._controller:
                    self._controller.disconnect()


def main():
    """Entry point."""
    server = MobilityMCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
