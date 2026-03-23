"""Hearing MCP Server - always-on listening for embodied-claude.

Standalone MCP server that manages audio capture and transcription
in a separate worker subprocess. Completely isolated from other
MCP servers to avoid GIL/subprocess interference.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from ._behavior import get_behavior
from .config import BUFFER_FILE, PID_FILE, SEGMENT_DIR, HearingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HearingMCPServer:
    """MCP Server for always-on listening."""

    def __init__(self):
        self._server = Server("hearing-mcp")
        self._worker_proc: asyncio.subprocess.Process | None = None
        self._running = False
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        @self._server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="start_listening",
                    description=(
                        "Start continuous background listening. Audio is captured "
                        "and transcribed in the background, and results are "
                        "automatically injected into your context via the hearing "
                        "hook. Use this to passively monitor sounds in your "
                        "environment."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": (
                                    "Audio source: 'local' (PC microphone) or "
                                    "'camera' (RTSP camera mic). "
                                    "Default: from mcpBehavior.toml"
                                ),
                            },
                            "debug": {
                                "type": "boolean",
                                "default": False,
                                "description": (
                                    "Enable detailed debug logging for the "
                                    "hearing pipeline (inference time, filter "
                                    "results, segment processing)"
                                ),
                            },
                        },
                        "required": [],
                    },
                ),
                Tool(
                    name="stop_listening",
                    description="Stop continuous background listening.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                ),
            ]

        @self._server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            match name:
                case "start_listening":
                    return await self._handle_start(arguments)
                case "stop_listening":
                    return await self._handle_stop()
                case _:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]

    async def _handle_start(self, arguments: dict) -> list[TextContent]:
        if self._running:
            return [TextContent(
                type="text",
                text="Already listening. Use stop_listening to stop first.",
            )]

        config = HearingConfig.from_toml()

        # Resolve audio source
        source_arg = arguments.get("source")
        source = get_behavior("hearing", "source", "local")
        if source_arg:
            source = source_arg

        # If source is "camera", read RTSP URL from mcpBehavior.toml
        if source == "camera":
            rtsp_url = get_behavior("hearing", "rtsp_url", "")
            if rtsp_url:
                source = str(rtsp_url)
            else:
                return [TextContent(
                    type="text",
                    text="Error: source='camera' but [hearing] rtsp_url "
                    "not set in mcpBehavior.toml",
                )]

        debug = arguments.get("debug", False)

        # Spawn worker subprocess
        cmd = [
            sys.executable, "-m", "hearing",
            "--source", source,
            "--model", config.whisper_model,
            "--language", config.language,
            "--segment-seconds", str(config.segment_seconds),
            "--vad-energy-threshold", str(config.vad_energy_threshold),
        ]
        if debug:
            cmd.append("--debug")

        self._worker_proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        # Drain worker stderr as background task
        asyncio.get_event_loop().create_task(self._drain_worker_stderr())

        # Clear stale buffer and offset
        BUFFER_FILE.write_text("")
        Path("/tmp/hearing_stop_offset").unlink(missing_ok=True)
        Path("/tmp/hearing-stop-counter").unlink(missing_ok=True)

        self._running = True

        debug_msg = " (debug=ON)" if debug else ""
        return [TextContent(
            type="text",
            text=(
                f"Started continuous listening (source={source}){debug_msg}.\n"
                "Transcriptions will appear via [hearing] hook."
            ),
        )]

    async def _handle_stop(self) -> list[TextContent]:
        if not self._running:
            return [TextContent(type="text", text="Not currently listening.")]

        if self._worker_proc and self._worker_proc.returncode is None:
            try:
                os.killpg(self._worker_proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                await asyncio.wait_for(self._worker_proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                try:
                    os.killpg(self._worker_proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

        PID_FILE.unlink(missing_ok=True)
        SEGMENT_DIR.joinpath(".worker_ready").unlink(missing_ok=True)
        # Clear buffer and offset on stop
        BUFFER_FILE.write_text("")
        Path("/tmp/hearing_stop_offset").unlink(missing_ok=True)
        Path("/tmp/hearing-stop-counter").unlink(missing_ok=True)
        self._worker_proc = None
        self._running = False

        return [TextContent(type="text", text="Stopped continuous listening.")]

    async def _drain_worker_stderr(self) -> None:
        proc = self._worker_proc
        if proc is None or proc.stderr is None:
            return
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                text = line.decode(errors="replace").rstrip()
                if text:
                    logger.debug("worker: %s", text)
        except (asyncio.CancelledError, OSError):
            pass

    async def run(self) -> None:
        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(),
            )


def main() -> None:
    import setproctitle
    setproctitle.setproctitle("hearing-mcp")

    # jurigged hot-reload (watches src/ for live code changes)
    try:
        import jurigged
        jurigged.watch(poll=True)
        logging.getLogger(__name__).info("jurigged hot-reload enabled")
    except Exception:
        pass

    server = HearingMCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
