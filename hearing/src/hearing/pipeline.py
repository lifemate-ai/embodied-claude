"""HearingPipeline - async subprocess launcher for MCP servers.

Spawns a single worker subprocess (python -m hearing) that handles
everything: ffmpeg capture, segment watching, and Whisper transcription.

The MCP server process has ZERO threads and ZERO child process management.
Only uses asyncio.create_subprocess_exec for non-blocking subprocess launch.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from .config import (
    PID_FILE,
    SEGMENT_DIR,
    HearingConfig,
)

logger = logging.getLogger(__name__)


def set_debug(enabled: bool) -> None:
    """Toggle DEBUG logging for hearing modules in this process."""
    level = logging.DEBUG if enabled else logging.INFO
    logging.getLogger("hearing.pipeline").setLevel(level)


class HearingPipeline:
    """Async hearing pipeline launcher.

    Usage::

        pipeline = HearingPipeline(source="local")
        await pipeline.start()   # non-blocking, spawns worker subprocess
        ...
        await pipeline.stop()    # graceful shutdown
    """

    def __init__(
        self,
        source: str = "local",
        config: HearingConfig | None = None,
    ):
        self._source = source
        self._config = config or HearingConfig.from_toml()
        self._worker_proc: asyncio.subprocess.Process | None = None
        self._running = False
        self._debug = False

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start the hearing worker subprocess (non-blocking)."""
        if self._running:
            logger.warning("Hearing pipeline is already running")
            return

        cmd = [
            sys.executable, "-m", "hearing",
            "--source", self._source,
            "--model", self._config.whisper_model,
            "--language", self._config.language,
            "--segment-seconds", str(self._config.segment_seconds),
        ]
        if self._debug:
            cmd.append("--debug")

        self._worker_proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        # Start background task to drain stderr (non-blocking)
        asyncio.get_event_loop().create_task(
            self._drain_worker_stderr()
        )

        self._running = True
        logger.info(
            "Hearing worker started (pid=%d, source=%s, model=%s, lang=%s, seg=%ds)",
            self._worker_proc.pid,
            self._source,
            self._config.whisper_model,
            self._config.language,
            self._config.segment_seconds,
        )

    async def _drain_worker_stderr(self) -> None:
        """Drain worker stderr as async task."""
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

    async def stop(self) -> None:
        """Stop the hearing worker subprocess gracefully."""
        if not self._running:
            return

        logger.info("Hearing pipeline stopping...")

        if self._worker_proc and self._worker_proc.returncode is None:
            try:
                # Send SIGTERM to the worker's process group
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
                try:
                    await asyncio.wait_for(
                        self._worker_proc.wait(), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    pass

        PID_FILE.unlink(missing_ok=True)
        SEGMENT_DIR.joinpath(".worker_ready").unlink(missing_ok=True)
        self._running = False
        logger.info("Hearing pipeline stopped")
