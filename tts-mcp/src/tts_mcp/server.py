"""MCP Server for text-to-speech."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .go2rtc import Go2RTCProcess

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from . import playback
from ._behavior import load_behavior
from .config import ServerConfig, TTSConfig
from .engines import TTSEngine
from .engines.elevenlabs import ElevenLabsEngine

logger = logging.getLogger(__name__)


class TTSMCP:
    """MCP server that speaks text using multiple TTS engines."""

    def __init__(self) -> None:
        self._server_config = ServerConfig.from_env()
        self._config = TTSConfig.from_env()
        self._engines: dict[str, TTSEngine] = {}
        self._server = Server(self._server_config.name)
        self._go2rtc: Go2RTCProcess | None = None
        self._init_engines()
        self._setup_handlers()

    def _init_engines(self) -> None:
        """Initialize available TTS engines."""
        if self._config.elevenlabs:
            el = self._config.elevenlabs
            self._engines["elevenlabs"] = ElevenLabsEngine(
                api_key=el.api_key,
                voice_id=el.voice_id,
                model_id=el.model_id,
                output_format=el.output_format,
            )

        if self._config.voicevox:
            from .engines.voicevox import VoicevoxEngine

            vv = self._config.voicevox
            self._engines["voicevox"] = VoicevoxEngine(
                url=vv.url,
                speaker=vv.speaker,
            )

        if self._config.atlascloud:
            from .engines.atlascloud import AtlasCloudEngine

            atlas = self._config.atlascloud
            self._engines["atlascloud"] = AtlasCloudEngine(
                api_key=atlas.api_key,
                model_id=atlas.model_id,
                voice_id=atlas.voice_id,
                output_format=atlas.output_format,
                base_url=atlas.base_url,
                poll_interval=atlas.poll_interval,
                timeout=atlas.timeout,
            )

        if not self._engines:
            logger.warning(
                "No TTS engine configured. Set ELEVENLABS_API_KEY, VOICEVOX_URL, "
                "or ATLASCLOUD_API_KEY."
            )

    def _get_engine(self, requested: str | None = None) -> TTSEngine:
        """Get the appropriate TTS engine."""
        # TOML > explicit request > env default
        behavior = load_behavior("tts")
        toml_engine = behavior.get("default_engine", "") or None
        effective_requested = requested or toml_engine
        name = self._config.resolve_engine(effective_requested)
        engine = self._engines.get(name)
        if engine is None:
            available = list(self._engines.keys())
            raise ValueError(
                f"Engine '{name}' not available. "
                f"Available: {available or 'none (check env vars)'}"
            )
        return engine

    def _setup_handlers(self) -> None:
        @self._server.list_tools()
        async def list_tools() -> list[Tool]:
            available_engines = list(self._engines.keys())
            engine_desc = ", ".join(available_engines) if available_engines else "none configured"
            return [
                Tool(
                    name="say",
                    description=(
                        f"Speak text out loud using TTS. "
                        f"Available engines: {engine_desc}. "
                        f"Use this when you want to say something aloud."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to speak",
                            },
                            "engine": {
                                "type": "string",
                                "description": (
                                    f"TTS engine to use ({engine_desc}). "
                                    "If omitted, uses default."
                                ),
                                "enum": available_engines
                                or ["elevenlabs", "voicevox", "atlascloud"],
                            },
                            "voice_id": {
                                "type": "string",
                                "description": (
                                    "Override voice ID (ElevenLabs or Atlas Cloud, optional)"
                                ),
                            },
                            "model_id": {
                                "type": "string",
                                "description": (
                                    "Override model ID (ElevenLabs or Atlas Cloud, optional)"
                                ),
                            },
                            "output_format": {
                                "type": "string",
                                "description": (
                                    "Override output format (ElevenLabs or Atlas Cloud, optional)"
                                ),
                            },
                            "voicevox_speaker": {
                                "type": "integer",
                                "description": "VOICEVOX speaker/style ID (optional)",
                            },
                            "speed_scale": {
                                "type": "number",
                                "description": "Speech speed (VOICEVOX only, 0.5-2.0, optional)",
                            },
                            "pitch_scale": {
                                "type": "number",
                                "description": "Pitch shift (VOICEVOX only, -0.15 to 0.15, optional)",
                            },
                            "play_audio": {
                                "type": "boolean",
                                "description": "Play audio on this machine (default: true)",
                                "default": True,
                            },
                            "speaker": {
                                "type": "string",
                                "description": (
                                    "Where to play: 'camera' (camera speaker only), "
                                    "'local' (PC only), 'both' (default if go2rtc configured)"
                                ),
                                "enum": ["camera", "local", "both"],
                            },
                        },
                        "required": ["text"],
                    },
                )
            ]

        @self._server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            if name != "say":
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

            text = (arguments.get("text") or "").strip()
            if not text:
                return [TextContent(type="text", text="Error: 'text' is required")]

            pb = self._config.playback
            behavior = load_behavior("tts")
            play_audio = arguments.get(
                "play_audio", behavior.get("play_audio", pb.play_audio),
            )
            speaker_target = arguments.get("speaker") or (
                "both" if pb.go2rtc_url else "local"
            )
            use_local = speaker_target in {"local", "both"}
            use_camera = speaker_target in {"camera", "both"} and pb.go2rtc_url

            try:
                engine = self._get_engine(arguments.get("engine"))
                engine_name = engine.engine_name

                # Build engine-specific kwargs
                kwargs: dict[str, Any] = {}
                if engine_name in {"elevenlabs", "atlascloud"}:
                    for key in ("voice_id", "model_id", "output_format"):
                        if arguments.get(key):
                            kwargs[key] = arguments[key]
                elif engine_name == "voicevox":
                    if arguments.get("voicevox_speaker") is not None:
                        kwargs["speaker"] = arguments["voicevox_speaker"]
                    if arguments.get("speed_scale") is not None:
                        kwargs["speed_scale"] = arguments["speed_scale"]
                    if arguments.get("pitch_scale") is not None:
                        kwargs["pitch_scale"] = arguments["pitch_scale"]

                # Try streaming for ElevenLabs
                playback_mode = (
                    behavior.get("playback", pb.playback) or "auto"
                ).strip().lower()
                use_streaming = (
                    play_audio
                    and use_local
                    and engine_name == "elevenlabs"
                    and playback_mode in {"auto", "stream"}
                    and playback.can_stream()
                )

                if use_streaming:
                    el_engine: ElevenLabsEngine = engine  # type: ignore[assignment]
                    sentences = el_engine.stream_sentences(text)
                    if len(sentences) > 1:
                        sentence_streams = [
                            (s, el_engine.stream(s, **kwargs)) for s in sentences
                        ]
                        audio_bytes, play_status = await asyncio.to_thread(
                            playback.stream_sentences_with_mpv,
                            sentence_streams,
                            pb.pulse_sink,
                            pb.pulse_server,
                        )
                    else:
                        audio_stream = await asyncio.to_thread(
                            el_engine.stream, text, **kwargs,
                        )
                        audio_bytes, play_status = await asyncio.to_thread(
                            playback.stream_with_mpv,
                            audio_stream,
                            pb.pulse_sink,
                            pb.pulse_server,
                        )
                    audio_format = (
                        kwargs.get("output_format", "mp3_44100_128").split("_", 1)[0]
                    )
                    file_path = playback.save_audio(audio_bytes, audio_format, pb.save_dir)
                else:
                    # Standard synthesis
                    audio_bytes, audio_format = await asyncio.to_thread(
                        engine.synthesize, text, **kwargs,
                    )
                    file_path = playback.save_audio(audio_bytes, audio_format, pb.save_dir)

                    play_status = "skipped"
                    if play_audio and use_local:
                        play_status = await asyncio.to_thread(
                            playback.play_audio,
                            audio_bytes,
                            file_path,
                            pb.playback,
                            pb.pulse_sink,
                            pb.pulse_server,
                        )

                camera_status = "not configured"
                if use_camera:
                    ok, cam_msg = await asyncio.to_thread(
                        playback.play_with_go2rtc,
                        file_path,
                        pb.go2rtc_url,
                        pb.go2rtc_stream,
                        pb.go2rtc_ffmpeg,
                    )
                    camera_status = cam_msg

                message = (
                    f"Spoken via {engine_name}\n"
                    f"File: {file_path}\n"
                    f"Speaker: {speaker_target}\n"
                    f"Playback: {play_status}\n"
                    f"Camera: {camera_status}"
                )
                return [TextContent(type="text", text=message)]
            except Exception as exc:  # noqa: BLE001
                return [TextContent(type="text", text=f"Error: {exc}")]

    async def _ensure_go2rtc(self) -> None:
        """Auto-download and start go2rtc if configured."""
        pb = self._config.playback
        if not pb.go2rtc_url or not pb.go2rtc_auto_start:
            return

        from .go2rtc import Go2RTCProcess, default_config_path, ensure_binary, generate_config

        try:
            bin_path = Path(pb.go2rtc_bin) if pb.go2rtc_bin else None
            bin_path = ensure_binary(bin_path)
        except Exception as exc:
            logger.warning("go2rtc binary not available: %s", exc)
            return

        if pb.go2rtc_config:
            config_path = Path(pb.go2rtc_config)
        elif pb.go2rtc_camera_host and pb.go2rtc_camera_password:
            config_path = generate_config(
                config_path=default_config_path(),
                stream_name=pb.go2rtc_stream,
                camera_host=pb.go2rtc_camera_host,
                username=pb.go2rtc_camera_username or "",
                password=pb.go2rtc_camera_password,
                cloud_password=pb.go2rtc_camera_cloud_password,
                ffmpeg_bin=pb.go2rtc_ffmpeg,
            )
        else:
            logger.warning("go2rtc: no config and no camera credentials, skipping auto-start")
            return

        try:
            self._go2rtc = Go2RTCProcess(bin_path, config_path, pb.go2rtc_url)
            await self._go2rtc.start()
        except Exception as exc:
            logger.warning("go2rtc failed to start: %s", exc)
            self._go2rtc = None

    async def run(self) -> None:
        try:
            await self._ensure_go2rtc()
            async with stdio_server() as (read_stream, write_stream):
                await self._server.run(
                    read_stream,
                    write_stream,
                    self._server.create_initialization_options(),
                )
        finally:
            if self._go2rtc:
                self._go2rtc.stop()


def main() -> None:
    try:
        import jurigged
        jurigged.watch(pattern="src/**/*.py", logger=None)
    except ImportError:
        pass
    asyncio.run(TTSMCP().run())


if __name__ == "__main__":
    main()
