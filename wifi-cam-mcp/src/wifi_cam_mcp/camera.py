"""ONVIF Camera Controller - The eyes of AI."""

import asyncio
import base64
import io
import locale
import logging
import re
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from PIL import Image

from ._behavior import get_behavior
from .config import CameraConfig

logger = logging.getLogger(__name__)


class Direction(str, Enum):
    """Pan/Tilt directions."""

    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


@dataclass(frozen=True)
class CaptureResult:
    """Result of image capture."""

    image_base64: str
    file_path: str | None
    timestamp: str
    width: int
    height: int
    # Where the camera was pointing when this was taken, in the user's frame.
    # None when the device would not report it: a picture with no coordinates
    # is better than a picture with guessed ones, because a guess accumulates
    # into a map you then trust.
    pose: "CameraPosition | None" = None


@dataclass(frozen=True)
class AudioResult:
    """Result of audio capture."""

    audio_base64: str
    file_path: str | None
    timestamp: str
    duration: float
    transcript: str | None = None


@dataclass(frozen=True)
class MoveResult:
    """Result of camera movement."""

    direction: Direction
    degrees: int
    success: bool
    message: str


@dataclass
class CameraPosition:
    """Current camera PTZ position.

    When hardware position is available via ONVIF GetStatus, those values
    are used. Otherwise falls back to software tracking.
    """

    pan: float = 0.0  # normalized -1.0 to +1.0 (ONVIF) or degrees (software)
    tilt: float = 0.0  # normalized -1.0 to +1.0 (ONVIF) or degrees (software)


def to_user_frame(x: float, y: float, mount_mode: str) -> CameraPosition:
    """Put a raw ONVIF position into the frame of the person in the room.

    Positive pan is to their right, positive tilt is up. Both axes, not one.

    The device disagrees on both counts: on a Tapo C220 positive x is
    physically left (which is why `_move_impl` sends +x for Direction.LEFT)
    and positive y is physically down. Tilt was already being flipped here;
    pan was not, so the returned pose was half in the user's frame and half in
    the device's. Read as a coordinate, its sign said the opposite of where the
    camera was pointing -- and `body_contingency` maps look_left to pan -1.0
    and look_right to +1.0, so connecting the two would have inverted every
    camera agency verdict.

    Ceiling mounting turns the camera upside-down, mirroring both axes again.
    """

    pan = -x
    tilt = -y
    if mount_mode == "ceiling":
        pan = -pan
        tilt = -tilt
    return CameraPosition(pan=pan, tilt=tilt)


def describe_pose(pose: "CameraPosition | None") -> str:
    """Say where the camera is aimed, in degrees and in words.

    Degrees rather than the raw normalized value because -0.40 does not
    accumulate into a sense of the room, and 72 degrees left does. The words
    are there so a sign error is visible as a contradiction rather than hiding
    in a minus.

    Said as an offset from centre, because that is what it is: the zero of the
    normalized range is the middle of the camera's travel, not anything about
    the room. Reading it as "the right-hand side of the room" would put every
    remembered heading in a frame the camera does not have.
    """

    if pose is None:
        return "aim unknown (the camera did not report its position)"

    pan_deg = pose.pan * PAN_RANGE_DEGREES
    tilt_deg = pose.tilt * TILT_RANGE_DEGREES

    parts = []
    if abs(pan_deg) >= 3:
        parts.append(f"{abs(pan_deg):.0f} deg {'right' if pan_deg > 0 else 'left'}")
    if abs(tilt_deg) >= 3:
        parts.append(f"{abs(tilt_deg):.0f} deg {'up' if tilt_deg > 0 else 'down'}")
    if not parts:
        return "aimed at centre"
    return "aimed " + " and ".join(parts) + " from centre"


# ---------------------------------------------------------------------------
# Degree <-> ONVIF normalized conversion helpers
# ---------------------------------------------------------------------------
# Tapo PTZ cameras typically report pan in [-1.0, 1.0] mapping to [-180, 180]
# and tilt in [-1.0, 1.0] mapping to roughly [-45, 90] (varies by model).
#
# The pan figure is measured, not assumed: commanding a 30 degree turn on the
# camera in this room moved the reported position by 0.165, which is 29.7
# degrees at this scale, and the reading returned to its exact prior value on
# the way back. So a heading in degrees means the same thing as a movement in
# degrees, and the two can be reasoned about together.
#
# The tilt figure is still the conservative default -- the same check has not
# been run on that axis, so tilt degrees may be off by a scale factor.

PAN_RANGE_DEGREES = 180.0
TILT_RANGE_DEGREES = 90.0


def _degrees_to_normalized_pan(degrees: float) -> float:
    """Convert degrees to ONVIF normalized pan value."""
    return max(-1.0, min(1.0, degrees / PAN_RANGE_DEGREES))


def _degrees_to_normalized_tilt(degrees: float) -> float:
    """Convert degrees to ONVIF normalized tilt value."""
    return max(-1.0, min(1.0, degrees / TILT_RANGE_DEGREES))


# ---------------------------------------------------------------------------
# Maximum retries for ONVIF reconnection
# ---------------------------------------------------------------------------
MAX_RECONNECT_RETRIES = 2
RECONNECT_DELAY = 1.0  # seconds


# ---------------------------------------------------------------------------
# Whisper model cache
# ---------------------------------------------------------------------------
# Loading a Whisper model takes ~1.3s, so cache it at module level and reuse
# it across listen calls. Models are loaded lazily on first transcription.
# The lock guards against concurrent loads from asyncio.to_thread.

_whisper_models: dict[tuple[str, str], object] = {}
_whisper_models_lock = threading.Lock()


def _get_whisper_model(backend: str, model_size: str):
    """Load a Whisper model, or return the cached one (blocking).

    Args:
        backend: "openai-whisper" or "faster-whisper"
        model_size: Model size (e.g. "tiny", "base", "small")

    Returns:
        Loaded model instance (cached per backend/size pair)
    """
    key = (backend, model_size)
    with _whisper_models_lock:
        model = _whisper_models.get(key)
        if model is None:
            if backend == "faster-whisper":
                import ctranslate2
                from faster_whisper import WhisperModel

                # CTranslate2: float16 on GPU, int8 quantization on CPU
                if ctranslate2.get_cuda_device_count() > 0:
                    device, compute_type = "cuda", "float16"
                else:
                    device, compute_type = "cpu", "int8"
                model = WhisperModel(model_size, device=device, compute_type=compute_type)
            else:
                import whisper

                # load_model picks cuda automatically when available
                model = whisper.load_model(model_size)
            _whisper_models[key] = model
    return model


def _transcribe_with_model(backend: str, model, audio_path: str) -> str:
    """Transcribe an audio file with a loaded model (blocking).

    Both backends return the same shape of result: the transcript text.
    """
    if backend == "faster-whisper":
        segments, _info = model.transcribe(audio_path, language="ja")
        return "".join(segment.text for segment in segments).strip()
    result = model.transcribe(audio_path, language="ja")
    return result.get("text", "").strip()


async def _find_dshow_audio_device() -> str:
    """Find the first DirectShow audio device on Windows via ffmpeg.

    ffmpeg prints the device list to stderr and exits nonzero; that is
    expected behavior, so we parse stderr regardless of the return code.

    Returns:
        Name of the first audio device (e.g. "マイク (Realtek Audio)")
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-list_devices", "true",
        "-f", "dshow",
        "-i", "dummy",
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr_data = await asyncio.wait_for(process.communicate(), timeout=10.0)
    except asyncio.TimeoutError:
        process.kill()
        raise RuntimeError("DirectShow device listing timed out after 10s")

    # Device names are often non-ASCII on Japanese Windows. Modern ffmpeg
    # outputs UTF-8; fall back to the ANSI code page for older builds.
    try:
        listing = stderr_data.decode("utf-8")
    except UnicodeDecodeError:
        listing = stderr_data.decode(locale.getpreferredencoding(False), errors="replace")

    # ffmpeg lists devices as: [dshow @ ...] "Device Name" (audio)
    devices = re.findall(r'"([^"]+)"\s+\(audio\)', listing)
    if not devices:
        raise RuntimeError(
            "No DirectShow audio device found. Set the MIC_DEVICE environment "
            "variable to your microphone's device name "
            "(list devices with: ffmpeg -list_devices true -f dshow -i dummy)"
        )
    return devices[0]


class TapoCamera:
    """Controller for Tapo cameras via ONVIF protocol.

    Supports C210, C220, and other ONVIF-compatible Tapo PTZ cameras.
    """

    def __init__(
        self,
        config: CameraConfig,
        capture_dir: str = "/tmp/wifi-cam-mcp",
        mic_device: str | None = None,
        transcribe_backend: str = "openai-whisper",
        transcribe_model: str = "base",
    ):
        self._config = config
        self._capture_dir = Path(capture_dir)
        self._mic_device = mic_device
        self._transcribe_backend = transcribe_backend
        self._transcribe_model = transcribe_model
        self._lock = asyncio.Lock()

        # ONVIF objects (set on connect)
        self._cam = None  # ONVIFCamera instance
        self._media_service = None
        self._ptz_service = None
        self._devicemgmt_service = None
        self._profile_token: str | None = None

        # Software position tracking (fallback when GetStatus unavailable)
        self._sw_position = CameraPosition()
        self._connected = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish ONVIF connection to camera."""
        async with self._lock:
            if self._connected:
                return
            await self._do_connect()

    async def _do_connect(self) -> None:
        """Internal connect (must be called under lock)."""
        import os

        import onvif
        from onvif import ONVIFCamera

        logger.info(
            "Connecting to camera at %s:%d via ONVIF...",
            self._config.host,
            self._config.onvif_port,
        )

        # onvif-zeep-async has a bug in its default wsdl_dir calculation:
        # it uses dirname(dirname(__file__)) which resolves to
        # site-packages/wsdl/ instead of the correct site-packages/onvif/wsdl/.
        # We compute the correct path from the onvif package location.
        onvif_dir = os.path.dirname(onvif.__file__)
        wsdl_dir = os.path.join(onvif_dir, "wsdl")
        if not os.path.isdir(wsdl_dir):
            # Fallback: try the path one level up
            wsdl_dir = os.path.join(os.path.dirname(onvif_dir), "wsdl")

        self._cam = ONVIFCamera(
            self._config.host,
            self._config.onvif_port,
            self._config.username,
            self._config.password,
            wsdl_dir=wsdl_dir,
            adjust_time=True,
        )
        await self._cam.update_xaddrs()

        # Create services
        self._media_service = await self._cam.create_media_service()
        self._ptz_service = await self._cam.create_ptz_service()
        self._devicemgmt_service = await self._cam.create_devicemgmt_service()

        # Get first media profile
        profiles = await self._media_service.GetProfiles()
        if not profiles:
            raise RuntimeError("No media profiles found on camera")
        self._profile_token = profiles[0].token

        self._capture_dir.mkdir(parents=True, exist_ok=True)
        self._connected = True

        logger.info(
            "Connected to camera at %s (profile=%s, mount=%s)",
            self._config.host,
            self._profile_token,
            self._config.mount_mode,
        )

    async def disconnect(self) -> None:
        """Close ONVIF connection."""
        async with self._lock:
            if self._cam is not None:
                try:
                    await self._cam.close()
                except Exception:
                    pass
            self._cam = None
            self._media_service = None
            self._ptz_service = None
            self._devicemgmt_service = None
            self._profile_token = None
            self._connected = False
            logger.info("Disconnected from camera at %s", self._config.host)

    async def _ensure_connected(self) -> None:
        """Ensure camera is connected, attempt reconnect if needed."""
        if self._connected and self._cam is not None:
            return
        # Try to reconnect
        async with self._lock:
            if self._connected and self._cam is not None:
                return
            logger.warning("Camera not connected, attempting reconnect...")
            for attempt in range(1, MAX_RECONNECT_RETRIES + 1):
                try:
                    await self._do_connect()
                    logger.info("Reconnected on attempt %d", attempt)
                    return
                except Exception as e:
                    logger.error("Reconnect attempt %d failed: %s", attempt, e)
                    if attempt < MAX_RECONNECT_RETRIES:
                        await asyncio.sleep(RECONNECT_DELAY)
            raise RuntimeError(
                f"Camera not connected after {MAX_RECONNECT_RETRIES} attempts. "
                "Call connect() first."
            )

    async def _with_reconnect(self, operation, *args, **kwargs):
        """Execute an operation, retrying once on connection failure."""
        try:
            await self._ensure_connected()
            return await operation(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            is_connection_error = any(
                keyword in error_str
                for keyword in ["connection", "timeout", "refused", "reset", "broken"]
            )
            if is_connection_error:
                logger.warning("Connection error during operation, reconnecting: %s", e)
                self._connected = False
                self._cam = None
                await self._ensure_connected()
                return await operation(*args, **kwargs)
            raise

    # ------------------------------------------------------------------
    # Image capture
    # ------------------------------------------------------------------

    async def capture_image(self, save_to_file: bool = True) -> CaptureResult:
        """Capture a snapshot from the camera.

        First tries ONVIF snapshot (fast, ~300ms). Falls back to RTSP
        capture via ffmpeg if ONVIF snapshot is unavailable.

        Args:
            save_to_file: If True, save image to disk as well

        Returns:
            CaptureResult with base64 encoded image and metadata
        """
        return await self._with_reconnect(self._capture_image_impl, save_to_file)

    async def _capture_image_impl(self, save_to_file: bool) -> CaptureResult:
        """Internal capture implementation."""
        image_data = None

        # If a custom stream URL is set, prefer RTSP (higher resolution).
        # ONVIF snapshot often returns lower resolution (e.g. 640x480).
        if self._config.stream_url:
            try:
                image_data = await self._capture_via_rtsp()
            except Exception as e:
                logger.info("RTSP capture failed: %s, trying ONVIF snapshot", e)

        # Try ONVIF snapshot (default path, or fallback from RTSP)
        if image_data is None:
            onvif_error = None
            try:
                image_data = await self._try_onvif_snapshot()
            except Exception as e:
                onvif_error = str(e)

            # Fall back to RTSP if ONVIF snapshot also fails
            if image_data is None:
                logger.info(
                    "ONVIF snapshot unavailable (reason: %s), falling back to RTSP capture",
                    onvif_error or "empty response",
                )
                image_data = await self._capture_via_rtsp()

        # Process image
        image = Image.open(io.BytesIO(image_data))

        # In ceiling mount mode the image is upside-down, so rotate 180°.
        mount_mode = get_behavior("wifi-cam", "mount_mode", self._config.mount_mode)
        if mount_mode == "ceiling":
            image = image.rotate(180)

        # Resize if needed
        if image.width > self._config.max_width or image.height > self._config.max_height:
            image.thumbnail(
                (self._config.max_width, self._config.max_height),
                Image.LANCZOS,
            )

        width, height = image.size

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        image_base64 = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = None

        if save_to_file:
            file_path = str(self._capture_dir / f"capture_{timestamp}.jpg")
            with open(file_path, "wb") as f:
                f.write(buffer.getvalue())

        return CaptureResult(
            image_base64=image_base64,
            file_path=file_path,
            timestamp=timestamp,
            width=width,
            height=height,
            # Asked for here rather than left to the caller: an image and the
            # angle it was taken at are two facts that have to be stapled
            # together by whoever wants a map, and a caller under pressure
            # forgets. Arriving together, they cannot come apart.
            pose=await self.get_hw_position(),
        )

    async def _try_onvif_snapshot(self) -> bytes | None:
        """Try to get snapshot via ONVIF GetSnapshotUri."""
        try:
            image_bytes = await self._cam.get_snapshot(self._profile_token)
            if image_bytes and len(image_bytes) > 0:
                return image_bytes
        except Exception as e:
            logger.debug("ONVIF snapshot failed: %s", e)
        return None

    async def _capture_via_rtsp(self) -> bytes:
        """Capture a frame via RTSP using ffmpeg (fallback).

        Tries main stream (stream1, high quality) first, then falls back
        to sub stream (stream2) for low-bandwidth environments.
        """
        # Try main stream first (higher quality)
        try:
            return await self._capture_rtsp_stream(self._get_rtsp_url(sub_stream=False))
        except Exception as e:
            logger.info("Main stream (stream1) failed: %s, trying sub stream", e)
        return await self._capture_rtsp_stream(self._get_rtsp_url(sub_stream=True))

    async def _capture_rtsp_stream(self, rtsp_url: str) -> bytes:
        """Capture a single frame from an RTSP stream."""

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = [
                "ffmpeg",
                "-rtsp_transport",
                "tcp",
                "-i",
                rtsp_url,
                "-frames:v",
                "1",
                "-f",
                "image2",
                "-y",
                tmp_path,
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr_data = await asyncio.wait_for(
                    process.communicate(), timeout=10.0
                )
            except asyncio.TimeoutError:
                process.kill()
                raise RuntimeError("RTSP capture timed out after 10s")

            if process.returncode != 0:
                stderr_msg = stderr_data.decode(errors="replace").strip()[-500:]
                raise RuntimeError(
                    f"ffmpeg RTSP capture failed (rc={process.returncode}): {stderr_msg}"
                )

            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _get_rtsp_url(self, sub_stream: bool = False) -> str:
        """Get RTSP stream URL.

        Args:
            sub_stream: If True, use low-quality sub stream (stream2)
                        for low-bandwidth environments.
        """
        if self._config.stream_url:
            return self._config.stream_url
        stream = "stream2" if sub_stream else "stream1"
        return (
            f"rtsp://{self._config.username}:{self._config.password}"
            f"@{self._config.host}:554/{stream}"
        )

    # ------------------------------------------------------------------
    # PTZ control
    # ------------------------------------------------------------------

    async def move(self, direction: Direction, degrees: int = 30) -> MoveResult:
        """Move the camera in specified direction via ONVIF RelativeMove.

        Args:
            direction: Direction to move (left, right, up, down)
            degrees: Degrees to move (default: 30)

        Returns:
            MoveResult with operation status
        """
        return await self._with_reconnect(self._move_impl, direction, degrees)

    async def _move_impl(self, direction: Direction, degrees: int) -> MoveResult:
        """Internal move implementation."""
        degrees = max(1, min(degrees, 90))

        # Convert degrees to ONVIF normalized values
        pan_delta = 0.0
        tilt_delta = 0.0

        match direction:
            case Direction.LEFT:
                # Tapo C220: positive x = physical left
                pan_delta = _degrees_to_normalized_pan(degrees)
            case Direction.RIGHT:
                pan_delta = -_degrees_to_normalized_pan(degrees)
            case Direction.UP:
                # Tapo C220 ONVIF: y+ = physical DOWN, y- = physical UP
                # (confirmed: y=1.0 is the lower limit when desk-mounted)
                tilt_delta = -_degrees_to_normalized_tilt(degrees)
            case Direction.DOWN:
                tilt_delta = _degrees_to_normalized_tilt(degrees)

        # In ceiling mount mode the camera is upside-down:
        # - Tilt inverts (y=+1.0 becomes the upper limit)
        # - Pan mirrors (left/right swap)
        mount_mode = get_behavior("wifi-cam", "mount_mode", self._config.mount_mode)
        if mount_mode == "ceiling":
            pan_delta = -pan_delta
            tilt_delta = -tilt_delta

        try:
            ptz_mode = get_behavior("wifi-cam", "ptz_mode", self._config.ptz_mode)

            if ptz_mode == "relative":
                await self._ptz_relative_move(pan_delta, tilt_delta)
            elif ptz_mode == "continuous":
                await self._ptz_continuous_move(pan_delta, tilt_delta, degrees)
            else:
                # Auto: try RelativeMove, fall back to ContinuousMove
                try:
                    await self._ptz_relative_move(pan_delta, tilt_delta)
                except Exception:
                    await self._ptz_continuous_move(pan_delta, tilt_delta, degrees)

            # Update software tracking as well
            match direction:
                case Direction.LEFT:
                    self._sw_position.pan = max(-180.0, self._sw_position.pan - degrees)
                case Direction.RIGHT:
                    self._sw_position.pan = min(180.0, self._sw_position.pan + degrees)
                case Direction.UP:
                    self._sw_position.tilt = min(90.0, self._sw_position.tilt + degrees)
                case Direction.DOWN:
                    self._sw_position.tilt = max(-90.0, self._sw_position.tilt - degrees)

            # Give the motor time to settle
            await asyncio.sleep(0.5)

            return MoveResult(
                direction=direction,
                degrees=degrees,
                success=True,
                message=f"Moved {direction.value} by {degrees} degrees",
            )
        except Exception as e:
            return MoveResult(
                direction=direction,
                degrees=degrees,
                success=False,
                message=f"Failed to move: {e!s}",
            )

    async def _ptz_relative_move(self, pan_delta: float, tilt_delta: float) -> None:
        """Move camera using ONVIF RelativeMove (Tapo, etc.)."""
        await self._ptz_service.RelativeMove(
            {
                "ProfileToken": self._profile_token,
                "Translation": {
                    "PanTilt": {"x": pan_delta, "y": tilt_delta},
                },
            }
        )

    async def _ptz_continuous_move(
        self, pan_delta: float, tilt_delta: float, degrees: int
    ) -> None:
        """Move camera using ONVIF ContinuousMove + Stop (Imou, etc.)."""
        # Normalize velocity direction to -1..1 range
        mag = max(abs(pan_delta), abs(tilt_delta), 0.001)
        # Invert pan direction: ContinuousMove convention is opposite to
        # RelativeMove on some cameras (e.g. Imou vs Tapo)
        vx = -pan_delta / mag
        vy = tilt_delta / mag
        # Duration proportional to degrees (calibrated for Imou Ranger 2C)
        move_duration = max(0.3, degrees / 36.0)

        await self._ptz_service.ContinuousMove(
            {
                "ProfileToken": self._profile_token,
                "Velocity": {
                    "PanTilt": {"x": vx * 0.5, "y": vy * 0.5},
                },
            }
        )
        await asyncio.sleep(move_duration)
        await self._ptz_service.Stop(
            {"ProfileToken": self._profile_token, "PanTilt": True, "Zoom": True}
        )

    def get_position(self) -> CameraPosition:
        """Get current camera position (software-tracked).

        For hardware position, use get_hw_position() instead.
        """
        return CameraPosition(pan=self._sw_position.pan, tilt=self._sw_position.tilt)

    async def get_hw_position(self) -> CameraPosition | None:
        """Get actual hardware PTZ position via ONVIF GetStatus.

        Returns:
            CameraPosition with hardware-reported values, or None if unavailable.
            Both axes are in the user's frame: positive pan is to their right,
            positive tilt is up, whatever the mount mode. See `to_user_frame`.
        """
        try:
            await self._ensure_connected()
            status = await self._ptz_service.GetStatus({"ProfileToken": self._profile_token})
            if status.Position and status.Position.PanTilt:
                mount_mode = get_behavior("wifi-cam", "mount_mode", self._config.mount_mode)
                return to_user_frame(
                    status.Position.PanTilt.x,
                    status.Position.PanTilt.y,
                    mount_mode,
                )
        except Exception as e:
            logger.debug("Failed to get hardware position: %s", e)
        return None

    def reset_position_tracking(self) -> None:
        """Reset software position tracking to center (0, 0)."""
        self._sw_position = CameraPosition()

    async def pan_left(self, degrees: int = 30) -> MoveResult:
        """Pan camera to the left."""
        return await self.move(Direction.LEFT, degrees)

    async def pan_right(self, degrees: int = 30) -> MoveResult:
        """Pan camera to the right."""
        return await self.move(Direction.RIGHT, degrees)

    async def tilt_up(self, degrees: int = 20) -> MoveResult:
        """Tilt camera upward."""
        return await self.move(Direction.UP, degrees)

    async def tilt_down(self, degrees: int = 20) -> MoveResult:
        """Tilt camera downward."""
        return await self.move(Direction.DOWN, degrees)

    async def look_around(self) -> list[CaptureResult]:
        """Look around the room by capturing multiple angles.

        Captures: center, left, right, up-center positions.

        Returns:
            List of CaptureResults from different angles
        """
        captures: list[CaptureResult] = []

        center = await self.capture_image()
        captures.append(center)

        await self.pan_left(45)
        left = await self.capture_image()
        captures.append(left)

        await self.pan_right(90)
        right = await self.capture_image()
        captures.append(right)

        await self.pan_left(45)
        await self.tilt_up(20)
        up = await self.capture_image()
        captures.append(up)

        await self.tilt_down(20)

        return captures

    # ------------------------------------------------------------------
    # Device info & presets
    # ------------------------------------------------------------------

    async def get_device_info(self) -> dict:
        """Get camera device information via ONVIF."""
        await self._ensure_connected()
        try:
            info = await self._devicemgmt_service.GetDeviceInformation()
            # Convert zeep object to a plain dict
            import zeep.helpers

            return zeep.helpers.serialize_object(info, dict)
        except Exception as e:
            logger.error("Failed to get device info: %s", e)
            return {"error": str(e)}

    async def get_presets(self) -> list[dict]:
        """Get saved camera presets via ONVIF."""
        await self._ensure_connected()
        try:
            result = await self._ptz_service.GetPresets({"ProfileToken": self._profile_token})
            return [
                {"token": p.token, "name": getattr(p, "Name", None) or p.token}
                for p in (result or [])
            ]
        except Exception as e:
            logger.error("Failed to get presets: %s", e)
            return []

    async def go_to_preset(self, preset_id: str) -> MoveResult:
        """Move camera to a saved preset position via ONVIF."""
        await self._ensure_connected()
        try:
            await self._ptz_service.GotoPreset(
                {
                    "ProfileToken": self._profile_token,
                    "PresetToken": preset_id,
                }
            )
            await asyncio.sleep(1)
            return MoveResult(
                direction=Direction.LEFT,
                degrees=0,
                success=True,
                message=f"Moved to preset {preset_id}",
            )
        except Exception as e:
            return MoveResult(
                direction=Direction.LEFT,
                degrees=0,
                success=False,
                message=f"Failed to go to preset: {e!s}",
            )

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    async def listen_audio(
        self, duration: float = 5.0, transcribe: bool = False, mic_source: str = "camera"
    ) -> AudioResult:
        """Record audio from the camera's microphone or local PC microphone.

        Args:
            duration: Duration in seconds to record (default: 5.0)
            transcribe: If True, transcribe audio using Whisper (default: False)
            mic_source: Audio source - "camera" (RTSP) or "local" (PC microphone)

        Returns:
            AudioResult with base64 encoded audio and optional transcript
        """
        import platform

        if mic_source != "local":
            await self._ensure_connected()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = str(self._capture_dir / f"audio_{timestamp}.wav")
        self._capture_dir.mkdir(parents=True, exist_ok=True)

        try:
            if mic_source == "local":
                system = platform.system()
                if system == "Darwin":
                    cmd = [
                        "ffmpeg",
                        "-f", "avfoundation",
                        "-i", ":0",
                        "-ar", "16000",
                        "-ac", "1",
                        "-t", str(duration),
                        "-y", file_path,
                    ]
                elif system == "Linux":
                    cmd = [
                        "ffmpeg",
                        "-f", "alsa",
                        "-i", "default",
                        "-ar", "16000",
                        "-ac", "1",
                        "-t", str(duration),
                        "-y", file_path,
                    ]
                elif system == "Windows":
                    # dshow requires the literal device name. Use MIC_DEVICE
                    # if set, otherwise pick the first audio device found.
                    device = self._mic_device or await _find_dshow_audio_device()
                    # Some devices (notably Bluetooth headsets) need a moment to
                    # start delivering samples, which truncates the beginning of
                    # the recording. A smaller buffer reduces the loss.
                    cmd = [
                        "ffmpeg",
                        "-f", "dshow",
                        "-audio_buffer_size", "50",
                        "-i", f"audio={device}",
                        "-ar", "16000",
                        "-ac", "1",
                        "-t", str(duration),
                        "-y", file_path,
                    ]
                else:
                    raise RuntimeError(f"Unsupported platform for local microphone: {system}")
            else:
                rtsp_url = self._get_rtsp_url()
                cmd = [
                    "ffmpeg",
                    "-rtsp_transport",
                    "tcp",
                    "-i",
                    rtsp_url,
                    "-vn",  # No video
                    "-acodec",
                    "pcm_s16le",  # PCM 16-bit
                    "-ar",
                    "16000",  # 16kHz sample rate (good for speech)
                    "-ac",
                    "1",  # Mono
                    "-t",
                    str(duration),
                    "-y",
                    file_path,
                ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(process.wait(), timeout=duration + 10.0)

            with open(file_path, "rb") as f:
                audio_data = f.read()

            audio_base64 = base64.standard_b64encode(audio_data).decode("utf-8")

            transcript = None
            if transcribe:
                transcript = await self._transcribe_audio(file_path)

            return AudioResult(
                audio_base64=audio_base64,
                file_path=file_path,
                timestamp=timestamp,
                duration=duration,
                transcript=transcript,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to record audio: {e!s}") from e

    async def _transcribe_audio(self, audio_path: str) -> str | None:
        """Transcribe audio file using Whisper.

        The backend (openai-whisper or faster-whisper) and model size are
        configured via TRANSCRIBE_BACKEND / TRANSCRIBE_MODEL.

        Args:
            audio_path: Path to the audio file

        Returns:
            Transcribed text or None if transcription fails
        """
        backend = self._transcribe_backend
        if backend == "faster-whisper":
            try:
                import faster_whisper  # noqa: F401
            except ImportError:
                return "[faster-whisper not installed. Run: pip install faster-whisper]"
        else:
            try:
                import whisper  # noqa: F401
            except ImportError:
                return "[Whisper not installed. Run: pip install openai-whisper]"

        try:
            model = await asyncio.to_thread(
                _get_whisper_model, backend, self._transcribe_model
            )
            return await asyncio.to_thread(
                _transcribe_with_model, backend, model, audio_path
            )
        except Exception as e:
            return f"[Transcription failed: {e!s}]"
