"""Tests for legacy tool alias compatibility in wifi-cam-mcp."""

from __future__ import annotations

import pytest
from mcp.types import CallToolRequest, ImageContent, ListToolsRequest, TextContent

from wifi_cam_mcp.camera import AudioResult, CaptureResult, Direction, MoveResult
from wifi_cam_mcp.server import CameraMCPServer


class DummyCamera:
    """Minimal async camera stub for tool dispatch tests."""

    def __init__(self) -> None:
        self.last_pan_left_degrees: int | None = None

    async def capture_image(self) -> CaptureResult:
        return CaptureResult(
            image_base64="abc123",
            file_path=None,
            timestamp="2026-02-17T00:00:00Z",
            width=640,
            height=480,
        )

    async def pan_left(self, degrees: int) -> MoveResult:
        self.last_pan_left_degrees = degrees
        return MoveResult(
            direction=Direction.LEFT,
            degrees=degrees,
            success=True,
            message=f"left {degrees}",
        )

    async def pan_right(self, degrees: int) -> MoveResult:
        return MoveResult(
            direction=Direction.RIGHT,
            degrees=degrees,
            success=True,
            message=f"right {degrees}",
        )

    async def tilt_up(self, degrees: int) -> MoveResult:
        return MoveResult(
            direction=Direction.UP,
            degrees=degrees,
            success=True,
            message=f"up {degrees}",
        )

    async def tilt_down(self, degrees: int) -> MoveResult:
        return MoveResult(
            direction=Direction.DOWN,
            degrees=degrees,
            success=True,
            message=f"down {degrees}",
        )

    async def look_around(self) -> list[CaptureResult]:
        return [await self.capture_image(), await self.capture_image()]

    async def get_device_info(self) -> dict[str, str]:
        return {"model": "dummy"}

    async def get_presets(self) -> list[str]:
        return ["home"]

    async def go_to_preset(self, preset_id: str) -> MoveResult:
        return MoveResult(
            direction=Direction.LEFT,
            degrees=0,
            success=True,
            message=f"preset {preset_id}",
        )

    async def listen_audio(self, duration: int, transcribe: bool) -> AudioResult:
        transcript = "hello" if transcribe else None
        return AudioResult(
            audio_base64="audio",
            file_path="/tmp/audio.wav",
            timestamp="2026-02-17T00:00:00Z",
            duration=float(duration),
            transcript=transcript,
        )


async def _list_tool_names(server: CameraMCPServer) -> list[str]:
    handler = server._server.request_handlers[ListToolsRequest]
    req = ListToolsRequest(method="tools/list", params={})
    res = await handler(req)
    return [tool.name for tool in res.root.tools]


async def _call_tool(
    server: CameraMCPServer, name: str, arguments: dict[str, object] | None = None
) -> list[TextContent | ImageContent]:
    handler = server._server.request_handlers[CallToolRequest]
    req = CallToolRequest(method="tools/call", params={"name": name, "arguments": arguments or {}})
    res = await handler(req)
    return res.root.content


@pytest.mark.asyncio
async def test_list_tools_includes_canonical_and_legacy_aliases() -> None:
    server = CameraMCPServer()
    tool_names = await _list_tool_names(server)

    for name in (
        "see",
        "look_left",
        "look_right",
        "look_up",
        "look_down",
        "look_around",
    ):
        assert name in tool_names

    for alias in (
        "camera_capture",
        "camera_pan_left",
        "camera_pan_right",
        "camera_tilt_up",
        "camera_tilt_down",
        "camera_look_around",
    ):
        assert alias in tool_names


@pytest.mark.asyncio
async def test_legacy_capture_alias_returns_deprecation_message_and_image() -> None:
    server = CameraMCPServer()
    server._camera = DummyCamera()

    content = await _call_tool(server, "camera_capture")
    assert isinstance(content[0], TextContent)
    assert "Deprecated tool alias 'camera_capture'" in content[0].text
    assert any(isinstance(item, ImageContent) for item in content)
    assert any(
        isinstance(item, TextContent) and "Captured image at" in item.text for item in content[1:]
    )


@pytest.mark.asyncio
async def test_canonical_name_does_not_emit_deprecation_message() -> None:
    server = CameraMCPServer()
    server._camera = DummyCamera()

    content = await _call_tool(server, "see")
    assert all(
        not (isinstance(item, TextContent) and "Deprecated tool alias" in item.text)
        for item in content
    )


@pytest.mark.asyncio
async def test_legacy_pan_left_alias_forwards_arguments() -> None:
    server = CameraMCPServer()
    camera = DummyCamera()
    server._camera = camera

    content = await _call_tool(server, "camera_pan_left", {"degrees": 12})
    assert camera.last_pan_left_degrees == 12
    assert isinstance(content[0], TextContent)
    assert "Deprecated tool alias 'camera_pan_left'" in content[0].text
    assert isinstance(content[1], TextContent)
    assert content[1].text == "left 12"
