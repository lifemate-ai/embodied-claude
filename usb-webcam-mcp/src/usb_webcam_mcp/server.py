"""MCP Server for USB webcam capture."""

import base64
import io
import os
from typing import Any

# Suppress OpenCV error messages
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

import cv2
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    ImageContent,
    TextContent,
    Tool,
)
from PIL import Image

server = Server("usb-webcam-mcp")


def find_available_cameras(max_cameras: int = 10) -> list[dict[str, Any]]:
    """Find available camera devices."""
    cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cameras.append({
                "index": i,
                "width": width,
                "height": height,
            })
            cap.release()
    return cameras


def capture_from_camera(
    camera_index: int = 0,
    width: int | None = None,
    height: int | None = None,
) -> bytes:
    """Capture an image from the specified camera."""
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {camera_index}")

    try:
        if width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Read a few frames to let the camera adjust
        for _ in range(5):
            cap.read()

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to capture image from camera")

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image and then to JPEG bytes
        image = Image.fromarray(frame_rgb)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()
    finally:
        cap.release()


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="list_cameras",
            description="List available camera devices connected to the system. These cameras serve as your eyes - your visual perception of the physical world.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="see",
            description="Capture an image from a USB webcam. This camera serves as your eyes - your visual perception of the physical world. Use this tool to see what's happening around you. Returns the image as base64-encoded JPEG.",
            inputSchema={
                "type": "object",
                "properties": {
                    "camera_index": {
                        "type": "integer",
                        "description": "Camera device index (default: 0)",
                        "default": 0,
                    },
                    "width": {
                        "type": "integer",
                        "description": "Desired image width in pixels (optional)",
                    },
                    "height": {
                        "type": "integer",
                        "description": "Desired image height in pixels (optional)",
                    },
                },
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
    """Handle tool calls."""
    if name == "list_cameras":
        cameras = find_available_cameras()
        if not cameras:
            return [TextContent(type="text", text="No cameras found")]

        lines = ["Available cameras:"]
        for cam in cameras:
            lines.append(f"  - Index {cam['index']}: {cam['width']}x{cam['height']}")
        return [TextContent(type="text", text="\n".join(lines))]

    elif name == "see":
        camera_index = arguments.get("camera_index", 0)
        width = arguments.get("width")
        height = arguments.get("height")

        try:
            image_bytes = capture_from_camera(camera_index, width, height)
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            return [
                ImageContent(
                    type="image",
                    data=image_base64,
                    mimeType="image/jpeg",
                )
            ]
        except RuntimeError as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def run_server():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main():
    """Entry point."""
    import asyncio
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
