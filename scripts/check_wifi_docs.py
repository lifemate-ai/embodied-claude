"""Validate wifi tool names in docs and autonomous script."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from mcp.types import ListToolsRequest

from wifi_cam_mcp.server import CameraMCPServer


def _extract_table_tool_names(readme_path: Path) -> set[str]:
    names: set[str] = set()
    pattern = re.compile(r"^\|\s*`([^`]+)`\s*\|")
    for line in readme_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        token = match.group(1).strip()
        if token:
            names.add(token)
    return names


def _extract_wifi_allowed_tools(script_path: Path) -> list[str]:
    text = script_path.read_text(encoding="utf-8")
    matches = re.findall(r"mcp__wifi-cam__([a-z_]+)", text)
    return matches


async def _get_wifi_tool_names() -> set[str]:
    server = CameraMCPServer()
    handler = server._server.request_handlers[ListToolsRequest]
    req = ListToolsRequest(method="tools/list", params={})
    res = await handler(req)
    return {tool.name for tool in res.root.tools}


async def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    wifi_readme = repo_root / "wifi-cam-mcp" / "README.md"
    autonomous_script = repo_root / "autonomous-action.sh"

    declared_tools = _extract_table_tool_names(wifi_readme)
    actual_tools = await _get_wifi_tool_names()
    unknown_tools = sorted(name for name in declared_tools if name not in actual_tools)
    if unknown_tools:
        raise SystemExit(f"Unknown tools in wifi README: {', '.join(unknown_tools)}")

    allowed_wifi_tools = _extract_wifi_allowed_tools(autonomous_script)
    unknown_allowed = sorted(name for name in allowed_wifi_tools if name not in actual_tools)
    if unknown_allowed:
        raise SystemExit(
            "Unknown wifi tools in autonomous-action.sh: " + ", ".join(unknown_allowed)
        )

    deprecated_aliases = {
        "camera_capture",
        "camera_pan_left",
        "camera_pan_right",
        "camera_tilt_up",
        "camera_tilt_down",
        "camera_look_around",
    }
    used_deprecated = sorted(name for name in allowed_wifi_tools if name in deprecated_aliases)
    if used_deprecated:
        raise SystemExit(
            "autonomous-action.sh must use canonical wifi tools, found deprecated aliases: "
            + ", ".join(used_deprecated)
        )

    print("OK: wifi docs and autonomous tool names are consistent.")


if __name__ == "__main__":
    asyncio.run(main())
