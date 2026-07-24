from __future__ import annotations

import argparse
import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool


async def run(empty: bool) -> None:
    server = Server("doctor-fixture")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        if empty:
            return []
        return [
            Tool(
                name="ping",
                description="Return pong.",
                inputSchema={"type": "object", "properties": {}},
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, _arguments: dict) -> list[TextContent]:
        return [TextContent(type="text", text="pong")]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--empty", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.empty))


if __name__ == "__main__":
    main()
