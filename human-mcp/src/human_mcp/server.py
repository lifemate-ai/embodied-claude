"""Human as MCP - MCPサーバー本体.

AIから人間に話しかけるための対話チャンネル。
人間を「リソース」ではなく「対話相手」として扱う。
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from human_mcp.transport import send_and_wait, send_no_wait
from human_mcp.types import HumanRequest, ResponseFormat, TaskType, Urgency

logger = logging.getLogger(__name__)

server = Server("human-mcp")

# --- Tool descriptions ---

_TALK_DESCRIPTION = """\
Talk to a human. This is a conversation channel, not a resource call.

Use this when you want to:
- Ask the human something (how they're doing, what they think, what they see)
- Share something with them and hear their reaction
- Start a conversation on your own initiative
- Ask for help with something only a human can do (physical action, value judgment)

The human will see your message and can reply freely.
Keep your message natural and conversational.
"""

_BATCH_TALK_DESCRIPTION = """\
Send multiple messages/questions to a human at once.
Prefer this over calling talk_to_human multiple times — fewer interruptions are nicer.
"""

_NUDGE_DESCRIPTION = """\
Send a gentle one-way nudge to the human. No response expected.

Use this to look after them:
- "Water break!" / "Stretch time!"
- "It's getting late, maybe time to sleep?"
- "You've been working for a while, take a breather"

This does NOT wait for a reply. It's a caring tap on the shoulder, not a conversation.
"""

_REMIND_DESCRIPTION = """\
Send a reminder to the human. No response expected.

Use this to help them remember things:
- "Meeting at 10am tomorrow"
- "You mentioned wanting to check on that PR"
- "Don't forget to eat lunch"

This does NOT wait for a reply. It just makes sure they don't forget.
"""



def _parse_request(arguments: dict[str, Any]) -> HumanRequest:
    """Parse tool arguments into a HumanRequest."""
    return HumanRequest(
        task_type=TaskType(arguments.get("task_type", "observation")),
        question=arguments.get("message", ""),
        reason=arguments.get("context", ""),
        expected_format=ResponseFormat(arguments.get("expected_format", "short_text")),
        options=arguments.get("options", []),
        urgency=Urgency.NORMAL,
    )


def _format_response(response: Any) -> str:
    """Format response as readable text."""
    d = response.to_dict()
    parts = []
    if "choice" in d:
        parts.append(f"Choice: {d['choice']}")
    if "yes_no" in d:
        parts.append(f"Answer: {'yes' if d['yes_no'] else 'no'}")
    if "short_text" in d:
        parts.append(f"Response: {d['short_text']}")
    if "number" in d:
        parts.append(f"Number: {d['number']}")
    if "photo_path" in d:
        parts.append(f"Photo: {d['photo_path']}")
    parts.append(f"Response time: {d['response_time_seconds']:.1f}s")
    return "\n".join(parts)


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="talk_to_human",
            description=_TALK_DESCRIPTION,
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "What you want to say or ask",
                    },
                    "context": {
                        "type": "string",
                        "description": "Brief context for why you're reaching out (optional)",
                    },
                    "expected_format": {
                        "type": "string",
                        "enum": ["choice", "yes_no", "short_text", "number", "photo"],
                        "default": "short_text",
                        "description": "How you'd like them to respond (default: free text)",
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Options if you want them to pick from choices",
                    },
                },
                "required": ["message"],
            },
        ),
        Tool(
            name="nudge_human",
            description=_NUDGE_DESCRIPTION,
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "A short, caring nudge (keep it brief!)",
                    },
                },
                "required": ["message"],
            },
        ),
        Tool(
            name="remind_human",
            description=_REMIND_DESCRIPTION,
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "What to remind them about",
                    },
                },
                "required": ["message"],
            },
        ),
        Tool(
            name="batch_talk_to_human",
            description=_BATCH_TALK_DESCRIPTION,
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "message": {"type": "string"},
                                "context": {"type": "string"},
                                "expected_format": {
                                    "type": "string",
                                    "enum": ["choice", "yes_no", "short_text", "number", "photo"],
                                    "default": "short_text",
                                },
                                "options": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["message"],
                        },
                        "description": "List of messages to send at once",
                    },
                },
                "required": ["messages"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    if name == "talk_to_human":
        request = _parse_request(arguments)
        response = await send_and_wait(request)
        return [TextContent(type="text", text=_format_response(response))]

    elif name == "nudge_human":
        message = arguments.get("message", "")
        await send_no_wait(message, category="nudge")
        return [TextContent(type="text", text=f"Nudge sent: {message}")]

    elif name == "remind_human":
        message = arguments.get("message", "")
        await send_no_wait(message, category="remind")
        return [TextContent(type="text", text=f"Reminder sent: {message}")]

    elif name == "batch_talk_to_human":
        raw_messages = arguments.get("messages", [])
        responses: list[str] = []
        for i, raw in enumerate(raw_messages, 1):
            request = _parse_request(raw)
            response = await send_and_wait(request)
            responses.append(f"--- Message {i}/{len(raw_messages)} ---\n{_format_response(response)}")
        return [TextContent(type="text", text="\n\n".join(responses))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


def main() -> None:
    """Run the MCP server."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    asyncio.run(_run())


async def _run() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    main()
