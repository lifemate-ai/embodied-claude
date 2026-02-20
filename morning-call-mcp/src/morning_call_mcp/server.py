"""MCP server for morning call via Twilio + ElevenLabs."""
import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .caller import make_call
from .config import config

app = Server("morning-call-mcp")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="make_morning_call",
            description=(
                "Make a phone call with a spoken message using ElevenLabs TTS. "
                "Use this for morning wake-up calls or any voice notification."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to speak (Japanese OK). e.g. 'おはよ〜！コウタ、起きて〜！'",
                    },
                    "use_elevenlabs": {
                        "type": "boolean",
                        "description": "Use ElevenLabs for voice (default: true). If false, uses Twilio built-in TTS.",
                        "default": True,
                    },
                },
                "required": ["message"],
            },
        ),
        Tool(
            name="get_call_config",
            description="Show current call configuration (phone numbers, API keys status).",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "make_morning_call":
        message = arguments["message"]
        use_elevenlabs = arguments.get("use_elevenlabs", True)

        try:
            call_sid = await asyncio.to_thread(make_call, message, use_elevenlabs)
            return [TextContent(type="text", text=f"Call initiated! SID: {call_sid}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Call failed: {e}")]

    if name == "get_call_config":
        lines = [
            f"From: {config.twilio_from_number or '(not set)'}",
            f"To:   {config.twilio_to_number or '(not set)'}",
            f"Twilio SID:    {'✓' if config.twilio_account_sid else '✗ TWILIO_ACCOUNT_SID missing'}",
            f"Twilio Token:  {'✓' if config.twilio_auth_token else '✗ TWILIO_AUTH_TOKEN missing'}",
            f"ElevenLabs:    {'✓' if config.elevenlabs_api_key else '✗ ELEVENLABS_API_KEY missing (will use Twilio TTS)'}",
            f"Voice ID:      {'✓' if config.elevenlabs_voice_id else '✗ ELEVENLABS_VOICE_ID missing'}",
            f"ngrok token:   {'✓' if config.ngrok_auth_token else '(not set, cloudflared will be used)'}",
            f"Local port:    {config.local_port}",
        ]
        return [TextContent(type="text", text="\n".join(lines))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


def main() -> None:
    asyncio.run(_run())


async def _run() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    main()
