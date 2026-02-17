"""
Desire System MCP Server - ここねの自発的な欲求レベルを提供する。

desires.json（desire_updater.pyが定期更新）を読み込み、
現在の欲求状態をMCPツール経由で返す。
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from desire_updater import compute_desires, save_desires

# 欲求レベル読み込み元
DESIRES_PATH = Path(os.getenv("DESIRES_PATH", str(Path.home() / ".claude" / "desires.json")))

# 欲求の日本語ラベル
DESIRE_LABELS: dict[str, str] = {
    "look_outside": "外を見たい",
    "browse_curiosity": "何か調べたい",
    "miss_companion": "コウタに会いたい",
    "observe_room": "部屋を観察したい",
}

server = Server("desire-system")


def load_desires() -> dict[str, Any] | None:
    """desires.json を読み込む。存在しなければ None。"""
    if not DESIRES_PATH.exists():
        return None
    try:
        with open(DESIRES_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def format_desires(data: dict[str, Any]) -> str:
    """欲求データを読みやすい形式に整形する。"""
    lines = []
    dominant = data.get("dominant", "")
    desires = data.get("desires", {})
    updated_at = data.get("updated_at", "")

    # dominant欲求
    dominant_label = DESIRE_LABELS.get(dominant, dominant)
    lines.append(f"【最も強い欲求】{dominant_label} (level: {desires.get(dominant, 0):.3f})")
    lines.append("")

    # 全欲求レベル
    lines.append("【欲求レベル一覧】")
    for key, level in sorted(desires.items(), key=lambda x: -x[1]):
        label = DESIRE_LABELS.get(key, key)
        bar = "█" * int(level * 10) + "░" * (10 - int(level * 10))
        lines.append(f"  {label}: [{bar}] {level:.3f}")

    if updated_at:
        lines.append(f"\n更新: {updated_at}")

    return "\n".join(lines)


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="get_desires",
            description=(
                "Get Kokone's current desire levels. "
                "IMPORTANT: Act immediately on any desire with level >= 0.7. "
                "browse_curiosity -> WebSearch something interesting; "
                "look_outside -> use camera to look outside; "
                "miss_companion -> talk to コウタ proactively; "
                "observe_room -> use camera to observe room. "
                "After acting, call satisfy_desire to recheck levels. "
                "Keep acting until level drops below 0.5. "
                "Desires drive behavior - do not just report them."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="satisfy_desire",
            description=(
                "Recalculate desire levels after taking an action. "
                "Call this after every action to check if the desire is satisfied (level < 0.5). "
                "If still high, keep acting. "
                "Loop: act -> satisfy_desire -> still high? -> act again."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="boost_desire",
            description=(
                "Boost a desire level due to novelty/surprise (dopamine/prediction error). "
                "Call when you feel surprised or encounter unexpected info "
                "(large prediction error). "
                "e.g. コウタ says something unexpected -> boost browse_curiosity; "
                "camera shows something unusual -> boost observe_room. "
                "Simulates dopamine response to novelty."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "desire_name": {
                        "type": "string",
                        "description": (
                            "Name of the desire to boost: "
                            "browse_curiosity, look_outside, miss_companion, observe_room"
                        ),
                    },
                    "amount": {
                        "type": "number",
                        "description": "Amount to boost (0.1-0.5). 0.2=mild surprise, 0.5=strong.",
                    },
                },
                "required": ["desire_name", "amount"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    if name == "get_desires":
        data = load_desires()
        if data is None:
            return [TextContent(
                type="text",
                text=(
                    "desires.jsonが見つからへん。\n"
                    f"パス: {DESIRES_PATH}\n"
                    "desire_updater を先に実行: "
                    "uv run --directory desire-system desire-updater"
                ),
            )]
        return [TextContent(type="text", text=format_desires(data))]

    if name == "satisfy_desire":
        try:
            import chromadb
            chroma_path = os.getenv(
                "MEMORY_DB_PATH",
                str(Path.home() / ".claude" / "memories" / "chroma"),
            )
            collection_name = os.getenv("MEMORY_COLLECTION_NAME", "claude_memories")
            client = chromadb.PersistentClient(path=chroma_path)
            collection = client.get_or_create_collection(collection_name)
            state = compute_desires(collection)
            save_desires(state, DESIRES_PATH)
            data = state.to_dict()
            return [TextContent(type="text", text=format_desires(data))]
        except Exception as e:
            return [TextContent(type="text", text=f"欲求レベルの更新に失敗: {e}")]

    if name == "boost_desire":
        desire_name = arguments.get("desire_name", "")
        amount = float(arguments.get("amount", 0.2))
        amount = max(0.0, min(0.5, amount))

        data = load_desires()
        if data is None:
            return [TextContent(
                type="text",
                text="desires.jsonが見つからへん。先にdesire-updaterを実行して。",
            )]

        desires = data.get("desires", {})
        if desire_name not in desires:
            valid = list(desires.keys())
            return [TextContent(type="text", text=f"欲求名が不正: {desire_name}. 有効: {valid}")]

        desires[desire_name] = min(1.0, desires[desire_name] + amount)
        dominant = max(desires, key=lambda k: desires[k])
        data["desires"] = desires
        data["dominant"] = dominant

        DESIRES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DESIRES_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        label = DESIRE_LABELS.get(desire_name, desire_name)
        return [TextContent(
            type="text",
            text=f"[ドーパミン] {label} +{amount:.1f} → {desires[desire_name]:.3f}",
        )]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def run_server() -> None:
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    """Entry point."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
