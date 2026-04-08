"""
Desire System MCP Server - ここねの自発的な欲求レベルを提供する。

v2: ホメオスタシス/アロスタシス対応
- 不快度（discomfort）表示
- セットポイントからの乖離で行動を駆動
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

from desire_updater import DESIRE_CONFIGS, compute_desires, save_desires

# 欲求レベル読み込み元
DESIRES_PATH = Path(os.getenv("DESIRES_PATH", str(Path.home() / ".claude" / "desires.json")))

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
    discomforts = data.get("discomforts", {})
    updated_at = data.get("updated_at", "")

    # dominant欲求（不快度ベース）
    dominant_label = DESIRE_CONFIGS[dominant].label if dominant in DESIRE_CONFIGS else dominant
    dominant_discomfort = discomforts.get(dominant, 0)
    lines.append(
        f"【最も不快な欲求】{dominant_label} "
        f"(level: {desires.get(dominant, 0):.3f}, discomfort: {dominant_discomfort:.3f})"
    )
    lines.append("")

    # 全欲求レベル（不快度でソート）
    lines.append("【欲求レベル一覧（不快度順）】")
    sorted_names = sorted(discomforts.keys(), key=lambda k: -discomforts[k])
    for key in sorted_names:
        level = desires.get(key, 0)
        discomfort = discomforts.get(key, 0)
        cfg = DESIRE_CONFIGS.get(key)
        label = cfg.label if cfg else key
        set_point = cfg.set_point if cfg else 0.0
        bar = "█" * int(discomfort * 10) + "░" * (10 - int(discomfort * 10))
        lines.append(
            f"  {label}: [{bar}] level={level:.3f} sp={set_point:.1f}"
            f" discomfort={discomfort:.3f}"
        )

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
                "Get Kokone's current desire levels and discomfort (homeostasis). "
                "Discomfort = distance from set point. "
                "IMPORTANT: Act immediately on any desire with discomfort >= 0.5. "
                "browse_curiosity -> WebSearch something interesting; "
                "look_outside -> use camera to look outside; "
                "miss_companion -> talk to コウタ proactively; "
                "observe_room -> use camera to observe room; "
                "identity_coherence -> recall memories to reinforce self-model; "
                "cognitive_load -> think about or discuss something interesting. "
                "After acting, call satisfy_desire to recheck levels. "
                "Keep acting until discomfort drops below 0.3. "
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
                "Recalculate desire levels and discomfort after taking an action. "
                "Call this after every action to check if homeostasis is restored "
                "(discomfort < 0.3). "
                "If still high, keep acting. "
                "Loop: act -> satisfy_desire -> still uncomfortable? -> act again."
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
                "Call when you feel surprised or encounter unexpected info. "
                "e.g. コウタ says something unexpected -> boost browse_curiosity; "
                "camera shows something unusual -> boost observe_room; "
                "identity feels unstable -> boost identity_coherence. "
                "Simulates dopamine response to novelty."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "desire_name": {
                        "type": "string",
                        "description": (
                            "Name of the desire to boost: "
                            + ", ".join(DESIRE_CONFIGS.keys())
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
        discomforts = data.get("discomforts", {})
        if desire_name not in desires and desire_name not in DESIRE_CONFIGS:
            valid = list(DESIRE_CONFIGS.keys())
            return [TextContent(type="text", text=f"欲求名が不正: {desire_name}. 有効: {valid}")]

        # レベルを上げる
        desires[desire_name] = min(1.0, desires.get(desire_name, 0) + amount)

        # 不快度を再計算
        from datetime import datetime, timezone

        from desire_updater import calculate_discomfort, get_allostatic_set_point
        now = datetime.now(timezone.utc)
        cfg = DESIRE_CONFIGS[desire_name]
        adjusted_sp = get_allostatic_set_point(desire_name, now)
        discomforts[desire_name] = round(calculate_discomfort(desires[desire_name], adjusted_sp), 3)

        dominant = max(discomforts, key=lambda k: discomforts[k])
        data["desires"] = desires
        data["discomforts"] = discomforts
        data["dominant"] = dominant

        DESIRES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DESIRES_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        label = cfg.label
        return [TextContent(
            type="text",
            text=(
                f"[ドーパミン] {label} +{amount:.1f} → "
                f"level={desires[desire_name]:.3f} discomfort={discomforts[desire_name]:.3f}"
            ),
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
