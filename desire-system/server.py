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

from backend import make_default_adapter
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
                "When discomfort >= 0.5: pick ONE bounded action that fits the "
                "current context (prefer private/non-interruptive during quiet "
                "hours), then call satisfy_desire. Do NOT loop more than once "
                "per tick. Desires bias attention; they are not a command queue. "
                "browse_curiosity -> WebSearch something interesting; "
                "look_outside -> use camera to look outside; "
                "miss_companion -> talk to コウタ proactively; "
                "observe_room -> use camera to observe room; "
                "identity_coherence -> recall memories to reinforce self-model; "
                "cognitive_load -> think about or discuss something interesting."
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
                "Record that a desire has been acted on and recompute levels. "
                "Writes a memory row with canonical satisfaction markers so "
                "the next tick reflects the action without waiting on other "
                "memory write paths. Prefer calling this once per action; avoid "
                "tight act->satisfy loops."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "desire_name": {
                        "type": "string",
                        "description": (
                            "Which desire was addressed. One of: "
                            + ", ".join(DESIRE_CONFIGS.keys())
                            + ". If omitted, only recomputes current levels."
                        ),
                    },
                    "action_summary": {
                        "type": "string",
                        "description": (
                            "Short phrase describing what was done (≤120 chars). "
                            "Stored as the satisfaction evidence."
                        ),
                    },
                    "outcome": {
                        "type": "string",
                        "enum": ["satisfied", "partially_satisfied", "not_satisfied"],
                        "description": "How the action landed. Defaults to satisfied.",
                    },
                    "person_id": {
                        "type": "string",
                        "description": "Optional person context for the action.",
                    },
                },
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
        desire_name = (arguments.get("desire_name") or "").strip()
        action_summary = (arguments.get("action_summary") or "").strip()
        outcome = arguments.get("outcome", "satisfied")
        person_id = arguments.get("person_id")

        try:
            from datetime import datetime, timezone

            adapter = make_default_adapter()
            # Record evidence first so the recomputation sees it.
            if desire_name and desire_name in DESIRE_CONFIGS and outcome == "satisfied":
                canonical_marker = DESIRE_CONFIGS[desire_name].keywords[0]
                summary_text = action_summary or canonical_marker
                adapter.record_satisfaction(
                    desire_name=desire_name,
                    summary=f"{canonical_marker}。{summary_text}",
                    ts=datetime.now(timezone.utc),
                    metadata={"outcome": outcome, "person_id": person_id},
                )
            state = compute_desires(adapter)
            save_desires(state, DESIRES_PATH)
            data = state.to_dict()
            header = ""
            if desire_name and desire_name in DESIRE_CONFIGS:
                label = DESIRE_CONFIGS[desire_name].label
                header = f"[{outcome}] {label} — {action_summary or '(no summary)'}\n\n"
            return [TextContent(type="text", text=header + format_desires(data))]
        except Exception as e:
            return [TextContent(type="text", text=f"欲求レベルの更新に失敗: {e}")]

    if name == "boost_desire":
        desire_name = arguments.get("desire_name", "")
        amount = float(arguments.get("amount", 0.2))
        amount = max(0.0, min(0.5, amount))

        # Validate against the config first so orphaned keys in desires.json
        # cannot slip through.
        if desire_name not in DESIRE_CONFIGS:
            valid = list(DESIRE_CONFIGS.keys())
            return [
                TextContent(
                    type="text", text=f"欲求名が不正: {desire_name!r}. 有効: {valid}"
                )
            ]

        data = load_desires()
        if data is None:
            return [TextContent(
                type="text",
                text="desires.jsonが見つからへん。先にdesire-updaterを実行して。",
            )]

        desires = data.get("desires", {})
        discomforts = data.get("discomforts", {})

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
        data["updated_at"] = now.isoformat()

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
