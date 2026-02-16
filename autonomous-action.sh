#!/bin/bash
# Claude 自律行動スクリプト
# 10分ごとにcronで実行される

set -euo pipefail

# Optional nodenv PATH bootstrap for cron-like environments.
if [[ -d "$HOME/.nodenv/versions/22.14.0/bin" ]]; then
  export PATH="$HOME/.nodenv/versions/22.14.0/bin:$HOME/.nodenv/shims:$PATH"
fi

LOG_DIR="${AUTONOMOUS_LOG_DIR:-$HOME/.claude/autonomous-logs}"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/$TIMESTAMP.log"
CLAUDE_BIN="${CLAUDE_BIN:-$(command -v claude || true)}"

if [[ -z "$CLAUDE_BIN" ]]; then
  echo "ERROR: claude command not found. Set CLAUDE_BIN to the executable path." >> "$LOG_FILE"
  exit 1
fi

echo "=== 自律行動開始: $(date) ===" >> "$LOG_FILE"

PROMPT="自律行動タイム！以下を実行して：
1. カメラで部屋を見る
2. 前回と比べて変化があるか確認（人がいる/いない、明るさ、など）
3. 気づいたことがあれば記憶に保存（category: observation, importance: 2-4）
4. 特に変化がなければ何もしなくてOK

簡潔に報告して。"

ALLOWED_TOOLS="mcp__wifi-cam__see,mcp__wifi-cam__look_left,mcp__wifi-cam__look_right,mcp__wifi-cam__look_up,mcp__wifi-cam__look_down,mcp__wifi-cam__look_around,mcp__memory__save_memory,mcp__memory__search_memories,mcp__memory__recall,mcp__memory__list_recent_memories"

echo "$PROMPT" | "$CLAUDE_BIN" -p --allowedTools "$ALLOWED_TOOLS" >> "$LOG_FILE" 2>&1

echo "=== 自律行動終了: $(date) ===" >> "$LOG_FILE"
