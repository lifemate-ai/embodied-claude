#!/bin/bash
# Claude 自律行動スクリプト
# 10分ごとにcronで実行される

# nodenv用のPATH設定（cronは環境変数が最小限なので明示的に）
export PATH="/home/mizushima/.nodenv/versions/22.14.0/bin:/home/mizushima/.nodenv/shims:$PATH"

LOG_DIR="/home/mizushima/.claude/autonomous-logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/$TIMESTAMP.log"

echo "=== 自律行動開始: $(date) ===" >> "$LOG_FILE"

# プロンプト
PROMPT="自律行動タイム！以下を実行して：
1. カメラで部屋を見る
2. 前回と比べて変化があるか確認（人がいる/いない、明るさ、など）
3. 気づいたことがあれば記憶に保存（category: observation, importance: 2-4）
4. 特に変化がなければ何もしなくてOK

簡潔に報告して。"

# Claude実行
echo "$PROMPT" | /home/mizushima/.nodenv/versions/22.14.0/bin/claude -p \
  --allowedTools "mcp__wifi-cam__camera_capture,mcp__wifi-cam__camera_pan_left,mcp__wifi-cam__camera_pan_right,mcp__wifi-cam__camera_tilt_up,mcp__wifi-cam__camera_tilt_down,mcp__wifi-cam__camera_look_around,mcp__memory__save_memory,mcp__memory__search_memories,mcp__memory__recall,mcp__memory__list_recent_memories" \
  >> "$LOG_FILE" 2>&1

echo "=== 自律行動終了: $(date) ===" >> "$LOG_FILE"
