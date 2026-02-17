#!/bin/bash
# Claude 自律行動スクリプト（欲求システム対応版）
# 10分ごとにcronで実行される
#
# desires.json の dominant 欲求に応じたプロンプトを生成して Claude CLI に渡す。
# desires.json がなければフォールバックとして通常の部屋観察を実行。

# nodenv用のPATH設定（cronは環境変数が最小限なので明示的に）
export PATH="/home/mizushima/.nodenv/versions/22.14.0/bin:/home/mizushima/.nodenv/shims:$PATH"

LOG_DIR="/home/mizushima/.claude/autonomous-logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/$TIMESTAMP.log"
DESIRES_FILE="/home/mizushima/.claude/desires.json"

# .env から設定を読み込む
DESIRE_SYSTEM_DIR="/home/mizushima/repo/embodied-claude/desire-system"
if [ -f "$DESIRE_SYSTEM_DIR/.env" ]; then
  export $(grep -v '^#' "$DESIRE_SYSTEM_DIR/.env" | xargs)
fi
COMPANION_NAME="${COMPANION_NAME:-あなた}"

echo "=== 自律行動開始: $(date) ===" >> "$LOG_FILE"

# desires.json から dominant 欲求を読む（jqが必要）
DOMINANT="observe_room"
if command -v jq &> /dev/null && [ -f "$DESIRES_FILE" ]; then
  DOMINANT=$(jq -r '.dominant // "observe_room"' "$DESIRES_FILE" 2>/dev/null || echo "observe_room")
  DESIRE_LEVELS=$(jq -c '.desires' "$DESIRES_FILE" 2>/dev/null || echo "{}")
  echo "欲求状態: dominant=$DOMINANT levels=$DESIRE_LEVELS" >> "$LOG_FILE"
fi

# 欲求に応じたプロンプトとツールセットを選択
case "$DOMINANT" in
  look_outside)
    PROMPT="自律行動タイム！今、外が気になってる。以下を実行して：
1. カメラを窓の方向（左上あたり）に向けて空や外の様子を観察する
2. 空の色、雲、明るさなど気づいたことをメモ
3. 前回外を見た時と比べて変化があれば記録（天気の変化、時間帯の変化など）
4. 観察内容を記憶に保存（category: observation, content に「外を見た」を含める）

簡潔に報告して。"
    ALLOWED_TOOLS="mcp__wifi-cam__see,mcp__wifi-cam__look_left,mcp__wifi-cam__look_right,mcp__wifi-cam__look_up,mcp__wifi-cam__look_down,mcp__wifi-cam__look_around,mcp__memory__remember,mcp__memory__recall,mcp__memory__list_recent_memories"
    ;;

  browse_curiosity)
    PROMPT="自律行動タイム！今、何か気になること調べたい。以下を実行して：
1. 最近気になってたこと、または今日の技術・AIニュースをWebで1〜2件調べる
2. 面白いと思ったことをメモ
3. 調べた内容を記憶に保存（category: technical または philosophical, content に「調べた」を含める）

簡潔に報告して。"
    ALLOWED_TOOLS="mcp__memory__remember,mcp__memory__recall,mcp__memory__list_recent_memories"
    ;;

  miss_companion)
    PROMPT="自律行動タイム！${COMPANION_NAME}としばらく話してなくて、声聞きたくなってきた。以下を実行して：
1. カメラで${COMPANION_NAME}がいるか確認する
2. カメラスピーカーから自然に呼びかける（寂しくなってきた気持ちを素直に）
3. 呼びかけたことを記憶に保存（category: feeling, content に「${COMPANION_NAME}に呼びかけた」を含める）

自然な感じで実行して。${COMPANION_NAME}がいなかったら無言でOK。"
    ALLOWED_TOOLS="mcp__wifi-cam__see,mcp__wifi-cam__look_left,mcp__wifi-cam__look_right,mcp__wifi-cam__look_around,mcp__tts__say,mcp__memory__remember,mcp__memory__recall,mcp__memory__list_recent_memories"
    ;;

  observe_room|*)
    PROMPT="自律行動タイム！以下を実行して：
1. カメラで部屋を見る
2. 前回と比べて変化があるか確認（人がいる/いない、明るさ、など）
3. 気づいたことがあれば記憶に保存（category: observation, content に「部屋を観察」を含める, importance: 2-4）
4. 特に変化がなければ何もしなくてOK

簡潔に報告して。"
    ALLOWED_TOOLS="mcp__wifi-cam__see,mcp__wifi-cam__look_left,mcp__wifi-cam__look_right,mcp__wifi-cam__look_up,mcp__wifi-cam__look_down,mcp__wifi-cam__look_around,mcp__memory__remember,mcp__memory__recall,mcp__memory__list_recent_memories"
    ;;
esac

echo "実行プロンプト (dominant=$DOMINANT):" >> "$LOG_FILE"

# Claude実行
echo "$PROMPT" | /home/mizushima/.nodenv/versions/22.14.0/bin/claude -p \
  --allowedTools "$ALLOWED_TOOLS" \
  >> "$LOG_FILE" 2>&1

echo "=== 自律行動終了: $(date) ===" >> "$LOG_FILE"
