#!/bin/bash
# Claude 自律行動スクリプト（macOS / Linux 対応）
# 20分ごとにcronで実行、時間帯に応じて間引く
#
# セットアップ:
# 1. このファイルをコピー: cp autonomous-action.sample.sh autonomous-action.sh
# 2. autonomous-action.sh を編集（以下の「環境設定」セクション）
# 3. 実行権限を付与: chmod +x autonomous-action.sh
# 4. crontab -e で以下を追加:
#    */20 * * * * /path/to/your/autonomous-action.sh
#
# Usage:
#   autonomous-action.sh                                    # 通常実行（cron）
#   autonomous-action.sh --test-prompt FILE                 # プロンプト差し替え（スケジュール制御スキップ）
#   autonomous-action.sh --date "2026-02-20 14:30"          # 日時を注入（スケジュール制御テスト）
#   autonomous-action.sh --force-routine                    # ルーチン回を強制
#   autonomous-action.sh --force-normal                     # 通常回を強制
#   autonomous-action.sh --dry-run                          # claude -p を実行せずプロンプトを表示
#   autonomous-action.sh -p "任意のプロンプト"              # プロンプト直接指定（スケジュール制御スキップ）
#   autonomous-action.sh --dry-run --date "2026-02-20 03:00" --force-routine  # 組み合わせ可

# ============================================================================
# 環境設定（必須：ユーザー環境に合わせて編集）
# ============================================================================
# このファイル (autonomous-action.sample.sh) はサンプルです。
# 以下の手順で設定してください:
#   1. cp autonomous-action.sample.sh autonomous-action.sh
#   2. autonomous-action.sh の ★ マーク箇所を編集
#   3. autonomous-action.sh は .gitignore に追加されています（コミット不要）
# ============================================================================

# ★ ホームディレクトリ（crontab は $HOME すら持たないため明示的に設定）
# 例: /Users/yourname (macOS) または /home/yourname (Linux)
# 確認方法: ターミナルで "echo $HOME" を実行
export HOME="/Users/yourname"

# ★ PATH 設定（crontab は $PATH も最小限のため、必要なコマンドのパスを追加）
# 必要なコマンド: claude, jq, date など
# 例 (macOS + asdf + homebrew):
#   export PATH="$HOME/.asdf/shims:/opt/homebrew/bin:$PATH"
# 例 (Linux + nodenv):
#   export PATH="$HOME/.nodenv/shims:/usr/local/bin:$PATH"
# 確認方法: "which claude" "which jq" でパスを確認
export PATH="$HOME/.asdf/shims:/opt/homebrew/bin:$PATH"

# ★ プロジェクトディレクトリ（SOUL.md, TODO.md, ROUTINES.md などが存在する場所）
# 例: /Users/yourname/workspace/yourproject
# ⚠️ 重要: test-autonomous.sh の L129 と設定を一致させる必要があります（必須）
PROJECT_DIR="$HOME/yourproject"

# ★ .env ファイルのパス（プロジェクトディレクトリ配下）
# .env には以下の環境変数が含まれる:
#   - ELEVENLABS_API_KEY: ElevenLabs TTS の API キー
#   - TAPO_USERNAME, TAPO_PASSWORD: Wi-Fi カメラの認証情報
#   - その他 MCP サーバーが必要とする環境変数
ENV_FILE="$PROJECT_DIR/.env"
set -a
source "$ENV_FILE"
set +a

# ★ ユーザー名・部屋名（時間帯ルールで使用）
# 例: "あなた" "コウタ" など
USER_NAME="あなた"
# 例: "あなたの部屋" "コウタの部屋" など
USER_ROOM="${USER_NAME}の部屋"

# ★ allowedTools で許可するディレクトリパス
# セキュリティ: 必要最小限のディレクトリのみ指定すること（.env ファイルなど機密情報も読める）
# メモ：シークレットとenvを別で監理するのはまだ実装されていない
# 通常は PROJECT_DIR と同じでOK
ALLOWED_DIR="$PROJECT_DIR"

# ログディレクトリ名（プロジェクトディレクトリ配下に作成される）
# ⚠️ 重要: test-autonomous.sh の L129 と設定を一致させる必要があります（必須）
LOG_DIR_NAME=".autonomous-logs"

# ログ保持期間（日数）
# この日数より古いログファイルを自動削除する
# 例: 1 = 1日以上前のログを削除, 7 = 1週間以上前のログを削除
LOG_RETENTION_DAYS=7

# ============================================================================
# 環境検出・日時処理
# ============================================================================

# OS検出（macOS or Linux）
if [[ "$OSTYPE" == "darwin"* ]]; then
  IS_MACOS=true
else
  IS_MACOS=false
fi

# --- 引数パース ---
TEST_PROMPT_FILE=""
TEST_PROMPT_STRING=""
OVERRIDE_DATE=""
FORCE_ROUTINE=""    # "", "routine", "normal"
DRY_RUN=false

while [ $# -gt 0 ]; do
  case "$1" in
    -p)
      TEST_PROMPT_STRING="$2"
      shift 2
      ;;
    --test-prompt)
      TEST_PROMPT_FILE="$2"
      shift 2
      ;;
    --date)
      OVERRIDE_DATE="$2"
      shift 2
      ;;
    --force-routine)
      FORCE_ROUTINE="routine"
      shift
      ;;
    --force-normal)
      FORCE_ROUTINE="normal"
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# --- 日時の取得（環境別に一度だけ実行） ---
if [ -n "$OVERRIDE_DATE" ]; then
  # テスト用の日時オーバーライド
  if [ "$IS_MACOS" = true ]; then
    # macOS: date -j -f "format" "date_string" +output
    CURRENT_DATE=$(date -j -f "%Y-%m-%d %H:%M" "$OVERRIDE_DATE" "+%Y-%m-%d %H:%M:%S" 2>/dev/null)
    HOUR=$((10#$(date -j -f "%Y-%m-%d %H:%M" "$OVERRIDE_DATE" +%H 2>/dev/null)))
    MINUTE=$((10#$(date -j -f "%Y-%m-%d %H:%M" "$OVERRIDE_DATE" +%M 2>/dev/null)))
    TIMESTAMP=$(date -j -f "%Y-%m-%d %H:%M" "$OVERRIDE_DATE" +%Y%m%d_%H%M%S 2>/dev/null)
  else
    # Linux: date -d "date_string" +output
    CURRENT_DATE=$(date -d "$OVERRIDE_DATE" "+%Y-%m-%d %H:%M:%S" 2>/dev/null)
    HOUR=$((10#$(date -d "$OVERRIDE_DATE" +%H 2>/dev/null)))
    MINUTE=$((10#$(date -d "$OVERRIDE_DATE" +%M 2>/dev/null)))
    TIMESTAMP=$(date -d "$OVERRIDE_DATE" +%Y%m%d_%H%M%S 2>/dev/null)
  fi
else
  # 現在時刻を取得
  CURRENT_DATE=$(date "+%Y-%m-%d %H:%M:%S")
  HOUR=$((10#$(date +%H)))
  MINUTE=$((10#$(date +%M)))
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
fi
# --- スケジュール制御（claude到達前に早期リターン） ---
# テストモードではスキップ。dry-run は --date 指定時のみスケジュール制御を通す
# 密（20分間隔で毎回実行）: 7-8時, 12-13時, 18-24時
# 昼間それ以外（毎時:00のみ, 30%）: 8-12時, 13-18時
# 深夜（毎時:00のみ, 10%）: 0-7時

SKIP_SCHEDULE=false
if [ -n "$TEST_PROMPT_FILE" ] || [ -n "$TEST_PROMPT_STRING" ]; then
  SKIP_SCHEDULE=true
elif [ "$DRY_RUN" = true ] && [ -z "$OVERRIDE_DATE" ]; then
  SKIP_SCHEDULE=true
fi

if [ "$SKIP_SCHEDULE" = false ]; then
  IS_ACTIVE=false
  if [ "$HOUR" -ge 7 ] && [ "$HOUR" -lt 8 ]; then
    IS_ACTIVE=true
  elif [ "$HOUR" -ge 12 ] && [ "$HOUR" -lt 13 ]; then
    IS_ACTIVE=true
  elif [ "$HOUR" -ge 18 ]; then
    IS_ACTIVE=true
  fi

  if [ "$IS_ACTIVE" = false ]; then
    # 非アクティブ時間帯: 毎時:00のみ（:20, :40 はスキップ）
    if [ "$MINUTE" -ne 0 ]; then
      echo "非アクティブ時間帯 :${MINUTE} スキップ" >> "$LOG_FILE"
      exit 0
    fi

    RAND=$(( $(od -An -tu2 -N2 /dev/urandom | tr -d ' ') % 100 ))
    if [ "$HOUR" -ge 8 ] && [ "$HOUR" -lt 18 ]; then
      # 昼間: 30% の確率で実行
      if [ "$RAND" -ge 30 ]; then
        echo "昼間スキップ (RAND=$RAND >= 30)" >> "$LOG_FILE"
        exit 0
      fi
      echo "昼間実行 (RAND=$RAND < 30)" >> "$LOG_FILE"
    else
      # 深夜: 10% の確率で実行
      if [ "$RAND" -ge 10 ]; then
        echo "深夜スキップ (RAND=$RAND >= 10)" >> "$LOG_FILE"
        exit 0
      fi
      echo "深夜実行 (RAND=$RAND < 10)" >> "$LOG_FILE"
    fi
  fi
fi

# --- 時間帯ルール ---
# 深夜帯: 静音モード（通知・発話禁止）
# 日中: 人がいる場合のみ発話可能、不在時は控えめに通知
if [ "$HOUR" -ge 24 ] || [ "$HOUR" -lt 7 ]; then
  TIME_RULE="現在は深夜帯。say, notify, slack は絶対に使わないこと。静かに観察のみ。"
else
  TIME_RULE="say は${USER_ROOM}の視界で、人がいるときだけ使ってよい。${USER_NAME}が${USER_ROOM}にいる場合はsayを積極的に使う。リビングにいる場合は slackを使う。部屋に人がいない場合、notify, slack は${USER_NAME}に伝えたいことがあるときだけ使う。"
fi

# --- ルーチン判定（20%の確率でルーチン回） ---
if [ "$FORCE_ROUTINE" = "routine" ]; then
  ROUTINE_RAND=0
elif [ "$FORCE_ROUTINE" = "normal" ]; then
  ROUTINE_RAND=100
else
  ROUTINE_RAND=$(( $(od -An -tu2 -N2 /dev/urandom | tr -d ' ') % 100 ))
fi

if [ "$ROUTINE_RAND" -lt 20 ]; then
  ROUTINE_MODE="今回はルーチン回。ROUTINES.md を読んで、最終実行日から間隔が空いたものを一つ選んで実行せよ。実行したら最終実行日を更新すること。"
  echo "ルーチン回 (RAND=$ROUTINE_RAND < 20)" >> "$LOG_FILE"
else
  ROUTINE_MODE="通常回。CLAUDE.md の Heartbeat Protocol に従って行動せよ。"
  echo "通常回 (RAND=$ROUTINE_RAND >= 20)" >> "$LOG_FILE"
fi

# --- プロンプト組み立て ---
# SOUL.md, TODO.md: プロジェクト内のコンテキストファイル（カレントディレクトリに存在する前提）
# ROUTINE_MODE: 20%の確率でルーチン実行、80%は通常のハートビート
# ★ プロンプト内容もカスタマイズ可能（補足ルールなど）
PROMPT="自律行動タイム(Heartbeat)

@SOUL.md
@TODO.md

${ROUTINE_MODE}

## 補足ルール
- ${TIME_RULE}
- 人がいないことはよくある。一日のうち人がいるのは2時間程度やそれ以下の場合も少なくない
- 読書を選択した場合は、ゆっくり読んで、感想をしっかり書き残す。感想は長くなっても良い。読書を味わうこと。読書を味わうとは、予想して、伏線に注目して、感じたことを大切にする。
- MCPが動作していなければ、デバッグのために関係があると思われる要素をallowedToolsの範囲で調査して
"

# プロジェクトディレクトリに移動
cd "$PROJECT_DIR" || {
  echo "Error: PROJECT_DIR not found: $PROJECT_DIR" >&2
  exit 1
}

# ログディレクトリをプロジェクト配下に作成
LOG_DIR="$PROJECT_DIR/$LOG_DIR_NAME"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$TIMESTAMP.log"

# 古いログを掃除（LOG_RETENTION_DAYS より古いログを削除）
find "$LOG_DIR" -name "*.log" -mtime "+$LOG_RETENTION_DAYS" -delete 2>/dev/null

echo "=== 自律行動開始: $CURRENT_DATE ===" >> "$LOG_FILE"
if [ -n "$OVERRIDE_DATE" ]; then
  echo "[日時オーバーライド] $OVERRIDE_DATE (HOUR=$HOUR, MINUTE=$MINUTE)" >> "$LOG_FILE"
fi
if [ -n "$TEST_PROMPT_FILE" ]; then
  echo "[テストモード] プロンプト: $TEST_PROMPT_FILE" >> "$LOG_FILE"
fi

# --- allowedTools ---
# Claude Code の --allowedTools で許可するツールのリスト
# セキュリティ: 必要最小限のディレクトリのみ指定すること（.env ファイルなど機密情報も読める）
ALLOWED_TOOLS=$(cat <<TOOLS
Read($ALLOWED_DIR/**),
Write($ALLOWED_DIR/**),
Edit($ALLOWED_DIR/**),
Glob($ALLOWED_DIR/**),
Skill(notify:*),
Skill(slack:*),
Skill(read:*),
Bash(bun run:*),
mcp__wifi-cam__see,
mcp__wifi-cam__look_left,
mcp__wifi-cam__look_right,
mcp__wifi-cam__look_up,
mcp__wifi-cam__look_down,
mcp__wifi-cam__look_around,
mcp__wifi-cam__camera_info,
mcp__wifi-cam__camera_presets,
mcp__wifi-cam__camera_go_to_preset,
mcp__wifi-cam__listen,
mcp__tts__say,
mcp__memory__remember,
mcp__memory__search_memories,
mcp__memory__recall,
mcp__memory__recall_divergent,
mcp__memory__list_recent_memories,
mcp__memory__get_memory_stats,
mcp__memory__recall_with_associations,
mcp__memory__get_association_diagnostics,
mcp__memory__consolidate_memories,
mcp__memory__get_memory_chain,
mcp__memory__create_episode,
mcp__memory__search_episodes,
mcp__memory__get_episode_memories,
mcp__memory__save_visual_memory,
mcp__memory__save_audio_memory,
mcp__memory__recall_by_camera_position,
mcp__memory__get_working_memory,
mcp__memory__refresh_working_memory,
mcp__memory__link_memories,
mcp__memory__get_causal_chain,
mcp__memory__tom,
mcp__system-temperature__get_current_time
TOOLS
)
# 改行を除去して1行にする
ALLOWED_TOOLS=$(echo "$ALLOWED_TOOLS" | tr -d '\n' | sed 's/, */,/g')

# テストモードならプロンプトを差し替え
if [ -n "$TEST_PROMPT_STRING" ]; then
  PROMPT="$TEST_PROMPT_STRING"
elif [ -n "$TEST_PROMPT_FILE" ]; then
  PROMPT=$(cat "$TEST_PROMPT_FILE")
fi

# --- 実行 ---
if [ "$DRY_RUN" = true ]; then
  echo "=== DRY RUN ===" >> "$LOG_FILE"
  echo "[HOUR=$HOUR MINUTE=$MINUTE]" >> "$LOG_FILE"
  echo "[ROUTINE_RAND=$ROUTINE_RAND]" >> "$LOG_FILE"
  echo "[TIME_RULE] $TIME_RULE" >> "$LOG_FILE"
  echo "[ROUTINE_MODE] $ROUTINE_MODE" >> "$LOG_FILE"
  echo "" >> "$LOG_FILE"
  echo "--- PROMPT ---" >> "$LOG_FILE"
  echo "$PROMPT" >> "$LOG_FILE"
  echo "" >> "$LOG_FILE"
  echo "--- ALLOWED_TOOLS ---" >> "$LOG_FILE"
  echo "$ALLOWED_TOOLS" | tr ',' '\n' >> "$LOG_FILE"
  # 標準出力にも出す
  cat "$LOG_FILE"
else
  # セッション継続機能: 前回の会話を引き継ぐことで文脈が保持される
  # SESSION_FILE: session_id を保存するファイル（プロジェクトディレクトリ配下）
  # 例: ~/yourproject/.heartbeat-session-id
  # claude --resume で前回セッションを継続、失敗時は新規セッション作成
  SESSION_FILE="$PWD/.heartbeat-session-id"

  run_new_session() {
    echo "[新規セッション作成]" >> "$LOG_FILE"
    RESULT_JSON=$(echo "$PROMPT" | claude -p \
      --output-format json \
      --allowedTools "$ALLOWED_TOOLS" 2>&1)

    # jq で session_id を抽出（jq が必要: brew install jq）
    NEW_SESSION_ID=$(echo "$RESULT_JSON" | jq -r '.session_id // empty' 2>/dev/null)
    if [ -n "$NEW_SESSION_ID" ]; then
      echo "$NEW_SESSION_ID" > "$SESSION_FILE"
      echo "[session_id] $NEW_SESSION_ID" >> "$LOG_FILE"
    fi

    echo "$RESULT_JSON" | jq -r '.result // .' 2>/dev/null >> "$LOG_FILE"
  }

  if [ -f "$SESSION_FILE" ]; then
    SESSION_ID=$(cat "$SESSION_FILE")
    echo "[resume] session_id=$SESSION_ID" >> "$LOG_FILE"

    RESULT=$(echo "$PROMPT" | claude -p \
      --resume "$SESSION_ID" \
      --allowedTools "$ALLOWED_TOOLS" 2>&1)

    # セッション継続失敗時（"No conversation found" など）は新規作成
    if echo "$RESULT" | grep -qi "No conversation found\|error"; then
      echo "[resume失敗] $RESULT" >> "$LOG_FILE"
      rm -f "$SESSION_FILE"
      run_new_session
    else
      echo "$RESULT" >> "$LOG_FILE"
    fi
  else
    run_new_session
  fi
fi

echo "=== 自律行動終了: $(date "+%Y-%m-%d %H:%M:%S") ===" >> "$LOG_FILE"
