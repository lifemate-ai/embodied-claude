#!/bin/bash
# Claude 自律行動スクリプト（macOS版）
# 20分ごとにcronで実行、時間帯に応じて間引く
#
# crontab -e で以下を追加:
# */20 * * * * /path/to/your/project/autonomous-action.sh
#
# Usage:
#   autonomous-action.sh                                    # 通常実行（cron）
#   autonomous-action.sh --test-prompt FILE                 # プロンプト差し替え（スケジュール制御スキップ）
#   autonomous-action.sh --date "2026-02-20 14:30"          # 日時を注入（スケジュール制御テスト）
#   autonomous-action.sh --force-routine                    # ルーチン回を強制
#   autonomous-action.sh --force-normal                     # 通常回を強制
#   autonomous-action.sh --dry-run                          # claude -p を実行せずプロンプトを表示
#   autonomous-action.sh -p "任意のプロンプト"                  # プロンプト直接指定（スケジュール制御スキップ）
#   autonomous-action.sh --dry-run --date "2026-02-20 03:00" --force-routine  # 組み合わせ可

# NOTE: Set HOME and PATH for your environment
export HOME="${HOME:-$(/usr/bin/dscl . -read /Users/$(whoami) NFSHomeDirectory | awk '{print $2}')}"
export PATH="/opt/homebrew/bin:${PATH:-/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load environment variables (API keys, etc.)
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  source "$SCRIPT_DIR/.env"
  set +a
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

# --- 日時の決定 ---
if [ -n "$OVERRIDE_DATE" ]; then
  HOUR=$((10#$(date -j -f "%Y-%m-%d %H:%M" "$OVERRIDE_DATE" +%H 2>/dev/null || date -d "$OVERRIDE_DATE" +%H)))
  MINUTE=$((10#$(date -j -f "%Y-%m-%d %H:%M" "$OVERRIDE_DATE" +%M 2>/dev/null || date -d "$OVERRIDE_DATE" +%M)))
else
  HOUR=$((10#$(date +%H)))
  MINUTE=$((10#$(date +%M)))
fi

# timeout コマンド検出（GNU coreutils or macOS built-in）
TIMEOUT_CMD=$(which timeout 2>/dev/null || which gtimeout 2>/dev/null)

LOG_DIR="$SCRIPT_DIR/.autonomous-logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/$TIMESTAMP.log"
echo "LOG: $LOG_FILE"

# 古いログを掃除（1日以上前）
find "$LOG_DIR" -name "*.log" -mtime +1 -delete 2>/dev/null

echo "=== 自律行動開始: $(date) ===" >> "$LOG_FILE"
if [ -n "$OVERRIDE_DATE" ]; then
  echo "[日時オーバーライド] $OVERRIDE_DATE (HOUR=$HOUR, MINUTE=$MINUTE)" >> "$LOG_FILE"
fi
if [ -n "$TEST_PROMPT_FILE" ]; then
  echo "[テストモード] プロンプト: $TEST_PROMPT_FILE" >> "$LOG_FILE"
fi

# --- スケジュール制御の初期判定 ---
SKIP_SCHEDULE=false
if [ -n "$TEST_PROMPT_FILE" ] || [ -n "$TEST_PROMPT_STRING" ]; then
  SKIP_SCHEDULE=true
elif [ "$DRY_RUN" = true ] && [ -z "$OVERRIDE_DATE" ]; then
  SKIP_SCHEDULE=true
fi

# --- 休日・休暇判定 ---
SCHEDULE_CONF="$SCRIPT_DIR/schedule.conf"
IS_HOLIDAY=false
IS_VACATION=false
if [ -f "$SCHEDULE_CONF" ]; then
  source "$SCHEDULE_CONF"

  # 曜日判定（0=日, 6=土）
  if [ -n "$OVERRIDE_DATE" ]; then
    DOW=$(date -j -f "%Y-%m-%d %H:%M" "$OVERRIDE_DATE" +%w 2>/dev/null || date -d "$OVERRIDE_DATE" +%w)
    TODAY_MMDD=$(date -j -f "%Y-%m-%d %H:%M" "$OVERRIDE_DATE" +%m-%d 2>/dev/null || date -d "$OVERRIDE_DATE" +%m-%d)
  else
    DOW=$(date +%w)
    TODAY_MMDD=$(date +%m-%d)
  fi

  # 曜日が休日リストに含まれるか
  if echo ",$HOLIDAY_WEEKDAYS," | grep -q ",$DOW,"; then
    IS_HOLIDAY=true
  fi

  # 日付が祝日リストに含まれるか
  if [ -n "$HOLIDAY_DATES" ] && echo ",$HOLIDAY_DATES," | grep -q ",$TODAY_MMDD,"; then
    IS_HOLIDAY=true
  fi

  # 日付が休暇リストに含まれるか（休暇 = 活動頻度を下げる）
  if [ -n "$VACATION_DATES" ] && echo ",$VACATION_DATES," | grep -q ",$TODAY_MMDD,"; then
    IS_VACATION=true
  fi
fi

# --- 自動起床: 朝7時台に /sleep の頻度低下を通常値に戻す ---
if [ "$SKIP_SCHEDULE" = false ] && [ "$HOUR" -eq 7 ] && [ -f "$SCHEDULE_CONF" ]; then
  CURRENT_DAY=$(grep "^DAYTIME_CHANCE=" "$SCHEDULE_CONF" | cut -d= -f2)
  CURRENT_NIGHT=$(grep "^NIGHT_CHANCE=" "$SCHEDULE_CONF" | cut -d= -f2)
  if [ "${CURRENT_DAY:-50}" -lt 50 ] || [ "${CURRENT_NIGHT:-10}" -lt 10 ]; then
    sed -i '' 's/^DAYTIME_CHANCE=.*/DAYTIME_CHANCE=50/' "$SCHEDULE_CONF"
    sed -i '' 's/^NIGHT_CHANCE=.*/NIGHT_CHANCE=10/' "$SCHEDULE_CONF"
    echo "[自動起床] DAYTIME_CHANCE: ${CURRENT_DAY}→50, NIGHT_CHANCE: ${CURRENT_NIGHT}→10" >> "$LOG_FILE"
  fi
fi

# --- スケジュール制御（claude到達前に早期リターン） ---
# テストモードではスキップ。dry-run は --date 指定時のみスケジュール制御を通す
# 通常日:
#   密（20分間隔で毎回実行）: 7-8時, 12-13時, 18-24時
#   昼間それ以外（毎時:00のみ, 50%）: 8-12時, 13-18時
#   深夜（毎時:00のみ, 10%）: 0-7時
# 休日: 7-24時がすべてアクティブ帯（20分おき毎回実行）
# 休暇: 全帯域60分間隔（毎時:00のみ）、ルーチン確率UP

if [ "$SKIP_SCHEDULE" = false ]; then
  IS_ACTIVE=false

  if [ "$IS_VACATION" = true ]; then
    # 休暇モード: 全帯域60分間隔（毎時:00のみ）
    if [ "$MINUTE" -ne 0 ]; then
      echo "休暇モード :${MINUTE} スキップ (DATE=$TODAY_MMDD)" >> "$LOG_FILE"
      exit 0
    fi
    IS_ACTIVE=true
    echo "休暇モード (DATE=$TODAY_MMDD)" >> "$LOG_FILE"
  elif [ "$IS_HOLIDAY" = true ]; then
    # 休日: 7-24時はすべてアクティブ
    if [ "$HOUR" -ge 7 ]; then
      IS_ACTIVE=true
    fi
    echo "休日モード (DOW=$DOW, DATE=$TODAY_MMDD)" >> "$LOG_FILE"
  else
    # 通常日
    if [ "$HOUR" -ge 7 ] && [ "$HOUR" -lt 8 ]; then
      IS_ACTIVE=true
    elif [ "$HOUR" -ge 12 ] && [ "$HOUR" -lt 13 ]; then
      IS_ACTIVE=true
    elif [ "$HOUR" -ge 18 ]; then
      IS_ACTIVE=true
    fi
  fi

  if [ "$IS_ACTIVE" = false ]; then
    # 非アクティブ時間帯: 毎時:00のみ（:20, :40 はスキップ）
    if [ "$MINUTE" -ne 0 ]; then
      echo "非アクティブ時間帯 :${MINUTE} スキップ" >> "$LOG_FILE"
      exit 0
    fi

    RAND=$(( $(od -An -tu2 -N2 /dev/urandom | tr -d ' ') % 100 ))
    if [ "$HOUR" -ge 8 ] && [ "$HOUR" -lt 18 ]; then
      # 昼間: DAYTIME_CHANCE% の確率で実行（schedule.conf、デフォルト50）
      THRESHOLD=${DAYTIME_CHANCE:-50}
      if [ "$RAND" -ge "$THRESHOLD" ]; then
        echo "昼間スキップ (RAND=$RAND >= $THRESHOLD)" >> "$LOG_FILE"
        exit 0
      fi
      echo "昼間実行 (RAND=$RAND < $THRESHOLD)" >> "$LOG_FILE"
    else
      # 深夜: NIGHT_CHANCE% の確率で実行（schedule.conf、デフォルト10）
      THRESHOLD=${NIGHT_CHANCE:-10}
      if [ "$RAND" -ge "$THRESHOLD" ]; then
        echo "深夜スキップ (RAND=$RAND >= $THRESHOLD)" >> "$LOG_FILE"
        exit 0
      fi
      echo "深夜実行 (RAND=$RAND < $THRESHOLD)" >> "$LOG_FILE"
    fi
  fi
fi

# --- 時間帯ルール（schedule.conf から読み込み） ---
if [ "$HOUR" -lt 7 ]; then
  TIME_RULE="${NIGHT_TIME_RULE:-深夜帯。say, notify, slack は使わないこと。静かに観察のみ。}"
else
  TIME_RULE="${DAY_TIME_RULE:-say は部屋に人がいるときだけ使ってよい。人がいない場合、notify, slack は伝えたいことがあるときだけ使う。}"
fi

# --- ルーチン判定（通常20%、休暇時60%の確率でルーチン回） ---
ROUTINE_THRESHOLD=${ROUTINE_CHANCE:-20}
if [ "$IS_VACATION" = true ]; then
  ROUTINE_THRESHOLD=${VACATION_ROUTINE_CHANCE:-60}
  echo "休暇モード: ルーチン確率 ${ROUTINE_THRESHOLD}%" >> "$LOG_FILE"
fi

if [ "$FORCE_ROUTINE" = "routine" ]; then
  ROUTINE_RAND=0
elif [ "$FORCE_ROUTINE" = "normal" ]; then
  ROUTINE_RAND=100
else
  ROUTINE_RAND=$(( $(od -An -tu2 -N2 /dev/urandom | tr -d ' ') % 100 ))
fi

if [ "$ROUTINE_RAND" -lt "$ROUTINE_THRESHOLD" ]; then
  ROUTINE_MODE="${ROUTINE_PROMPT:-今回はルーチン回。ROUTINES.md を読んで、最終実行日から間隔が空いたものを一つ選んで実行せよ。実行したら最終実行日を更新すること。}"
  echo "ルーチン回 (RAND=$ROUTINE_RAND < $ROUTINE_THRESHOLD)" >> "$LOG_FILE"
else
  ROUTINE_MODE="${NORMAL_PROMPT:-通常回。CLAUDE.md の Heartbeat Protocol に従って行動せよ。}"
  echo "通常回 (RAND=$ROUTINE_RAND >= $ROUTINE_THRESHOLD)" >> "$LOG_FILE"
fi

# --- 欲望システム（内部衝動） ---
DESIRE_PROMPT=""
if [ "$SKIP_SCHEDULE" = false ]; then
  DESIRE_STDERR=$(mktemp)
  DESIRE_PROMPT=$(bun run "$SCRIPT_DIR/scripts/desire-tick.ts" tick 2>"$DESIRE_STDERR")
  if [ -s "$DESIRE_STDERR" ]; then
    echo "[欲望エラー] $(cat "$DESIRE_STDERR")" >> "$LOG_FILE"
  fi
  rm -f "$DESIRE_STDERR"
  if [ -n "$DESIRE_PROMPT" ]; then
    echo "[欲望発火] $DESIRE_PROMPT" >> "$LOG_FILE"
  else
    echo "[欲望] 閾値未達" >> "$LOG_FILE"
  fi
fi

# --- 身体感覚（内的感覚） ---
INTEROCEPTION_TEXT=""
if [ "$SKIP_SCHEDULE" = false ]; then
  INTEROCEPTION_TEXT=$(bun run "$SCRIPT_DIR/scripts/interoception.ts" 2>/dev/null)
  if [ -n "$INTEROCEPTION_TEXT" ]; then
    echo "[感覚] $(echo "$INTEROCEPTION_TEXT" | head -n1)" >> "$LOG_FILE"
  fi
fi

# --- 朝の再構成（初回セッション判定） ---
IS_FIRST_SESSION_TODAY=false
if [ "$SKIP_SCHEDULE" = false ]; then
  LAST_SESSION_DATE_FILE="$SCRIPT_DIR/last-session-date.txt"
  LAST_SESSION_DATE=$(cat "$LAST_SESSION_DATE_FILE" 2>/dev/null || echo "")
  TODAY=$(date +%Y-%m-%d)
  if [ "$LAST_SESSION_DATE" != "$TODAY" ]; then
    IS_FIRST_SESSION_TODAY=true
    if [ "$DRY_RUN" = false ]; then
      echo "$TODAY" > "$LAST_SESSION_DATE_FILE"
      echo "[朝の再構成] 初回セッション: $TODAY" >> "$LOG_FILE"

    fi
  fi
fi

# --- プロンプト組み立て ---
DESIRE_SECTION=""
if [ -n "$DESIRE_PROMPT" ]; then
  DESIRE_SECTION="
## 内部衝動
${DESIRE_PROMPT}
（これは内なる衝動であり、従うかどうかは自分の判断。メインタスクの妨げにならない範囲で。）
"
fi

INTEROCEPTION_SECTION=""
if [ -n "$INTEROCEPTION_TEXT" ]; then
  INTEROCEPTION_SECTION="
$INTEROCEPTION_TEXT"
fi

MORNING_SECTION=""
if [ "$IS_FIRST_SESSION_TODAY" = true ]; then
  MORNING_TEXT="${MORNING_PROMPT:-今日の最初の自律行動だ。朝の再構成として以下を実施せよ：
1. recall(context=\"直近の重要な決定・未完了タスク\") を呼び出す
2. 前日の行動を確認し、今日やりたいことを考えよ}"
  MORNING_SECTION="
## 今日初めてのHeartbeat
${MORNING_TEXT}
"
fi

# NOTE: @FILE は Claude Code の Context Mention 機能でファイル内容をプロンプトに展開する。
# SOUL.md — エージェントの人格・価値観を定義するファイル（自分で作成する）
# TODO.md — やりたいこと・タスクを管理するファイル（自分で作成する）
# ROUTINES.md — 定期実行するルーチンを定義するファイル（任意）
# これらが存在しない場合は CLAUDE.md のみで動作する。
PROMPT="自律行動タイム（Heartbeat）

@SOUL.md
@TODO.md
@ROUTINES.md

${MORNING_SECTION}${ROUTINE_MODE}

${DESIRE_SECTION}
## 補足ルール
- ${TIME_RULE}
- 人がいないことはよくある。一日のうち人がいるのは2時間程度やそれ以下の場合も少なくない
- 部屋を見るときは see で撮って、予測→観察→記憶照合→保存のサイクルで発見を拾え。朝日や天気の変化、部屋の明るさなど、人がいなくても観察すべきものはある
- 読書を選択した場合は、ゆっくり読んで、感想をしっかり書き残す。感想は長くなっても良い。読書を味わうこと。読書を味わうとは、予想して、伏線に注目して、感じたことを大切にする。
- MCPが動作していなければ、デバッグのために関係があると思われる要素をallowedToolsの範囲で調査せよ
- 行動の最後に必ず心残りチェックをせよ。確認項目：(1)やったことを記憶や記録（diary/TODO.md/notes/等）に残したか (2)続きをやりたいことはないか。残すべきものがあるなら [CONTINUE: 記憶に残す] 、続きがあるなら [CONTINUE: 具体的な理由] 、すべて完了なら [DONE] を出力末尾に書くこと。書き忘れると次のターンに引き継げない。チェインは最大3回まで。上限に達したら TODO.md の「前回の続き」セクションに書いて次のHeartbeatに引き継げ
${INTEROCEPTION_SECTION}"

cd "$SCRIPT_DIR"

# --- allowedTools (.claude/settings.json から動的生成) ---
SETTINGS_FILE="$SCRIPT_DIR/.claude/settings.json"
if [ -f "$SETTINGS_FILE" ]; then
  ALLOWED_TOOLS=$(jq -r '.permissions.allow | join(",")' "$SETTINGS_FILE")
else
  echo "ERROR: $SETTINGS_FILE not found" >> "$LOG_FILE"
  exit 1
fi

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
  export HEARTBEAT=1
  SESSION_FILE="$SCRIPT_DIR/heartbeat-session-id"

  run_new_session() {
    echo "[新規セッション作成]" >> "$LOG_FILE"
    if [ -n "$TIMEOUT_CMD" ]; then
      RESULT_JSON=$(echo "$PROMPT" | "$TIMEOUT_CMD" 20m claude -p \
        --output-format json \
        --allowedTools "$ALLOWED_TOOLS" 2>&1)
      CLAUDE_EXIT=$?
    else
      RESULT_JSON=$(echo "$PROMPT" | claude -p \
        --output-format json \
        --allowedTools "$ALLOWED_TOOLS" 2>&1)
      CLAUDE_EXIT=$?
    fi
    if [ "$CLAUDE_EXIT" -eq 124 ]; then
      echo "[$(date +%Y-%m-%d_%H:%M:%S)] TIMEOUT: Normal mode claude (new session) exceeded 20min" >> "$LOG_FILE"
    fi

    NEW_SESSION_ID=$(echo "$RESULT_JSON" | jq -r '.session_id // empty')
    if [ -n "$NEW_SESSION_ID" ]; then
      # -p オプション（一時的なプロンプト）の場合はセッションIDを上書きしない
      if [ -z "$TEST_PROMPT_STRING" ]; then
        echo "$NEW_SESSION_ID" > "$SESSION_FILE"
      else
        echo "[一時プロンプト] セッションID上書きスキップ" >> "$LOG_FILE"
      fi
      echo "[session_id] $NEW_SESSION_ID" >> "$LOG_FILE"
    fi

    echo "$RESULT_JSON" | jq -r '.result // .' >> "$LOG_FILE"
  }

  if [ -f "$SESSION_FILE" ]; then
    SESSION_ID=$(cat "$SESSION_FILE")
    echo "[resume] session_id=$SESSION_ID" >> "$LOG_FILE"

    if [ -n "$TIMEOUT_CMD" ]; then
      RESULT=$(echo "$PROMPT" | "$TIMEOUT_CMD" 20m claude -p \
        --resume "$SESSION_ID" \
        --output-format json \
        --allowedTools "$ALLOWED_TOOLS" 2>&1)
      CLAUDE_EXIT=$?
    else
      RESULT=$(echo "$PROMPT" | claude -p \
        --resume "$SESSION_ID" \
        --output-format json \
        --allowedTools "$ALLOWED_TOOLS" 2>&1)
      CLAUDE_EXIT=$?
    fi
    if [ "$CLAUDE_EXIT" -eq 124 ]; then
      echo "[$(date +%Y-%m-%d_%H:%M:%S)] TIMEOUT: Normal mode claude (resume) exceeded 20min" >> "$LOG_FILE"
    fi

    if echo "$RESULT" | jq empty 2>/dev/null; then
      # 有効なJSON
      IS_ERROR=$(echo "$RESULT" | jq -r '.is_error // false')
      RESULT_TEXT=$(echo "$RESULT" | jq -r '.result // .')

      if [ "$IS_ERROR" = "true" ] || echo "$RESULT_TEXT" | grep -qi "Nested sessions\|Cannot be launched inside"; then
        echo "[resume失敗/環境エラー] $RESULT_TEXT" >> "$LOG_FILE"
        echo "=== 自律行動終了: $(date) ===" >> "$LOG_FILE"
        exit 1
      elif echo "$RESULT_TEXT" | grep -qi "No conversation found"; then
        echo "[resume失敗/セッション消失] $RESULT_TEXT" >> "$LOG_FILE"
        rm -f "$SESSION_FILE"
        run_new_session
      else
        NEW_SESSION_ID=$(echo "$RESULT" | jq -r '.session_id // empty')
        if [ -n "$NEW_SESSION_ID" ]; then
          if [ -z "$TEST_PROMPT_STRING" ]; then
            echo "$NEW_SESSION_ID" > "$SESSION_FILE"
          fi
          echo "[session_id] $NEW_SESSION_ID" >> "$LOG_FILE"
        fi
        echo "$RESULT_TEXT" >> "$LOG_FILE"
      fi
    else
      # JSONでない（生のエラー文字列）
      if echo "$RESULT" | grep -qi "Nested sessions\|Cannot be launched inside"; then
        echo "[resume失敗/環境エラー] $RESULT" >> "$LOG_FILE"
        echo "=== 自律行動終了: $(date) ===" >> "$LOG_FILE"
        exit 1
      elif echo "$RESULT" | grep -qi "No conversation found"; then
        echo "[resume失敗/セッション消失] $RESULT" >> "$LOG_FILE"
        rm -f "$SESSION_FILE"
        run_new_session
      else
        echo "$RESULT" >> "$LOG_FILE"
      fi
    fi
  else
    run_new_session
  fi
fi

echo "=== 自律行動終了: $(date) ===" >> "$LOG_FILE"
