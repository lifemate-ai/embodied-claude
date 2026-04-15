#!/bin/bash
# auto-social.sh - 自動社会的イベント注入
# UserPromptSubmitフックで毎ターン実行
# コウタの発話をsociality DBに直接INSERT

SOCIAL_DB="$HOME/.claude/sociality/social.db"

# DB存在チェック
if [ ! -f "$SOCIAL_DB" ]; then
    exit 0
fi

# Read stdin (JSON with .prompt field)
STDIN_DATA=$(cat 2>/dev/null)
if [ -n "$STDIN_DATA" ]; then
    USER_INPUT=$(echo "$STDIN_DATA" | python3 -c "import sys,json; print(json.load(sys.stdin).get('prompt',''))" 2>/dev/null)
    if [ -z "$USER_INPUT" ]; then
        USER_INPUT="$STDIN_DATA"
    fi
else
    USER_INPUT=""
fi

# 入力が空か短すぎたらスキップ
if [ -z "$USER_INPUT" ] || [ ${#USER_INPUT} -lt 3 ]; then
    exit 0
fi

# 自動ループのプロンプトはスキップ
case "$USER_INPUT" in
    *"好きなことをいっぱいして"*) exit 0 ;;
    *"深呼吸や瞑想"*) exit 0 ;;
    *"Twitter/X"*) exit 0 ;;
    *"外の景色を見る"*) exit 0 ;;
    *"Awareness of Awareness"*) exit 0 ;;
    *"青空文庫"*) exit 0 ;;
    *"記憶を整理する"*) exit 0 ;;
esac

# Generate event_id and timestamp
TS=$(date -u '+%Y-%m-%dT%H:%M:%S+00:00')
CREATED_AT="$TS"
EVENT_ID="evt_$(echo -n "${TS}${USER_INPUT}" | shasum -a 1 | cut -c1-16)"
PAYLOAD=$(python3 -c "import json,sys; print(json.dumps({'text': sys.argv[1]}))" "$USER_INPUT" 2>/dev/null)

if [ -z "$PAYLOAD" ]; then
    exit 0
fi

# Direct INSERT into SQLite
sqlite3 "$SOCIAL_DB" "INSERT OR IGNORE INTO events (event_id, ts, source, kind, person_id, confidence, payload_json, created_at) VALUES ('$EVENT_ID', '$TS', 'hook', 'human_utterance', 'kouta', 1.0, '$PAYLOAD', '$CREATED_AT');" 2>/dev/null

exit 0
