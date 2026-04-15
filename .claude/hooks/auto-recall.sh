#!/bin/bash
# auto-recall.sh - 自動連想想起
exec 2>/tmp/auto-recall-debug.log
# UserPromptSubmitフックで毎ターン実行
# ユーザーの入力からmemory-mcpのHTTPエンドポイントで関連記憶を検索し注入

MEMORY_HTTP_PORT="${MEMORY_HTTP_PORT:-18900}"
# Read stdin into variable (JSON with .prompt field, or plain text)
STDIN_DATA=$(cat 2>/dev/null)
if [ -n "$STDIN_DATA" ]; then
    USER_INPUT=$(echo "$STDIN_DATA" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('prompt',''))" 2>/dev/null)
    # fallback: use raw stdin if JSON parse fails
    if [ -z "$USER_INPUT" ]; then
        USER_INPUT="$STDIN_DATA"
    fi
else
    USER_INPUT=""
fi

# 入力が空か短すぎたらスキップ
if [ -z "$USER_INPUT" ] || [ ${#USER_INPUT} -lt 5 ]; then
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

# URL encode the query
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$USER_INPUT" 2>/dev/null)

if [ -z "$ENCODED" ]; then
    exit 0
fi

# memory-mcp HTTP endpoint に問い合わせ（タイムアウト3秒）
RESULT=$(curl -s --max-time 3 "http://127.0.0.1:${MEMORY_HTTP_PORT}/recall?q=${ENCODED}&n=2" 2>/dev/null)

# 結果が空か [] ならスキップ
if [ -z "$RESULT" ] || [ "$RESULT" = "[]" ]; then
    exit 0
fi

echo "DEBUG: USER_INPUT='$USER_INPUT' ENCODED='$ENCODED' RESULT='$RESULT'" >&2
echo "[associative_recall] $RESULT"
