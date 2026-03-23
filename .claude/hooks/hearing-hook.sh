#!/bin/bash
# hearing-hook.sh - 聴覚バッファを Claude のコンテキストに注入する UserPromptSubmit フック
#
# hearing-daemon.py が /tmp/hearing_buffer.jsonl に蓄積した文字起こし結果を
# UserPromptSubmit のたびに読み取り、[hearing] プレフィックス付きで stdout に出力する。
# 読み取り後はバッファをアトミックに空にする。
#
# デーモンが未稼働の場合は何も出力せず静かに終了する。

BUFFER_FILE="/tmp/hearing_buffer.jsonl"
PID_FILE="/tmp/hearing-daemon.pid"
TIMING_LOG="/tmp/hearing_timing.log"
USER_PROMPT_FILE="/tmp/hearing_user_prompt.txt"

# stdinからユーザーのプロンプトを保存
jq -r '.prompt // empty' > "$USER_PROMPT_FILE"

# タイミング記録
NOW=$(python3 -c "import time; print(f'{time.time():.3f}')")
PREV=$(cat /tmp/hearing_hook_last_ts 2>/dev/null || echo "$NOW")
DELTA=$(python3 -c "print(f'{$NOW - $PREV:.1f}')")
echo "$NOW" > /tmp/hearing_hook_last_ts
echo "[$(date +%H:%M:%S)] submit-hook  delta=${DELTA}s" >> "$TIMING_LOG"

# ── デーモン稼働確認 ──────────────────────────────────────────────────────────

DAEMON_RUNNING=false
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null)
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        DAEMON_RUNNING=true
    fi
fi

# デーモンが稼働していなければ何も出力しない
if [ "$DAEMON_RUNNING" = "false" ]; then
    exit 0
fi

# ── バッファをアトミックにドレインして出力 ────────────────────────────────────

python3 - <<'PYEOF' 2>/dev/null
import json
import os
import sys
from pathlib import Path

BUFFER = Path("/tmp/hearing_buffer.jsonl")
DRAIN_TMP = Path("/tmp/hearing_buffer_drain.jsonl")

# バッファが空なら何もしない
if not BUFFER.exists() or BUFFER.stat().st_size == 0:
    sys.exit(0)

# os.rename はアトミック操作。rename 後にデーモンが書き込む新エントリは
# open("a") によって新しい BUFFER ファイルへ書かれるため、データ欠損なし。
try:
    os.rename(str(BUFFER), str(DRAIN_TMP))
except OSError as e:
    print(f"[hearing] drain_error={e}", file=sys.stderr)
    sys.exit(0)

# エントリを読み取る
entries = []
try:
    with open(DRAIN_TMP, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
finally:
    DRAIN_TMP.unlink(missing_ok=True)

# Stopフックのoffsetをリセット（バッファがdrainされたので）
Path("/tmp/hearing_stop_offset").unlink(missing_ok=True)

if not entries:
    sys.exit(0)

# 時刻は "HH:MM:SS" の形式に整形
def fmt_time(ts: str) -> str:
    if "T" in ts:
        return ts.split("T")[1][:8]
    return ts

n = len(entries)
first_ts = fmt_time(entries[0]["ts"])
last_ts  = fmt_time(entries[-1]["ts"])
texts    = [e["text"] for e in entries]
combined = " / ".join(texts)

# チェーン保証フラグ: stop hookが最低1回は待つようにする
Path("/tmp/hearing_had_speech").write_text(str(n))

# interoception.sh に合わせた key=value 形式で出力
print(f"[hearing] chunks={n} span={first_ts}~{last_ts} text={combined}")
PYEOF
