#!/bin/bash
# hearing-hook.sh - 聴覚バッファを Claude のコンテキストに注入する UserPromptSubmit フック
#
# hearing ライブラリ（別リポジトリ lifemate-ai/embodied-codex の hearing/）が
# $HEARING_DIR/hearing_buffer.jsonl に蓄積した文字起こし結果を
# UserPromptSubmit のたびに読み取り、[hearing] プレフィックス付きで stdout に出力する。
# 読み取り後はバッファをアトミックに空にする。
#
# デーモンが未稼働の場合は何も出力せず静かに終了する。
#
# 移植性メモ（#139）:
#   - バッファ等の置き場所は HEARING_DIR（既定: $TMPDIR または /tmp）。シェル側で
#     解決して環境変数で Python に渡す。Windows (Git Bash) では Python の
#     Path("/tmp") がカレントドライブ直下を指してしまうため、Python 内に
#     "/tmp/..." を直接書かない。
#   - python3 は Microsoft Store のエイリアスに取られていることがあるので、
#     実際に動くインタプリタを探す（HEARING_PYTHON で固定も可）。
#   - MSYS の kill -0 はネイティブプロセスを見られないので tasklist にフォールバック。

HEARING_DIR="${HEARING_DIR:-${TMPDIR:-/tmp}}"
export HEARING_DIR
# Windows の Python は既定で stdout/ファイルをロケール（cp932 等）で扱うので、
# [hearing] 行と JSON が UTF-8 で往復するよう UTF-8 モードを強制する（POSIX では実質無変化）。
export PYTHONUTF8=1

BUFFER_FILE="$HEARING_DIR/hearing_buffer.jsonl"
PID_FILE="$HEARING_DIR/hearing-daemon.pid"
TIMING_LOG="$HEARING_DIR/hearing_timing.log"
USER_PROMPT_FILE="$HEARING_DIR/hearing_user_prompt.txt"
LAST_TS_FILE="$HEARING_DIR/hearing_hook_last_ts"

# ── 動く Python を選ぶ ───────────────────────────────────────────────────────
# HEARING_PYTHON が設定されていればそれを使う。なければ python3 → python の順で
# "import sys" が通る最初のものを採用する（Store のエイリアスは終了コード 49 で弾かれる）。
find_python() {
    local candidate
    if [ -n "${HEARING_PYTHON:-}" ]; then
        # 明示指定はそれだけを試す（動かなければ黙って終了）
        if "$HEARING_PYTHON" -c "import sys" >/dev/null 2>&1; then
            printf '%s\n' "$HEARING_PYTHON"
            return 0
        fi
        return 1
    fi
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 &&
           "$candidate" -c "import sys" >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

PY="$(find_python)" || exit 0
[ -z "$PY" ] && exit 0

# ── PID 生存確認（kill -0 → tasklist フォールバック）────────────────────────
# MSYS の kill はネイティブプロセス（Windows で起動したデーモン）を見られないので、
# kill -0 が失敗したら tasklist に聞く。MSYS_NO_PATHCONV / MSYS2_ARG_CONV_EXCL は
# "/FI" が "C:/Program Files/Git/FI" に変換されるのを止めるため（POSIX では無視される）。
pid_alive() {
    kill -0 "$1" 2>/dev/null && return 0
    if command -v tasklist >/dev/null 2>&1; then
        MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' tasklist /FI "PID eq $1" /FO CSV /NH 2>/dev/null | grep -q "\"$1\"" && return 0
    fi
    return 1
}

# stdinからユーザーのプロンプトを保存（jq がなければ Python で読む）
if command -v jq >/dev/null 2>&1; then
    jq -r '.prompt // empty' > "$USER_PROMPT_FILE"
else
    "$PY" -c 'import json,sys
try:
    print(json.load(sys.stdin).get("prompt") or "")
except Exception:
    pass' > "$USER_PROMPT_FILE" 2>/dev/null
fi

# タイミング記録
NOW=$("$PY" -c "import time; print(f'{time.time():.3f}')")
PREV=$(cat "$LAST_TS_FILE" 2>/dev/null)
PREV=${PREV:-$NOW}
DELTA=$("$PY" -c "print(f'{$NOW - $PREV:.1f}')")
echo "$NOW" > "$LAST_TS_FILE"
echo "[$(date +%H:%M:%S)] submit-hook  delta=${DELTA}s" >> "$TIMING_LOG"

# ── デーモン稼働確認 ──────────────────────────────────────────────────────────

DAEMON_RUNNING=false
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null)
    if [ -n "$PID" ] && pid_alive "$PID"; then
        DAEMON_RUNNING=true
    fi
fi

# デーモンが稼働していなければ何も出力しない
if [ "$DAEMON_RUNNING" = "false" ]; then
    exit 0
fi

# ── バッファをアトミックにドレインして出力 ────────────────────────────────────

"$PY" - <<'PYEOF' 2>/dev/null
import json
import os
import sys
import tempfile
from pathlib import Path

HEARING_DIR = Path(os.environ.get("HEARING_DIR") or tempfile.gettempdir())

BUFFER = HEARING_DIR / "hearing_buffer.jsonl"
DRAIN_TMP = HEARING_DIR / "hearing_buffer_drain.jsonl"

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
(HEARING_DIR / "hearing_stop_offset").unlink(missing_ok=True)

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
(HEARING_DIR / "hearing_had_speech").write_text(str(n))

# interoception.sh に合わせた key=value 形式で出力
print(f"[hearing] chunks={n} span={first_ts}~{last_ts} text={combined}")
PYEOF
