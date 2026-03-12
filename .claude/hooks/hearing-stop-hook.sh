#!/bin/bash
# hearing-stop-hook.sh — Stop hook で聴覚バッファをチェックし、
# 新しい発話があればターンを延長する。
#
# バッファ管理: 行番号ベース
#   - バッファは消さずに読む（offset以降の新しい行だけ処理）
#   - 有効 → offset更新 & 処理済み行を削除 & block
#   - 無効 → offset据え置き → 短いsleep後に即リトライ（新データが溜まるのを待つ）

BUFFER_FILE="/tmp/hearing_buffer.jsonl"
PID_FILE="/tmp/hearing-daemon.pid"
OFFSET_FILE="/tmp/hearing_stop_offset"
TIMING_LOG="/tmp/hearing_timing.log"
GUARANTEED_COUNTER="/tmp/hearing-guaranteed-counter"
CONTEXT_FILE="/tmp/hearing_context.json"

# stdinからコンテキストを保存（1回だけ読める）
cat > "$CONTEXT_FILE"

# タイミング記録
NOW=$(python3 -c "import time; print(f'{time.time():.3f}')")
PREV=$(cat /tmp/hearing_stop_last_ts 2>/dev/null || echo "$NOW")
DELTA=$(python3 -c "print(f'{$NOW - $PREV:.1f}')")
echo "$NOW" > /tmp/hearing_stop_last_ts
echo "[$(date +%H:%M:%S)] stop-hook-start delta=${DELTA}s count=${COUNT:-?}" >> "$TIMING_LOG"
MAX_HEARING_CONTINUES=${MAX_HEARING_CONTINUES:-20}
COUNTER_FILE="/tmp/hearing-stop-counter"
WAIT_SECONDS=${HEARING_WAIT_SECONDS:-5}
RETRY_WAIT=${HEARING_RETRY_WAIT:-3}
NO_SPEECH_THRESHOLD=${HEARING_NO_SPEECH_THRESHOLD:-0.6}

# mcpBehavior.toml から hearing 設定を読む（uv経由でtomllib使用）
# MCP_BEHAVIOR_TOML が未設定ならプロジェクトルートから探す
if [ -z "$MCP_BEHAVIOR_TOML" ]; then
    _PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
    [ -f "$_PROJECT_ROOT/mcpBehavior.toml" ] && MCP_BEHAVIOR_TOML="$_PROJECT_ROOT/mcpBehavior.toml"
fi
if [ -n "$MCP_BEHAVIOR_TOML" ] && [ -f "$MCP_BEHAVIOR_TOML" ]; then
    HEARING_DIR="$(cd "$(dirname "$0")/../.." && pwd)/embodied-claude/hearing"
    [ ! -d "$HEARING_DIR" ] && HEARING_DIR="$(cd "$(dirname "$0")/../.." && pwd)/hearing"
    eval "$(uv run --directory "$HEARING_DIR" python3 -c "
import tomllib
try:
    with open('$MCP_BEHAVIOR_TOML', 'rb') as f:
        h = tomllib.load(f).get('hearing', {})
    print(f'HEARING_MIN_GUARANTEED={h.get(\"min_guaranteed\", 5)}')
    print(f'HEARING_GUARANTEED_SLEEP={h.get(\"guaranteed_sleep\", 5)}')
except Exception:
    print('HEARING_MIN_GUARANTEED=5')
    print('HEARING_GUARANTEED_SLEEP=5')
" 2>/dev/null)"
fi
HEARING_MIN_GUARANTEED=${HEARING_MIN_GUARANTEED:-5}
HEARING_GUARANTEED_SLEEP=${HEARING_GUARANTEED_SLEEP:-5}

# ── デーモン稼働確認 ──────────────────────────────────────────
DAEMON_RUNNING=false
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null)
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        DAEMON_RUNNING=true
    fi
fi

[ "$DAEMON_RUNNING" = "false" ] && exit 0

# ── カウンタ読み込み & 上限チェック ────────────────────────────
COUNT=$(cat "$COUNTER_FILE" 2>/dev/null || echo 0)

if [ "$COUNT" -ge "$MAX_HEARING_CONTINUES" ]; then
    rm -f "$COUNTER_FILE"
    exit 0
fi

# ── 応答を待つ ────────────────────────────────────────────────
sleep "$WAIT_SECONDS"

# ── バッファを行番号ベースで読み取り・判定 ─────────────────────
RESULT=$(python3 - "$NO_SPEECH_THRESHOLD" "$OFFSET_FILE" "$BUFFER_FILE" "$RETRY_WAIT" "$COUNT" <<'PYEOF' 2>>/tmp/hearing_timing.log
import json
import os
import sys
import time
from pathlib import Path

threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
offset_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/hearing_stop_offset")
buffer_file = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/tmp/hearing_buffer.jsonl")
retry_wait = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0

def read_offset():
    try:
        return int(offset_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return 0

def write_offset(n):
    offset_file.write_text(str(n))

def read_buffer_from(start_line):
    """バッファのstart_line行目以降を読む（0-indexed）"""
    if not buffer_file.exists() or buffer_file.stat().st_size == 0:
        return [], 0
    lines = []
    total = 0
    with open(buffer_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            total = i + 1
            if i >= start_line:
                lines.append((i, line))
    return lines, total

def filter_entries(lines):
    entries = []
    for line_no, line in lines:
        line_s = line.strip()
        if not line_s:
            continue
        try:
            e = json.loads(line_s)
            if e.get("no_speech_prob", 1.0) <= threshold:
                entries.append(e)
        except json.JSONDecodeError:
            pass
    return entries

def truncate_buffer(up_to_line):
    """処理済み行をバッファから削除（up_to_line行目まで削除、それ以降を残す）"""
    if not buffer_file.exists():
        return
    with open(buffer_file, encoding="utf-8") as f:
        all_lines = f.readlines()
    remaining = all_lines[up_to_line:]
    with open(buffer_file, "w", encoding="utf-8") as f:
        f.writelines(remaining)
    # offset をリセット（バッファが切り詰められたので）
    write_offset(0)

def fmt_time(ts):
    if "T" in ts:
        return ts.split("T")[1][:8]
    return ts

def _read_toml_hearing(key, default=None):
    """mcpBehavior.toml の [hearing] から値を読む"""
    toml_path = Path(os.environ.get("MCP_BEHAVIOR_TOML", ""))
    if not toml_path.is_file():
        return default
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            return default
    try:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        return data.get("hearing", {}).get(key, default)
    except Exception:
        return default

def llm_filter(texts):
    """claude -p でハルシネーション判定。実際の発話だけ返す。opt-in。"""
    import subprocess as sp

    # opt-in チェック
    if not _read_toml_hearing("llm_filter", False):
        return " / ".join(texts)  # フィルタなしでそのまま返す

    timeout = int(_read_toml_hearing("llm_filter_timeout", 20))
    combined = " / ".join(texts)

    # コンテキスト読み込み（短く切る。[hearing]やhookメタ情報を除去）
    context_parts = []
    user_prompt = Path("/tmp/hearing_user_prompt.txt")
    context_json = Path("/tmp/hearing_context.json")
    if user_prompt.exists():
        up = user_prompt.read_text().strip()
        # LLMプロンプトの再帰ネスト除去: 【タスク】マーカーがあれば手前を切る
        if "【タスク】" in up:
            up = ""
        # hook由来のメタ情報を除去
        lines = [l for l in up.splitlines()
                 if not l.startswith("[hearing]")
                 and not l.startswith("Stop hook")
                 and not l.startswith("チェイン")
                 and "聞き取り待機中" not in l
                 and "音声認識フィルタ" not in l]
        up = "\n".join(lines).strip()[:150]
        if up:
            context_parts.append(f"ユーザー: {up}")
    if context_json.exists():
        try:
            import json as _j
            ctx = _j.loads(context_json.read_text())
            lam = ctx.get("last_assistant_message", "")
            if lam:
                context_parts.append(f"Claude: {lam[:150]}")
        except Exception:
            pass
    context_str = "\n".join(context_parts) if context_parts else "(コンテキストなし)"

    prompt = (
        "あなたは音声認識フィルタです。会話の文脈と音声認識結果を見て判定してください。\n\n"
        f"【会話の文脈】\n{context_str}\n\n"
        f"【音声認識(Whisper)の出力】\n{combined}\n\n"
        "【タスク】\n"
        "1. 音声認識結果から、実際に人が喋った言葉だけを抽出してください\n"
        "2. Whisperのハルシネーション(「ご視聴ありがとう」「さようなら」「チャンネル登録」等の定型句、"
        "意味不明な繰り返し、文脈と無関係なフレーズ)は除去してください\n"
        "3. 実際の発話が1つもなければ EMPTY とだけ返してください\n"
        "4. 実際の発話があれば、そのテキストだけを返してください。余計な説明は不要です"
    )
    try:
        env = {**os.environ, "CLAUDECODE": ""}
        env.pop("CLAUDECODE", None)
        r = sp.run(
            ["claude", "-p", "--model", "haiku", prompt],
            capture_output=True, text=True, timeout=timeout,
            env=env,
        )
        out = r.stdout.strip()
        if not out or out == "EMPTY":
            return None
        return out
    except Exception as e:
        print(f"[hearing-debug] llm_filter error: {e}", file=sys.stderr)
        return combined  # フォールバック: フィルタなしで通す

def try_read():
    offset = read_offset()
    lines, total = read_buffer_from(offset)
    if not lines:
        return None, total, False
    entries = filter_entries(lines)
    if not entries:
        return None, total, False
    # 末尾エントリが発話途中かチェック
    tail_speaking = entries[-1].get("tail_speech", False)
    # 有効 → LLMフィルタ
    texts = [e["text"] for e in entries]
    filtered = llm_filter(texts)
    if not filtered:
        # LLMがハルシネーションと判定 → バッファは切り詰めるが結果なし
        last_line_no = lines[-1][0]
        truncate_buffer(last_line_no + 1)
        return None, total, False
    # 有効 → 出力 & バッファ切り詰め
    last_line_no = lines[-1][0]
    truncate_buffer(last_line_no + 1)
    n = len(entries)
    first_ts = fmt_time(entries[0]["ts"])
    last_ts = fmt_time(entries[-1]["ts"])
    return f"[hearing] chunks={n} span={first_ts}~{last_ts} text={filtered}", total, tail_speaking

# チェーン保証回数: バッファが空でもこの回数まではリトライする
MIN_GUARANTEED = int(_read_toml_hearing("min_guaranteed", 5))
count = int(sys.argv[5]) if len(sys.argv) > 5 else 0
t_start = time.time()

debug_lines = []
pending_result = None  # tail_speech で保留中の結果
# 最大3回リトライ（retry_wait間隔）
for attempt in range(3):
    result, total, tail_speaking = try_read()
    elapsed = time.time() - t_start
    tag = "HIT" if result else "empty"
    if result and tail_speaking:
        tag = "HIT+tail"
    debug_lines.append(f"  retry{attempt}: {tag} buf_lines={total} +{elapsed:.1f}s")
    if result:
        if tail_speaking and attempt < 2:
            # 末尾に音声あり → まだ喋ってる途中。保留して次のチャンクを待つ
            pending_result = result
            time.sleep(retry_wait)
            continue
        # tail_speech なし or 最終リトライ → 確定出力
        # pending があれば今回の結果に統合済み（truncate_bufferで処理済み）
        debug = " | ".join(debug_lines)
        print(f"{result} [debug: count={count} {debug}]")
        sys.exit(0)
    # バッファ空だが保留結果あり → 保留結果を出力
    if pending_result:
        debug = " | ".join(debug_lines)
        print(f"{pending_result} [debug: count={count} {debug} (flushed pending)]")
        sys.exit(0)
    # バッファ空 → リトライ（保証判定はbash側で行う）
    time.sleep(retry_wait)

# 保留結果が残っていれば出力
if pending_result:
    debug = " | ".join(debug_lines)
    print(f"{pending_result} [debug: count={count} {debug} (flushed pending)]")
    sys.exit(0)

# 全リトライ失敗時もデバッグ出力
debug = " | ".join(debug_lines)
print(f"[hearing-debug] no speech. count={count} {debug}", file=sys.stderr)
sys.exit(0)
PYEOF
)

# ── 判定 ──────────────────────────────────────────────────────
END_TS=$(python3 -c "import time; print(f'{time.time():.3f}')")
GCOUNT=$(cat "$GUARANTEED_COUNTER" 2>/dev/null || echo 0)
MIN_GUARANTEED=${HEARING_MIN_GUARANTEED:-5}

if [ -n "$RESULT" ]; then
    echo $((COUNT + 1)) > "$COUNTER_FILE"
    # HIT → 保証カウンターリセット（発話があったので）
    rm -f "$GUARANTEED_COUNTER"
    ELAPSED=$(python3 -c "print(f'{$END_TS - $NOW:.1f}')")
    echo "[$(date +%H:%M:%S)] stop-hook-block elapsed=${ELAPSED}s chain=$((COUNT+1))" >> "$TIMING_LOG"
    ESCAPED=$(echo "$RESULT" | sed 's/"/\\"/g')
    echo "{\"decision\": \"block\", \"reason\": \"Stop hook feedback:\n${ESCAPED}\nチェイン($((COUNT+1))/${MAX_HEARING_CONTINUES}) 保証(0/${MIN_GUARANTEED})\"}"
else
    ELAPSED=$(python3 -c "print(f'{$END_TS - $NOW:.1f}')")
    # チェーン保証: 連続空回数が保証回数以内なら待機
    if [ "$GCOUNT" -lt "$MIN_GUARANTEED" ]; then
        # 保証待機: 次のセグメントが来るまで待つ
        sleep "$HEARING_GUARANTEED_SLEEP"
        echo $((COUNT + 1)) > "$COUNTER_FILE"
        echo $((GCOUNT + 1)) > "$GUARANTEED_COUNTER"
        echo "[$(date +%H:%M:%S)] stop-hook-wait elapsed=${ELAPSED}s chain=$((COUNT+1)) guaranteed=$((GCOUNT+1))/${MIN_GUARANTEED}" >> "$TIMING_LOG"
        echo "{\"decision\": \"block\", \"reason\": \"Stop hook feedback:\n[hearing] 聞き取り待機中... 保証($((GCOUNT+1))/${MIN_GUARANTEED}) チェイン($((COUNT+1))/${MAX_HEARING_CONTINUES})\"}"
    else
        echo "[$(date +%H:%M:%S)] stop-hook-pass elapsed=${ELAPSED}s (no speech, guaranteed exhausted)" >> "$TIMING_LOG"
        rm -f "$COUNTER_FILE" "$GUARANTEED_COUNTER"
        exit 0
    fi
fi
