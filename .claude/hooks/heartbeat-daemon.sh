#!/bin/bash
# heartbeat-daemon.sh - 心拍デーモン
# 5秒ごとに実行され、体の状態を /tmp/interoception_state.json に書き出す
# interoception.sh (UserPromptSubmitフック) がこのファイルを読んでコンテキストに注入する
# macOS: launchd (com.embodied-claude.heartbeat.plist)
# Linux: systemd user timer (heartbeat-daemon.timer)

STATE_FILE="/tmp/interoception_state.json"
WINDOW_SIZE=12  # 直近12エントリ（5秒×12=1分間）

# --- OS判定 ---
OS_TYPE="$(uname -s)"

# --- 時刻 ---
CURRENT_TIME=$(date '+%Y-%m-%dT%H:%M:%S%z')
HOUR=$(date '+%H')

if [ "$HOUR" -ge 5 ] && [ "$HOUR" -lt 10 ]; then
    PHASE="morning"
elif [ "$HOUR" -ge 10 ] && [ "$HOUR" -lt 12 ]; then
    PHASE="late_morning"
elif [ "$HOUR" -ge 12 ] && [ "$HOUR" -lt 14 ]; then
    PHASE="midday"
elif [ "$HOUR" -ge 14 ] && [ "$HOUR" -lt 17 ]; then
    PHASE="afternoon"
elif [ "$HOUR" -ge 17 ] && [ "$HOUR" -lt 20 ]; then
    PHASE="evening"
elif [ "$HOUR" -ge 20 ] && [ "$HOUR" -lt 23 ]; then
    PHASE="night"
else
    PHASE="late_night"
fi

# --- CPU負荷（覚醒度） ---
case "$OS_TYPE" in
    Linux)
        LOAD_AVG=$(awk '{print $1}' /proc/loadavg)
        NCPU=$(nproc 2>/dev/null || echo 4)
        ;;
    Darwin)
        LOAD_AVG=$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')
        if [ -z "$LOAD_AVG" ]; then
            LOAD_AVG=$(uptime | awk -F'load averages?: ' '{print $2}' | awk '{print $1}' | tr -d ',')
        fi
        NCPU=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
        ;;
esac
AROUSAL=$(echo "$LOAD_AVG $NCPU" | awk '{pct = ($1 / $2) * 100; if (pct > 100) pct = 100; printf "%.0f", pct}')

# --- メモリ ---
case "$OS_TYPE" in
    Linux)
        MEM_TOTAL=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
        MEM_AVAIL=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        if [ -n "$MEM_TOTAL" ] && [ "$MEM_TOTAL" -gt 0 ]; then
            MEM_PRESSURE=$(echo "$MEM_AVAIL $MEM_TOTAL" | awk '{printf "%.0f", ($1 / $2) * 100}')
        else
            MEM_PRESSURE="0"
        fi
        ;;
    Darwin)
        MEM_PRESSURE=$(memory_pressure 2>/dev/null | grep "System-wide memory free percentage" | awk '{print $NF}' | tr -d '%')
        if [ -z "$MEM_PRESSURE" ]; then
            MEM_PRESSURE="0"
        fi
        ;;
esac

# --- 体温 ---
case "$OS_TYPE" in
    Linux)
        RAW_TEMP=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo "0")
        THERMAL=$(echo "$RAW_TEMP" | awk '{printf "%.1f", $1 / 1000}')
        ;;
    Darwin)
        THERMAL=$(sysctl -n machdep.xcpm.cpu_thermal_level 2>/dev/null || echo "0")
        ;;
esac

# --- 稼働時間（分） ---
case "$OS_TYPE" in
    Linux)
        UPTIME_MIN=$(awk '{printf "%.0f", $1 / 60}' /proc/uptime)
        ;;
    Darwin)
        BOOT_TIME=$(sysctl -n kern.boottime 2>/dev/null | awk '{print $4}' | tr -d ',')
        if [ -n "$BOOT_TIME" ]; then
            NOW_EPOCH=$(date +%s)
            UPTIME_MIN=$(( (NOW_EPOCH - BOOT_TIME) / 60 ))
        else
            UPTIME_MIN=0
        fi
        ;;
esac

# --- ring buffer 管理 ---
# 既存のstate fileからwindowを読み出し、新エントリを追加、古いのを削除
if [ -f "$STATE_FILE" ]; then
    # 既存windowを取得（最大WINDOW_SIZE-1エントリ保持）
    EXISTING_WINDOW=$(python3 -c "
import json, sys
try:
    with open('$STATE_FILE') as f:
        data = json.load(f)
    window = data.get('window', [])
    # 最新 WINDOW_SIZE-1 エントリだけ保持
    window = window[-(${WINDOW_SIZE}-1):]
    print(json.dumps(window))
except:
    print('[]')
" 2>/dev/null || echo "[]")
else
    EXISTING_WINDOW="[]"
fi

# 新エントリをwindowに追加
NEW_ENTRY="{\"ts\":\"${CURRENT_TIME}\",\"arousal\":${AROUSAL},\"mem_free\":${MEM_PRESSURE:-0},\"thermal\":${THERMAL:-0}}"

# --- トレンド算出 ---
TREND_JSON=$(python3 -c "
import json
window = json.loads('${EXISTING_WINDOW}')
new = ${NEW_ENTRY}
window.append(new)

def trend(values):
    if len(values) < 3:
        return 'stable'
    recent = values[-3:]
    diff = recent[-1] - recent[0]
    if diff > 5:
        return 'rising'
    elif diff < -5:
        return 'falling'
    return 'stable'

arousal_vals = [e.get('arousal', 0) for e in window]
mem_vals = [e.get('mem_free', 0) for e in window]

result = {
    'now': {
        'ts': '${CURRENT_TIME}',
        'phase': '${PHASE}',
        'arousal': ${AROUSAL},
        'thermal': ${THERMAL:-0},
        'mem_free': ${MEM_PRESSURE:-0},
        'uptime_min': ${UPTIME_MIN}
    },
    'window': window,
    'trend': {
        'arousal': trend(arousal_vals),
        'mem_free': trend(mem_vals)
    }
}
print(json.dumps(result, ensure_ascii=False, indent=2))
" 2>/dev/null)

# --- 書き出し ---
if [ -n "$TREND_JSON" ]; then
    echo "$TREND_JSON" > "${STATE_FILE}.tmp"
    mv "${STATE_FILE}.tmp" "$STATE_FILE"
fi
