#!/bin/bash
# install-heartbeat.sh - heartbeat-daemon の launchd 登録を行う
#
# このスクリプトは embodied-claude リポジトリのルートを自動検出し、
# plist template のプレースホルダ (__EMBODIED_CLAUDE_ROOT__) を置換して
# ~/Library/LaunchAgents/com.embodied-claude.heartbeat.plist に配置、
# launchctl で load する。
#
# 使い方:
#   bash .claude/hooks/install-heartbeat.sh           # install & load
#   bash .claude/hooks/install-heartbeat.sh --uninstall  # unload & remove

set -e

HOOKS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HOOKS_DIR/../.." && pwd)"
TEMPLATE="$HOOKS_DIR/com.embodied-claude.heartbeat.plist"
TARGET_DIR="$HOME/Library/LaunchAgents"
TARGET="$TARGET_DIR/com.embodied-claude.heartbeat.plist"

uninstall() {
    if [ -f "$TARGET" ]; then
        launchctl unload "$TARGET" 2>/dev/null || true
        rm -f "$TARGET"
        echo "[install-heartbeat] unloaded and removed: $TARGET"
    else
        echo "[install-heartbeat] not installed: $TARGET"
    fi
}

if [ "$1" = "--uninstall" ]; then
    uninstall
    exit 0
fi

case "$(uname -s)" in
    Darwin) ;;
    *)
        echo "[install-heartbeat] このスクリプトは macOS (launchd) 専用です" >&2
        echo "[install-heartbeat] Linux では systemd user timer を使ってください" >&2
        exit 1
        ;;
esac

if [ ! -f "$TEMPLATE" ]; then
    echo "[install-heartbeat] template not found: $TEMPLATE" >&2
    exit 1
fi

mkdir -p "$TARGET_DIR"

# __EMBODIED_CLAUDE_ROOT__ を実パスに置換
sed "s|__EMBODIED_CLAUDE_ROOT__|${REPO_ROOT}|g" "$TEMPLATE" > "$TARGET"

# すでに load されていたら unload してから load し直す
launchctl unload "$TARGET" 2>/dev/null || true
launchctl load "$TARGET"

echo "[install-heartbeat] installed: $TARGET"
echo "[install-heartbeat] repo root:  $REPO_ROOT"
echo "[install-heartbeat] 動作確認: ls /tmp/interoception_state.json"
