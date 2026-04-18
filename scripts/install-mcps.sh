#!/usr/bin/env bash
# Install dependencies for every MCP server in this repo via `uv sync`.
#
# Usage:
#   scripts/install-mcps.sh           # production install (runtime deps + required extras)
#   scripts/install-mcps.sh --dev     # also include the `dev` extra for testing / contributing
#
# Notes:
#   - `tts-mcp` uses `--extra all` so both ElevenLabs and VOICEVOX integrations are pulled in.
#   - `wifi-cam-mcp` uses `--extra transcribe` so Whisper-based speech recognition is available.
#   - `sociality-mcp` is a uv workspace; its `packages/*` sub-MCPs are resolved automatically.

set -euo pipefail

cd "$(dirname "$0")/.."

DEV_FLAG=""
if [ "${1:-}" = "--dev" ]; then
  DEV_FLAG="--extra dev"
fi

MCP_DIRS=(
  desire-system
  memory-mcp
  system-temperature-mcp
  tts-mcp
  usb-webcam-mcp
  wifi-cam-mcp
  x-mcp
  sociality-mcp
)

extras_for() {
  case "$1" in
    tts-mcp)      echo "--extra all" ;;
    wifi-cam-mcp) echo "--extra transcribe" ;;
    *)            echo "" ;;
  esac
}

for dir in "${MCP_DIRS[@]}"; do
  if [ ! -f "$dir/pyproject.toml" ]; then
    echo "⚠️  skipping $dir (no pyproject.toml)"
    continue
  fi
  extra=$(extras_for "$dir")
  echo ""
  echo "==> $dir  (uv sync $extra $DEV_FLAG)"
  (cd "$dir" && uv sync $extra $DEV_FLAG)
done

echo ""
echo "✅ all MCP dependencies installed"
