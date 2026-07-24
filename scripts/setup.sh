#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it with:" >&2
  echo "curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  echo "More options: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 127
fi

exec uv run --no-project --python 3.13 python scripts/setup.py "$@"
