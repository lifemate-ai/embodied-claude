#!/usr/bin/env bash
# Install the unified workspace and warm the memory embedding model.
#
# Usage:
#   scripts/install-mcps.sh
#   scripts/install-mcps.sh --dev     # compatibility alias; dev dependencies are installed by default
#
# Notes:
#   - New users should prefer `scripts/setup.sh`, which also creates a safe
#     Core `.mcp.json` and runs diagnostics.
#   - This compatibility installer preserves the historical "install everything"
#     behavior. New installations should select capabilities through setup.
#   - `memory-mcp` pre-downloads its embedding model so the first remember() doesn't lazy-fail.

set -euo pipefail

cd "$(dirname "$0")/.."

case "${1:-}" in
  "" | --dev) ;;
  *)
    echo "usage: scripts/install-mcps.sh [--dev]" >&2
    exit 2
    ;;
esac

echo "==> syncing the embodied-claude workspace"
uv sync --locked --all-extras --group dev

echo "==> pre-downloading embedding model (honors \$MEMORY_EMBEDDING_MODEL)"
uv run --package memory-mcp python -c "
from memory_mcp.config import MemoryConfig
from memory_mcp.embedding import E5EmbeddingFunction
model = MemoryConfig.from_env().embedding_model
print(f'  warming {model}')
E5EmbeddingFunction(model)._load_model()
print('  done')
"

echo ""
echo "all MCP dependencies installed in the root .venv"
