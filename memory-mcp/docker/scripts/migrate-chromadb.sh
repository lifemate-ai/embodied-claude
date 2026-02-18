#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKER_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$DOCKER_DIR")"

echo "=== ChromaDB -> PostgreSQL migration ==="

# ChromaDB data path
CHROMA_PATH="${1:-$HOME/.claude/memories/chroma}"

if [ ! -d "$CHROMA_PATH" ]; then
    echo "ChromaDB data not found: $CHROMA_PATH"
    echo "Usage: $0 [path-to-chromadb-data]"
    exit 1
fi

echo "ChromaDB data: $CHROMA_PATH"

# Check Docker is running
if ! docker compose -f "${DOCKER_DIR}/docker-compose.yml" ps postgres 2>/dev/null | grep -q "running"; then
    echo "PostgreSQL container is not running."
    echo "Run: cd ${DOCKER_DIR} && docker compose up -d"
    exit 1
fi

# Build PG DSN from .env
if [ -f "${DOCKER_DIR}/.env" ]; then
    PG_USER=$(grep PG_USER "${DOCKER_DIR}/.env" | cut -d= -f2)
    PG_PASS=$(grep PG_PASSWORD "${DOCKER_DIR}/.env" | cut -d= -f2)
    PG_DB=$(grep PG_DATABASE "${DOCKER_DIR}/.env" | cut -d= -f2)
    PG_DSN="postgresql://${PG_USER}:${PG_PASS}@localhost:5432/${PG_DB}"
else
    PG_DSN="postgresql://memory_mcp:changeme@localhost:5432/embodied_claude"
fi

echo "PostgreSQL DSN: ${PG_DSN}"

# Run migration
cd "${PROJECT_DIR}"
uv run memory-migrate \
    --chroma-path "$CHROMA_PATH" \
    --pg-dsn "$PG_DSN" \
    --batch-size 100

echo ""
echo "Migration complete!"
