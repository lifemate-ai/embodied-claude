#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKER_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== embodied-claude Docker setup ==="
echo ""

# ── 1. Docker Desktop check ──
if ! docker info &>/dev/null; then
    echo "Docker Desktop is not running. Please start it first."
    exit 1
fi
echo "[ok] Docker Desktop running"

# ── 2. Generate .env ──
if [ ! -f "${DOCKER_DIR}/.env" ]; then
    echo ""
    echo "Generating .env file..."
    cp "${DOCKER_DIR}/.env.example" "${DOCKER_DIR}/.env"

    # Generate random password
    PG_PASS=$(openssl rand -base64 24 | tr -dc 'a-zA-Z0-9' | head -c 24)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/your_secure_password_here/${PG_PASS}/" "${DOCKER_DIR}/.env"
    else
        sed -i "s/your_secure_password_here/${PG_PASS}/" "${DOCKER_DIR}/.env"
    fi

    echo "  PostgreSQL password generated automatically"
else
    echo "[ok] .env file exists"
fi

# ── 3. Build Docker images ──
echo ""
echo "Building Docker images (first build may take 10-20 minutes)..."
cd "${DOCKER_DIR}"
docker compose build

# ── 4. Start containers ──
echo ""
echo "Starting containers..."
docker compose up -d

# ── 5. Wait for services ──
echo ""
echo "Waiting for services..."

# PostgreSQL
echo -n "  PostgreSQL: "
for i in $(seq 1 30); do
    if docker compose exec -T postgres pg_isready -U memory_mcp -d embodied_claude &>/dev/null; then
        echo "[ok]"
        break
    fi
    if [ "$i" -eq 30 ]; then echo "[TIMEOUT]"; exit 1; fi
    sleep 2
done

# Extensions
echo -n "  pgvector: "
if docker compose exec -T postgres psql -U memory_mcp -d embodied_claude \
    -c "SELECT extversion FROM pg_extension WHERE extname='vector'" -t 2>/dev/null | grep -q .; then
    echo "[ok]"
else
    echo "[FAIL]"
fi

echo -n "  pgroonga: "
if docker compose exec -T postgres psql -U memory_mcp -d embodied_claude \
    -c "SELECT extversion FROM pg_extension WHERE extname='pgroonga'" -t 2>/dev/null | grep -q .; then
    echo "[ok]"
else
    echo "[FAIL]"
fi

# Embedding API
echo -n "  Embedding API: "
for i in $(seq 1 60); do
    if curl -sf http://localhost:8100/health &>/dev/null; then
        echo "[ok]"
        break
    fi
    if [ "$i" -eq 60 ]; then echo "[TIMEOUT] (model may still be downloading)"; break; fi
    sleep 3
done

# ── 6. Connection info ──
PG_USER=$(grep PG_USER "${DOCKER_DIR}/.env" | cut -d= -f2)
PG_PASS=$(grep PG_PASSWORD "${DOCKER_DIR}/.env" | cut -d= -f2)
PG_DB=$(grep PG_DATABASE "${DOCKER_DIR}/.env" | cut -d= -f2)
PG_DSN="postgresql://${PG_USER}:${PG_PASS}@localhost:5432/${PG_DB}"

echo ""
echo "========================================="
echo "Setup complete!"
echo ""
echo "  PostgreSQL:    localhost:5432"
echo "  Embedding API: localhost:8100"
echo "  PG DSN:        ${PG_DSN}"
echo ""
echo "Set the following environment variable for memory-mcp:"
echo "  export MEMORY_PG_DSN='${PG_DSN}'"
echo ""
echo "To use the embedding API, set:"
echo "  export MEMORY_EMBEDDING_API_URL='http://localhost:8100'"
echo "========================================="
