"""Pytest fixtures for Memory MCP tests (PostgreSQL backend)."""

import os

import asyncpg
import pytest
import pytest_asyncio
from dotenv import load_dotenv

from memory_mcp.config import MemoryConfig
from memory_mcp.memory import MemoryStore

load_dotenv()

_PG_ERROR_MSG = (
    "PostgreSQL is required to run tests. "
    "Start the database with: cd docker && docker compose up -d postgres\n"
    "Set TEST_PG_DSN to override the connection string."
)


@pytest.fixture
def pg_dsn() -> str:
    """Get PostgreSQL DSN for testing."""
    return os.getenv(
        "TEST_PG_DSN",
        "postgresql://memory_mcp:changeme@localhost:5432/embodied_claude_test",
    )


@pytest.fixture
def memory_config(pg_dsn: str) -> MemoryConfig:
    """Create test memory config."""
    return MemoryConfig(
        pg_dsn=pg_dsn,
        pool_min_size=1,
        pool_max_size=5,
        embedding_model=os.getenv(
            "TEST_EMBEDDING_MODEL", "intfloat/multilingual-e5-base"
        ),
        embedding_api_url=os.getenv("TEST_EMBEDDING_API_URL"),
        vector_weight=0.7,
        text_weight=0.3,
        half_life_days=30.0,
        db_path="",
        collection_name="",
    )


@pytest_asyncio.fixture
async def memory_store(memory_config: MemoryConfig) -> MemoryStore:
    """Create and connect a memory store, clean up tables between tests."""
    store = MemoryStore(memory_config)
    try:
        await store.connect()
    except (OSError, asyncpg.PostgresError, asyncpg.InterfaceError) as e:
        pytest.fail(f"{_PG_ERROR_MSG}\n\nOriginal error: {e}")

    # Clean all tables before each test
    pool = store._store._pool
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM coactivation_weights")
        await conn.execute("DELETE FROM memory_links")
        await conn.execute("DELETE FROM memories")
        await conn.execute("DELETE FROM episodes")

    yield store
    await store.disconnect()
