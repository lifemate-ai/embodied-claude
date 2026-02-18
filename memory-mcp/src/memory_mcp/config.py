"""Configuration for Memory MCP Server."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class MemoryConfig:
    """Memory storage configuration."""

    # PostgreSQL connection
    pg_dsn: str
    pool_min_size: int
    pool_max_size: int

    # Embedding model
    embedding_model: str
    embedding_api_url: str | None

    # Search parameters
    vector_weight: float
    text_weight: float
    half_life_days: float

    # Legacy ChromaDB (for migration only)
    db_path: str
    collection_name: str

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """Create config from environment variables."""
        default_chroma_path = str(Path.home() / ".claude" / "memories" / "chroma")
        default_dsn = "postgresql://memory_mcp:changeme@localhost:5432/embodied_claude"

        return cls(
            pg_dsn=os.getenv("MEMORY_PG_DSN", default_dsn),
            pool_min_size=int(os.getenv("MEMORY_PG_POOL_MIN", "2")),
            pool_max_size=int(os.getenv("MEMORY_PG_POOL_MAX", "10")),
            embedding_model=os.getenv(
                "MEMORY_EMBEDDING_MODEL", "intfloat/multilingual-e5-base"
            ),
            embedding_api_url=os.getenv("MEMORY_EMBEDDING_API_URL"),
            vector_weight=float(os.getenv("MEMORY_VECTOR_WEIGHT", "0.7")),
            text_weight=float(os.getenv("MEMORY_TEXT_WEIGHT", "0.3")),
            half_life_days=float(os.getenv("MEMORY_HALF_LIFE_DAYS", "30.0")),
            db_path=os.getenv("MEMORY_DB_PATH", default_chroma_path),
            collection_name=os.getenv("MEMORY_COLLECTION_NAME", "claude_memories"),
        )


@dataclass(frozen=True)
class ServerConfig:
    """MCP Server configuration."""

    name: str = "memory-mcp"
    version: str = "0.2.0"

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create config from environment variables."""
        return cls(
            name=os.getenv("MCP_SERVER_NAME", "memory-mcp"),
            version=os.getenv("MCP_SERVER_VERSION", "0.2.0"),
        )
