"""Configuration for Memory MCP Server."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class MemoryConfig:
    """Memory storage configuration."""

    db_path: str
    collection_name: str

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """Create config from environment variables."""
        default_path = str(Path.home() / ".claude" / "memories" / "chroma")

        return cls(
            db_path=os.getenv("MEMORY_DB_PATH", default_path),
            collection_name=os.getenv("MEMORY_COLLECTION_NAME", "claude_memories"),
        )


@dataclass(frozen=True)
class ServerConfig:
    """MCP Server configuration."""

    name: str = "memory-mcp"
    version: str = "0.1.0"

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create config from environment variables."""
        return cls(
            name=os.getenv("MCP_SERVER_NAME", "memory-mcp"),
            version=os.getenv("MCP_SERVER_VERSION", "0.1.0"),
        )

@dataclass(frozen=True)
class JobIsolationConfig:
    """Job isolation configuration for persistence."""

    jobs_db_path: str

    @classmethod
    def from_memory_config(cls, memory_config: MemoryConfig) -> "JobIsolationConfig":
        """Create job isolation config from memory config."""
        # Store job configs in the same directory as ChromaDB but in a separate JSON file
        jobs_db_path = os.path.join(
            os.path.dirname(memory_config.db_path),
            "job_configs.json"
        )
        return cls(jobs_db_path=jobs_db_path)
