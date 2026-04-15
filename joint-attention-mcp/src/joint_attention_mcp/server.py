"""FastMCP server for joint attention."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from mcp.server.fastmcp import FastMCP

from .store import JointAttentionStore

mcp = FastMCP("joint-attention-mcp")


@lru_cache(maxsize=1)
def _store() -> JointAttentionStore:
    return JointAttentionStore()


@mcp.tool()
def ingest_scene_parse(scene: dict[str, Any]) -> dict[str, str]:
    """Store a structured scene parse from an adapter or orchestrator."""

    return _store().ingest_scene_parse(scene)


@mcp.tool()
def resolve_reference(
    expression: str, person_id: str | None = None, lookback_frames: int = 5
) -> dict[str, Any]:
    """Resolve a deictic or descriptive expression against recent scene objects."""

    return (
        _store()
        .resolve_reference(
            expression=expression, person_id=person_id, lookback_frames=lookback_frames
        )
        .model_dump(mode="json")
    )


@mcp.tool()
def get_current_joint_focus(person_id: str | None = None) -> dict[str, Any]:
    """Infer the current joint focus target."""

    return _store().get_current_joint_focus(person_id=person_id)


@mcp.tool()
def set_joint_focus(person_id: str | None, target_id: str, initiator: str) -> dict[str, str]:
    """Record an explicit joint focus target."""

    return _store().set_joint_focus(person_id=person_id, target_id=target_id, initiator=initiator)


@mcp.tool()
def compare_recent_scenes(person_id: str | None = None, window_minutes: int = 30) -> dict[str, Any]:
    """Return compact changes across recent scene parses."""

    return _store().compare_recent_scenes(person_id=person_id, window_minutes=window_minutes)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
