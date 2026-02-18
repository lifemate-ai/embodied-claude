"""Associative graph expansion for divergent recall."""

from __future__ import annotations

from dataclasses import dataclass

from .predictive import query_ambiguity_score


def adaptive_search_params(
    context: str,
    requested_branches: int,
    requested_depth: int,
    seed_count: int,
) -> tuple[int, int]:
    """Adapt branch/depth based on query ambiguity and seed confidence."""
    ambiguity = query_ambiguity_score(context)
    if seed_count <= 1:
        ambiguity = min(1.0, ambiguity + 0.2)

    branch_scale = 0.8 + ambiguity
    depth_scale = 0.9 + 0.5 * ambiguity

    branches = int(round(requested_branches * branch_scale))
    depth = int(round(requested_depth * depth_scale))

    branches = max(1, min(8, branches))
    depth = max(1, min(5, depth))
    return branches, depth


@dataclass(frozen=True)
class AssociationDiagnostics:
    """Association traversal diagnostics."""

    branches_used: int
    depth_used: int
    traversed_edges: int
    expanded_nodes: int
    avg_branching_factor: float

