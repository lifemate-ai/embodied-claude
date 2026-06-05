"""Primitive types and constructors used by grammar rules.

Re-exports EpistemicClaim and EvidenceType from social-core so grammar code
imports a single namespace, and provides small constructors so rule actions
can produce claims without verbose kwargs.
"""

from __future__ import annotations

from social_core.models import EpistemicClaim, EvidenceType

__all__ = [
    "EpistemicClaim",
    "EvidenceType",
    "derive",
    "observe",
]


def observe(content: str, *, source: str, confidence: float = 1.0) -> EpistemicClaim:
    """Build an Observed claim. Use for direct sensor or tool results."""
    return EpistemicClaim(
        content=content,
        evidence_type="observed",
        source=source,
        confidence=confidence,
    )


def derive(
    content: str,
    *,
    from_claims: list[EpistemicClaim],
    confidence: float = 0.6,
) -> EpistemicClaim:
    """Build an Inferred claim from upstream observations.

    Confidence defaults to 0.6 to discourage over-claiming on derived facts.
    derived_from is populated from the upstream claim contents when no
    explicit IDs are tracked.
    """
    derived_ids = [c.source or c.content for c in from_claims]
    return EpistemicClaim(
        content=content,
        evidence_type="inferred",
        derived_from=derived_ids,
        confidence=confidence,
    )
