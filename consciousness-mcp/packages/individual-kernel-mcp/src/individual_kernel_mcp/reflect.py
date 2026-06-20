"""reflect_attention_schema — on-demand reflection over the attention tracker.

ON-DEMAND ONLY. Iterates the tracker's ring buffer (and optionally pulls
extra history from SQLite); too expensive for per-tick invocation. Phase
2.5's introspect() will use this as one of its substrates.

Returns an AttentionReflection summarizing modality distribution, focus
changes, dwell aggregates, and the dominant focal target. The single
canonical first-person read-out string is constructed in introspect()
(Phase 2.5) — this function returns structured data only.

See consciousness-mcp/AGENTS.md for vocabulary discipline.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field
from social_core.db import SocialDB

from individual_kernel_mcp.attention_schema import (
    AttentionSchema,
    AttentionSchemaTracker,
    Modality,
)


class ModalityDwell(BaseModel):
    """Dwell statistics for a single modality over the reflection window."""

    model_config = ConfigDict(extra="ignore")

    modality: Modality
    average_dwell_seconds: float
    max_dwell_seconds: float
    count: int


class AttentionReflection(BaseModel):
    """Structured output of reflect_attention_schema.

    `current` is the most-recent schema in the considered samples.
    `modality_distribution[m]` is the fraction of samples in modality m
    (sums to 1.0 when non-empty). `focus_changes` counts transitions in
    focal_target_ref across the chronologically-ordered window.
    """

    model_config = ConfigDict(extra="ignore")

    owner_id: str
    current: AttentionSchema | None
    window_size: int
    modality_distribution: dict[Modality, float] = Field(default_factory=dict)
    modality_dwells: list[ModalityDwell] = Field(default_factory=list)
    focus_changes: int
    dominant_focal: str | None


def reflect_attention_schema(
    tracker: AttentionSchemaTracker,
    *,
    db: SocialDB | None = None,
    extra_history: int = 0,
) -> AttentionReflection:
    """Build an AttentionReflection over the tracker's buffer + optional history.

    db must be passed when extra_history > 0 (otherwise no historical
    schemas are pulled). Historical samples are deduped by schema_id.
    """
    samples: list[AttentionSchema] = list(tracker.buffer)

    effective_db = db if db is not None else tracker.db
    if effective_db is not None and extra_history > 0:
        rows = effective_db.fetchall(
            """
            SELECT schema_id, ts, owner_id, focal_target_ref, modality,
                   intensity, dwell_seconds, predicted_next_focus,
                   control_handle, source_tick_id
            FROM attention_schemas
            WHERE owner_id = ?
            ORDER BY ts DESC
            LIMIT ?
            """,
            (tracker.owner_id, extra_history),
        )
        seen = {s.schema_id for s in samples}
        for row in rows:
            d = dict(row)
            if d["schema_id"] in seen:
                continue
            samples.append(AttentionSchema(**d))

    if not samples:
        return AttentionReflection(
            owner_id=tracker.owner_id,
            current=None,
            window_size=0,
            modality_distribution={},
            modality_dwells=[],
            focus_changes=0,
            dominant_focal=None,
        )

    current = max(samples, key=lambda s: s.ts)

    modality_counts: Counter[str] = Counter(s.modality for s in samples)
    total = sum(modality_counts.values())
    distribution: dict[Modality, float] = {
        m: count / total for m, count in modality_counts.items()  # type: ignore[misc]
    }

    by_modality: dict[Modality, list[AttentionSchema]] = {}
    for s in samples:
        by_modality.setdefault(s.modality, []).append(s)
    modality_dwells: list[ModalityDwell] = []
    for m, ss in by_modality.items():
        dwells = [s.dwell_seconds for s in ss]
        modality_dwells.append(
            ModalityDwell(
                modality=m,
                average_dwell_seconds=sum(dwells) / len(dwells),
                max_dwell_seconds=max(dwells),
                count=len(ss),
            )
        )

    ordered = sorted(samples, key=lambda s: s.ts)
    focus_changes = 0
    for prev, curr in zip(ordered, ordered[1:], strict=False):
        if prev.focal_target_ref != curr.focal_target_ref:
            focus_changes += 1

    focal_counts: Counter[str] = Counter(
        s.focal_target_ref for s in samples if s.focal_target_ref
    )
    dominant = focal_counts.most_common(1)[0][0] if focal_counts else None

    return AttentionReflection(
        owner_id=tracker.owner_id,
        current=current,
        window_size=len(samples),
        modality_distribution=distribution,
        modality_dwells=modality_dwells,
        focus_changes=focus_changes,
        dominant_focal=dominant,
    )
