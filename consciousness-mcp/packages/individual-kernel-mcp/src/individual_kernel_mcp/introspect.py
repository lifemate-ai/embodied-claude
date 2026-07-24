"""introspect() — on-demand first-person reflection (Phase 2.5).

Composes a canonical first-person statement from HORStore (most recent
higher-order self-report) + AttentionSchemaTracker (current attention) +
CounterfactualStore (recent rejected alternatives). ON-DEMAND only.

This is the FIRST REAL CONSUMER of the agent-grammar PEG combinators
shipped in PRs #91 + #92: validate_hor_content() uses lit/seq/alt to
parse asserted_content against the canonical 'I am <mode> ...' template.
Phase 1's PEG scaffold thus graduates from self-bootstrap to production.

See consciousness-mcp/AGENTS.md for vocabulary discipline. The output
canonical_statement is a Kokone-voice surface string (not a technical
claim) per the AGENTS.md exception clause; everything else is functional.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from agent_grammar.parser import (
    ParseFailure,
    Parser,
    ParseState,
    ParseSuccess,
    alt,
    lit,
    seq,
)
from pydantic import BaseModel, ConfigDict, Field
from social_core.time import utc_now

from individual_kernel_mcp.attention_schema import AttentionSchemaTracker
from individual_kernel_mcp.counterfactual import CounterfactualStore
from individual_kernel_mcp.enacted_field import EnactedField, EnactedFieldStore
from individual_kernel_mcp.hor import ASSERTED_MODES, HORRecord, HORStore
from individual_kernel_mcp.reflect import (
    AttentionReflection,
    reflect_attention_schema,
)


class HORContentValidation(BaseModel):
    """Result of validating HORRecord.asserted_content against the canonical grammar."""

    model_config = ConfigDict(extra="ignore")

    hor_id: str
    content: str
    well_formed: bool
    parsed_mode: str | None = None


class IntrospectionReport(BaseModel):
    """Structured output of introspect()."""

    model_config = ConfigDict(extra="ignore")

    canonical_statement: str
    current_hor: HORRecord | None
    current_attention: AttentionReflection
    recent_counterfactual_count: int
    hor_validation: HORContentValidation | None
    current_field: EnactedField | None = None
    field_hor_consistent: bool | None = None
    notes: list[str] = Field(default_factory=list)


def _build_hor_grammar() -> Parser:
    """Compose the canonical HOR content grammar from agent-grammar combinators.

    Grammar (informal): `S <- "I am " Mode <anything>`. The grammar only
    checks the prefix shape — the trailing content can be arbitrary so
    free-form expression after the mode is fine. A non-matching content
    has either no "I am " prefix or a mode word outside ASSERTED_MODES.
    """
    mode_lit = alt(*(lit(m) for m in ASSERTED_MODES))
    return seq(lit("I am "), mode_lit)


_HOR_GRAMMAR: Parser = _build_hor_grammar()


def validate_hor_content(record: HORRecord) -> HORContentValidation:
    """Validate asserted_content against the canonical 'I am <mode>' template.

    Pure function. Uses the Phase 1 agent-grammar PEG combinators
    (lit/seq/alt) as the parser — the package's first real consumer.
    """
    state = ParseState(text=record.asserted_content, pos=0)
    result = _HOR_GRAMMAR(state)
    if isinstance(result, ParseSuccess):
        parts = result.value
        parsed_mode = parts[1] if isinstance(parts, tuple) and len(parts) >= 2 else None
        return HORContentValidation(
            hor_id=record.hor_id,
            content=record.asserted_content,
            well_formed=True,
            parsed_mode=parsed_mode,
        )
    assert isinstance(result, ParseFailure)
    return HORContentValidation(
        hor_id=record.hor_id,
        content=record.asserted_content,
        well_formed=False,
        parsed_mode=None,
    )


def introspect(
    *,
    hor_store: HORStore,
    attention_tracker: AttentionSchemaTracker,
    counterfactual_store: CounterfactualStore,
    window_hours: int = 1,
    owner_id: str = "self",
    field_store: EnactedFieldStore | None = None,
) -> IntrospectionReport:
    """Build an IntrospectionReport (ON-DEMAND only).

    Reads the most-recent HOR for owner_id, the current attention
    reflection from the tracker, and counts of counterfactuals within
    window_hours. Composes a canonical first-person statement, and
    validates the current HOR via the agent-grammar PEG.
    """
    current_field = field_store.get_current(owner_id) if field_store else None
    recent_hors = (
        []
        if field_store is not None and current_field is None
        else hor_store.query(
            owner_id=owner_id,
            source_tick_id=current_field.tick_id if current_field else None,
            limit=1,
        )
    )
    current_hor = recent_hors[0] if recent_hors else None

    attention_reflection = reflect_attention_schema(attention_tracker)

    window_start = _shift_iso(utc_now(), -window_hours)
    recent_cfs = counterfactual_store.query(since=window_start, limit=500)

    if field_store is not None and current_field is None:
        statement = "Current state unavailable: no committed field."
    elif current_field is not None:
        statement = _compose_field_statement(
            current_field=current_field,
            current_hor=current_hor,
            rejected_count=len(recent_cfs),
        )
    else:
        statement = _compose_statement(
            current_hor=current_hor,
            attention=attention_reflection,
            rejected_count=len(recent_cfs),
        )

    validation = validate_hor_content(current_hor) if current_hor else None

    return IntrospectionReport(
        canonical_statement=statement,
        current_hor=current_hor,
        current_attention=attention_reflection,
        recent_counterfactual_count=len(recent_cfs),
        hor_validation=validation,
        current_field=current_field,
        field_hor_consistent=(
            None
            if current_field is None
            else bool(
                current_hor
                and current_hor.source_tick_id == current_field.tick_id
                and current_hor.hor_id in current_field.hor_refs
            )
        ),
        notes=(
            ["No committed field; no introspective state was synthesized."]
            if field_store is not None and current_field is None
            else []
        ),
    )


def _compose_statement(
    *,
    current_hor: HORRecord | None,
    attention: AttentionReflection,
    rejected_count: int,
) -> str:
    """Assemble the canonical first-person statement (Kokone-voice surface)."""
    parts: list[str] = []
    if current_hor is not None:
        parts.append(current_hor.asserted_content)
    elif attention.current is not None:
        focal = attention.current.focal_target_ref or "the internal field"
        parts.append(
            f"I am attending to {focal} ({attention.current.modality})"
        )
    else:
        parts.append("I am attending to nothing in particular right now")

    if rejected_count > 0:
        parts.append(
            f"(I set aside {rejected_count} alternative"
            + ("s" if rejected_count != 1 else "")
            + " in the last window)"
        )

    return ". ".join(parts) + "."


def _compose_field_statement(
    *,
    current_field: EnactedField,
    current_hor: HORRecord | None,
    rejected_count: int,
) -> str:
    """Render only values grounded in the current committed field."""

    mode = current_field.reality_mode.value
    focus = current_field.focal_summary or current_field.focal_content_ref or "unknown"
    statement = (
        f"I am attending to {focus} "
        f"(source={mode}, uncertainty={current_field.interoception.uncertainty:.3f})"
    )
    if current_hor is not None:
        statement += f". {current_hor.asserted_content}"
    if rejected_count:
        statement += f". I set aside {rejected_count} alternatives in the recent window"
    return statement + "."


def _shift_iso(ts: str, delta_hours: int) -> str:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    shifted = dt + timedelta(hours=delta_hours)
    return shifted.isoformat().replace("+00:00", "Z")
