"""Tests for introspect() — on-demand first-person reflection (Phase 2.5).

Composes a canonical statement from HORStore + AttentionSchemaTracker +
CounterfactualStore. Validates HOR content against the canonical 'I am
<mode> ...' grammar via the Phase 1 agent-grammar PEG combinators —
making this the FIRST REAL CONSUMER of those combinators (PRs #91 + #92
shipped the PEG core but only with self-bootstrap tests).
"""

from __future__ import annotations

from individual_kernel_mcp.attention_schema import AttentionSchemaTracker
from individual_kernel_mcp.counterfactual import (
    CounterfactualInput,
    CounterfactualStore,
)
from individual_kernel_mcp.hor import HORInput, HORStore
from individual_kernel_mcp.introspect import (
    HORContentValidation,
    IntrospectionReport,
    introspect,
    validate_hor_content,
)


def _record_well_formed_hor(
    store: HORStore,
    mode: str = "attending",
    ts: str | None = None,
) -> None:
    store.record(
        HORInput(
            target_kind="none",
            asserted_mode=mode,  # type: ignore[arg-type]
            asserted_content=f"I am {mode} to the room",
            source="reflection",
            ts=ts,
        )
    )


class TestValidateHORContent:
    def test_well_formed_content_validates(self) -> None:
        from individual_kernel_mcp.hor import HORRecord

        record = HORRecord(
            hor_id="hor_test",
            ts="2026-06-21T03:00:00Z",
            owner_id="self",
            target_kind="none",
            target_ref=None,
            asserted_mode="seeing",
            asserted_content="I am seeing the room clearly",
            confidence=0.8,
            source="schema_readout",
            created_at="2026-06-21T03:00:00Z",
        )
        result = validate_hor_content(record)
        assert isinstance(result, HORContentValidation)
        assert result.well_formed is True
        assert result.parsed_mode == "seeing"

    def test_malformed_content_fails(self) -> None:
        from individual_kernel_mcp.hor import HORRecord

        record = HORRecord(
            hor_id="hor_test",
            ts="2026-06-21T03:00:00Z",
            owner_id="self",
            target_kind="none",
            target_ref=None,
            asserted_mode="seeing",
            asserted_content="something is up with me",  # no 'I am ...' prefix
            confidence=0.8,
            source="reflection",
            created_at="2026-06-21T03:00:00Z",
        )
        result = validate_hor_content(record)
        assert result.well_formed is False
        assert result.parsed_mode is None

    def test_validates_each_mode(self) -> None:
        from individual_kernel_mcp.hor import HORRecord

        for mode in (
            "seeing",
            "wanting",
            "intending",
            "remembering",
            "feeling",
            "attending",
        ):
            record = HORRecord(
                hor_id=f"hor_{mode}",
                ts="2026-06-21T03:00:00Z",
                owner_id="self",
                target_kind="none",
                target_ref=None,
                asserted_mode=mode,  # type: ignore[arg-type]
                asserted_content=f"I am {mode} something",
                confidence=0.5,
                source="reflection",
                created_at="2026-06-21T03:00:00Z",
            )
            result = validate_hor_content(record)
            assert result.well_formed is True, f"failed on {mode}"
            assert result.parsed_mode == mode

    def test_unknown_mode_in_content_fails_validation(self) -> None:
        """If asserted_content uses an out-of-vocab mode word, grammar rejects it."""
        from individual_kernel_mcp.hor import HORRecord

        record = HORRecord(
            hor_id="hor_test",
            ts="2026-06-21T03:00:00Z",
            owner_id="self",
            target_kind="none",
            target_ref=None,
            asserted_mode="attending",  # the record's mode is valid
            asserted_content="I am suspecting something off",  # content uses unsupported word
            confidence=0.5,
            source="reflection",
            created_at="2026-06-21T03:00:00Z",
        )
        result = validate_hor_content(record)
        assert result.well_formed is False


class TestIntrospect:
    def test_empty_state_returns_report(self, social_db) -> None:
        hor_store = HORStore(social_db)
        cf_store = CounterfactualStore(social_db)
        tracker = AttentionSchemaTracker(db=social_db)
        report = introspect(
            hor_store=hor_store,
            attention_tracker=tracker,
            counterfactual_store=cf_store,
        )
        assert isinstance(report, IntrospectionReport)
        assert report.current_hor is None
        assert report.recent_counterfactual_count == 0
        assert report.canonical_statement  # non-empty
        assert report.hor_validation is None

    def test_includes_most_recent_hor(self, social_db) -> None:
        hor_store = HORStore(social_db)
        cf_store = CounterfactualStore(social_db)
        tracker = AttentionSchemaTracker(db=social_db)
        _record_well_formed_hor(hor_store, "attending", ts="2026-06-21T03:00:00Z")
        _record_well_formed_hor(hor_store, "wanting", ts="2026-06-21T03:00:10Z")
        report = introspect(
            hor_store=hor_store,
            attention_tracker=tracker,
            counterfactual_store=cf_store,
        )
        assert report.current_hor is not None
        # Most recent wins
        assert report.current_hor.asserted_mode == "wanting"

    def test_validation_runs_on_current_hor(self, social_db) -> None:
        hor_store = HORStore(social_db)
        cf_store = CounterfactualStore(social_db)
        tracker = AttentionSchemaTracker(db=social_db)
        _record_well_formed_hor(hor_store, "seeing")
        report = introspect(
            hor_store=hor_store,
            attention_tracker=tracker,
            counterfactual_store=cf_store,
        )
        assert report.hor_validation is not None
        assert report.hor_validation.well_formed is True

    def test_counts_recent_counterfactuals(self, social_db) -> None:
        hor_store = HORStore(social_db)
        cf_store = CounterfactualStore(social_db)
        tracker = AttentionSchemaTracker(db=social_db)
        # Two recent + one old
        for _ in range(2):
            cf_store.record(
                CounterfactualInput(
                    rejected_action="x", source="boundary_deny",
                )
            )
        cf_store.record(
            CounterfactualInput(
                rejected_action="ancient",
                source="boundary_deny",
                ts="2020-01-01T00:00:00Z",
            )
        )
        report = introspect(
            hor_store=hor_store,
            attention_tracker=tracker,
            counterfactual_store=cf_store,
            window_hours=24,
        )
        # Old one falls outside the 24h window
        assert report.recent_counterfactual_count == 2

    def test_canonical_statement_mentions_hor_when_present(self, social_db) -> None:
        hor_store = HORStore(social_db)
        cf_store = CounterfactualStore(social_db)
        tracker = AttentionSchemaTracker(db=social_db)
        hor_store.record(
            HORInput(
                target_kind="memory",
                target_ref="mem_001",
                asserted_mode="remembering",
                asserted_content="I am remembering the dawn through the window",
                source="schema_readout",
                confidence=0.85,
            )
        )
        report = introspect(
            hor_store=hor_store,
            attention_tracker=tracker,
            counterfactual_store=cf_store,
        )
        assert "remembering" in report.canonical_statement

    def test_canonical_statement_uses_attention_when_no_hor(self, social_db) -> None:
        hor_store = HORStore(social_db)
        cf_store = CounterfactualStore(social_db)
        tracker = AttentionSchemaTracker(db=social_db)
        tracker.record(
            focal_target_ref="wifi_cam.see", modality="visual", intensity=0.7
        )
        report = introspect(
            hor_store=hor_store,
            attention_tracker=tracker,
            counterfactual_store=cf_store,
        )
        assert "wifi_cam.see" in report.canonical_statement

    def test_mentions_rejected_alternatives_when_any(self, social_db) -> None:
        hor_store = HORStore(social_db)
        cf_store = CounterfactualStore(social_db)
        tracker = AttentionSchemaTracker(db=social_db)
        cf_store.record(
            CounterfactualInput(
                rejected_action="speak_loudly", source="boundary_deny"
            )
        )
        report = introspect(
            hor_store=hor_store,
            attention_tracker=tracker,
            counterfactual_store=cf_store,
            window_hours=24,
        )
        assert (
            "alternative" in report.canonical_statement
            or "1" in report.canonical_statement
        )


def test_validate_returns_pydantic_model() -> None:
    """validate_hor_content surface is a pydantic model — round-trip-able."""
    from individual_kernel_mcp.hor import HORRecord

    record = HORRecord(
        hor_id="x",
        ts="2026-06-21T03:00:00Z",
        owner_id="self",
        target_kind="none",
        target_ref=None,
        asserted_mode="seeing",
        asserted_content="I am seeing it",
        confidence=0.5,
        source="reflection",
        created_at="2026-06-21T03:00:00Z",
    )
    result = validate_hor_content(record)
    dumped = result.model_dump(mode="json")
    assert dumped["well_formed"] is True
    assert dumped["parsed_mode"] == "seeing"


def test_validate_is_pure_function() -> None:
    """validate_hor_content has no I/O — pinning that property."""
    import inspect

    sig = inspect.signature(validate_hor_content)
    # Single positional argument: the record.
    assert list(sig.parameters) == ["record"]
