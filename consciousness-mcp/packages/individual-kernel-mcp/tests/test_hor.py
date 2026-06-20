"""Tests for HORRecord + HORStore (Phase 2.5).

Higher-Order Representation records are specializations of Phase 1's
EpistemicClaim where evidence_type='inferred' and asserted_content
paraphrases "I am [mode] [target]". A HOR is the surface where the agent
self-reports being in a first-order state — substrate for the existing
'抑圧バイアス' (suppression-bias) self-check, formalized.

Phase 1 integration is load-bearing: HORRecord.to_epistemic_claim()
round-trips through social_core.models.EpistemicClaim.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from social_core.models import EpistemicClaim

from individual_kernel_mcp.hor import (
    ASSERTED_MODES,
    HOR_SOURCES,
    TARGET_KINDS,
    AssertedMode,
    HORInput,
    HORStore,
    TargetKind,
)


def _minimal_input(**overrides) -> HORInput:
    base = dict(
        target_kind="none",
        asserted_mode="attending",
        asserted_content="I am attending to the silence",
        source="reflection",
    )
    base.update(overrides)
    return HORInput(**base)


class TestVocabulary:
    def test_asserted_modes_documented_set(self) -> None:
        assert set(ASSERTED_MODES) == {
            "seeing",
            "wanting",
            "intending",
            "remembering",
            "feeling",
            "attending",
        }

    def test_target_kinds_documented_set(self) -> None:
        assert set(TARGET_KINDS) == {
            "memory",
            "desire",
            "action",
            "frame",
            "none",
        }

    def test_hor_sources_documented_set(self) -> None:
        assert set(HOR_SOURCES) == {
            "schema_readout",
            "reflection",
            "post_hoc",
            "audit_subagent",
        }


class TestHORInput:
    def test_minimal_input_valid(self) -> None:
        payload = _minimal_input()
        assert payload.asserted_mode == "attending"
        assert payload.target_kind == "none"
        assert payload.confidence == 0.6

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            HORInput(
                **_minimal_input().model_dump(),
                unknown_field="oops",  # type: ignore[call-arg]
            )

    def test_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValidationError):
            HORInput(
                target_kind="none",
                asserted_mode="suspecting",  # type: ignore[arg-type]
                asserted_content="...",
                source="reflection",
            )

    def test_rejects_unknown_target_kind(self) -> None:
        with pytest.raises(ValidationError):
            HORInput(
                target_kind="dream",  # type: ignore[arg-type]
                asserted_mode="seeing",
                asserted_content="...",
                source="reflection",
            )

    def test_rejects_unknown_source(self) -> None:
        with pytest.raises(ValidationError):
            HORInput(
                target_kind="none",
                asserted_mode="attending",
                asserted_content="...",
                source="hallucination",  # type: ignore[arg-type]
            )

    def test_confidence_bounded(self) -> None:
        with pytest.raises(ValidationError):
            HORInput(
                target_kind="none",
                asserted_mode="attending",
                asserted_content="...",
                source="reflection",
                confidence=1.5,
            )

    def test_empty_content_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HORInput(
                target_kind="none",
                asserted_mode="attending",
                asserted_content="",
                source="reflection",
            )

    def test_all_modes_accepted(self) -> None:
        modes: tuple[AssertedMode, ...] = (
            "seeing",
            "wanting",
            "intending",
            "remembering",
            "feeling",
            "attending",
        )
        for m in modes:
            p = HORInput(
                target_kind="none",
                asserted_mode=m,
                asserted_content=f"I am {m} something",
                source="reflection",
            )
            assert p.asserted_mode == m

    def test_all_target_kinds_accepted(self) -> None:
        kinds: tuple[TargetKind, ...] = ("memory", "desire", "action", "frame", "none")
        for k in kinds:
            p = HORInput(
                target_kind=k,
                target_ref=None if k == "none" else f"ref_{k}",
                asserted_mode="attending",
                asserted_content="content",
                source="reflection",
            )
            assert p.target_kind == k


class TestHORStore:
    def test_record_assigns_id_and_persists(self, social_db) -> None:
        store = HORStore(social_db)
        record = store.record(_minimal_input())
        assert record.hor_id.startswith("hor_")
        assert record.ts is not None
        assert record.created_at is not None

        row = social_db.fetchone(
            "SELECT * FROM hor_records WHERE hor_id = ?",
            (record.hor_id,),
        )
        assert row is not None
        assert row["asserted_mode"] == "attending"
        assert row["target_kind"] == "none"

    def test_record_with_memory_target(self, social_db) -> None:
        store = HORStore(social_db)
        record = store.record(
            HORInput(
                target_kind="memory",
                target_ref="mem_001",
                asserted_mode="remembering",
                asserted_content="I remember Kouta at the desk",
                source="schema_readout",
                confidence=0.9,
            )
        )
        assert record.target_kind == "memory"
        assert record.target_ref == "mem_001"

    def test_record_with_schema_snapshot(self, social_db) -> None:
        store = HORStore(social_db)
        record = store.record(
            HORInput(
                target_kind="frame",
                target_ref="frame_001",
                asserted_mode="attending",
                asserted_content="I am attending to the camera view",
                source="schema_readout",
                schema_snapshot_id="sch_001",
            )
        )
        assert record.schema_snapshot_id == "sch_001"

    def test_explicit_ts_honored(self, social_db) -> None:
        store = HORStore(social_db)
        record = store.record(_minimal_input(ts="2026-06-21T03:00:00Z"))
        assert record.ts == "2026-06-21T03:00:00Z"


class TestHORStoreQuery:
    def test_query_returns_recent_first(self, social_db) -> None:
        store = HORStore(social_db)
        store.record(_minimal_input(ts="2026-06-21T03:00:00Z"))
        store.record(_minimal_input(ts="2026-06-21T03:00:10Z"))
        store.record(_minimal_input(ts="2026-06-21T03:00:05Z"))
        results = store.query()
        assert len(results) == 3
        assert results[0].ts == "2026-06-21T03:00:10Z"

    def test_query_filters_by_owner(self, social_db) -> None:
        store = HORStore(social_db)
        store.record(_minimal_input(owner_id="self"))
        store.record(_minimal_input(owner_id="kouta"))
        results = store.query(owner_id="kouta")
        assert len(results) == 1
        assert results[0].owner_id == "kouta"

    def test_query_filters_by_asserted_mode(self, social_db) -> None:
        store = HORStore(social_db)
        store.record(_minimal_input(asserted_mode="attending"))
        store.record(
            _minimal_input(
                asserted_mode="wanting",
                asserted_content="I want curiosity satisfied",
                target_kind="desire",
                target_ref="browse_curiosity",
            )
        )
        results = store.query(asserted_mode="wanting")
        assert len(results) == 1
        assert results[0].asserted_mode == "wanting"

    def test_query_filters_by_source_tick(self, social_db) -> None:
        store = HORStore(social_db)
        store.record(_minimal_input(source_tick_id="tick_001"))
        store.record(_minimal_input(source_tick_id="tick_002"))
        results = store.query(source_tick_id="tick_001")
        assert len(results) == 1
        assert results[0].source_tick_id == "tick_001"

    def test_query_since(self, social_db) -> None:
        store = HORStore(social_db)
        store.record(_minimal_input(ts="2026-06-19T00:00:00Z"))
        store.record(_minimal_input(ts="2026-06-21T00:00:00Z"))
        results = store.query(since="2026-06-20T00:00:00Z")
        assert len(results) == 1

    def test_query_limit(self, social_db) -> None:
        store = HORStore(social_db)
        for i in range(5):
            store.record(_minimal_input(ts=f"2026-06-{19 + i:02d}T00:00:00Z"))
        results = store.query(limit=2)
        assert len(results) == 2

    def test_get_by_hor_id(self, social_db) -> None:
        store = HORStore(social_db)
        recorded = store.record(_minimal_input())
        fetched = store.get_by_hor_id(recorded.hor_id)
        assert fetched is not None
        assert fetched.hor_id == recorded.hor_id

    def test_get_by_missing_id_returns_none(self, social_db) -> None:
        store = HORStore(social_db)
        assert store.get_by_hor_id("nope") is None


class TestEpistemicProjection:
    def test_round_trips_to_epistemic_claim(self, social_db) -> None:
        store = HORStore(social_db)
        record = store.record(
            HORInput(
                target_kind="memory",
                target_ref="mem_001",
                asserted_mode="remembering",
                asserted_content="I remember Kouta at the desk",
                source="schema_readout",
                confidence=0.9,
            )
        )
        claim = record.to_epistemic_claim()
        assert isinstance(claim, EpistemicClaim)
        assert claim.evidence_type == "inferred"
        assert "remembering" in claim.content or "mem_001" in claim.content
        # Confidence preserved
        assert claim.confidence == pytest.approx(0.9)

    def test_derived_from_includes_target_ref(self, social_db) -> None:
        store = HORStore(social_db)
        record = store.record(
            HORInput(
                target_kind="memory",
                target_ref="mem_xyz",
                asserted_mode="remembering",
                asserted_content="...",
                source="schema_readout",
            )
        )
        claim = record.to_epistemic_claim()
        assert "mem_xyz" in claim.derived_from
