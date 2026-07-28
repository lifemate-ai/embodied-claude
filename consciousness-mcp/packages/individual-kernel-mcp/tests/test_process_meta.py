"""Higher-order records become consequential, and the processing gets recorded."""

from __future__ import annotations

import pytest

from individual_kernel_mcp.enacted_field import PrecisionState
from individual_kernel_mcp.process_meta import (
    HOR_PRECISION_CAP,
    HOR_PRECISION_GAIN,
    PrecisionBias,
    ProcessMetaRepresentation,
    apply_precision_bias,
    precision_bias_from_hors,
)


class _Record:
    """Minimal stand-in for HORRecord: only the two fields the bias reads."""

    def __init__(self, asserted_mode: str, confidence: float) -> None:
        self.asserted_mode = asserted_mode
        self.confidence = confidence


class TestBias:
    def test_no_records_is_no_bias(self):
        bias = precision_bias_from_hors([])
        assert bias.is_empty

    def test_attending_raises_the_self_model_channel(self):
        bias = precision_bias_from_hors([_Record("attending", 1.0)])
        assert bias.self_model == pytest.approx(HOR_PRECISION_GAIN)
        assert bias.extero == 0.0

    def test_each_mode_reaches_its_own_channel(self):
        assert precision_bias_from_hors([_Record("seeing", 1.0)]).extero > 0.0
        assert precision_bias_from_hors([_Record("feeling", 1.0)]).intero > 0.0
        assert precision_bias_from_hors([_Record("remembering", 1.0)]).mnemonic > 0.0

    def test_confidence_scales_the_contribution(self):
        weak = precision_bias_from_hors([_Record("attending", 0.2)])
        strong = precision_bias_from_hors([_Record("attending", 0.9)])
        assert weak.self_model < strong.self_model

    def test_repetition_cannot_exceed_the_cap(self):
        many = precision_bias_from_hors([_Record("attending", 1.0)] * 20)
        assert many.self_model == pytest.approx(HOR_PRECISION_CAP)

    def test_unknown_mode_contributes_nothing(self):
        assert precision_bias_from_hors([_Record("guessing", 1.0)]).is_empty


class TestApply:
    def test_empty_bias_returns_the_same_precision(self):
        base = PrecisionState(self_model=0.4)
        assert apply_precision_bias(base, PrecisionBias()) is base

    def test_bias_raises_the_channel(self):
        base = PrecisionState(self_model=0.4)
        raised = apply_precision_bias(base, PrecisionBias(self_model=0.15))
        assert raised.self_model == pytest.approx(0.55)
        assert raised.extero == base.extero

    def test_result_stays_inside_the_model_bounds(self):
        base = PrecisionState(self_model=0.95)
        raised = apply_precision_bias(base, PrecisionBias(self_model=0.2))
        assert raised.self_model == pytest.approx(1.0)


class TestCanonicalStatement:
    def _record(self, **overrides) -> ProcessMetaRepresentation:
        payload = {
            "process_meta_id": "pmr_test",
            "tick_id": "tick_test",
            "field_id": "field_test",
            "trigger_kind": "user_prompt",
            "candidate_count": 11,
            "competition_margin": 0.195,
            "competition_entropy": 0.92,
            "ignited": True,
            "conflicted": True,
            "attention_intensity": 0.58,
        }
        payload.update(overrides)
        return ProcessMetaRepresentation(**payload)

    def test_reports_the_processing_not_the_content(self):
        statement = self._record().canonical_statement()
        assert "11 candidates competed" in statement
        assert "settled" in statement
        assert "contested" in statement
        assert "0.195" in statement

    def test_a_quiet_tick_reads_differently(self):
        statement = self._record(ignited=False, conflicted=False).canonical_statement()
        assert "did not settle" in statement
        assert "uncontested" in statement
