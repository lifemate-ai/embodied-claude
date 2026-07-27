"""Affect modulates what reaches the field, and never what is permitted.

Valence was previously computed and then ignored. These tests pin the shape of
its influence: it scales competition weights, so a bad mood narrows attention
onto needs and a good one widens it toward exploration. The boundary gate is
deliberately excluded — what is allowed must not depend on how it feels.
"""

from __future__ import annotations

import pytest

from individual_kernel_mcp.valence_coupling import (
    AffectState,
    modulate_weights,
)
from individual_kernel_mcp.workspace import CompetitionWeights

BASE = CompetitionWeights()
NEUTRAL = AffectState(valence=0.0, arousal=0.0)


class TestNeutralAffectChangesNothing:
    def test_zero_valence_reproduces_the_base_weights(self) -> None:
        assert modulate_weights(BASE, NEUTRAL) == BASE

    def test_arousal_alone_does_not_change_weights(self) -> None:
        loud = AffectState(valence=0.0, arousal=1.0)
        assert modulate_weights(BASE, loud) == BASE


class TestNarrowingAndWidening:
    def test_negative_valence_raises_need_and_lowers_information(self) -> None:
        bad = modulate_weights(BASE, AffectState(valence=-0.8, arousal=0.5))
        assert bad.need > BASE.need
        assert bad.information < BASE.information

    def test_positive_valence_raises_information_without_raising_need(self) -> None:
        good = modulate_weights(BASE, AffectState(valence=0.8, arousal=0.5))
        assert good.information > BASE.information
        assert good.need == pytest.approx(BASE.need)

    def test_modulation_is_monotonic_in_valence(self) -> None:
        weights = [
            modulate_weights(BASE, AffectState(valence=v, arousal=0.0)).information
            for v in (-1.0, -0.5, 0.0, 0.5, 1.0)
        ]
        assert weights == sorted(weights)

    def test_weights_never_go_negative(self) -> None:
        worst = modulate_weights(BASE, AffectState(valence=-1.0, arousal=1.0))
        assert all(
            getattr(worst, name) >= 0.0
            for name in ("surprise", "need", "goal", "information", "control")
        )

    def test_thresholds_are_left_alone(self) -> None:
        """Ignition and conflict thresholds are structural, not affective."""
        shifted = modulate_weights(BASE, AffectState(valence=-1.0, arousal=1.0))
        assert shifted.ignition_threshold == BASE.ignition_threshold
        assert shifted.conflict_margin == BASE.conflict_margin
        assert shifted.entropy_conflict_threshold == BASE.entropy_conflict_threshold


class TestAffectStateBounds:
    def test_out_of_range_valence_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            AffectState(valence=1.5, arousal=0.0)

    def test_out_of_range_arousal_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            AffectState(valence=0.0, arousal=-0.2)
