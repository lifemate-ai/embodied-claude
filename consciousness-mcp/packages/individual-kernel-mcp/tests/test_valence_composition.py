"""Valence is composed from appetitive and aversive terms.

The legacy rule was `-0.6 * max_discomfort`, whose upper bound is zero: no
outcome could ever produce a positive body state. These tests pin the replacement
contract, above all that a well-predicted, controllable, improving context does
produce positive valence.
"""

from __future__ import annotations

import pytest

from individual_kernel_mcp.allostasis import AllostaticVariable, compose_valence


def _compose(**overrides: float) -> AllostaticVariable:
    base: dict[str, float] = {
        "expected_valence_delta": 0.0,
        "mean_discomfort": 0.0,
        "controllability": 0.5,
        "uncertainty": 0.5,
        "unresolved_error": 0.5,
    }
    base.update(overrides)
    return compose_valence(**base)


class TestValenceRange:
    def test_valence_stays_within_bounds(self) -> None:
        best = _compose(
            expected_valence_delta=1.0,
            mean_discomfort=0.0,
            controllability=1.0,
            uncertainty=0.0,
            unresolved_error=0.0,
        )
        worst = _compose(
            expected_valence_delta=-1.0,
            mean_discomfort=1.0,
            controllability=0.0,
            uncertainty=1.0,
            unresolved_error=1.0,
        )
        assert -1.0 <= worst.valence <= best.valence <= 1.0

    def test_good_context_produces_positive_valence(self) -> None:
        """The property the legacy formula could not express."""
        state = _compose(
            expected_valence_delta=0.5,
            mean_discomfort=0.05,
            controllability=0.9,
            uncertainty=0.1,
            unresolved_error=0.1,
        )
        assert state.valence > 0.0

    def test_bad_context_produces_negative_valence(self) -> None:
        state = _compose(
            expected_valence_delta=-0.3,
            mean_discomfort=0.8,
            controllability=0.1,
            uncertainty=0.9,
            unresolved_error=0.9,
        )
        assert state.valence < 0.0


class TestMonotonicity:
    def test_more_expected_improvement_raises_valence(self) -> None:
        low = _compose(expected_valence_delta=0.0)
        high = _compose(expected_valence_delta=0.6)
        assert high.valence > low.valence

    def test_more_discomfort_lowers_valence(self) -> None:
        low = _compose(mean_discomfort=0.1)
        high = _compose(mean_discomfort=0.9)
        assert high.valence < low.valence

    def test_control_raises_valence_when_improvement_is_expected(self) -> None:
        weak = _compose(expected_valence_delta=0.5, controllability=0.1)
        strong = _compose(expected_valence_delta=0.5, controllability=0.9)
        assert strong.valence > weak.valence

    def test_unresolved_error_lowers_valence(self) -> None:
        settled = _compose(unresolved_error=0.0)
        surprised = _compose(unresolved_error=1.0)
        assert surprised.valence < settled.valence


class TestComponentsAreInspectable:
    def test_terms_are_reported_separately(self) -> None:
        state = _compose(expected_valence_delta=0.4, mean_discomfort=0.2)
        assert state.appetitive >= 0.0
        assert state.aversive >= 0.0
        assert state.valence == pytest.approx(state.appetitive - state.aversive, abs=1e-9)

    def test_snapshot_round_trips_as_plain_json_types(self) -> None:
        snapshot = _compose(expected_valence_delta=0.4).as_snapshot()
        assert set(snapshot) >= {"valence", "appetitive", "aversive", "controllability"}
        assert all(isinstance(value, float) for value in snapshot.values())
