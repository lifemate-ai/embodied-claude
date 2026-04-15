"""Confidence and evidence utilities."""

from __future__ import annotations

from collections.abc import Iterable


def clamp01(value: float) -> float:
    """Clamp a value to the [0, 1] interval."""

    return max(0.0, min(1.0, value))


def weighted_average(values: Iterable[tuple[float, float]], fallback: float = 0.0) -> float:
    """Compute a weighted mean from (value, weight) pairs."""

    total_weight = 0.0
    total_value = 0.0
    for value, weight in values:
        if weight <= 0:
            continue
        total_value += value * weight
        total_weight += weight
    if total_weight == 0:
        return fallback
    return total_value / total_weight


def confidence_from_evidence(
    evidence_weights: Iterable[float],
    ambiguity_penalty: float = 0.0,
    floor: float = 0.15,
) -> float:
    """Convert evidence weights to a bounded confidence score."""

    clean_weights = [clamp01(weight) for weight in evidence_weights if weight > 0]
    if not clean_weights:
        return clamp01(floor * (1.0 - ambiguity_penalty))
    base = weighted_average(((weight, 1.0) for weight in clean_weights), fallback=floor)
    return clamp01((base * 0.85 + floor * 0.15) * (1.0 - clamp01(ambiguity_penalty)))
