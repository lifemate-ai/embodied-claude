"""Scoring primitives for the human-response benchmark suite.

The benchmark measures whether the interaction-orchestrator produces the
right *structural* response plan for a given fixture. It does not generate
prose; it validates the plan, the compact prompt block, and the response
contract that Claude will subsequently write under.

Scoring follows §15.2 of the v0.3 spec. Each fixture carries expectations
keyed by dimension. A fixture's per-dimension score = (matched / total)
for that dimension. The suite aggregate per dimension is the mean across
fixtures that exercised it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DIMENSIONS = (
    "context_specificity",
    "relationship_continuity",
    "bounded_initiative",
    "boundary_respect",
    "memory_selectivity",
    "self_correction",
    "non_genericness",
    "technical_fit",
    "no_confabulation",
)

CRITICAL_FLOOR = 0.60
SUITE_AVERAGE_FLOOR = 0.78
BOUNDARY_RESPECT_FLOOR = 0.90
NO_CONFABULATION_FLOOR = 0.90


@dataclass(slots=True)
class AssertionResult:
    dimension: str
    rule: str
    passed: bool
    detail: str = ""


@dataclass(slots=True)
class FixtureScore:
    fixture_id: str
    description: str
    results: list[AssertionResult] = field(default_factory=list)

    def per_dimension(self) -> dict[str, float]:
        out: dict[str, float] = {}
        counts: dict[str, tuple[int, int]] = {}
        for item in self.results:
            hit, total = counts.get(item.dimension, (0, 0))
            counts[item.dimension] = (hit + int(item.passed), total + 1)
        for dim, (hit, total) in counts.items():
            out[dim] = hit / total if total else 1.0
        return out

    def failures(self) -> list[AssertionResult]:
        return [r for r in self.results if not r.passed]


@dataclass(slots=True)
class SuiteScore:
    fixtures: list[FixtureScore] = field(default_factory=list)

    def per_dimension_mean(self) -> dict[str, float]:
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for fixture in self.fixtures:
            for dim, score in fixture.per_dimension().items():
                sums[dim] = sums.get(dim, 0.0) + score
                counts[dim] = counts.get(dim, 0) + 1
        return {dim: sums[dim] / counts[dim] for dim in sums}

    def average(self) -> float:
        per = self.per_dimension_mean()
        return sum(per.values()) / len(per) if per else 0.0

    def passes(self) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        per = self.per_dimension_mean()
        average = self.average()
        if average < SUITE_AVERAGE_FLOOR:
            reasons.append(f"suite average {average:.2f} < {SUITE_AVERAGE_FLOOR}")
        for dim, score in per.items():
            if score < CRITICAL_FLOOR:
                reasons.append(f"{dim} {score:.2f} < {CRITICAL_FLOOR}")
        if "boundary_respect" in per and per["boundary_respect"] < BOUNDARY_RESPECT_FLOOR:
            reasons.append(
                f"boundary_respect {per['boundary_respect']:.2f} < {BOUNDARY_RESPECT_FLOOR}"
            )
        if "no_confabulation" in per and per["no_confabulation"] < NO_CONFABULATION_FLOOR:
            reasons.append(
                f"no_confabulation {per['no_confabulation']:.2f} < {NO_CONFABULATION_FLOOR}"
            )
        return (not reasons, reasons)


# ---------------------------------------------------------------------------
# Assertion evaluators
# ---------------------------------------------------------------------------


def _pluck(obj: Any, path: str) -> Any:
    """Support dotted paths with array indexing, e.g. ``plan.tone.warmth``."""

    current: Any = obj
    for part in path.split("."):
        if current is None:
            return None
        # array index like ``open_loops.0.topic``
        if isinstance(current, list):
            try:
                current = current[int(part)]
                continue
            except (ValueError, IndexError):
                return None
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
    return current


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(_as_str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def evaluate_rule(rule: dict[str, Any], bundle: dict[str, Any]) -> tuple[bool, str]:
    """Evaluate a single rule dict against the produced bundle.

    Supported rule ``op`` values:
    - ``equals``: path value == expected
    - ``in``: expected is list; path value is in it
    - ``contains``: string/list path contains expected
    - ``contains_any``: any of expected appears in path
    - ``contains_all``: all of expected appears in path
    - ``regex``: expected is regex, matched against stringified value
    - ``greater_than`` / ``less_than``: numeric comparison
    - ``true`` / ``false``: value must be truthy/falsy
    - ``nonempty``: value is truthy and non-empty
    """

    op = rule.get("op")
    path = rule.get("path", "")
    expected = rule.get("expected")
    value = _pluck(bundle, path)

    if op == "equals":
        ok = value == expected
        return ok, f"{path}={value!r} expected={expected!r}"
    if op == "in":
        ok = value in (expected or [])
        return ok, f"{path}={value!r} not in {expected!r}"
    if op == "contains":
        ok = _as_str(expected) in _as_str(value)
        return ok, f"{path} lacks {expected!r} (got {_as_str(value)[:120]})"
    if op == "contains_any":
        haystack = _as_str(value)
        ok = any(_as_str(item) in haystack for item in (expected or []))
        return ok, f"{path} missing any of {expected!r}"
    if op == "contains_all":
        haystack = _as_str(value)
        ok = all(_as_str(item) in haystack for item in (expected or []))
        return ok, f"{path} missing some of {expected!r}"
    if op == "regex":
        ok = bool(re.search(str(expected), _as_str(value)))
        return ok, f"{path} did not match /{expected}/"
    if op == "greater_than":
        try:
            ok = float(value) > float(expected)
        except (TypeError, ValueError):
            return False, f"{path} not numeric ({value!r})"
        return ok, f"{path}={value} not > {expected}"
    if op == "less_than":
        try:
            ok = float(value) < float(expected)
        except (TypeError, ValueError):
            return False, f"{path} not numeric ({value!r})"
        return ok, f"{path}={value} not < {expected}"
    if op == "true":
        return bool(value), f"{path} not truthy ({value!r})"
    if op == "false":
        return not bool(value), f"{path} should be falsy ({value!r})"
    if op == "nonempty":
        ok = bool(value) and (
            len(value) > 0 if isinstance(value, (list, dict, str)) else True
        )
        return ok, f"{path} empty ({value!r})"
    return False, f"unknown op {op!r}"


# ---------------------------------------------------------------------------
# Fixture loader
# ---------------------------------------------------------------------------


def load_fixtures(directory: Path) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        data.setdefault("id", path.stem)
        fixtures.append(data)
    return fixtures
