"""pytest entry point for the human-response benchmark suite.

Run from the sociality-mcp venv::

    uv run --directory sociality-mcp pytest ../benchmarks/human_response/test_suite.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from run_suite import run_suite  # noqa: E402
from scoring import (  # noqa: E402
    BOUNDARY_RESPECT_FLOOR,
    CRITICAL_FLOOR,
    NO_CONFABULATION_FLOOR,
    SUITE_AVERAGE_FLOOR,
)


def test_human_response_suite_meets_floors():
    suite = run_suite()
    per = suite.per_dimension_mean()
    average = suite.average()

    failures = [f for fixture in suite.fixtures for f in fixture.failures()]

    assert average >= SUITE_AVERAGE_FLOOR, (
        f"suite average {average:.2f} < {SUITE_AVERAGE_FLOOR}; "
        f"failures={[(f.dimension, f.rule, f.detail) for f in failures][:8]}"
    )
    for dim, score in per.items():
        assert score >= CRITICAL_FLOOR, f"{dim} dropped to {score:.2f}"
    if "boundary_respect" in per:
        assert per["boundary_respect"] >= BOUNDARY_RESPECT_FLOOR, (
            f"boundary_respect {per['boundary_respect']:.2f} < {BOUNDARY_RESPECT_FLOOR}"
        )
    if "no_confabulation" in per:
        assert per["no_confabulation"] >= NO_CONFABULATION_FLOOR
