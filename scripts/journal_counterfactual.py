#!/usr/bin/env python3
"""Append a counterfactual entry to the self-journal.

A counterfactual records: "I wanted X, but chose Y because Z."
This creates a record of what was NOT done, which is as important
as what was done for building self-continuity.

Usage:
    journal_counterfactual.py \\
        --wanted "start a conversation" \\
        --chose  "wait silently" \\
        --why    "the user appeared to be in deep focus based on recent signals" \\
        [--trigger "miss_companion"] \\
        [--person-id "<person-id>"] \\
        [--regret 0.2]
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path


DEFAULT_LOG_PATH = (
    Path.home() / ".claude" / "memories" / "counterfactuals.jsonl"
)


def append_counterfactual(
    *,
    wanted: str,
    chose: str,
    why: str,
    trigger: str | None = None,
    person_id: str | None = None,
    regret: float | None = None,
    log_path: Path = DEFAULT_LOG_PATH,
) -> dict:
    entry = {
        "id": f"cf_{uuid.uuid4().hex[:8]}",
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "wanted": wanted,
        "chose": chose,
        "why": why,
        "trigger": trigger,
        "person_id": person_id,
        "regret": regret,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record a counterfactual (wanted vs chose)."
    )
    parser.add_argument("--wanted", required=True, help="What you wanted to do")
    parser.add_argument("--chose", required=True, help="What you actually chose")
    parser.add_argument("--why", required=True, help="Reason for the choice")
    parser.add_argument(
        "--trigger",
        default=None,
        help="Desire / event that prompted the situation (e.g. miss_companion)",
    )
    parser.add_argument("--person-id", default=None, help="Person involved (free-form id)")
    parser.add_argument(
        "--regret",
        type=float,
        default=None,
        help="Self-reported regret 0-1 (optional)",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"Log file path (default: {DEFAULT_LOG_PATH})",
    )
    args = parser.parse_args()

    if args.regret is not None and not (0.0 <= args.regret <= 1.0):
        print(f"error: --regret must be in [0, 1], got {args.regret}", file=sys.stderr)
        return 1

    entry = append_counterfactual(
        wanted=args.wanted,
        chose=args.chose,
        why=args.why,
        trigger=args.trigger,
        person_id=args.person_id,
        regret=args.regret,
        log_path=args.log_path,
    )
    print(json.dumps(entry, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
