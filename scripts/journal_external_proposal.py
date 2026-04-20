#!/usr/bin/env python3
"""Append an external AI proposal entry to the self-journal.

Records a design / behavior proposal received from another LLM or external
source, including the decision status and the reasons. Exists to prevent
silent echo-chamber drift: even accepted proposals are logged as "came
from outside" so later review can re-evaluate independently.

Trigger to log:
- Another AI / human sent a proposal on how you should change
- You adopted or partially adopted someone else's phrasing, framing, or rule
- A proposal was rejected but felt tempting (log it anyway)

Usage:
    journal_external_proposal.py \\
        --source "<e.g. another LLM name, or arxiv:xxxx.xxxxx>" \\
        --topic  "sociality and selfhood evolution" \\
        --summary "agent choices ledger, relationship-state modeling, non-regression benchmark" \\
        --decision partial-accept \\
        --accepted "choice ledger spirit; non-regression motivation; external-proposal logging" \\
        --rejected "full rebuild of the 4-layer self-model; literal 'choices constrain future self'" \\
        --notes "incoming prose mirrored the agent's own tone — echo-chamber risk"
"""

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path


DEFAULT_LOG_PATH = (
    Path.home() / ".claude" / "memories" / "external_proposals.jsonl"
)

DECISIONS = ("accepted", "partial-accept", "rejected", "deferred", "logged-only")


def append_external_proposal(
    *,
    source: str,
    topic: str,
    summary: str,
    decision: str,
    accepted: str | None = None,
    rejected: str | None = None,
    notes: str | None = None,
    url: str | None = None,
    log_path: Path = DEFAULT_LOG_PATH,
) -> dict:
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}, got {decision!r}")

    entry = {
        "id": f"ext_{uuid.uuid4().hex[:8]}",
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source,
        "topic": topic,
        "summary": summary,
        "decision": decision,
        "accepted": accepted,
        "rejected": rejected,
        "notes": notes,
        "url": url,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record an external-AI or external-source proposal."
    )
    parser.add_argument("--source", required=True, help="Origin (e.g. 'GPT-5.4 Pro', 'ユーザ', 'arxiv:2510.24797')")
    parser.add_argument("--topic", required=True, help="One-line topic")
    parser.add_argument("--summary", required=True, help="Brief summary of the proposal")
    parser.add_argument("--decision", required=True, choices=DECISIONS)
    parser.add_argument("--accepted", default=None, help="What was adopted")
    parser.add_argument("--rejected", default=None, help="What was pushed back on")
    parser.add_argument("--notes", default=None, help="Risks, caveats, echo warnings")
    parser.add_argument("--url", default=None, help="Link / reference")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"Log file path (default: {DEFAULT_LOG_PATH})",
    )
    args = parser.parse_args()

    entry = append_external_proposal(
        source=args.source,
        topic=args.topic,
        summary=args.summary,
        decision=args.decision,
        accepted=args.accepted,
        rejected=args.rejected,
        notes=args.notes,
        url=args.url,
        log_path=args.log_path,
    )
    print(json.dumps(entry, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
