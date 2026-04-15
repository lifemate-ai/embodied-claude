#!/usr/bin/env python3
"""Example adapter that forwards human-mcp style utterances into the shared social DB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for package in ("social-core", "social-state-mcp", "relationship-mcp"):
    sys.path.insert(0, str(ROOT / package / "src"))

from relationship_mcp.store import RelationshipStore  # noqa: E402
from social_state_mcp.store import SocialStateStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="JSON file with ts/person_id/text/channel")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    social_state = SocialStateStore()
    relationship = RelationshipStore()
    social_state.ingest_social_event(
        {
            "ts": payload["ts"],
            "source": payload.get("channel", "human_mcp"),
            "kind": "human_utterance",
            "person_id": payload["person_id"],
            "confidence": payload.get("confidence", 0.98),
            "payload": {"text": payload["text"]},
        }
    )
    relationship.ingest_interaction(
        person_id=payload["person_id"],
        channel=payload.get("channel", "human_mcp"),
        direction="human_to_ai",
        text=payload["text"],
        ts=payload["ts"],
    )
    print(json.dumps({"status": "ok", "person_id": payload["person_id"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
