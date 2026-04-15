#!/usr/bin/env python3
"""Example adapter that forwards Garmin summary JSON into the social event store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "social-core" / "src"))
sys.path.insert(0, str(ROOT / "social-state-mcp" / "src"))

from social_state_mcp.store import SocialStateStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="JSON file with ts/person_id/body_battery/etc")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    store = SocialStateStore()
    result = store.ingest_social_event(
        {
            "ts": payload["ts"],
            "source": "garmin",
            "kind": "health_summary",
            "person_id": payload.get("person_id"),
            "confidence": payload.get("confidence", 0.95),
            "payload": payload,
        }
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
