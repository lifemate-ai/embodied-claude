#!/usr/bin/env python3
"""Replay deterministic fixture events into the shared social DB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "social-core" / "src"))

from social_core import EventStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="JSON file containing an array of social events")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    store = EventStore()
    results = store.replay(payload["events"] if isinstance(payload, dict) and "events" in payload else payload)
    print(json.dumps({"replayed": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
