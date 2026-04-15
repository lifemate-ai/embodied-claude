#!/usr/bin/env python3
"""Example adapter that forwards a structured scene parse into joint-attention and social-state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for package in ("social-core", "social-state-mcp", "joint-attention-mcp"):
    sys.path.insert(0, str(ROOT / package / "src"))

from joint_attention_mcp.store import JointAttentionStore  # noqa: E402
from social_state_mcp.store import SocialStateStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="JSON file matching the scene parse schema")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    joint_attention = JointAttentionStore()
    social_state = SocialStateStore()
    frame = joint_attention.ingest_scene_parse(payload)
    social_state.ingest_social_event(
        {
            "ts": payload["ts"],
            "source": "camera",
            "kind": "scene_parse",
            "person_id": payload.get("people", [{}])[0].get("person_id") if payload.get("people") else None,
            "confidence": 0.9,
            "payload": {"scene_summary": payload["scene_summary"]},
        }
    )
    print(json.dumps(frame, ensure_ascii=False))


if __name__ == "__main__":
    main()
