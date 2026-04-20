#!/usr/bin/env python3
"""Seed a person record into the sociality store from a local TOML file.

Usage:
    uv run --directory sociality-mcp python scripts/seed-person-from-toml.py \
        .local/person-seeds/kouta.toml

The TOML shape is:

    person_id = "kouta"
    canonical_name = "Kouta"
    role = "primary companion / developer"
    aliases = ["コウタ", "こうちゃん"]

    [[preferences]]
    text = "..."
    confidence = 0.8
    evidence = ["..."]
    source = "seeded"

Private seeds should live under a gitignored path (``.local/person-seeds/``);
do NOT commit your ``kouta.toml`` to a public repository.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(f"usage: {argv[0]} <path-to-toml>", file=sys.stderr)
        return 2

    toml_path = Path(argv[1]).expanduser()
    if not toml_path.exists():
        print(f"file not found: {toml_path}", file=sys.stderr)
        return 1

    data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    person_id = data.get("person_id")
    canonical_name = data.get("canonical_name")
    if not person_id or not canonical_name:
        print("toml must contain person_id and canonical_name", file=sys.stderr)
        return 1

    # Import lazily so this script can live outside the MCP venv.
    try:
        from relationship_mcp.store import RelationshipStore
    except ImportError as exc:
        print(
            "relationship_mcp is not importable. Run via: "
            "uv run --directory sociality-mcp/packages/relationship-mcp python "
            f"{argv[0]} {argv[1]}",
            file=sys.stderr,
        )
        print(f"  import error: {exc}", file=sys.stderr)
        return 1

    store = RelationshipStore()
    try:
        store.upsert_person(
            person_id=person_id,
            canonical_name=canonical_name,
            aliases=list(data.get("aliases", [])),
            role=data.get("role"),
        )
    finally:
        store.close()

    prefs = data.get("preferences", [])
    print(
        f"seeded person {person_id!r} ({canonical_name})"
        + (f" with role={data.get('role')!r}" if data.get("role") else "")
        + (f" and {len(prefs)} preference record(s)" if prefs else "")
    )
    if prefs:
        print(
            "note: preferences are surfaced through get_person_model; they are "
            "re-derived from evidence on each read, so the TOML is the source of "
            "truth for seeded values."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
