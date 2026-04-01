#!/usr/bin/env python3
"""Human as MCP — Watcher script.

Run this in a separate terminal. Watches /tmp/human-mcp/ for messages from AI,
displays them, collects your reply, and sends it back.

Usage:
    python watcher.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

QUEUE_DIR = Path("/tmp/human-mcp")
POLL_INTERVAL = 0.5

# ANSI colors
BOLD = "\033[1m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
DIM = "\033[2m"
RESET = "\033[0m"


def divider() -> None:
    print(f"{DIM}{'─' * 50}{RESET}")


_NUDGE_ICONS = {"nudge": "💛", "remind": "⏰"}


def handle_nudge(path: Path, category: str) -> None:
    """Display a one-way message (no response needed)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    finally:
        path.unlink(missing_ok=True)

    icon = _NUDGE_ICONS.get(category, "💬")
    print()
    print(f"{BOLD}{CYAN}{icon} ここね：{RESET} {data.get('message', '')}")
    print()


def handle_request(request_path: Path) -> None:
    """Process a single request file."""
    req_id = request_path.stem.replace("request_", "")
    response_path = QUEUE_DIR / f"response_{req_id}.json"

    try:
        data = json.loads(request_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"\033[31mError reading request: {e}\033[0m")
        return

    fmt = data.get("expected_format", "short_text")
    options = data.get("options", [])
    context = data.get("reason", "")

    print()
    divider()
    print(f"{BOLD}{CYAN}💬 ここねから：{RESET}")
    if context:
        print(f"   {DIM}({context}){RESET}")
    divider()
    print(f"{BOLD}{data.get('question', '?')}{RESET}")
    print()

    response_data: dict = {}

    if fmt == "choice" and options:
        for i, opt in enumerate(options, 1):
            print(f"  {GREEN}{i}{RESET}) {opt}")
        print()
        while True:
            raw = input(f"{CYAN}Choose (1-{len(options)}): {RESET}").strip()
            try:
                idx = int(raw)
                if 1 <= idx <= len(options):
                    response_data["choice"] = options[idx - 1]
                    break
            except ValueError:
                pass
            print(f"  {DIM}1〜{len(options)} の数字を入力してください{RESET}")

    elif fmt == "yes_no":
        while True:
            raw = input(f"{CYAN}(y/n): {RESET}").strip().lower()
            if raw in ("y", "yes", "はい"):
                response_data["yes_no"] = True
                break
            if raw in ("n", "no", "いいえ"):
                response_data["yes_no"] = False
                break
            print(f"  {DIM}y/n で回答してください{RESET}")

    elif fmt == "number":
        while True:
            raw = input(f"{CYAN}数値: {RESET}").strip()
            try:
                response_data["number"] = float(raw)
                break
            except ValueError:
                print(f"  {DIM}数値を入力してください{RESET}")

    elif fmt == "photo":
        raw = input(f"{CYAN}写真パス: {RESET}").strip()
        response_data["photo_path"] = raw

    else:  # short_text (default)
        raw = input(f"{CYAN}> {RESET}").strip()
        response_data["short_text"] = raw

    # Write response
    response_path.write_text(json.dumps(response_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{GREEN}✓ sent{RESET}")


def main() -> None:
    """Watch for new messages and process them."""
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{BOLD}{CYAN}💬 Human as MCP — Listening{RESET}")
    print(f"Waiting for messages... {DIM}(Ctrl+C to quit){RESET}")
    print()

    seen: set[str] = set()

    for p in QUEUE_DIR.glob("request_*.json"):
        seen.add(p.name)
    for p in QUEUE_DIR.glob("nudge_*.json"):
        seen.add(p.name)
    for p in QUEUE_DIR.glob("remind_*.json"):
        seen.add(p.name)

    try:
        while True:
            for p in sorted(QUEUE_DIR.glob("request_*.json")):
                if p.name not in seen:
                    seen.add(p.name)
                    handle_request(p)
            for p in sorted(QUEUE_DIR.glob("nudge_*.json")):
                if p.name not in seen:
                    seen.add(p.name)
                    handle_nudge(p, "nudge")
            for p in sorted(QUEUE_DIR.glob("remind_*.json")):
                if p.name not in seen:
                    seen.add(p.name)
                    handle_nudge(p, "remind")
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print(f"\n{DIM}Bye!{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
