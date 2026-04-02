"""CLI interface for human responses.

Presents structured requests to the human in the terminal
and collects structured responses.
"""

from __future__ import annotations

import sys
import time

from human_mcp.types import HumanRequest, HumanResponse, ResponseFormat, Urgency

# ANSI colors
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_DIM = "\033[2m"
_RESET = "\033[0m"

_URGENCY_COLORS = {
    Urgency.LOW: _DIM,
    Urgency.NORMAL: _YELLOW,
    Urgency.HIGH: _RED,
}

_URGENCY_LABELS = {
    Urgency.LOW: "LOW",
    Urgency.NORMAL: "NORMAL",
    Urgency.HIGH: "⚡ URGENT",
}


def _print_divider() -> None:
    print(f"{_DIM}{'─' * 50}{_RESET}")


def _print_request(request: HumanRequest) -> None:
    """Display a formatted request to the human."""
    urgency_color = _URGENCY_COLORS[request.urgency]
    urgency_label = _URGENCY_LABELS[request.urgency]

    print()
    _print_divider()
    print(f"{_BOLD}{_CYAN}📋 Human callback requested{_RESET}")
    print(f"   Type: {request.task_type.value}")
    print(f"   Urgency: {urgency_color}{urgency_label}{_RESET}")
    print(f"   Reason: {_DIM}{request.reason}{_RESET}")
    _print_divider()
    print(f"{_BOLD}{request.question}{_RESET}")
    print()


def prompt_choice(request: HumanRequest) -> HumanResponse:
    """Prompt human to pick from options."""
    _print_request(request)

    if not request.options:
        print(f"{_RED}Error: no options provided for choice format{_RESET}")
        return HumanResponse(short_text="error: no options")

    for i, option in enumerate(request.options, 1):
        print(f"  {_GREEN}{i}{_RESET}) {option}")
    print()

    start = time.time()
    while True:
        raw = input(f"{_CYAN}Choose (1-{len(request.options)}): {_RESET}").strip()
        try:
            idx = int(raw)
            if 1 <= idx <= len(request.options):
                elapsed = time.time() - start
                return HumanResponse(
                    choice=request.options[idx - 1],
                    response_time_seconds=elapsed,
                )
        except ValueError:
            pass
        print(f"  {_DIM}1〜{len(request.options)} の数字を入力してください{_RESET}")


def prompt_yes_no(request: HumanRequest) -> HumanResponse:
    """Prompt human for yes/no."""
    _print_request(request)

    start = time.time()
    while True:
        raw = input(f"{_CYAN}(y/n): {_RESET}").strip().lower()
        if raw in ("y", "yes", "はい"):
            return HumanResponse(yes_no=True, response_time_seconds=time.time() - start)
        if raw in ("n", "no", "いいえ"):
            return HumanResponse(yes_no=False, response_time_seconds=time.time() - start)
        print(f"  {_DIM}y/n で回答してください{_RESET}")


def prompt_short_text(request: HumanRequest) -> HumanResponse:
    """Prompt human for short text."""
    _print_request(request)

    start = time.time()
    raw = input(f"{_CYAN}回答（短文）: {_RESET}").strip()
    return HumanResponse(short_text=raw, response_time_seconds=time.time() - start)


def prompt_number(request: HumanRequest) -> HumanResponse:
    """Prompt human for a number."""
    _print_request(request)

    start = time.time()
    while True:
        raw = input(f"{_CYAN}数値: {_RESET}").strip()
        try:
            val = float(raw)
            return HumanResponse(number=val, response_time_seconds=time.time() - start)
        except ValueError:
            print(f"  {_DIM}数値を入力してください{_RESET}")


def prompt_photo(request: HumanRequest) -> HumanResponse:
    """Prompt human for a photo path."""
    _print_request(request)

    start = time.time()
    raw = input(f"{_CYAN}写真パス: {_RESET}").strip()
    return HumanResponse(photo_path=raw, response_time_seconds=time.time() - start)


_PROMPTERS = {
    ResponseFormat.CHOICE: prompt_choice,
    ResponseFormat.YES_NO: prompt_yes_no,
    ResponseFormat.SHORT_TEXT: prompt_short_text,
    ResponseFormat.NUMBER: prompt_number,
    ResponseFormat.PHOTO: prompt_photo,
}


def prompt_human(request: HumanRequest) -> HumanResponse:
    """Route to the appropriate prompt based on expected format."""
    prompter = _PROMPTERS.get(request.expected_format, prompt_short_text)
    try:
        response = prompter(request)
    except (EOFError, KeyboardInterrupt):
        print(f"\n{_DIM}(cancelled){_RESET}")
        sys.exit(130)
    _print_divider()
    return response
