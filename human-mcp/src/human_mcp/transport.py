"""File-based transport for Human as MCP.

Communication flow:
1. Server writes request to /tmp/human-mcp/request_<id>.json
2. Watcher script detects new request, displays it, collects human input
3. Watcher writes response to /tmp/human-mcp/response_<id>.json
4. Server polls for response file, reads it, returns to agent
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from human_mcp.types import HumanRequest, HumanResponse

QUEUE_DIR = Path("/tmp/human-mcp")
POLL_INTERVAL = 0.5  # seconds
DEFAULT_TIMEOUTS = {"low": 3600, "normal": 300, "high": 60}


def _ensure_queue_dir() -> None:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)


def _request_to_dict(request: HumanRequest) -> dict:
    return {
        "task_type": request.task_type.value,
        "question": request.question,
        "reason": request.reason,
        "expected_format": request.expected_format.value,
        "options": request.options,
        "urgency": request.urgency.value,
        "created_at": request.created_at.isoformat(),
    }


async def send_no_wait(message: str, category: str = "nudge") -> None:
    """Write a one-way message. No response expected."""
    _ensure_queue_dir()

    msg_id = uuid.uuid4().hex[:8]
    msg_path = QUEUE_DIR / f"{category}_{msg_id}.json"
    msg_path.write_text(
        json.dumps({"message": message, "category": category}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def send_and_wait(request: HumanRequest) -> HumanResponse:
    """Write request file and poll for response."""
    _ensure_queue_dir()

    req_id = uuid.uuid4().hex[:8]
    request_path = QUEUE_DIR / f"request_{req_id}.json"
    response_path = QUEUE_DIR / f"response_{req_id}.json"

    # Write request
    request_path.write_text(json.dumps(_request_to_dict(request), ensure_ascii=False, indent=2), encoding="utf-8")

    # Poll for response
    timeout = DEFAULT_TIMEOUTS.get(request.urgency.value, 300)
    start = time.time()

    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            # Timeout — clean up and return timeout response
            request_path.unlink(missing_ok=True)
            return HumanResponse(
                short_text=f"[TIMEOUT after {timeout}s — human did not respond]",
                response_time_seconds=elapsed,
            )

        if response_path.exists():
            try:
                data = json.loads(response_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                await asyncio.sleep(POLL_INTERVAL)
                continue

            # Clean up files
            request_path.unlink(missing_ok=True)
            response_path.unlink(missing_ok=True)

            return HumanResponse(
                choice=data.get("choice"),
                yes_no=data.get("yes_no"),
                short_text=data.get("short_text"),
                number=data.get("number"),
                photo_path=data.get("photo_path"),
                response_time_seconds=time.time() - start,
            )

        await asyncio.sleep(POLL_INTERVAL)
