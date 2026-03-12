"""JSONL buffer for passing transcriptions to hearing-hook.sh."""

from __future__ import annotations

import fcntl
import json
import logging
from typing import Any

from .config import BUFFER_FILE

logger = logging.getLogger(__name__)


def append_to_buffer(entry: dict[str, Any]) -> None:
    """Atomically append a JSON entry to the hearing buffer file.

    Uses flock for mutual exclusion with hearing-hook.sh's os.rename drain.
    """
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    with open(BUFFER_FILE, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
