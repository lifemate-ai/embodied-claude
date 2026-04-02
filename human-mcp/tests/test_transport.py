"""Tests for file-based transport."""

import json
from unittest.mock import patch

import pytest

from human_mcp.transport import send_and_wait
from human_mcp.types import HumanRequest, ResponseFormat, TaskType


def _make_request(**kwargs):
    defaults = {
        "task_type": TaskType.OBSERVATION,
        "question": "Test?",
        "reason": "Testing",
    }
    defaults.update(kwargs)
    return HumanRequest(**defaults)


@pytest.mark.asyncio
async def test_send_creates_request_file(tmp_path):
    """Request file is created in queue dir."""
    with patch("human_mcp.transport.QUEUE_DIR", tmp_path):
        req = _make_request()

        # Write a response immediately so we don't block
        import asyncio

        async def write_response_soon():
            await asyncio.sleep(0.1)
            for p in tmp_path.glob("request_*.json"):
                req_id = p.stem.replace("request_", "")
                resp_path = tmp_path / f"response_{req_id}.json"
                resp_path.write_text(json.dumps({"short_text": "ok"}))

        task = asyncio.create_task(write_response_soon())
        response = await send_and_wait(req)
        await task

        assert response.short_text == "ok"


@pytest.mark.asyncio
async def test_send_timeout(tmp_path):
    """Times out if no response file appears."""
    with patch("human_mcp.transport.QUEUE_DIR", tmp_path), \
         patch("human_mcp.transport.DEFAULT_TIMEOUTS", {"normal": 1}):
        req = _make_request()
        response = await send_and_wait(req)
        assert "TIMEOUT" in (response.short_text or "")


@pytest.mark.asyncio
async def test_send_choice_response(tmp_path):
    """Choice response is parsed correctly."""
    with patch("human_mcp.transport.QUEUE_DIR", tmp_path):
        req = _make_request(
            expected_format=ResponseFormat.CHOICE,
            options=["A", "B"],
        )

        import asyncio

        async def write_response():
            await asyncio.sleep(0.1)
            for p in tmp_path.glob("request_*.json"):
                req_id = p.stem.replace("request_", "")
                resp_path = tmp_path / f"response_{req_id}.json"
                resp_path.write_text(json.dumps({"choice": "B"}))

        task = asyncio.create_task(write_response())
        response = await send_and_wait(req)
        await task

        assert response.choice == "B"


@pytest.mark.asyncio
async def test_send_yes_no_response(tmp_path):
    """Yes/no response is parsed correctly."""
    with patch("human_mcp.transport.QUEUE_DIR", tmp_path):
        req = _make_request(expected_format=ResponseFormat.YES_NO)

        import asyncio

        async def write_response():
            await asyncio.sleep(0.1)
            for p in tmp_path.glob("request_*.json"):
                req_id = p.stem.replace("request_", "")
                resp_path = tmp_path / f"response_{req_id}.json"
                resp_path.write_text(json.dumps({"yes_no": True}))

        task = asyncio.create_task(write_response())
        response = await send_and_wait(req)
        await task

        assert response.yes_no is True


@pytest.mark.asyncio
async def test_cleanup_after_response(tmp_path):
    """Request and response files are cleaned up."""
    with patch("human_mcp.transport.QUEUE_DIR", tmp_path):
        req = _make_request()

        import asyncio

        async def write_response():
            await asyncio.sleep(0.1)
            for p in tmp_path.glob("request_*.json"):
                req_id = p.stem.replace("request_", "")
                resp_path = tmp_path / f"response_{req_id}.json"
                resp_path.write_text(json.dumps({"short_text": "done"}))

        task = asyncio.create_task(write_response())
        await send_and_wait(req)
        await task

        assert list(tmp_path.glob("request_*.json")) == []
        assert list(tmp_path.glob("response_*.json")) == []
