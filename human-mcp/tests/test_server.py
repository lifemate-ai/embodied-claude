"""Tests for MCP server module."""

from unittest.mock import AsyncMock, patch

import pytest

from human_mcp.server import call_tool, list_tools
from human_mcp.types import HumanResponse


@pytest.mark.asyncio
async def test_list_tools_returns_four_tools():
    tools = await list_tools()
    assert len(tools) == 4
    names = {t.name for t in tools}
    assert "talk_to_human" in names
    assert "nudge_human" in names
    assert "remind_human" in names
    assert "batch_talk_to_human" in names


@pytest.mark.asyncio
async def test_talk_to_human_only_requires_message():
    tools = await list_tools()
    talk_tool = next(t for t in tools if t.name == "talk_to_human")
    required = talk_tool.inputSchema["required"]
    assert required == ["message"]


@pytest.mark.asyncio
async def test_nudge_only_requires_message():
    tools = await list_tools()
    nudge_tool = next(t for t in tools if t.name == "nudge_human")
    required = nudge_tool.inputSchema["required"]
    assert required == ["message"]


@pytest.mark.asyncio
async def test_remind_only_requires_message():
    tools = await list_tools()
    remind_tool = next(t for t in tools if t.name == "remind_human")
    required = remind_tool.inputSchema["required"]
    assert required == ["message"]


@pytest.mark.asyncio
async def test_talk_description_is_conversational():
    tools = await list_tools()
    talk_tool = next(t for t in tools if t.name == "talk_to_human")
    desc = talk_tool.description
    assert "conversation" in desc.lower()


@pytest.mark.asyncio
async def test_nudge_description_says_no_response():
    tools = await list_tools()
    nudge_tool = next(t for t in tools if t.name == "nudge_human")
    assert "NOT wait" in nudge_tool.description


@pytest.mark.asyncio
async def test_remind_description_says_no_response():
    tools = await list_tools()
    remind_tool = next(t for t in tools if t.name == "remind_human")
    assert "NOT wait" in remind_tool.description


@pytest.mark.asyncio
@patch(
    "human_mcp.server.send_and_wait",
    return_value=HumanResponse(short_text="Doing great!", response_time_seconds=3.0),
)
async def test_call_talk_to_human(mock_send):
    result = await call_tool("talk_to_human", {
        "message": "How are you doing?",
    })
    assert len(result) == 1
    assert "Doing great!" in result[0].text


@pytest.mark.asyncio
@patch("human_mcp.server.send_no_wait", new_callable=AsyncMock)
async def test_call_nudge_human(mock_send):
    result = await call_tool("nudge_human", {
        "message": "Drink some water!",
    })
    assert "Nudge sent" in result[0].text
    assert "Drink some water!" in result[0].text
    mock_send.assert_called_once_with("Drink some water!", category="nudge")


@pytest.mark.asyncio
@patch("human_mcp.server.send_no_wait", new_callable=AsyncMock)
async def test_call_remind_human(mock_send):
    result = await call_tool("remind_human", {
        "message": "Meeting at 10am",
    })
    assert "Reminder sent" in result[0].text
    assert "Meeting at 10am" in result[0].text
    mock_send.assert_called_once_with("Meeting at 10am", category="remind")



@pytest.mark.asyncio
@patch(
    "human_mcp.server.send_and_wait",
    side_effect=[
        HumanResponse(short_text="Good", response_time_seconds=1.0),
        HumanResponse(short_text="Sure", response_time_seconds=2.0),
    ],
)
async def test_call_batch_talk(mock_send):
    result = await call_tool("batch_talk_to_human", {
        "messages": [
            {"message": "How are you?"},
            {"message": "Want to try something?"},
        ],
    })
    text = result[0].text
    assert "Message 1/2" in text
    assert "Message 2/2" in text


@pytest.mark.asyncio
async def test_call_unknown_tool():
    result = await call_tool("nonexistent_tool", {})
    assert "Unknown tool" in result[0].text
