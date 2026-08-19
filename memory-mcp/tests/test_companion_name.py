"""The default ``person`` of the social tools follows COMPANION_NAME (#135).

The literal default used to be one specific person's name, so every other
deployment's theory-of-mind and joint-attention calls were about someone they
had never met.
"""

from __future__ import annotations

import importlib

import pytest
from mcp.types import ListToolsRequest


async def _tool_schemas(server_module) -> dict[str, dict]:
    mcp_server = server_module.MemoryMCPServer()
    handler = mcp_server._server.request_handlers[ListToolsRequest]
    result = await handler(ListToolsRequest(method="tools/list"))
    return {tool.name: tool.inputSchema for tool in result.root.tools}


@pytest.fixture
def reloaded_server(monkeypatch):
    """Reload memory_mcp.server under a controlled COMPANION_NAME and restore it after."""

    import memory_mcp.server as server_module

    def _reload(value: str | None):
        if value is None:
            monkeypatch.delenv("COMPANION_NAME", raising=False)
        else:
            monkeypatch.setenv("COMPANION_NAME", value)
        return importlib.reload(server_module)

    yield _reload
    monkeypatch.delenv("COMPANION_NAME", raising=False)
    importlib.reload(server_module)


async def test_default_person_is_neutral(reloaded_server) -> None:
    server_module = reloaded_server(None)

    assert server_module.COMPANION_NAME == "あなた"
    schemas = await _tool_schemas(server_module)
    assert schemas["tom"]["properties"]["person"]["default"] == "あなた"
    assert schemas["joint_attention"]["properties"]["person"]["default"] == "あなた"
    assert "コウタ" not in schemas["tom"]["properties"]["person"]["description"]


async def test_default_person_follows_companion_name(reloaded_server) -> None:
    server_module = reloaded_server("コウタ")

    assert server_module.COMPANION_NAME == "コウタ"
    schemas = await _tool_schemas(server_module)
    assert schemas["tom"]["properties"]["person"]["default"] == "コウタ"
    assert "コウタ" in schemas["tom"]["properties"]["person"]["description"]
    assert schemas["joint_attention"]["properties"]["person"]["default"] == "コウタ"
