"""Tests for MCP server."""

from mobility_mcp.server import MobilityMCPServer


class TestMobilityMCPServer:
    def test_server_creation(self):
        server = MobilityMCPServer()
        assert server._server is not None
        assert server._controller is None

    def test_clamp_duration_none(self):
        server = MobilityMCPServer()
        assert server._clamp_duration(None) is None

    def test_clamp_duration_normal(self):
        server = MobilityMCPServer()
        assert server._clamp_duration(2.0) == 2.0

    def test_clamp_duration_too_small(self):
        server = MobilityMCPServer()
        assert server._clamp_duration(0.01) == 0.1

    def test_clamp_duration_too_large(self):
        server = MobilityMCPServer()
        result = server._clamp_duration(999.0)
        assert result == 10.0  # MAX_MOVE_DURATION default
