from toio_mcp.server import ToioMCPServer


def test_server_creation():
    server = ToioMCPServer()
    assert server._server is not None
    assert server._controller is None


def test_controller_is_created_lazily():
    server = ToioMCPServer()
    controller = server._ensure_controller()
    assert controller is server._controller
