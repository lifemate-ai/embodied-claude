import pytest

from toio_mcp.config import ToioConfig
from toio_mcp.controller import ToioHandController


@pytest.mark.asyncio
async def test_dry_run_connect_and_status():
    controller = ToioHandController(ToioConfig(dry_run=True))
    message = await controller.connect()
    assert "Connected to toio hand" in message
    status = await controller.status()
    assert status["connected"] is True
    assert status["dry_run"] is True
    assert status["battery_level"] == 100


@pytest.mark.asyncio
async def test_dry_run_relative_motion_changes_position():
    controller = ToioHandController(ToioConfig(dry_run=True))
    await controller.connect()
    before = await controller.status()
    await controller.move_forward(speed=40, duration=1.0)
    after = await controller.status()
    assert before["position"] != after["position"]


@pytest.mark.asyncio
async def test_dry_run_move_to_position():
    controller = ToioHandController(ToioConfig(dry_run=True))
    await controller.connect()
    message = await controller.move_to_position(120, 45)
    assert "Moved hand to position" in message
    status = await controller.status()
    assert status["position"] == {"x": 120, "y": 45}


@pytest.mark.asyncio
async def test_dry_run_orientation_updates():
    controller = ToioHandController(ToioConfig(dry_run=True))
    await controller.connect()
    await controller.set_orientation(270)
    status = await controller.status()
    assert status["orientation"] == 270
