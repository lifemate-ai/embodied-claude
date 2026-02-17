"""Tests for vacuum mobility controller."""

from unittest.mock import MagicMock, patch

import pytest

from mobility_mcp.config import TuyaDeviceConfig
from mobility_mcp.vacuum import (
    DIRECTION_BACKWARD,
    DIRECTION_FORWARD,
    DIRECTION_LEFT,
    DIRECTION_RIGHT,
    DIRECTION_STOP,
    VacuumMobilityController,
)


@pytest.fixture
def config():
    return TuyaDeviceConfig(
        device_id="test_dev",
        ip_address="192.168.1.100",
        local_key="testkey",
        version="3.3",
    )


@pytest.fixture
def controller(config):
    return VacuumMobilityController(config)


@pytest.fixture
def mock_device():
    device = MagicMock()
    device.set_value.return_value = {"dps": {"4": "forward"}}
    device.status.return_value = {"dps": {"1": True, "4": "stop", "101": 85}}
    return device


class TestVacuumMobilityController:
    @pytest.mark.asyncio
    async def test_move_forward_continuous(self, controller, mock_device):
        with patch("mobility_mcp.vacuum.tinytuya.Device", return_value=mock_device):
            controller._device = mock_device
            result = await controller.move_forward()
            mock_device.set_value.assert_called_once_with(4, DIRECTION_FORWARD)
            assert "Moving forward" in result

    @pytest.mark.asyncio
    async def test_move_forward_with_duration(self, controller, mock_device):
        with patch("mobility_mcp.vacuum.tinytuya.Device", return_value=mock_device):
            controller._device = mock_device
            result = await controller.move_forward(duration=0.1)
            assert mock_device.set_value.call_count == 2
            calls = mock_device.set_value.call_args_list
            assert calls[0].args == (4, DIRECTION_FORWARD)
            assert calls[1].args == (4, DIRECTION_STOP)
            assert "0.1s" in result

    @pytest.mark.asyncio
    async def test_move_backward(self, controller, mock_device):
        controller._device = mock_device
        result = await controller.move_backward()
        mock_device.set_value.assert_called_once_with(4, DIRECTION_BACKWARD)
        assert "Moving backward" in result

    @pytest.mark.asyncio
    async def test_turn_left(self, controller, mock_device):
        controller._device = mock_device
        result = await controller.turn_left()
        mock_device.set_value.assert_called_once_with(4, DIRECTION_LEFT)
        assert "Turning left" in result

    @pytest.mark.asyncio
    async def test_turn_right(self, controller, mock_device):
        controller._device = mock_device
        result = await controller.turn_right()
        mock_device.set_value.assert_called_once_with(4, DIRECTION_RIGHT)
        assert "Turning right" in result

    @pytest.mark.asyncio
    async def test_stop(self, controller, mock_device):
        controller._device = mock_device
        result = await controller.stop()
        mock_device.set_value.assert_called_once_with(4, DIRECTION_STOP)
        assert "Stopped" in result

    @pytest.mark.asyncio
    async def test_get_status(self, controller, mock_device):
        controller._device = mock_device
        status = await controller.get_status()
        mock_device.status.assert_called_once()
        assert "dps" in status
        assert status["dps"]["101"] == 85

    @pytest.mark.asyncio
    async def test_invalid_direction(self, controller, mock_device):
        controller._device = mock_device
        with pytest.raises(ValueError, match="Invalid direction"):
            await controller._send_direction("invalid_direction")

    def test_disconnect(self, controller, mock_device):
        controller._device = mock_device
        controller.disconnect()
        mock_device.close.assert_called_once()
        assert controller._device is None

    def test_disconnect_when_not_connected(self, controller):
        controller.disconnect()  # Should not raise

    def test_ensure_device_creates_connection(self, controller):
        with patch("mobility_mcp.vacuum.tinytuya.Device") as mock_cls:
            mock_cls.return_value = MagicMock()
            device = controller._ensure_device()
            mock_cls.assert_called_once_with(
                dev_id="test_dev",
                address="192.168.1.100",
                local_key="testkey",
                version="3.3",
            )
            assert device is not None

    def test_ensure_device_reuses_connection(self, controller, mock_device):
        controller._device = mock_device
        device = controller._ensure_device()
        assert device is mock_device
