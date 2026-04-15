"""Tests for image_utils module."""

import base64
import os
import tempfile

import pytest
from PIL import Image

from memory_mcp.image_utils import (
    RESOLUTION_PRESETS,
    encode_image_for_memory,
    resolve_resolution,
)


@pytest.fixture
def sample_image_path():
    """Create a temporary test image and return its path."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new("RGB", (1920, 1080), color=(100, 150, 200))
        img.save(f, format="JPEG")
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def sample_png_path():
    """Create a temporary RGBA PNG image."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGBA", (800, 600), color=(100, 150, 200, 128))
        img.save(f, format="PNG")
        path = f.name
    yield path
    os.unlink(path)


class TestEncodeImageForMemory:
    def test_returns_base64_string(self, sample_image_path):
        result = encode_image_for_memory(sample_image_path)
        assert result is not None
        # Verify it's valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_resizes_to_fit(self, sample_image_path):
        result = encode_image_for_memory(sample_image_path, max_width=320, max_height=240)
        assert result is not None
        # Decode and check dimensions
        decoded = base64.b64decode(result)
        from io import BytesIO

        img = Image.open(BytesIO(decoded))
        assert img.width <= 320
        assert img.height <= 240

    def test_maintains_aspect_ratio(self, sample_image_path):
        result = encode_image_for_memory(sample_image_path, max_width=320, max_height=240)
        assert result is not None
        decoded = base64.b64decode(result)
        from io import BytesIO

        img = Image.open(BytesIO(decoded))
        # 1920x1080 = 16:9, thumbnail to 320x240 → 320x180
        assert img.width == 320
        assert img.height == 180

    def test_handles_rgba_images(self, sample_png_path):
        result = encode_image_for_memory(sample_png_path)
        assert result is not None
        # Should be valid JPEG (converted from RGBA)
        decoded = base64.b64decode(result)
        from io import BytesIO

        img = Image.open(BytesIO(decoded))
        assert img.mode == "RGB"

    def test_returns_none_for_nonexistent_file(self):
        result = encode_image_for_memory("/nonexistent/path/image.jpg")
        assert result is None

    def test_returns_none_for_invalid_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"not an image")
            path = f.name
        try:
            result = encode_image_for_memory(path)
            assert result is None
        finally:
            os.unlink(path)

    def test_custom_resolution(self, sample_image_path):
        result = encode_image_for_memory(
            sample_image_path, max_width=160, max_height=120
        )
        assert result is not None
        decoded = base64.b64decode(result)
        from io import BytesIO

        img = Image.open(BytesIO(decoded))
        assert img.width <= 160
        assert img.height <= 120


class TestResolveResolution:
    def test_low(self):
        assert resolve_resolution("low") == (160, 120)

    def test_medium(self):
        assert resolve_resolution("medium") == (320, 240)

    def test_high(self):
        assert resolve_resolution("high") == (640, 480)

    def test_none_defaults_to_medium(self):
        assert resolve_resolution(None) == (320, 240)

    def test_unknown_defaults_to_medium(self):
        assert resolve_resolution("ultra") == (320, 240)

    def test_presets_dict(self):
        assert len(RESOLUTION_PRESETS) == 4
        assert "full_hd" in RESOLUTION_PRESETS
