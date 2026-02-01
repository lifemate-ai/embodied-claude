"""Tests for SensoryIntegration."""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.memory_mcp.config import MemoryConfig
from src.memory_mcp.memory import MemoryStore
from src.memory_mcp.sensory import SensoryIntegration
from src.memory_mcp.types import CameraPosition


@pytest.fixture
async def memory_store():
    """Create a MemoryStore instance for testing with isolated temp DB."""
    # Create unique temp directory for each test
    temp_dir = tempfile.mkdtemp(prefix="test_sensory_")

    config = MemoryConfig(
        db_path=temp_dir,
        collection_name="test_memories",
    )
    store = MemoryStore(config)
    await store.connect()
    yield store
    await store.disconnect()

    # Cleanup temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sensory_integration(memory_store):
    """Create a SensoryIntegration instance."""
    return SensoryIntegration(memory_store)


class TestVisualMemory:
    """Test visual memory operations."""

    @pytest.mark.asyncio
    async def test_save_visual_memory(self, memory_store, sensory_integration):
        """Test saving a visual memory."""
        camera_pos = CameraPosition(pan_angle=60, tilt_angle=-30)

        memory = await sensory_integration.save_visual_memory(
            content="Found the morning sky",
            image_path="/tmp/wifi-cam/2026-02-01_07-53-00.jpg",
            camera_position=camera_pos,
            emotion="excited",
            importance=5,
        )

        assert memory.content == "Found the morning sky"
        assert memory.emotion == "excited"
        assert memory.importance == 5
        assert memory.category == "observation"
        assert memory.camera_position == camera_pos
        assert len(memory.sensory_data) == 1
        assert memory.sensory_data[0].sensory_type == "visual"
        assert memory.sensory_data[0].file_path == "/tmp/wifi-cam/2026-02-01_07-53-00.jpg"

    @pytest.mark.asyncio
    async def test_visual_memory_metadata(self, sensory_integration):
        """Test visual memory sensory data metadata."""
        camera_pos = CameraPosition(pan_angle=45, tilt_angle=-20)

        memory = await sensory_integration.save_visual_memory(
            content="Test visual",
            image_path="/tmp/test.jpg",
            camera_position=camera_pos,
        )

        sensory = memory.sensory_data[0]
        assert sensory.metadata["camera_position"]["pan_angle"] == 45
        assert sensory.metadata["camera_position"]["tilt_angle"] == -20


class TestAudioMemory:
    """Test audio memory operations."""

    @pytest.mark.asyncio
    async def test_save_audio_memory(self, sensory_integration):
        """Test saving an audio memory."""
        memory = await sensory_integration.save_audio_memory(
            content="Heard the childhood friend's voice",
            audio_path="/tmp/wifi-cam/2026-02-01_07-52-00.wav",
            transcript="こんにちは、今日はいい天気ですね",
            emotion="happy",
            importance=4,
        )

        assert memory.content == "Heard the childhood friend's voice"
        assert memory.emotion == "happy"
        assert memory.importance == 4
        assert len(memory.sensory_data) == 1
        assert memory.sensory_data[0].sensory_type == "audio"
        assert memory.sensory_data[0].file_path == "/tmp/wifi-cam/2026-02-01_07-52-00.wav"
        assert memory.sensory_data[0].description == "こんにちは、今日はいい天気ですね"

    @pytest.mark.asyncio
    async def test_audio_memory_transcript_in_metadata(self, sensory_integration):
        """Test audio memory has transcript in metadata."""
        memory = await sensory_integration.save_audio_memory(
            content="Test audio",
            audio_path="/tmp/test.wav",
            transcript="Test transcript",
        )

        sensory = memory.sensory_data[0]
        assert sensory.metadata["transcript"] == "Test transcript"


class TestCameraPositionRecall:
    """Test recalling memories by camera position."""

    @pytest.mark.asyncio
    async def test_recall_by_camera_position_exact(
        self, memory_store, sensory_integration
    ):
        """Test recalling memory with exact camera position."""
        camera_pos = CameraPosition(pan_angle=60, tilt_angle=-30)

        # Save visual memory
        await sensory_integration.save_visual_memory(
            content="Morning sky at 60/-30",
            image_path="/tmp/test1.jpg",
            camera_position=camera_pos,
        )

        # Recall with same position
        results = await sensory_integration.recall_by_camera_position(
            pan_angle=60,
            tilt_angle=-30,
            tolerance=5,
        )

        assert len(results) == 1
        assert results[0].content == "Morning sky at 60/-30"

    @pytest.mark.asyncio
    async def test_recall_by_camera_position_within_tolerance(
        self, sensory_integration
    ):
        """Test recalling memory within tolerance."""
        camera_pos = CameraPosition(pan_angle=50, tilt_angle=-25)

        await sensory_integration.save_visual_memory(
            content="Test memory",
            image_path="/tmp/test.jpg",
            camera_position=camera_pos,
        )

        # Recall with slightly different position but within tolerance
        results = await sensory_integration.recall_by_camera_position(
            pan_angle=55,
            tilt_angle=-28,
            tolerance=10,
        )

        assert len(results) == 1
        assert results[0].content == "Test memory"

    @pytest.mark.asyncio
    async def test_recall_by_camera_position_outside_tolerance(
        self, sensory_integration
    ):
        """Test no recall when position is outside tolerance."""
        camera_pos = CameraPosition(pan_angle=50, tilt_angle=-25)

        await sensory_integration.save_visual_memory(
            content="Test memory",
            image_path="/tmp/test.jpg",
            camera_position=camera_pos,
        )

        # Recall with position outside tolerance
        results = await sensory_integration.recall_by_camera_position(
            pan_angle=80,  # 30 degrees away
            tilt_angle=-25,
            tolerance=10,
        )

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_recall_by_camera_position_multiple_memories(
        self, sensory_integration
    ):
        """Test recalling multiple memories at similar positions."""
        # Save 3 memories at similar positions
        for i in range(3):
            camera_pos = CameraPosition(pan_angle=60 + i, tilt_angle=-30)
            await sensory_integration.save_visual_memory(
                content=f"Memory {i}",
                image_path=f"/tmp/test{i}.jpg",
                camera_position=camera_pos,
            )

        # Recall all 3 with large tolerance
        results = await sensory_integration.recall_by_camera_position(
            pan_angle=60,
            tilt_angle=-30,
            tolerance=5,
        )

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_recall_excludes_memories_without_camera_position(
        self, memory_store, sensory_integration
    ):
        """Test that recall excludes memories without camera position."""
        # Save visual memory with camera position
        camera_pos = CameraPosition(pan_angle=60, tilt_angle=-30)
        await sensory_integration.save_visual_memory(
            content="With camera position",
            image_path="/tmp/test.jpg",
            camera_position=camera_pos,
        )

        # Save regular memory without camera position
        await memory_store.save(
            content="Without camera position",
            importance=3,
        )

        # Recall should only return the visual memory
        results = await sensory_integration.recall_by_camera_position(
            pan_angle=60,
            tilt_angle=-30,
            tolerance=5,
        )

        assert len(results) == 1
        assert results[0].content == "With camera position"

    @pytest.mark.asyncio
    async def test_recall_returns_newest_first(self, sensory_integration):
        """Test that recall returns newest memories first."""
        camera_pos = CameraPosition(pan_angle=60, tilt_angle=-30)

        # Save 3 memories (with small time gaps)
        mem1 = await sensory_integration.save_visual_memory(
            content="First",
            image_path="/tmp/1.jpg",
            camera_position=camera_pos,
        )
        mem2 = await sensory_integration.save_visual_memory(
            content="Second",
            image_path="/tmp/2.jpg",
            camera_position=camera_pos,
        )
        mem3 = await sensory_integration.save_visual_memory(
            content="Third",
            image_path="/tmp/3.jpg",
            camera_position=camera_pos,
        )

        results = await sensory_integration.recall_by_camera_position(
            pan_angle=60,
            tilt_angle=-30,
        )

        # Should be in reverse chronological order
        assert len(results) == 3
        assert results[0].content == "Third"
        assert results[1].content == "Second"
        assert results[2].content == "First"


class TestGetMemoriesWithSensoryData:
    """Test getting memories with sensory data."""

    @pytest.mark.asyncio
    async def test_get_memories_with_visual_data(
        self, memory_store, sensory_integration
    ):
        """Test getting memories with visual data."""
        # Save visual and audio memories
        await sensory_integration.save_visual_memory(
            content="Visual 1",
            image_path="/tmp/v1.jpg",
            camera_position=CameraPosition(0, 0),
        )
        await sensory_integration.save_audio_memory(
            content="Audio 1",
            audio_path="/tmp/a1.wav",
            transcript="Test",
        )
        await sensory_integration.save_visual_memory(
            content="Visual 2",
            image_path="/tmp/v2.jpg",
            camera_position=CameraPosition(0, 0),
        )

        # Get only visual memories
        results = await sensory_integration.get_memories_with_sensory_data(
            sensory_type="visual"
        )

        assert len(results) == 2
        assert all("Visual" in m.content for m in results)

    @pytest.mark.asyncio
    async def test_get_memories_with_audio_data(
        self, memory_store, sensory_integration
    ):
        """Test getting memories with audio data."""
        await sensory_integration.save_audio_memory(
            content="Audio 1",
            audio_path="/tmp/a1.wav",
            transcript="Test 1",
        )
        await sensory_integration.save_visual_memory(
            content="Visual 1",
            image_path="/tmp/v1.jpg",
            camera_position=CameraPosition(0, 0),
        )

        # Get only audio memories
        results = await sensory_integration.get_memories_with_sensory_data(
            sensory_type="audio"
        )

        assert len(results) == 1
        assert results[0].content == "Audio 1"

    @pytest.mark.asyncio
    async def test_get_all_memories_with_sensory_data(
        self, memory_store, sensory_integration
    ):
        """Test getting all memories with sensory data."""
        # Save mixed memories
        await sensory_integration.save_visual_memory(
            content="Visual",
            image_path="/tmp/v.jpg",
            camera_position=CameraPosition(0, 0),
        )
        await sensory_integration.save_audio_memory(
            content="Audio",
            audio_path="/tmp/a.wav",
            transcript="Test",
        )
        await memory_store.save(content="No sensory data", importance=3)

        # Get all with sensory data (no type filter)
        results = await sensory_integration.get_memories_with_sensory_data(
            sensory_type=None
        )

        assert len(results) == 2  # Excludes the one without sensory data
