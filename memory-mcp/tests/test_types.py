"""Tests for Phase 4 type definitions."""

import pytest
from datetime import datetime, timezone

from src.memory_mcp.types import (
    CameraPosition,
    Episode,
    Memory,
    SensoryData,
)


class TestCameraPosition:
    """Test CameraPosition dataclass."""

    def test_create_camera_position(self):
        """Test creating a CameraPosition."""
        pos = CameraPosition(pan_angle=60, tilt_angle=-30)

        assert pos.pan_angle == 60
        assert pos.tilt_angle == -30
        assert pos.preset_id is None

    def test_camera_position_with_preset(self):
        """Test CameraPosition with preset_id."""
        pos = CameraPosition(pan_angle=0, tilt_angle=0, preset_id="home")

        assert pos.preset_id == "home"

    def test_camera_position_to_dict(self):
        """Test to_dict serialization."""
        pos = CameraPosition(pan_angle=45, tilt_angle=-20, preset_id="window")

        data = pos.to_dict()

        assert data == {
            "pan_angle": 45,
            "tilt_angle": -20,
            "preset_id": "window",
        }

    def test_camera_position_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "pan_angle": 30,
            "tilt_angle": -15,
            "preset_id": None,
        }

        pos = CameraPosition.from_dict(data)

        assert pos.pan_angle == 30
        assert pos.tilt_angle == -15
        assert pos.preset_id is None

    def test_camera_position_roundtrip(self):
        """Test serialization roundtrip."""
        original = CameraPosition(pan_angle=90, tilt_angle=-45, preset_id="sky")

        data = original.to_dict()
        restored = CameraPosition.from_dict(data)

        assert restored == original


class TestSensoryData:
    """Test SensoryData dataclass."""

    def test_create_visual_sensory_data(self):
        """Test creating visual sensory data."""
        timestamp = datetime.now(timezone.utc).isoformat()
        data = SensoryData(
            sensory_type="visual",
            file_path="/tmp/image.jpg",
            metadata={"width": 640, "height": 480},
            description="A beautiful morning sky",
            timestamp=timestamp,
        )

        assert data.sensory_type == "visual"
        assert data.file_path == "/tmp/image.jpg"
        assert data.metadata["width"] == 640
        assert data.description == "A beautiful morning sky"

    def test_create_audio_sensory_data(self):
        """Test creating audio sensory data."""
        timestamp = datetime.now(timezone.utc).isoformat()
        data = SensoryData(
            sensory_type="audio",
            file_path="/tmp/audio.wav",
            metadata={"transcript": "Hello world"},
            description="Greeting",
            timestamp=timestamp,
        )

        assert data.sensory_type == "audio"
        assert data.metadata["transcript"] == "Hello world"

    def test_sensory_data_to_dict(self):
        """Test to_dict serialization."""
        timestamp = "2026-02-01T12:00:00+00:00"
        data = SensoryData(
            sensory_type="visual",
            file_path="/tmp/test.jpg",
            metadata={"camera": "Tapo"},
            description=None,
            timestamp=timestamp,
        )

        result = data.to_dict()

        assert result["sensory_type"] == "visual"
        assert result["file_path"] == "/tmp/test.jpg"
        assert result["metadata"] == {"camera": "Tapo"}
        assert result["description"] is None
        assert result["timestamp"] == timestamp

    def test_sensory_data_from_dict(self):
        """Test from_dict deserialization."""
        timestamp = "2026-02-01T12:00:00+00:00"
        dict_data = {
            "sensory_type": "visual",
            "file_path": "/tmp/image.jpg",
            "metadata": {"width": 1920, "height": 1080},
            "description": "Test image",
            "timestamp": timestamp,
        }

        data = SensoryData.from_dict(dict_data)

        assert data.sensory_type == "visual"
        assert data.file_path == "/tmp/image.jpg"
        assert data.metadata["width"] == 1920
        assert data.description == "Test image"

    def test_sensory_data_roundtrip(self):
        """Test serialization roundtrip."""
        timestamp = datetime.now(timezone.utc).isoformat()
        original = SensoryData(
            sensory_type="audio",
            file_path="/tmp/sound.wav",
            metadata={"duration": 5.2},
            description="Bird chirping",
            timestamp=timestamp,
        )

        dict_data = original.to_dict()
        restored = SensoryData.from_dict(dict_data)

        assert restored == original


class TestEpisode:
    """Test Episode dataclass."""

    def test_create_episode(self):
        """Test creating an Episode."""
        episode = Episode(
            id="ep1",
            title="Morning sky search",
            start_time="2026-02-01T07:52:00+00:00",
            end_time="2026-02-01T07:53:00+00:00",
            memory_ids=("mem1", "mem2", "mem3"),
            participants=("幼馴染",),
            location_context="Balcony",
            summary="Found the morning sky after searching",
            emotion="excited",
            importance=5,
        )

        assert episode.id == "ep1"
        assert episode.title == "Morning sky search"
        assert len(episode.memory_ids) == 3
        assert episode.participants == ("幼馴染",)
        assert episode.importance == 5

    def test_episode_to_metadata(self):
        """Test to_metadata serialization."""
        episode = Episode(
            id="ep1",
            title="Test Episode",
            start_time="2026-02-01T10:00:00+00:00",
            end_time="2026-02-01T11:00:00+00:00",
            memory_ids=("m1", "m2"),
            participants=("Alice", "Bob"),
            location_context="Room",
            summary="A test episode",
            emotion="happy",
            importance=4,
        )

        metadata = episode.to_metadata()

        assert metadata["title"] == "Test Episode"
        assert metadata["start_time"] == "2026-02-01T10:00:00+00:00"
        assert metadata["end_time"] == "2026-02-01T11:00:00+00:00"
        assert metadata["memory_ids"] == "m1,m2"
        assert metadata["participants"] == "Alice,Bob"
        assert metadata["location_context"] == "Room"
        assert metadata["emotion"] == "happy"
        assert metadata["importance"] == 4

    def test_episode_from_metadata(self):
        """Test from_metadata deserialization."""
        metadata = {
            "title": "Test Episode",
            "start_time": "2026-02-01T10:00:00+00:00",
            "end_time": "2026-02-01T11:00:00+00:00",
            "memory_ids": "m1,m2,m3",
            "participants": "User",
            "location_context": "Office",
            "emotion": "curious",
            "importance": 3,
        }

        episode = Episode.from_metadata(
            id="ep1",
            summary="Generated summary",
            metadata=metadata,
        )

        assert episode.id == "ep1"
        assert episode.title == "Test Episode"
        assert episode.memory_ids == ("m1", "m2", "m3")
        assert episode.participants == ("User",)
        assert episode.summary == "Generated summary"

    def test_episode_with_none_end_time(self):
        """Test episode with None end_time (ongoing)."""
        episode = Episode(
            id="ep1",
            title="Ongoing Episode",
            start_time="2026-02-01T10:00:00+00:00",
            end_time=None,
            memory_ids=("m1",),
            participants=(),
            location_context=None,
            summary="In progress",
            emotion="neutral",
            importance=3,
        )

        metadata = episode.to_metadata()

        assert metadata["end_time"] == ""
        assert metadata["location_context"] == ""


class TestMemoryPhase4Fields:
    """Test Memory with Phase 4 fields."""

    def test_memory_with_camera_position(self):
        """Test Memory with camera position."""
        timestamp = datetime.now(timezone.utc).isoformat()
        camera_pos = CameraPosition(pan_angle=60, tilt_angle=-30)

        memory = Memory(
            id="m1",
            content="Found the morning sky",
            timestamp=timestamp,
            emotion="excited",
            importance=5,
            category="observation",
            camera_position=camera_pos,
        )

        assert memory.camera_position == camera_pos
        assert memory.camera_position.pan_angle == 60

    def test_memory_with_sensory_data(self):
        """Test Memory with sensory data."""
        timestamp = datetime.now(timezone.utc).isoformat()
        sensory = SensoryData(
            sensory_type="visual",
            file_path="/tmp/sky.jpg",
            metadata={"camera": "Tapo"},
            description="Morning sky",
            timestamp=timestamp,
        )

        memory = Memory(
            id="m1",
            content="Captured the morning sky",
            timestamp=timestamp,
            emotion="happy",
            importance=4,
            category="observation",
            sensory_data=(sensory,),
        )

        assert len(memory.sensory_data) == 1
        assert memory.sensory_data[0].file_path == "/tmp/sky.jpg"

    def test_memory_with_episode_id(self):
        """Test Memory with episode_id."""
        timestamp = datetime.now(timezone.utc).isoformat()

        memory = Memory(
            id="m1",
            content="Part of an episode",
            timestamp=timestamp,
            emotion="neutral",
            importance=3,
            category="daily",
            episode_id="ep1",
        )

        assert memory.episode_id == "ep1"

    def test_memory_with_tags(self):
        """Test Memory with tags."""
        timestamp = datetime.now(timezone.utc).isoformat()

        memory = Memory(
            id="m1",
            content="Tagged memory",
            timestamp=timestamp,
            emotion="neutral",
            importance=3,
            category="daily",
            tags=("important", "work", "deadline"),
        )

        assert memory.tags == ("important", "work", "deadline")
        assert len(memory.tags) == 3

    def test_memory_to_metadata_with_phase4_fields(self):
        """Test to_metadata with Phase 4 fields."""
        timestamp = "2026-02-01T12:00:00+00:00"
        camera_pos = CameraPosition(pan_angle=45, tilt_angle=-20)
        sensory = SensoryData(
            sensory_type="visual",
            file_path="/tmp/test.jpg",
            metadata={},
            description=None,
            timestamp=timestamp,
        )

        memory = Memory(
            id="m1",
            content="Test memory",
            timestamp=timestamp,
            emotion="happy",
            importance=4,
            category="observation",
            episode_id="ep1",
            sensory_data=(sensory,),
            camera_position=camera_pos,
            tags=("test", "phase4"),
        )

        metadata = memory.to_metadata()

        assert metadata["episode_id"] == "ep1"
        assert "sensory_data" in metadata  # JSON string
        assert "camera_position" in metadata  # JSON string
        assert metadata["tags"] == "test,phase4"
