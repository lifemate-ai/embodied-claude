"""Configuration for hearing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._behavior import get_behavior

SEGMENT_DIR = Path("/tmp/hearing_segments")
SEGMENT_LIST = SEGMENT_DIR / "list.csv"
BUFFER_FILE = Path("/tmp/hearing_buffer.jsonl")
PID_FILE = Path("/tmp/hearing-daemon.pid")

SEGMENT_SECS_DEFAULT = 5
POLL_INTERVAL = 0.5
MAX_QUEUE_SIZE = 4


@dataclass
class HearingConfig:
    """Runtime configuration, merged from caller args and mcpBehavior.toml."""

    whisper_model: str = "small"
    language: str = "ja"
    segment_seconds: int = SEGMENT_SECS_DEFAULT
    vad_energy_threshold: float = 0.0

    @classmethod
    def from_toml(cls, **overrides: object) -> "HearingConfig":
        """Build config with priority: overrides > mcpBehavior.toml > defaults."""
        toml_model = get_behavior("hearing", "whisper_model", cls.whisper_model)
        toml_lang = get_behavior("hearing", "language", cls.language)
        toml_seg = get_behavior("hearing", "segment_seconds", cls.segment_seconds)
        toml_vad = get_behavior("hearing", "vad_energy_threshold", cls.vad_energy_threshold)

        return cls(
            whisper_model=str(overrides.get("whisper_model", toml_model)),
            language=str(overrides.get("language", toml_lang)),
            segment_seconds=int(overrides.get("segment_seconds", toml_seg)),
            vad_energy_threshold=float(overrides.get("vad_energy_threshold", toml_vad)),
        )
