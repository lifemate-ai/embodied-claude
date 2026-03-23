"""faster-whisper transcription wrapper."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class Transcriber:
    """Wraps faster-whisper model loading, warmup, and inference."""

    def __init__(self, model_name: str = "small", language: str = "ja"):
        self._language = language
        logger.info("Whisper モデル '%s' を読み込み中...", model_name)
        self._model = WhisperModel(model_name, device="cpu", compute_type="int8")
        logger.info("Whisper モデル '%s' の読み込み完了", model_name)
        self._warmup()

    def _warmup(self) -> None:
        """Run a dummy inference to reduce first-call latency."""
        dummy = np.zeros(16000, dtype=np.float32)
        list(self._model.transcribe(dummy, language=self._language))
        logger.info("Whisper ウォームアップ完了")

    def transcribe(self, audio_path: Path) -> tuple[str, float]:
        """Transcribe an audio file.

        Returns:
            (text, min_no_speech_prob) tuple.
            text is empty string if no speech detected.
        """
        segments, _info = self._model.transcribe(
            str(audio_path),
            language=self._language,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.2,
        )

        texts = []
        min_no_speech = 1.0
        for seg in segments:
            texts.append(seg.text)
            if seg.no_speech_prob < min_no_speech:
                min_no_speech = seg.no_speech_prob

        text = " ".join(texts).strip()
        return text, min_no_speech
