"""Atlas Cloud TTS engine."""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any


class AtlasCloudEngine:
    """Atlas Cloud asynchronous audio generation API."""

    def __init__(
        self,
        api_key: str,
        model_id: str = "minimax/speech-2.6-turbo",
        voice_id: str = "English_expressive_narrator",
        output_format: str = "mp3",
        base_url: str = "https://api.atlascloud.ai",
        poll_interval: float = 1.0,
        timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._voice_id = voice_id
        self._output_format = output_format
        self._base_url = base_url.rstrip("/")
        self._poll_interval = poll_interval
        self._timeout = timeout

    @property
    def engine_name(self) -> str:
        return "atlascloud"

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _request_json(self, request: urllib.request.Request) -> dict[str, Any]:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())

    def synthesize(self, text: str, **kwargs: Any) -> tuple[bytes, str]:
        """Submit once, then poll the returned prediction until it completes."""
        model_id = kwargs.get("model_id") or self._model_id
        voice_id = kwargs.get("voice_id") or self._voice_id
        output_format = kwargs.get("output_format") or self._output_format
        payload = {
            "model": model_id,
            "text": text,
            "voice": voice_id,
            "format": output_format,
        }
        request = urllib.request.Request(
            f"{self._base_url}/api/v1/model/generateAudio",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        prediction = self._request_json(request)
        prediction_id = prediction.get("id")
        if not prediction_id:
            raise RuntimeError("Atlas Cloud did not return a prediction ID")

        deadline = time.monotonic() + self._timeout
        while prediction.get("status") not in {"completed", "failed"}:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Atlas Cloud prediction {prediction_id} timed out")
            time.sleep(self._poll_interval)
            poll_request = urllib.request.Request(
                f"{self._base_url}/api/v1/model/prediction/{prediction_id}",
                headers={"Authorization": f"Bearer {self._api_key}"},
                method="GET",
            )
            prediction = self._request_json(poll_request)

        if prediction.get("status") == "failed":
            raise RuntimeError(f"Atlas Cloud prediction {prediction_id} failed")
        outputs = prediction.get("outputs") or []
        if not outputs:
            raise RuntimeError(f"Atlas Cloud prediction {prediction_id} returned no audio")

        audio_request = urllib.request.Request(outputs[0], method="GET")
        with urllib.request.urlopen(audio_request, timeout=60) as response:
            return response.read(), output_format
