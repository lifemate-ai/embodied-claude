"""Configuration for morning-call-mcp."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project directory
_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(_env_path)


def _require(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"Environment variable {key} is required")
    return val


class Config:
    # Twilio
    twilio_account_sid: str = os.environ.get("TWILIO_ACCOUNT_SID", "")
    twilio_auth_token: str = os.environ.get("TWILIO_AUTH_TOKEN", "")
    twilio_from_number: str = os.environ.get("TWILIO_FROM_NUMBER", "")
    twilio_to_number: str = os.environ.get("TWILIO_TO_NUMBER", "")

    # ElevenLabs
    elevenlabs_api_key: str = os.environ.get("ELEVENLABS_API_KEY", "")
    elevenlabs_voice_id: str = os.environ.get("ELEVENLABS_VOICE_ID", "")

    # ngrok (optional fallback if cloudflared is not installed)
    ngrok_auth_token: str = os.environ.get("NGROK_AUTH_TOKEN", "")

    # Local HTTP server port for serving audio
    local_port: int = int(os.environ.get("MORNING_CALL_PORT", "18765"))

    def validate(self) -> None:
        missing = []
        for key in ("twilio_account_sid", "twilio_auth_token", "twilio_from_number", "twilio_to_number"):
            if not getattr(self, key):
                missing.append(key.upper())
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


config = Config()
