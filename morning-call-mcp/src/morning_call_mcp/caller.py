"""Morning call logic: ElevenLabs audio generation + Twilio outbound call."""
import http.server
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from twilio.rest import Client

from .config import config


def generate_audio_elevenlabs(message: str) -> bytes:
    """Generate speech audio via ElevenLabs SDK."""
    if not config.elevenlabs_api_key or not config.elevenlabs_voice_id:
        raise RuntimeError("ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID are required for ElevenLabs TTS")

    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=config.elevenlabs_api_key)
    audio_iter = client.text_to_speech.convert(
        text=message,
        voice_id=config.elevenlabs_voice_id,
        model_id="eleven_v3",
        output_format="mp3_44100_128",
        language_code="ja",
    )
    if isinstance(audio_iter, (bytes, bytearray)):
        return bytes(audio_iter)
    return b"".join(audio_iter)


def normalize_audio(input_path: str) -> str:
    """Re-encode MP3 via ffmpeg to remove artifacts and add trailing silence.
    Returns path to cleaned file (caller must delete it).
    """
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    out = input_path.replace(".mp3", "_clean.mp3")
    subprocess.run(
        [
            ffmpeg, "-y", "-i", input_path,
            "-af", "apad=pad_dur=0.5",
            "-codec:a", "libmp3lame", "-b:a", "128k", "-ar", "44100",
            out,
        ],
        check=True,
        capture_output=True,
    )
    return out


class _AudioHandler(http.server.BaseHTTPRequestHandler):
    """Serves the MP3 file over HTTP."""

    audio_path: str = ""

    def do_GET(self) -> None:
        if self.path == "/audio.mp3":
            audio_bytes = Path(self.__class__.audio_path).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(len(audio_bytes)))
            self.end_headers()
            self.wfile.write(audio_bytes)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        self.do_GET()

    def log_message(self, fmt: str, *args: object) -> None:
        pass


def _start_tunnel(port: int) -> tuple[str, "subprocess.Popen[bytes]"]:
    """Start cloudflared tunnel. Returns (public_url, process).

    Falls back to ngrok if cloudflared is not installed.
    """
    # Try cloudflared first (free, no account needed for Quick Tunnels)
    cloudflared = shutil.which("cloudflared") or os.path.expanduser("~/.local/bin/cloudflared")
    if os.path.exists(cloudflared):
        proc = subprocess.Popen(
            [cloudflared, "tunnel", "--url", f"http://localhost:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        url = None
        for _ in range(120):
            line = proc.stderr.readline().decode().strip()  # type: ignore[union-attr]
            m = re.search(r"https://[\w-]+\.trycloudflare\.com", line)
            if m:
                url = m.group(0)
            if url and "Registered tunnel connection" in line:
                return url, proc
        proc.terminate()
        raise RuntimeError("cloudflared tunnel did not establish in time")

    # Fallback: ngrok
    if config.ngrok_auth_token:
        from pyngrok import conf, ngrok
        conf.get_default().auth_token = config.ngrok_auth_token
        tunnel = ngrok.connect(port, "http")
        # wrap in a dummy Popen-like object
        return tunnel.public_url, None  # type: ignore[return-value]

    raise RuntimeError(
        "cloudflared not found. Install: "
        "wget https://github.com/cloudflare/cloudflared/releases/latest/download/"
        "cloudflared-linux-amd64 -O ~/.local/bin/cloudflared && "
        "chmod +x ~/.local/bin/cloudflared"
    )


def make_call(message: str, use_elevenlabs: bool = True) -> str:
    """
    Make a phone call with the given message.

    ElevenLabs path:
      1. Generate audio via ElevenLabs
      2. Re-encode via ffmpeg (removes artifacts, adds trailing silence)
      3. Serve audio via local HTTP server exposed through cloudflared tunnel
      4. Initiate Twilio outbound call with TwiML <Play> pointing to the tunnel URL

    Fallback:
      Uses Twilio built-in TTS (<Say>).

    Note (Twilio trial accounts):
      The call starts with a trial announcement asking the recipient to press any key.
      Press any key on the phone to proceed to the actual audio.
    """
    config.validate()

    twilio_client = Client(config.twilio_account_sid, config.twilio_auth_token)

    if use_elevenlabs and config.elevenlabs_api_key:
        audio_bytes = generate_audio_elevenlabs(message)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            raw_path = f.name

        clean_path = None
        tunnel_proc = None
        try:
            clean_path = normalize_audio(raw_path)

            # Serve audio via local HTTP server
            _AudioHandler.audio_path = clean_path
            server = http.server.HTTPServer(("0.0.0.0", config.local_port), _AudioHandler)
            t = threading.Thread(target=server.serve_forever, daemon=True)
            t.start()

            public_url, tunnel_proc = _start_tunnel(config.local_port)
            audio_url = f"{public_url}/audio.mp3"

            twiml = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                "<Response>"
                f"<Play>{audio_url}</Play>"
                "<Hangup/>"
                "</Response>"
            )

            call = twilio_client.calls.create(
                twiml=twiml,
                to=config.twilio_to_number,
                from_=config.twilio_from_number,
            )

            # Wait for call to complete
            for _ in range(60):
                time.sleep(5)
                status = twilio_client.calls(call.sid).fetch().status
                if status in ("completed", "failed", "busy", "no-answer", "canceled"):
                    break

            return call.sid

        finally:
            server.shutdown()
            for p in [raw_path, clean_path]:
                if p:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
            if tunnel_proc:
                tunnel_proc.terminate()
            elif config.ngrok_auth_token:
                try:
                    from pyngrok import ngrok
                    ngrok.kill()
                except Exception:
                    pass

    else:
        # Fallback: use Twilio built-in TTS
        twiml = f'<Response><Say language="ja-JP">{message}</Say></Response>'
        call = twilio_client.calls.create(
            twiml=twiml,
            to=config.twilio_to_number,
            from_=config.twilio_from_number,
        )
        return call.sid
