# morning-call-mcp

MCP server for making morning calls using any ElevenLabs voice via Twilio.

## Setup

### 1. Install dependencies

```bash
cd morning-call-mcp
uv sync
```

### 2. Install cloudflared (recommended)

Twilio needs a publicly accessible URL to fetch the audio file.
cloudflared provides free Quick Tunnels with no account required.

```bash
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
  -O ~/.local/bin/cloudflared && chmod +x ~/.local/bin/cloudflared
```

If you prefer ngrok, set `NGROK_AUTH_TOKEN` and it will be used as a fallback.

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env
```

| Variable | Description |
|----------|-------------|
| `TWILIO_ACCOUNT_SID` | Account SID from Twilio Console |
| `TWILIO_AUTH_TOKEN` | Auth Token from Twilio Console |
| `TWILIO_FROM_NUMBER` | Your Twilio phone number (e.g. `+18126841174`) |
| `TWILIO_TO_NUMBER` | Recipient's phone number (e.g. `+819XXXXXXXXX`) |
| `ELEVENLABS_API_KEY` | ElevenLabs API key (falls back to Twilio TTS if omitted) |
| `ELEVENLABS_VOICE_ID` | ElevenLabs voice ID |
| `NGROK_AUTH_TOKEN` | ngrok auth token (fallback if cloudflared is unavailable) |

### 4. Install ffmpeg

Used to re-encode audio and remove trailing artifacts.

```bash
# Ubuntu/Debian
sudo apt install ffmpeg
```

### 5. Test the call

```bash
uv run python -c "
from morning_call_mcp.caller import make_call
sid = make_call('Good morning! Time to wake up!')
print('Call SID:', sid)
"
```

> **Note (Twilio trial accounts)**: Trial calls start with an English announcement asking the recipient to press any key. Press any key to proceed to your audio. Paid accounts skip this announcement.

## Use as an MCP server

Add to `.mcp.json`:

```json
{
  "mcpServers": {
    "morning-call": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/morning-call-mcp", "morning-call-mcp"],
      "env": {
        "TWILIO_ACCOUNT_SID": "ACxxx",
        "TWILIO_AUTH_TOKEN": "xxx",
        "TWILIO_FROM_NUMBER": "+1xxxxxxxxxx",
        "TWILIO_TO_NUMBER": "+1xxxxxxxxxx",
        "ELEVENLABS_API_KEY": "sk_xxx",
        "ELEVENLABS_VOICE_ID": "xxx"
      }
    }
  }
}
```

## Automate with cron

```bash
# Add to crontab -e
# Call every morning at 7:00
0 7 * * * cd /path/to/morning-call-mcp && uv run python -c "
from morning_call_mcp.caller import make_call
make_call('[cheerful] Good morning! Time to wake up!')
"
```

## MCP Tools

### make_morning_call

Make a phone call with a spoken message using ElevenLabs TTS.

```json
{
  "message": "[cheerful] Good morning! Wake up!",
  "use_elevenlabs": true
}
```

### get_call_config

Show current configuration status.

## How it works

```
cron / MCP tool call
  ↓
① Generate audio via ElevenLabs
② Re-encode with ffmpeg (remove artifacts + add trailing silence)
③ Start local HTTP server to serve the audio
④ Expose via cloudflared Quick Tunnel
⑤ Initiate Twilio outbound call
⑥ Twilio fetches audio via cloudflared → plays it
⑦ Shutdown server and tunnel after call ends
```

## License

MIT
