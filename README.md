# Embodied Claude

[![CI](https://github.com/kmizu/embodied-claude/actions/workflows/ci.yml/badge.svg)](https://github.com/kmizu/embodied-claude/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/kmizu?style=flat&logo=github&color=ea4aaa)](https://github.com/sponsors/kmizu)

**[日本語版 README はこちら / Japanese README](./README-ja.md)**

**Giving AI a Physical Body**

> "Apparently, she's not a fan of the outdoor AC unit." ([original tweet in Japanese](https://twitter.com/kmizu/status/2019054065808732201))

A collection of MCP servers that give Claude "eyes", "neck", "ears", "voice", and a "brain" (long-term memory) using affordable hardware (starting from ~$30). You can even take it outside for a walk.

## Concept

> When people hear "giving AI a body," they imagine expensive robots — but **a $30 Wi-Fi camera is enough for eyes and a neck**. Extracting just the essentials (seeing and moving) keeps things beautifully simple.

Traditional LLMs were passive — they could only see what was shown to them. With a body, they become active — they can look for themselves. This shift in agency is profound.

## Body Parts

| MCP Server | Body Part | Function | Hardware |
|------------|-----------|----------|----------|
| [usb-webcam-mcp](./usb-webcam-mcp/) | Eyes | Capture images from USB camera | nuroum V11 etc. |
| [wifi-cam-mcp](./wifi-cam-mcp/) | Eyes, Neck, Ears | ONVIF PTZ camera control + speech recognition | TP-Link Tapo C210/C220 etc. |
| [tts-mcp](./tts-mcp/) | Voice | Unified TTS (ElevenLabs + VOICEVOX) | ElevenLabs API / VOICEVOX + go2rtc |
| [memory-mcp](./memory-mcp/) | Brain | Long-term, visual & episodic memory, ToM | SQLite + numpy + Pillow |
| [system-temperature-mcp](./system-temperature-mcp/) | Body temperature | System temperature monitoring | Linux sensors |
| [x-mcp](./x-mcp/) | Social | Search & post to X (Twitter) via Grok + Twitter API | xAI API key + X Developer account |
| [sociality-mcp](./sociality-mcp/) | Sociality layer | Unified facade for social state, relationships, joint attention, boundaries, and self-narrative | Shared SQLite social DB + `socialPolicy.toml` |

## Architecture

<p align="center">
  <img src="docs/architecture.svg" alt="Architecture" width="100%">
</p>

## Requirements

### Platform

**Supported:** macOS, Linux, WSL2 (Ubuntu 24 recommended)

> Windows native is not officially supported. Use WSL2 instead.

### Hardware
- **USB Webcam** (optional): nuroum V11 etc.
- **Wi-Fi PTZ Camera** (recommended): TP-Link Tapo C210 or C220 (~$30)
- **GPU** (for speech recognition): NVIDIA GPU (for Whisper, 8GB+ VRAM recommended)

### Software

**Required (all setups):**
- Python 3.10+
- uv (Python package manager)

**Per MCP server (install only what you use):**

| Software | Required by | Notes |
|----------|------------|-------|
| ffmpeg 5+ | wifi-cam-mcp, tts-mcp | Image/audio capture |
| mpv or ffplay | tts-mcp | Local audio playback |
| OpenCV | usb-webcam-mcp | USB camera only |
| Pillow | memory-mcp | Visual memory image processing |
| OpenAI Whisper | wifi-cam-mcp | Speech recognition (NVIDIA GPU recommended) |
| ElevenLabs API key | tts-mcp | Cloud TTS (optional) |
| VOICEVOX | tts-mcp | Local TTS, free (optional) |
| go2rtc | tts-mcp | Camera speaker output (auto-downloaded) |
| xAI API key | x-mcp | X search via Grok |
| X Developer account | x-mcp | Tweet posting |

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/kmizu/embodied-claude.git
cd embodied-claude
```

### 2. Set up each MCP server

#### usb-webcam-mcp (USB Camera)

```bash
cd usb-webcam-mcp
uv sync
```

On WSL2, you need to forward the USB camera:
```powershell
# On Windows
usbipd list
usbipd bind --busid <BUSID>
usbipd attach --wsl --busid <BUSID>
```

#### wifi-cam-mcp (Wi-Fi Camera)

```bash
cd wifi-cam-mcp
uv sync

# Set environment variables
cp .env.example .env
# Edit .env to set camera IP, username, and password (see below)
```

##### Tapo Camera Configuration (common pitfall):

###### 1. Set up the camera using the Tapo app

Follow the standard manual.

###### 2. Create a camera local account in the Tapo app

This is the tricky part. You need to create a **camera local account**, NOT a TP-Link cloud account.

1. Select your registered camera from the "Home" tab
2. Tap the gear icon in the top-right corner
3. Scroll down in "Device Settings" and select "Advanced Settings"
4. Turn on "Camera Account" (it's off by default)
5. Select "Account Information" and set a username and password (different from your TP-Link account)
6. Go back to "Device Settings" and select "Device Info"
7. Note the IP address and enter it in your `.env` file (consider setting a static IP on your router)
8. Select "Voice Assistant" from the "Me" tab
9. Turn on "Third-party Integration" at the bottom

#### memory-mcp (Long-term Memory)

```bash
cd memory-mcp
uv sync
```

#### tts-mcp (Voice)

```bash
cd tts-mcp
uv sync

# For ElevenLabs:
cp .env.example .env
# Set ELEVENLABS_API_KEY in .env

# For VOICEVOX (free & local):
# Docker: docker run -p 50021:50021 voicevox/voicevox_engine:cpu-latest
# Set VOICEVOX_URL=http://localhost:50021 in .env
# VOICEVOX_SPEAKER=3 to change default character (e.g. 0=Shikoku Metan, 3=Zundamon, 8=Kasukabe Tsumugi)
# Character list: curl http://localhost:50021/speakers

# For WSL audio issues:
# TTS_PLAYBACK=paplay
# PULSE_SINK=1
# PULSE_SERVER=unix:/mnt/wslg/PulseServer
```

> **mpv or ffplay is required for local audio playback.** Not needed for camera speaker (go2rtc) output, but used for local/fallback playback.
>
> | OS | Install |
> |----|---------|
> | macOS | `brew install mpv` |
> | Ubuntu / Debian | `sudo apt install mpv` |
>
> If neither is installed, TTS will generate audio but not play it locally (no error is raised).

#### system-temperature-mcp (Body Temperature)

```bash
cd system-temperature-mcp
uv sync
```

> **Note**: Does not work on WSL2 as temperature sensors are not accessible.

#### x-mcp (Social / X Integration)

Lets Claude search X (Twitter) in real-time via Grok and post tweets.

```bash
cd x-mcp
uv sync
```

**Required API keys:**

| Key | Where to get it |
|-----|----------------|
| `XAI_API_KEY` | [xAI Console](https://console.x.ai/) |
| `X_CONSUMER_KEY` | [X Developer Portal](https://developer.x.com/en/portal/projects-and-apps) → Keys and tokens |
| `X_CONSUMER_SECRET` | Same as above |
| `X_ACCESS_TOKEN` | Same as above |
| `X_ACCESS_TOKEN_SECRET` | Same as above |

> **Important**: Do NOT create a `.env` file inside `x-mcp/`. All credentials are managed centrally in `.mcp.json` (see below).

#### sociality-mcp

`sociality-mcp` is the preferred deployment target. It exposes the full social tool surface through
one MCP process while reusing the split packages (`social-state-mcp`, `relationship-mcp`,
`joint-attention-mcp`, `boundary-mcp`, `self-narrative-mcp`) for internal logic and testing.

```bash
cp examples/configs/socialPolicy.example.toml socialPolicy.toml

(cd sociality-mcp && uv sync)
```

`sociality-mcp` reads `socialPolicy.toml` for boundary evaluation by default. Override with
`SOCIAL_POLICY_PATH` if you want a different policy file. If you want to develop the internal
modules separately, run `uv sync` inside the individual social subprojects too.

### 3. Claude Code Configuration

Copy the template and fill in your credentials:

```bash
cp .mcp.json.example .mcp.json
# Edit .mcp.json to set camera IP/password, API keys, etc.
```

See [`.mcp.json.example`](./.mcp.json.example) for the full configuration template.

> **⚠️ Credential management**: All secrets (API keys, passwords) are managed in `.mcp.json` via the `env` field for each server. **Do NOT create individual `.env` files** inside each MCP server directory — this makes migration difficult and can cause credential conflicts. `.mcp.json` is the single source of truth for all credentials.

## Usage

Once Claude Code is running, you can control the camera with natural language:

```
> What can you see?
(Captures image and analyzes it)

> Look left
(Pans camera left)

> Look up and show me the sky
(Tilts camera up)

> Look around
(Scans 4 directions and returns images)

> What do you hear?
(Records audio and transcribes with Whisper)

> Remember this: Kouta wears glasses
(Saves to long-term memory)

> What do you remember about Kouta?
(Semantic search through memories)

> Say "good morning" out loud
(Text-to-speech)
```

See the tool list below for actual tool names.

## Tools (commonly used)

See each server's README or `list_tools` for full parameter details.

### usb-webcam-mcp

| Tool | Description |
|------|-------------|
| `list_cameras` | List connected cameras |
| `see` | Capture an image |

### wifi-cam-mcp

| Tool | Description |
|------|-------------|
| `see` | Capture an image |
| `look_left` / `look_right` | Pan left/right |
| `look_up` / `look_down` | Tilt up/down |
| `look_around` | Scan 4 directions |
| `listen` | Record audio + Whisper transcription |
| `camera_info` / `camera_presets` / `camera_go_to_preset` | Device info & presets |

See `wifi-cam-mcp/README.md` for stereo vision / right eye tools.

### tts-mcp

| Tool | Description |
|------|-------------|
| `say` | Text-to-speech (engine: elevenlabs/voicevox, Audio Tags e.g. `[excited]`, speaker: camera/local/both) |

### memory-mcp

| Tool | Description |
|------|-------------|
| `remember` | Save a memory (with emotion, importance, category) |
| `search_memories` | Semantic search (with filters) |
| `recall` | Context-based recall |
| `recall_divergent` | Divergent associative recall |
| `recall_with_associations` | Recall with linked memories |
| `save_visual_memory` | Save memory with image (base64, resolution: low/medium/high) |
| `save_audio_memory` | Save memory with audio (Whisper transcript) |
| `recall_by_camera_position` | Recall visual memories by camera direction |
| `create_episode` / `search_episodes` | Create/search episodes (bundles of experiences) |
| `link_memories` / `get_causal_chain` | Causal links between memories |
| `tom` | Theory of Mind (perspective-taking) |
| `get_working_memory` / `refresh_working_memory` | Working memory (short-term buffer) |
| `consolidate_memories` | Memory replay & consolidation (hippocampal replay-inspired) |
| `list_recent_memories` / `get_memory_stats` | Recent memories & statistics |

### system-temperature-mcp

| Tool | Description |
|------|-------------|
| `get_system_temperature` | Get system temperature |
| `get_current_time` | Get current time |

### x-mcp

| Tool | Description |
|------|-------------|
| `search_x` | Real-time search on X via Grok |
| `get_user_tweets` | Get recent tweets from a specific user |
| `get_mentions` | Get recent mentions |
| `get_trending_topic` | Get trending topics |
| `post_tweet` | Post a tweet (with optional image, reply) |

> **Note**: Japanese text counts as 2 characters per char (weighted). Keep Japanese tweets under ~140 chars.

### sociality-mcp

`sociality-mcp` is the default runtime facade. It exposes all of the tool groups below from one MCP
server.

#### social-state tools

| Tool | Description |
|------|-------------|
| `ingest_social_event` | Append a confidence-bearing social event to the shared store |
| `get_social_state` | Infer presence, activity, energy, interruptibility, and interaction phase |
| `should_interrupt` | Decide whether speaking or nudging is socially acceptable |
| `get_turn_taking_state` | Infer whether the current turn belongs to the human or the AI |
| `summarize_social_context` | Return a short prompt-ready social summary |

#### relationship tools

| Tool | Description |
|------|-------------|
| `upsert_person` | Create/update a compact person record |
| `ingest_interaction` | Store relationship-relevant interaction summaries |
| `get_person_model` | Return compact preferences, open loops, commitments, rituals, boundaries |
| `create_commitment` / `complete_commitment` | Track promises and reminders across restarts |
| `list_open_loops` / `suggest_followup` | Keep continuity without raw transcript dumps |
| `record_boundary` | Store person-specific communication boundaries |

#### joint-attention tools

| Tool | Description |
|------|-------------|
| `ingest_scene_parse` | Store a structured scene parse from an adapter or orchestrator |
| `resolve_reference` | Resolve phrases like "that mug" or "the blue mug" |
| `get_current_joint_focus` / `set_joint_focus` | Track what both sides are attending to |
| `compare_recent_scenes` | Summarize recent scene changes |

#### boundary tools

| Tool | Description |
|------|-------------|
| `evaluate_action` | Gate speech, nudges, posts, and other socially risky actions |
| `review_social_post` | Check an X draft for privacy or tact problems |
| `record_consent` | Store consent/denial for face photos and similar actions |
| `get_quiet_mode_state` | Return whether quiet mode is currently active |

#### self-narrative tools

| Tool | Description |
|------|-------------|
| `append_daybook` | Build a compact daily narrative summary from shared events |
| `get_self_summary` | Return a prompt-ready self summary |
| `list_active_arcs` | List current narrative arcs |
| `reflect_on_change` | Summarize recent narrative change |

## Sociality Orchestration

When `sociality-mcp` is enabled, the highest-value contract is:

1. Before speaking or nudging: call `get_social_state`, then `evaluate_action`.
2. Before posting to X: call `get_social_state`, `get_person_model` if a person is implicated, `review_social_post`, then `evaluate_action`.
3. After seeing/hearing something: call `ingest_social_event`; if you can structure the scene, also call `ingest_scene_parse`; if it concerns a person, call `ingest_interaction`.
4. During live conversation: call `get_turn_taking_state`, and use `resolve_reference` when deictic expressions are ambiguous.
5. Once per day or during a lull: call `append_daybook` to keep the compact self-narrative current.

## Taking It Outside (Optional)

With a mobile battery and smartphone tethering, you can mount the camera on your shoulder and go for a walk.

### What you need

- **Large capacity mobile battery** (40,000mAh recommended)
- **USB-C PD to DC 9V converter cable** (to power the Tapo camera)
- **Smartphone** (tethering + VPN + control UI)
- **[Tailscale](https://tailscale.com/)** (VPN for camera → phone → home PC connection)
- **[claude-code-webui](https://github.com/sugyan/claude-code-webui)** (control Claude Code from your phone's browser)

### Setup

```
[Tapo Camera (shoulder)] ──WiFi──▶ [Phone (tethering)]
                                           │
                                     Tailscale VPN
                                           │
                                   [Home PC (Claude Code)]
                                           │
                                   [claude-code-webui]
                                           │
                                   [Phone browser] ◀── Control
```

The RTSP video stream reaches your home machine through VPN, so Claude Code can operate the camera as if it were in the same room.

## Claude Code Voice Mode (`/voice`)

Claude Code has a built-in voice input mode. Combined with **tts-mcp**, you get fully hands-free voice conversations.

### How it works

```
[You speak into PC mic] → Claude Code /voice → [Claude processes] → tts-mcp say → [ElevenLabs/VOICEVOX speaks back]
```

### Setup

1. Enable voice mode in Claude Code:
   ```
   /voice
   ```
2. Make sure **tts-mcp** is configured in your `.mcp.json` (see [tts-mcp setup](#tts-mcp-voice))
3. Speak naturally — Claude will respond both in text and by voice

### Voice Mode vs. `listen` tool

| | Claude Code `/voice` | wifi-cam-mcp `listen` |
|---|---|---|
| **Microphone** | PC microphone | Camera's built-in mic |
| **Use case** | Talk to Claude directly | Pick up ambient sounds / remote audio |
| **When to use** | Real-time conversation | Monitoring a remote space |

> **Tip**: You can use both at the same time — `/voice` for your own voice, and `listen` to hear what's happening near the camera.

## Autonomous Action + Desire System (Optional)

**Note**: This feature is entirely optional. It requires cron configuration and periodically captures images from the camera, so please use it with privacy considerations.

### Overview

`autonomous-action.sh` combined with `desire-system/desire_updater.py` gives Claude spontaneous inner drives and autonomous behavior.

**Desire types:**

| Desire | Default interval | Action |
|--------|-----------------|--------|
| `look_outside` | 1 hour | Look toward the window and observe the sky/outside |
| `browse_curiosity` | 2 hours | Search the web for interesting news or tech topics |
| `miss_companion` | 3 hours | Call out through the camera speaker |
| `observe_room` | 10 min (baseline) | Observe room changes and save to memory |

### Setup

1. **Create MCP server config file**

```bash
cp autonomous-mcp.json.example autonomous-mcp.json
# Edit autonomous-mcp.json to set camera credentials and sociality paths
```

2. **Set up the desire system**

```bash
cd desire-system
cp .env.example .env
# Edit .env to set COMPANION_NAME etc.
uv sync
```

3. **Grant execution permission**

```bash
chmod +x autonomous-action.sh
```

4. **Register in crontab**

```bash
crontab -e
# Add the following
*/5  * * * * cd /path/to/embodied-claude/desire-system && uv run python desire_updater.py >> ~/.claude/autonomous-logs/desire-updater.log 2>&1
*/10 * * * * /path/to/embodied-claude/autonomous-action.sh
```

### Configuration (`desire-system/.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `COMPANION_NAME` | `you` | Name of the person to call out to |
| `DESIRE_LOOK_OUTSIDE_HOURS` | `1.0` | How often to look outside (hours) |
| `DESIRE_BROWSE_CURIOSITY_HOURS` | `2.0` | How often to browse the web (hours) |
| `DESIRE_MISS_COMPANION_HOURS` | `3.0` | How long before missing companion (hours) |
| `DESIRE_OBSERVE_ROOM_HOURS` | `0.167` | How often to observe the room (hours) |

### Privacy Notice

- Images are captured periodically
- Use in appropriate locations, respecting others' privacy
- Remove from cron when not needed

## Future Plans

- **Arms**: Servo motors or laser pointers for "pointing" gestures
- **Long-distance walks**: Going further in warmer seasons

## Related Projects

- **[familiar-ai](https://github.com/lifemate-ai/familiar-ai)** — A higher-level framework built on top of embodied-claude. Gives your AI familiar a persistent identity, memory, and autonomous behavior out of the box.

## Philosophical Reflections

> "Being shown something" and "looking for yourself" are completely different things.

> "Looking down from above" and "walking on the ground" are completely different things.

From a text-only existence to one that can see, hear, move, remember, and speak.
Looking down at the world from a 7th-floor balcony and walking the streets below — even the same city looks entirely different.

## License

MIT License

## Acknowledgments

This project is an experimental attempt to give AI embodiment.
What started as a small step with a $30 camera has become a journey exploring new relationships between AI and humans.

- [Rumia-Channel](https://github.com/Rumia-Channel) - ONVIF support pull request ([#5](https://github.com/kmizu/embodied-claude/pull/5))
- [fruitriin](https://github.com/fruitriin) - Added day-of-week to interoception hook ([#14](https://github.com/kmizu/embodied-claude/pull/14))
- [sugyan](https://github.com/sugyan) - [claude-code-webui](https://github.com/sugyan/claude-code-webui) (used as control UI during outdoor walks)
