# toio-mcp

Use a **toio Core Cube** as a small programmable hand for embodied-codex.

This MCP server wraps the official `toio.py` library and exposes a compact tool surface for:

- connecting to a cube over BLE
- short forward/backward moves
- in-place rotation
- mat-based positioning
- RGB light control
- note playback
- status inspection

## Why this exists

A full-sized mobile body and a tiny visible actuator solve different problems.

`toio-mcp` is aimed at the second one: a small hand you can put on a desk, point with, nudge things
with, or use as a visible tool-use substrate.

## Hardware

- 1x toio Core Cube
- 1x charger or toio set that can charge the cube
- Bluetooth-capable host machine
- Optional: bundled simple play mat or another Position ID mat for precise positioning

## Setup

```bash
cd toio-mcp
uv sync
```

## Configuration

Environment variables are optional unless noted.

- `TOIO_CUBE_NAME`
  - Optional 3-digit cube suffix to target a specific cube.
- `TOIO_SCAN_TIMEOUT`
  - BLE scan timeout in seconds. Default: `5`
- `TOIO_MAX_SPEED`
  - Safety clamp for any speed value. Default: `70`
- `TOIO_MOVE_SPEED`
  - Default forward/backward speed. Default: `50`
- `TOIO_ROTATE_SPEED`
  - Default in-place rotation speed. Default: `60`
- `TOIO_MOVE_DURATION`
  - Default move duration in seconds. Default: `0.5`
- `TOIO_LIGHT_DURATION`
  - Default lamp duration in seconds. Default: `0.5`
- `TOIO_NOTE_DURATION`
  - Default sound duration in seconds. Default: `0.25`
- `TOIO_DRY_RUN`
  - Set to `1` to use a built-in software simulator until the real cube arrives.

## Claude Code config

Add this to `.mcp.json`:

```json
{
  "mcpServers": {
    "toio": {
      "command": "uv",
      "args": ["run", "--directory", "toio-mcp", "toio-mcp"],
      "env": {
        "TOIO_CUBE_NAME": "123"
      }
    }
  }
}
```

For design work before the cube arrives:

```json
{
  "mcpServers": {
    "toio": {
      "command": "uv",
      "args": ["run", "--directory", "toio-mcp", "toio-mcp"],
      "env": {
        "TOIO_DRY_RUN": "1"
      }
    }
  }
}
```

## Tools

- `connect_hand`
- `disconnect_hand`
- `hand_status`
- `move_hand_forward`
- `move_hand_backward`
- `rotate_hand_left`
- `rotate_hand_right`
- `stop_hand`
- `move_hand_to_position`
- `move_hand_to_grid_cell`
- `set_hand_orientation`
- `set_hand_light`
- `clear_hand_light`
- `play_hand_note`
- `stop_hand_sound`

## Notes

- Mat-aware tools require a Position ID mat under the cube.
- Without a mat, relative movement, light, sound, and battery inspection still work.
- The server intentionally favors short, safe movements rather than indefinite drive commands.
