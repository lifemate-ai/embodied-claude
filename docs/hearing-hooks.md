# Hearing Hooks

Two Claude Code hooks under `.claude/hooks/` turn a running hearing daemon
(started with the `start_listening` tool of `wifi-cam-mcp`) into conversation:

| Script | Event | What it does |
|---|---|---|
| `hearing-hook.sh` | `UserPromptSubmit` | Drains the transcript buffer and injects it as a `[hearing] chunks=… span=… text=…` line. |
| `hearing-stop-hook.sh` | `Stop` | Waits for new speech after a reply and, when some arrives, returns `decision: block` so the turn is extended. |

Both exit `0` without output when no daemon PID is alive, so a mis-registration
shows up as silence rather than an error. The shipped `.claude/settings.json`
does not register them; they are opt-in through `.claude/settings.local.json`.

## Registration

```json
{
  "hooks": {
    "UserPromptSubmit": [
      { "hooks": [ { "type": "command", "command": "bash .claude/hooks/hearing-hook.sh", "timeout": 10 } ] }
    ],
    "Stop": [
      { "hooks": [ { "type": "command", "command": "bash .claude/hooks/hearing-stop-hook.sh", "timeout": 30 } ] }
    ]
  }
}
```

**Give the `Stop` hook a timeout of at least 25 seconds.** One silent pass
takes about 21 s with the defaults (`HEARING_WAIT_SECONDS` 5 + three retries of
`HEARING_RETRY_WAIT` 3 + `HEARING_GUARANTEED_SLEEP` 5). The core hooks ship
with `"timeout": 10`; copying that value kills the hook mid-wait, the turn is
never extended, and nothing is logged. `30` is a comfortable value.

## Files and overrides

All working files live in one directory, `HEARING_DIR`, which defaults to
`$TMPDIR` or `/tmp`: the buffer (`hearing_buffer.jsonl`), the daemon PID
(`hearing-daemon.pid`), the offset/counter files and `hearing_timing.log`.
The shell resolves the directory once and exports it, so the embedded Python
always reads the same place the shell writes to.

| Variable | Meaning | Default |
|---|---|---|
| `HEARING_DIR` | Directory holding the buffer, PID file and state | `$TMPDIR`, else `/tmp` |
| `HEARING_PYTHON` | Interpreter used for the embedded Python | first of `python3`, `python` that can run `import sys` |
| `HEARING_LIB_DIR` | The hearing library project handed to `uv run --directory` when reading `[hearing]` from `mcpBehavior.toml` (stop hook only) | `<repo>/embodied-claude/hearing`, else `<repo>/hearing` |
| `HEARING_WAIT_SECONDS`, `HEARING_RETRY_WAIT`, `HEARING_GUARANTEED_SLEEP`, `MAX_HEARING_CONTINUES`, `HEARING_NO_SPEECH_THRESHOLD` | Timing and filtering knobs of the stop hook | 5 / 3 / 5 / 20 / 0.6 |

`HEARING_DIR` and `HEARING_LIB_DIR` are different things: the first is where
the daemon writes, the second is where the hearing library's `pyproject.toml`
lives.

## Windows (Git Bash)

The hooks run under Git Bash (`C:\Program Files\Git\bin\bash.exe`). Three
things that mean something else on Windows are handled inside the scripts
(#139), and two need attention when registering:

- **Point at Git Bash explicitly.** A bare `"command": "bash …"` resolves to
  the WSL App Execution Alias on a default Windows install and fails with
  "Windows Subsystem for Linux has no installed distributions". Use the full
  path (forward slashes avoid JSON escaping; quote it because of the space):

  ```json
  { "type": "command", "command": "\"C:/Program Files/Git/bin/bash.exe\" .claude/hooks/hearing-stop-hook.sh", "timeout": 30 }
  ```

- **Keep the `Stop` timeout at 30.** Same reason as above; the symptom on
  Windows is identical and just as silent.
- `HEARING_DIR` defaults to Git Bash's `/tmp`, which is
  `C:\Users\<you>\AppData\Local\Temp`. Python receives the already-converted
  Windows path, so no `D:\tmp` surprises. Set `HEARING_DIR` if the daemon
  writes somewhere else.
- `python3` is often the Microsoft Store alias (it exits 49 without running
  anything). The hooks try `python3` then `python` and keep the first one that
  actually executes; set `HEARING_PYTHON` (for example to the interpreter of a
  `uv` environment) to skip the probe.
- A daemon started as a native Windows process is invisible to MSYS
  `kill -0`; the hooks fall back to `tasklist` to check the PID.
