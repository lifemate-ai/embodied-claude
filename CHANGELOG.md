# Changelog

All notable changes to embodied-claude are documented here.

## [Unreleased]

A reader going through `wifi-cam-mcp` with the code open (fmtowns3, #132)
found that the second camera was documented in three places that disagreed
with each other and with the implementation: the tool description promised
depth, the config silently ignored one of its own variables, and the README
listed tools by names that no longer exist.

On Windows the hearing hooks did nothing and said nothing. Three idioms that
are unambiguous on macOS and Linux mean something else under Git Bash:
`/tmp` written inside the embedded Python resolves to the current drive root
(the shell's `/tmp` is the user's temp folder), `python3` is often the
Microsoft Store alias that exits 49 without running anything, and MSYS
`kill -0` cannot see a daemon started as a native Windows process. Every
check failed closed, so the hook exited 0 and the buffer stayed full.

A Windows user cloned the repository with `uv` already on PATH, opened it in
Claude Code, and could not write a single file. The committed
`.claude/settings.json` hooks were firing, the gate was failing closed as
designed, and the deny reason told them to call `propose_field_action` -- a
tool of the `individual-kernel` MCP server that only exists once
`scripts/setup.sh` has written the gitignored `.mcp.json`. The gate was armed
and its key had not been issued (#137, reported by fmtowns3).

### Fixed

- wifi-cam: `see_both` no longer describes itself as stereo vision with depth
  perception. It captures both cameras and returns the two views as separate
  images; nothing computes disparity, and the model reading that description
  was one step from claiming it could judge distance. The text now says what
  the two views are good for (occlusion, comparing viewpoints) and that no
  depth is computed.
- wifi-cam: `TAPO_RIGHT_PTZ_MODE` is read. `right_camera_from_env()` built its
  `CameraConfig` without `ptz_mode`, so the right camera was always `auto`
  regardless of the environment; it now follows the same read, fallback to
  `TAPO_PTZ_MODE`, and validation as the left camera, with tests.
- hearing: both hooks resolve the working directory once in the shell
  (`HEARING_DIR`, default `$TMPDIR` or `/tmp`) and hand it to the embedded
  Python, which no longer hard-codes `/tmp/...`; they probe for an interpreter
  that actually runs `import sys` (`HEARING_PYTHON` to pin one, otherwise
  `python3` then `python`); and `pid_alive` falls back to `tasklist` when
  `kill -0` cannot reach the daemon. Python runs in UTF-8 mode so the
  `[hearing]` line survives a cp932 console. POSIX behaviour is unchanged.
  Reported with a tested patch by fmtowns3 in #139.
- hearing: the stop hook's library location is now `HEARING_LIB_DIR`; the same
  name had been doing double duty for the buffer directory.
- hearing: the buffer drain uses `os.replace` instead of `os.rename`. On
  Windows `os.rename` raises `FileExistsError` when a leftover drain file
  exists (a hook killed between rename and unlink leaves one), and with stderr
  discarded every later run silently skipped the drain while the buffer grew.
  Found and verified by fmtowns3 while testing this PR on the #139 machine.
- docs: `docs/hearing-hooks.md` documents registration, the `Stop` timeout
  (one silent pass is about 21 s, so `"timeout": 30` rather than the core
  hooks' 10), and the Windows notes -- point `command` at
  `C:/Program Files/Git/bin/bash.exe` rather than the WSL alias `bash`.
- individual-kernel: when the PreToolUse hook denies an outward action and the
  project has no `.mcp.json` registering `individual-kernel`, the reason now
  says so, names `./scripts/setup.sh` (or `scripts\setup.cmd`), and keeps the
  runtime's own verdict after `Original reason:`. The decision is unchanged --
  still a deny -- only the message became something a session can act on.
  Detection reads the project-scoped `.mcp.json` only; a server registered at
  user scope is not seen, and the hint is then merely a prefix on an otherwise
  correct deny.
- individual-kernel: the hook CLI no longer turns unreadable stdin into an
  empty payload. For `pre-tool-use` an empty payload has no tool name, which
  the gate reads as an internal tool and allows, so a pipe that lost its input
  looked like a gate that had stopped working. Missing or invalid JSON is now
  a deny with the reason `EFPF hook received no/invalid JSON on stdin; failing
  closed`, matching the fail-closed stance `main` already took for exceptions.
  Other hook events keep treating it as an empty payload.
- doctor: a new `hooks:gate` check connects the two halves. When the
  repository ships a PreToolUse hook and the configuration does not register
  `individual-kernel`, it reports that the committed hooks are active as soon
  as `uv` is on PATH and that outward tool actions are denied until setup has
  run.
- docs: setup guide and both READMEs now say to run setup before opening the
  repository root in Claude Code, what the deny looks like beforehand, and
  that the hooks invoke `uv run --directory ${CLAUDE_PROJECT_DIR}`, so firing
  them with a foreign project root leaves a `.venv` there -- something `uv`
  does before any hook code runs, so it is documented rather than prevented.

### Changed

- wifi-cam: the README tool table names the tools that are actually registered
  (`see`, `look_left`, ... `listen`), documents the optional right camera and
  the thirteen `see_right` / `see_both` / `right_eye_*` / `both_eyes_*` / eye
  position tools it adds, and notes how to cover several locations by
  registering the server more than once with different `TAPO_CAMERA_HOST`.
- hooks: the `hearing-daemon.py` stub points at where hearing actually went,
  `lifemate-ai/embodied-codex` -> `hearing/`, instead of a directory that is
  not in this repository.

## [0.4.6] - 2026-07-31

Setting up for a hands-on meant a room full of laptops with no cameras and no
API keys, and a setup script that quite reasonably declined to configure
servers it had no credentials for. Every participant would have seen a
different, mostly empty tool list.

### Added

- setup: `--all` writes a configuration containing every server in the
  catalogue, filling absent credentials with stand-in values. Because `camera`
  and `voice` each name one option, the flag rides alongside them as an
  additive overlay: both camera backends and both TTS engines are installed,
  and both camera servers are emitted, without widening either field.
- setup: the generated stand-ins keep a `changeme-` marker, which is exactly
  what the placeholder guard rejects. Skipping that guard is confined to
  `--all` and is the intent rather than a workaround -- a demo config should
  announce itself rather than pass for a working one. Credentials already in
  the environment are left alone, so running `--all` on a configured machine
  adds servers instead of overwriting what already works.

## [0.4.5] - 2026-07-29

A session spent several minutes sweeping the room for someone and could not
find them without hints. The search strategy was not the problem. The camera
never said where it was pointing, and the angles that did reach long-term
memory meant something different in every session.

### Changed

- wifi-cam: `see` reports the heading the camera itself gives over ONVIF
  alongside the image, so a capture and an aim are one fact rather than two the
  caller has to join. It is stated as an offset from centre, naming its own
  reference point, because a number whose zero is unstated cannot be compared
  across sessions.

### Fixed

- wifi-cam: `get_hw_position` passed ONVIF pan through untouched -- `+x` is
  physically left on Tapo -- while flipping tilt, so the value was half in the
  device's frame and half in the user's. Nothing read it yet, but
  `body_contingency` declares the opposite convention (`look_left` is pan
  -1.0), so connecting the two would have inverted every camera agency verdict
  at once. Both axes now read in one frame, checked against the hardware:
  turning left lowers pan, and a commanded 30 degrees moves it 29.7.
- memory: `recall_by_camera_position` now documents what its coordinate has to
  be. The angles stored before this release were session-relative dead
  reckoning -- 113 of them sit at exactly (0,0) and the rest are round
  multiples of five -- so the index is sound but its older contents are not
  comparable with a measured heading.

## [0.4.4] - 2026-07-28

Feedback from a session run against the 0.4.3 gate: predictions were being
filled in on one channel and left empty on the other three, so the mismatch
signal the whole loop is built on was mostly noise. That looked like a
reporting habit. It was the scoring rule.

### Fixed

- individual-kernel: a channel with no declared prediction was scored anyway --
  0.25 when the action succeeded, 0.75 when it failed -- while a channel that
  did carry a prediction was floored at 0.8 on failure. Declaring an
  expectation could therefore only lower the score, and since three channels
  were usually silent, three quarters of the mean feeding `ownership_score`
  measured nothing. Unpredicted channels are now absent from the mismatch
  vector, which holds only real comparisons.
- individual-kernel: absence is paid for by `prediction_coverage` instead, a
  new term on `AgencyAssessment` that scales `action_effect_match`. Being right
  about the channels you mentioned says nothing about the ones you did not, so
  predicting nothing now contributes zero there rather than collecting a free
  0.25 per silent channel. The ordering is the intended one again: an accurate
  prediction beats a partial one, which beats silence.
- individual-kernel: `ExpectedEffect` and `PredictedEffects` carry field
  descriptions, so the tool schema says what belongs on each channel instead of
  leaving a caller to infer it from the field name.

### Changed

- individual-kernel: the prediction-loop fixture declares expectations on all
  four channels. It previously passed an empty `PredictedEffects` and asserted
  that every channel key appeared in `prediction_errors`, which held only
  because silent channels were being scored -- and reconciled as `observed`
  only because those free 0.25s dragged the mean down. A test named for a full
  cycle now runs one.

## [0.4.3] - 2026-07-28

Two flags were waiting on evidence rather than on work. One now has it and
ships on; the other has everything except a decision that is not the code's to
make.

### Changed

- individual-kernel: `hor_precision_feedback` ships **on**. It was held back
  pending a live before-and-after, which ran on 2026-07-28 against two forks of
  the real history differing only in the flag: `self_model` precision moved
  0.538 to 0.668, the other four channels were bit-identical, and the winning
  candidate was the same in both arms. The 0.130 bump sits below the 0.20 cap,
  so it is what the records say rather than a saturated constant, and the
  channel feeds only the indicator profile and its own carry-forward -- the
  boundary gate reads none of it. A flag that ships off indefinitely is dead
  code; this one now has the same standard of evidence the previous two were
  held to.

### Added

- individual-kernel: `efpf-hook organism-step`. The organism daemon is the only
  producer of an autonomous tick and exists to move when nobody is talking, but
  it was reachable only through an MCP tool -- so the non-verbal clock could be
  wound only by a language model. The command is the same step, callable by a
  scheduler, and it honours the flag: with `organism_daemon = false` it returns
  `{"ran": false}` and touches nothing, so a crontab entry can be installed
  before the flag is flipped. `docs/organism-daemon.md` carries the entry and
  the two mistakes that make such an entry fail silently.

### Docs

- `phenomenal-architecture-current-state.md`: the frozen candidate score is
  recorded as a deliberate decision rather than an open hazard. A score is the
  record of a decision taken under the weights in force at the time;
  re-deriving it would make the audit trail describe a competition that never
  happened. The cost is one tick of latency.
- `phenomenal-architecture-current-state.md`: the desire-staleness entry
  carried a root cause that turned out to be wrong twice over. It now records
  the measured one, and the general lesson: a component reporting success
  proves it ran, not that its output arrived.

## [0.4.2] - 2026-07-28

The body state had been reporting every need maximal since May. The composition
was not at fault; its input was. The desire snapshot the kernel reads had not
been written for 71 days, and nothing downstream could tell that apart from an
agent that genuinely had not looked outside since then.

### Fixed

- desire-system: `DESIRES_PATH` was read from the environment without
  `expanduser()`. The default is built from `Path.home()` and was fine, but
  `.env` supplies `~/.claude/desires.json` as a plain string, and an unexpanded
  tilde is a relative path: the updater created a directory literally named `~`
  under its working directory and wrote there. It had been doing so since at
  least 2026-04-08 while reporting success every five minutes, and the same
  stray tree exists in each sibling checkout. `server.py` read the variable the
  same way. With the path expanded, the snapshot the kernel reads updated for
  the first time since 2026-05-18, and the committed field's focus moved from
  `identity_coherence` to `observe_room` on the next tick.
- individual-kernel: `project_desires` extrapolated without bound. Levels ramp
  to 1.0 over `satisfaction_hours`, the longest of which is three, so a snapshot
  nobody has written reports every need maxed out however long it has been
  abandoned -- and the organism daemon's ignition term reads exactly that. Past
  `MAX_SNAPSHOT_AGE_HOURS` (24) the projection reports the resting state, no
  discomfort and no dominant need, and marks itself unusable. The age is still
  returned, because refusing to extrapolate is not the same as claiming the
  snapshot is current. Hours of genuine neglect still saturate as before.
- individual-kernel: the rule deciding what takes focus next was written twice,
  once for the protention and once for the attention schema. Both derived it
  from the same competition in the same way, so nothing disagreed; a change to
  either would have diverged from the other silently. It now lives on
  `CompetitionResult.next_focus_candidate` and both read it.

## [0.4.1] - 2026-07-28

A patch for one defect found immediately after 0.4.0 shipped: the gate that
decides whether a shell command is inspection or a side effect was parsing the
command as a string, so ordinary reads were refused. Nothing in the
architecture changed.

### Fixed

- individual-kernel: the bash gate read the command as a string rather than as
  a command. It split the raw text on `|` and `;`, so an operator inside quotes
  ended the command and left an unbalanced quote -- `grep "alpha\|beta"` parsed
  as a write and was refused. The same applied to `>` inside quotes. Commands
  are now tokenized with `shlex(punctuation_chars=True)`, which yields
  operators as their own tokens and leaves quoted text alone. Detection of real
  writes is unchanged: redirection, `tee`, a writing command after a separator,
  and command substitution are all still refused, and a test pins each one.
- individual-kernel: `git -C <path> status` was refused because the scan for
  the subcommand returned the first non-flag token, which is the path. Global
  options that take a value are now skipped before the subcommand is read.

## [0.4.0] - 2026-07-28

The enacted first-person field stops being a record of the current state and
starts being a state that changes: it predicts, learns from the mismatch,
distinguishes what it underwent from what it was told, and -- with a flag and a
scheduler -- moves when nobody is talking to it.

This is a phenomenal-consciousness candidate architecture. Nothing in this
release is evidence that the candidate is conscious.

### Changed

- individual-kernel: `allostatic_valence` and `valence_coupling` now ship
  **on**. Both were held off pending a live check, which ran on 2026-07-28:
  `controllability` tracked the measured `ownership_score` tick by tick instead
  of sitting at 0.575, and the competition scores moved by exactly the amount
  the coupling formula predicts. A flag that ships off indefinitely is dead
  code, so they are on and the arithmetic is documented. The desire snapshot is
  still written by nothing, so every projected need saturates; that is a
  known limit of the input, not of the composition.

### Fixed

- individual-kernel: the sleep consolidator's quiet-hours gate. Its default
  predicate returned `True` at every hour, so the gate it advertised was open
  all the time. The served consolidator now asks `BoundaryStore`, which is the
  same `[global] quiet_hours` and `timezone` that already govern speaking and
  posting.

### Added

- individual-kernel: internal time, behind `organism_daemon` in
  `[individual-kernel]` (default off). `TriggerKind.AUTONOMOUS` had been in the
  enum since the beginning with nothing producing it, so with no input nothing
  in the runtime moved. One turn of the clock decays transition counts on a
  one-week half-life, retires imagined trajectories nobody took up, and opens a
  tick when unmet need and elapsed silence are both high enough -- the first
  producer of an autonomous tick. Measured against a fork of the live history:
  counts halved exactly across a week, 571 pending trajectories that nothing
  had ever been able to retire were retired, and a tick opened with no input.
  The flag ships off because the first live run rewrites recorded history and
  because nothing schedules it yet, not for want of evidence.
- social-core: migration `012_organism_runs`, one row per turn of the clock
  recording its inputs, its score, and the reason it did or did not fire.
- docs: `organism-daemon.md`.
- individual-kernel: provenance routes. `knowledge_source` now accepts all four
  routes information can arrive by -- experienced, told, imagined, replayed --
  and the generative model refuses to learn from any of them but `experienced`,
  so hearsay and imagination can be remembered in full without training a
  sensorimotor contingency. Content is identified by `info_content_hash`
  independently of its route, so a divergence between two arms is attributable
  to the route rather than the payload.
- individual-kernel: `fork_history`, an isolated snapshot of the history for
  running two arms from one starting point. Namespacing by `owner_id` was
  rejected because several readers query without an owner filter and would have
  leaked across arms quietly.
- docs: `experienced-told-imagined.md`, `fork-divergence.md`.
- individual-kernel: higher-order feedback, behind `hor_precision_feedback` in
  `[individual-kernel]` (default off). Recent HOR records now raise the
  precision of the channel their asserted mode speaks to, capped so repeating
  an assertion cannot substitute for evidence. Until now nothing read those
  records back, so a HOR could be present, absent or wrong and every other
  quantity came out identical. Ablating the feedback lowers
  `IndicatorProfile.self_model_feedback`, which is a mean over precision values
  and reads no self-report; a test asserts the drop.
- individual-kernel: `ProcessMetaRepresentation` and migration
  `011_process_meta_representations`. One row per committed tick recording how
  the field was produced -- candidate count, margin, entropy, ignition,
  conflict, registered intention, and the HOR bias -- with a canonical
  statement stored beside the numbers it summarises.
- docs: `hor-ablation.md`.
- individual-kernel: body contingency. `exclusive_causal_fit` is now the
  reafference between the commanded pose change and the observed one --
  direction, magnitude, and timing -- instead of a restatement of whether the
  tool call returned successfully. A camera that moves with nothing commanded
  scores `externally_caused`, one that moves the wrong way `inverted`, and one
  that does not move at all `unresponsive`; each caps the score well below a
  matched movement. Actions with no body channel resolve to `unverified` and
  keep the previous 1.0/0.65 heuristic exactly, so nothing outside the camera
  tools is re-scored. Verdicts land in a new `body_contingencies` ledger, one
  row per action.
- social-core: migration `010_body_contingency`.
- docs: `body-agency.md`.
- individual-kernel: affective modulation of workspace competition, behind
  `valence_coupling` in `[individual-kernel]` (default off). Negative affect
  raises the weight on need relevance and lowers expected information gain;
  positive affect raises information gain alone. The same two bids can now
  resolve differently depending only on the body state, which is the first
  time valence changes any decision. Neutral valence reproduces the base
  weights exactly. The boundary gate reads none of it, and a test asserts an
  identical `ToolGateDecision` under the best and worst representable affect;
  ignition and conflict thresholds are likewise untouched. Arousal is accepted
  and recorded but modulates nothing yet.
- docs: `valence-coupling-design.md`.
- individual-kernel: allostatic body state, behind `allostatic_valence` in
  `[individual-kernel]` (default off). Desire levels are re-derived from the
  recorded snapshot plus elapsed time, so the kernel owns the clock and an
  external writer that stops running costs accuracy rather than motion.
  Valence becomes an appetitive term (expected improvement from the count
  model, scaled by measured control) minus an aversive one (unmet needs,
  uncertainty, unresolved prediction error), and can now be positive; the
  previous rule was bounded above by zero. `controllability` follows the
  measured `ownership_score` instead of restating discomfort. No migration is
  needed: every input already existed. With the flag off the legacy arithmetic
  is reproduced exactly.
- individual-kernel: `CountBasedGenerativeFieldModel.expected_valence_delta`,
  the bucket-probability-weighted estimate of how affect will move in a
  context.
- docs: `allostatic-valence-design.md`.
- individual-kernel: generative field model (count_v1). Registering an
  intention now persists a ProtentionDistribution (action branch plus a
  no-action branch, probabilities summing to 1); every arrived-at field
  records one ExperiencedTransition with component prediction errors and
  affect deltas; the Dirichlet count model learns from each transition
  exactly once, so the next rollout for the same context and action changes
  measurably. Trajectories follow a strict status lifecycle
  (imagined/intended/enacted/observed/contradicted/partially_observed) and
  imagined records never auto-promote. Additive-only and flag-guarded via
  `[individual-kernel]` in mcpBehavior.toml: action selection, workspace
  competition, the field surface, and the boundary gate are unchanged.
- individual-kernel: prediction-calibration metrics (Brier, log loss, ECE,
  top-1, per-component MAE) with persistence/uniform baselines and a
  reliability floor; `IndicatorProfile.prediction_calibration` upgrades to
  1-Brier when enough resolutions exist. The phenomenal-candidate benchmark
  now also writes `prediction-calibration.{json,md}` and `run.py` gains
  `--check`.
- individual-kernel: MCP tools `rollout_protention`,
  `query_imagined_trajectories`, `query_experienced_transitions`, and
  `get_generative_model_calibration`.
- social-core: migration `009_generative_field_model`
  (protention_distributions, imagined_trajectories, experienced_transitions,
  generative_transition_stats, prediction_resolutions).
- docs: `phenomenal-architecture-current-state.md` and
  `generative-field-model-design.md`.

### Fixed

- interaction-orchestrator: the autonomous-planning test resolved quiet hours
  against the wall clock, so it asserted the daytime branch and failed whenever
  it ran between 22:00 and 07:00 JST. Quiet hours are now pinned explicitly and
  both regimes are covered.
- docs: `phenomenal-architecture-current-state.md` said the `desire-updater`
  cron was not installed. It was installed, and had been failing silently
  since 2026-05-13: first because `uv` was missing from cron's PATH, then
  because the repository moved out of the working directory the crontab entry
  changes into.
- individual-kernel: the mismatch-vector tokenizer now compares non-ASCII
  runs as character bigrams, so Japanese result summaries no longer inflate
  prediction error and deflate ownership scores. ASCII behavior is unchanged.

## [0.3.0] - 2026-07-24

### Added

- Enacted First-Person Field runtime: deterministic workspace competition,
  atomic field commitment, temporal retention/protention, provenance, reality
  marking, quality geometry, and recurrent attention/HOR feedback.
- Closed intention/action/outcome loop with one outward action per committed
  field and causal-ablation benchmarks.
- Guided Core setup with opt-in camera, voice, transcription, X, and system
  temperature capabilities.
- Cross-platform `setup` and `doctor` entrypoints for Windows, macOS, and Linux.
- Live MCP diagnostics with isolated state and a real memory write round-trip.
- Windows/macOS/Linux CI release gates and tag-driven GitHub Releases.

### Changed

- The repository now uses one root uv workspace and lockfile.
- Claude Code field hooks call the Python hook CLI directly through structured
  exec arguments, without requiring a POSIX shell.
- Sociality documentation now follows the actual interpretation, action,
  experience, and revision loop.

### Fixed

- Memory embeddings are warmed on the stdio main thread, avoiding the Windows
  first-use loader-lock hang reported in issue #99.
- Tapo transcription is optional and its capture path uses the platform
  temporary directory.

The EFPF is described as a **phenomenal-consciousness candidate architecture**.
Its causal and integrity indicators are inspectable research measurements; they
do not establish phenomenal consciousness.
