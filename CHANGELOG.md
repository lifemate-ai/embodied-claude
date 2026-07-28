# Changelog

All notable changes to embodied-claude are documented here.

## [Unreleased]

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
