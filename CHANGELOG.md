# Changelog

All notable changes to embodied-claude are documented here.

## [Unreleased]

### Added

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
