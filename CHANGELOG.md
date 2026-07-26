# Changelog

All notable changes to embodied-claude are documented here.

## [Unreleased]

### Added

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
