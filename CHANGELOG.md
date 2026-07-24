# Changelog

All notable changes to embodied-claude are documented here.

## [Unreleased]

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
