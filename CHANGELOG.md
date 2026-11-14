# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Continuous integration: tests and linting on every pull request.
- `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, issue and pull-request templates.

## [0.1.0] — 2026-11-09

### Added

- **Stage B capture-quality gate (v3.1)** — pure-NumPy signal extraction
  (`sharpness`, `contrast`, `ink_ratio`, `brightness`, `exposure`) on a fixed,
  resolution-invariant analysis grid, mapped to `RETAKE | REVIEW | PASS`.
- **Supervised calibration** on the full 2000-image corpus with two
  load-bearing guarantees: zero false rejects on sharp-labelled images, and no
  blurred-labelled capture ever reaches `PASS`.
- **v3.1 readable-blur band** — hard-`RETAKE` sharpness cut lowered 0.10 → 0.08
  after field evidence; 0.08–0.13 is `REVIEW`.
- **`quality` CLI** — score any image file or folder, with `--json` output.
- **`stage-b` CLI** — streaming real-corpus runner with flat memory.
- **Deterministic degradation sweep** harness for gate verification.
- **Stage B-Q (experimental)** — tabular Q-learning gate, 16 states / 48
  parameters, learn-parity with the fixed gate (`0.8076` vs `0.8076` mean
  reward, `0.648` per-image agreement on 2000 labelled images), fail-closed on
  novel states.
- **Documentation** — `README.md`, `docs/ARCHITECTURE.md`,
  `docs/AI_CODING_INSTRUCTIONS.md`, `docs/AI_267_FULL_CONTEXT.md`, and
  `reports/AI-267_STAGE_B_RESULTS.md`.

### Removed

- Stage A (pre-registered policy study) — the repository ships the Stage B
  capture gate only.

[Unreleased]: https://github.com/reyyishreyas/smallai/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/reyyishreyas/smallai/releases/tag/v0.1.0
