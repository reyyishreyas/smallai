# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Continuous integration: tests and linting on every pull request.
- `CONTRIBUTING.md`, `SECURITY.md`, issue and pull-request templates.

### Changed

- CI actions updated to `actions/checkout@v7`, `actions/setup-python@v7` and
  `actions/upload-artifact@v7`; the runner image is pinned to `ubuntu-24.04`
  instead of the migrating `ubuntu-latest` label.
- Internal docs re-branded as standalone project documentation:
  `docs/AI_267_FULL_CONTEXT.md` → `docs/CONTEXT.md` and
  `reports/AI-267_STAGE_B_RESULTS.md` → `reports/STAGE_B_RESULTS.md`.
- README rewritten as standalone project documentation (no ticket framing).

### Removed

- `LICENSE` and the Apache-2.0 license metadata — no license is granted; all
  rights are reserved by default.

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
  `docs/AI_CODING_INSTRUCTIONS.md`, `docs/CONTEXT.md`, and
  `reports/STAGE_B_RESULTS.md`.

### Removed

- Stage A (pre-registered policy study) — the repository ships the Stage B
  capture gate only.

[Unreleased]: https://github.com/reyyishreyas/smallai/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/reyyishreyas/smallai/releases/tag/v0.1.0
