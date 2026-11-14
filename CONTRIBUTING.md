# Contributing to SmallAI

SmallAI is a small, deliberately constrained project. Read
[`docs/AI_CODING_INSTRUCTIONS.md`](docs/AI_CODING_INSTRUCTIONS.md) before
changing anything — it lists the hard invariants, the calibration procedure and
the docs-sync rules.

## Setup

```bash
git clone https://github.com/reyyishreyas/smallai.git
cd smallai
make setup          # venv + editable install with dev extras
make check          # lint + tests
```

Python 3.10+ is required. Runtime dependencies are `numpy` and `pillow` only.

## Hard invariants

These are not negotiable and are enforced by review:

1. **No LLM, neural network, or high-level ML library** in the inference path.
   Pure NumPy; Pillow for image IO only.
2. **Deterministic.** Fixed seeds, pure NumPy math, identical results on re-run.
3. **Stable verdict contract.** `RETAKE | REVIEW | PASS` for capture quality;
   `MATCH | REVIEW | MISMATCH | NO_SOURCE_DATA` per verification field.
4. **Offline and CPU-only.** No network, GPU, or model download at inference.
5. **Parameter budget.** Stage B: 0 learned parameters. Stage B-Q: ≤ 216 states.

## Workflow

1. Open an issue describing the behavior change and its evidence.
2. Create a short-lived branch: `feat/…`, `fix/…`, `docs/…`.
3. Make the change, then run `make check` (lint + tests).
4. Update docs in the **same** change (see below).
5. Open a pull request using the template.

## Threshold and calibration changes

A threshold may only move through the supervised procedure in
`docs/AI_CODING_INSTRUCTIONS.md` §4:

- calibrate against the full labelled corpus plus field evidence,
- keep the two load-bearing guarantees (zero false rejects on sharp-labelled
  images; nothing blurred-labelled reaches `PASS`),
- update `README.md`, `reports/STAGE_B_RESULTS.md` and `docs/ARCHITECTURE.md`
  in the same pull request.

Do not hand-tune thresholds to make a single example pass.

## Tests

- `make test` — full suite, must be green.
- `make lint` — `ruff check src tests`, must be clean.
- New behavior needs a test. Determinism and the fail-closed path are tested
  explicitly; keep them that way.

## Data

Corpora are **not** committed (`dataset/`, `data/`, `runs/` are ignored). If you
add a corpus, record its source, license and checksum in your pull request —
never commit the images themselves.

## Commits and pull requests

- Conventional commit subjects: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`,
  `chore:`.
- Keep pull requests small and single-purpose.
- Describe the **evidence** for any behavior change (commands run, numbers
  before/after), not just the code.

## License

By contributing you agree that your contribution is licensed under the
Apache License 2.0 (see [`LICENSE`](LICENSE)).
