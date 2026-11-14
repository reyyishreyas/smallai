# Developer & AI Coding Instructions — SmallAI Capture-Quality Gate

*How an AI coding assistant (or a human mentoring one) should work in this repository. These instructions exist so AI contributions follow the same rules as human ones — "the assistant does not get to invent its own process" (vibe-coding guardrails).*

## 1. What this project is

SmallAI is a non-LLM, pure-NumPy decision maker for one job — *"is this document capture usable?"* — for field officers photographing IDs/forms offline.

- **Stage B** — fixed offline capture-quality gate (`src/smallai/stage_b.py`, `src/smallai/env/image_quality.py`): real image signals → `RETAKE | REVIEW | PASS`.
- **Stage B-Q** — experimental tabular-Q variant of the gate (`src/smallai/stage_b_q.py`).

The repo's own docs for scope and design: `docs/ARCHITECTURE.md`, `reports/STAGE_B_RESULTS.md`, and `docs/CONTEXT.md` (current state).

## 2. Hard invariants — never break these

1. **No LLM / no neural net in the inference path.** Pure NumPy + Pillow (Pillow only for IO). No transformers, sklearn, torch, or LLM API calls inside the decision path. An LLM may be used only as a drafting aid *outside* the model — never as the decision-maker.
2. **Deterministic.** Fixed seeds everywhere; pure NumPy math; randomness only through seeded generators.
3. **Parameter budgets.** Stage B: 0 learned params (fixed thresholds). Stage B-Q: ≤216 states (observed 16 / 48 params).
4. **Stable verdict contract.** Stage B: `RETAKE | REVIEW | PASS`. v3.1 band: hard-RETAKE sharpness < 0.08; 0.08–0.13 readable-blur → REVIEW; else PASS. This contract is the stable interface the Android/Kotlin port will bind to — do not silently change the vocabulary or semantics.
5. **Offline / CPU-only.** Everything must run offline. Measured performance is **~14 ms/image** end-to-end; **< 10 ms** is the optimization target, so do not regress the current figure.

## 3. Repo map

- `src/smallai/env/image_quality.py` — signal extraction + gate thresholds (v3.1: `RETAKE_SHARP_MIN=0.08`, `REVIEW_SHARP_MIN=0.13`, …)
- `src/smallai/stage_b.py` — fixed offline gate: extract → decide, synthetic sweep (`run_sweep`), real-corpus run (`run_real_corpus`), `check_images`
- `src/smallai/stage_b_q.py` — experimental Q gate: quantization, samples, `QCaptureAgent`, `check_images_with_q`, `run_experiment`
- `src/smallai/cli.py` — `quality`, `stage-b` and `qgate` CLI commands (any image path; `--json`)
- `tests/` — Stage B gate and Stage B-Q suites
- `data/`, `runs/`, `reports/`, `docs/`, `dataset/` (gitignored corpus)
- `docs/CONTEXT.md` — project context and handoff (current state, invariants, next steps)
- `.github/workflows/ci.yml` — CI: lint + tests on every push/PR (3.10–3.13)
- `CONTRIBUTING.md` — contributor workflow, invariants, docs-sync rules
- `CHANGELOG.md`, `SECURITY.md`, `.editorconfig`, `.gitattributes`
- `pyproject.toml` — packaging metadata, console script, ruff + pytest config

## 4. How a behavior change should be made

- **Threshold change (gate):** must go through the supervised procedure (full corpus + labels + field evidence), update `env/image_quality.py`, `stage_b.py` + docs (`README.md` §2, `reports/STAGE_B_RESULTS.md`) in the same change, and keep the supervised guarantees documented (zero false rejects on sharp-labelled; nothing blurred-labelled may PASS).
- **Q-gate change:** re-run `run_experiment` / retrain via `stage_b_q.QCaptureAgent`, keep fail-closed semantics + L1 generalization, update README §6.
- **New signal / verifier / action:** update the verdict contract only if the product decision allows (it currently doesn't).

## 5. Commands

```bash
make setup     # venv + editable install with dev extras
make check     # lint + tests (what CI runs)
make test      # .venv/bin/python -m pytest
make lint      # .venv/bin/ruff check src tests
make build     # sdist + wheel
python -m smallai quality --image <file-or-dir> --json
python -m smallai qgate --image <file-or-dir>
```

## 6. AI assistant guardrails (vibe-coding)

- Do not add an LLM / transformer / sklearn into the decision or inference path.
- Do not change the `RETAKE | REVIEW | PASS` contract or its semantics silently.
- Do not retrain / re-calibrate thresholds without going through the documented supervised flow.
- Do not invent new verdict vocabularies without updating every consumer (CLI, Q-gate, tests, docs, Kotlin port contract).
- Keep PRs small; run tests + ruff before declaring done.
- When changing behavior, update `README.md`, `reports/STAGE_B_RESULTS.md`, and `docs/ARCHITECTURE.md` in the same PR.

## 7. Testing expectations

- `make test` green (gate + Q-gate suites), `ruff check src tests` clean.
- Determinism: fixed seeds; identical results on re-run.
- Offline: no network / no external services required by tests.

## 8. Docs sync checklist (update in same change as behavior)

- `README.md` (§2 how it works, §3 results, §6 Stage B-Q)
- `reports/STAGE_B_RESULTS.md`
- `docs/ARCHITECTURE.md`
- `src/smallai/**` docstrings
