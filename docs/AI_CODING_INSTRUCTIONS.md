# Developer & AI Coding Instructions — SmallAI Decision Model (AI-267)

*How an AI coding assistant (or a human mentoring one) should work in this repository. These instructions exist so AI contributions follow the same rules as human ones — "the assistant does not get to invent its own process" (vibe-coding guardrails).*

## 1. What this project is

SmallAI decision model (CLEF/Mifos AI-267 PoC): a tiny, non-LLM, interpretable decision model for one job — *"is this document capture usable?"* — for field officers photographing IDs/forms offline.

- **Stage A** — journey-learning decision policy (`src/smallai/`): tabular Q-table agent (env → predicted checkable state → solver → final action), trained on synthetic trajectories including deliberately-wrong ones.
- **Stage B** — fixed offline capture-quality gate (`src/smallai/stage_b.py`, `src/smallai/env/image_quality.py`): real image signals → `RETAKE | REVIEW | PASS`.
- **Stage B-Q** — experimental tabular-Q variant of the gate (`src/smallai/stage_b_q.py`, lives on branch `feat/stage-b-q-gate`).

The repo's own docs for scope and design: `docs/ARCHITECTURE.md`, `docs/EXPERIMENT.md`, `docs/SmallAI Decision Model PRD and Tech Plan.md`, `reports/AI-267_*.md`.

## 2. Hard invariants — never break these

1. **No LLM / no neural net in the inference path.** Pure NumPy + Pillow (Pillow only for IO). No transformers, sklearn, torch, or LLM API calls inside the decision path. GPT-4o-mini may be used only as a *shortcut* (drafting, comparison) outside the model — never as the decision-maker.
2. **Deterministic.** Fixed seeds everywhere; pure NumPy math; randomness only through seeded generators.
3. **Parameter budgets.** Stage A: ≤15,625 states / few legal actions. Stage B: 0 learned params (fixed thresholds). Stage B-Q: ≤216 states / 27 params.
4. **Stable verdict contract.** Stage B: `RETAKE | REVIEW | PASS`. v3.1 band: hard-RETAKE sharpness < 0.08; 0.08–0.13 readable-blur → REVIEW; else PASS. This contract is the stable interface the Android/Kotlin port will bind to — do not silently change the vocabulary or semantics.
5. **Offline / CPU-only.** Everything must run offline and at sub-10 ms in practice.

## 3. Repo map

- `src/smallai/env/image_quality.py` — signal extraction + gate thresholds (v3.1: `RETAKE_SHARP_MIN=0.08`, `REVIEW_SHARP_MIN=0.13`, …)
- `src/smallai/stage_b.py` — fixed offline gate: extract → decide, synthetic sweep (`run_sweep`), real-corpus run (`run_real_corpus`), `check_images`
- `src/smallai/stage_b_q.py` — experimental Q gate: quantization, samples, `QCaptureAgent`, `check_images_with_q`, `run_experiment` (branch `feat/stage-b-q-gate`)
- `src/smallai/cli.py` — `quality` and `qgate` CLI commands (any image path; `--json`)
- `src/smallai/` (root of package) — Stage A journey-learning components (env / agent / solver)
- `tests/` — Stage A, Stage B gate, Stage B-Q, env tests
- `configs/`, `data/`, `runs/`, `reports/`, `docs/`, `dataset/` (gitignored corpus)
- `docs/AI_267_FULL_CONTEXT.md` — full AI handoff/context for AI-267 (current state, invariants, next steps)

## 4. How a behavior change should be made

- **Threshold change (gate):** must go through the supervised procedure (full corpus + labels + field evidence), update `env/image_quality.py`, `stage_b.py` + docs (`README.md` §11, `reports/AI-267_STAGE_B_RESULTS.md`) in the same change, and keep the supervised guarantees documented (zero false rejects on sharp-labelled; nothing blurred-labelled may PASS).
- **Q-gate change:** re-run `run_experiment` / retrain via `stage_b_q.QCaptureAgent`, keep fail-closed semantics + L1 generalization, update README §12.
- **New signal / verifier / action:** keep the parameter budget; update verdict contract only if the product decision allows (it currently doesn't).

## 5. Commands

```bash
make setup     # venv + editable install
make test      # PYTHONPATH=src .venv/bin/python -m pytest tests -q
.venv/bin/ruff check src tests
python -m smallai quality --image <file-or-dir> --json
python -m smallai qgate --image <file-or-dir>
```

## 6. AI assistant guardrails (vibe-coding)

- Do not add an LLM / transformer / sklearn into the decision or inference path.
- Do not change the `RETAKE | REVIEW | PASS` contract or its semantics silently.
- Do not retrain / re-calibrate thresholds without going through the documented supervised flow.
- Do not invent new verdict vocabularies without updating every consumer (CLI, Q-gate, tests, docs, Kotlin port contract).
- Keep PRs small; run tests + ruff before declaring done.
- When changing behavior, update `README.md`, `reports/AI-267_STAGE_B_RESULTS.md`, and `docs/ARCHITECTURE.md` in the same PR.

## 7. Testing expectations

- `make test` green (all stage suites), `ruff check src tests` clean.
- Determinism: fixed seeds; identical results on re-run.
- Offline: no network / no external services required by tests.

## 8. Docs sync checklist (update in same change as behavior)

- `README.md` (§11 Stage B, §12 Stage B-Q)
- `reports/AI-267_STAGE_B_RESULTS.md`
- `docs/ARCHITECTURE.md`
- `src/smallai/**` docstrings
