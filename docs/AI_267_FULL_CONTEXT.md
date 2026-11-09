# AI-267 — Full Context & Handoff (for AI-assisted development)

> **Purpose of this file:** a single, AI-friendly entry point that explains what
> SmallAI is building (AI-267), where every piece lives, what the current
> delivered state is, and how to work forward safely. It complements (not
> replaces) `docs/AI_CODING_INSTRUCTIONS.md` (process), `docs/ARCHITECTURE.md`
> (design), and the `reports/AI-267_*.md` (results).

## 1. What AI-267 is

SmallAI is the lightweight, **non-LLM**, from-scratch (NumPy) decision model for
the Mifos X AI initiative (AI-267): *"is this document capture usable?"* — plus
the deterministic arms (capture gate, verification) delivered through the
Mifos X MCP servers.

Three stages:

| Stage | Scope | Status |
|---|---|---|
| **A** | Journey-learning decision policy (Tabular Q agent + solver + verifier + final head) over checkable states (energy / environment / nav) | Delivered — Test C delta **+15.54pp** (M 88.60% vs B2 73.06%), verifier AUC **0.938** (gate > 0.9 cleared) |
| **B** | Fixed offline capture-quality gate: real images → signals → `RETAKE \| REVIEW \| PASS` | Delivered (v3.1) — full 2000-image corpus: RETAKE **10.3%**, REVIEW **20.35%**, PASS **69.35%** |
| **B-Q** | Experimental tabular-Q variant of the capture gate (lightest RL: Q-table + bandit rewards) | Delivered — now **16 states / 48 params**, exact learn-parity with the v3.1 fixed gate (mean reward 0.8076 vs 0.8076; agreement 0.648 on all 2000 labeled images) |

## 2. Hard invariants (never break — full list in `docs/AI_CODING_INSTRUCTIONS.md` §2)

1. No LLM / no neural net in the inference path (pure NumPy + Pillow IO only).
2. Deterministic (fixed seeds; pure NumPy; seeded RNG only).
3. Parameter budgets (Stage A ≤ ~15.6K states / few legal actions; Stage B 0 learned params; Stage B-Q ≤ 216 states / 27 params — currently observed 16/48).
4. Stable verdict contract: capture `RETAKE | REVIEW | PASS`; verification
   per-field `MATCH | REVIEW | MISMATCH | NO_SOURCE_DATA`, overall
   `VERIFIED | NEEDS_REVIEW | FAILED | NOTHING_TO_VERIFY`.
5. Offline / CPU-only; sub-10 ms in practice.

## 3. Key code locations

- `src/smallai/env/image_quality.py` — signal extraction + v3.1 thresholds.
- `src/smallai/stage_b.py` — fixed offline gate, sweep, real-corpus runner, `check_images`.
- `src/smallai/stage_b_q.py` — Q gate: quantization, samples, `QCaptureAgent`, `check_images_with_q`, `run_experiment`.
- `src/smallai/env/oracle.py`, `agent.py`-family modules (Stage A), `train.py`, `evaluate.py`, `benchmark.py`.
- `src/smallai/cli.py` — commands: generate / train / evaluate / benchmark / predict / stage-b / quality / qgate.
- `configs/default.yaml` — every hyperparameter (no magic numbers).
- `tests/` — oracle, gradients, shapes, determinism, image quality (Stage B), Stage B-Q, env.
- `docs/` — AI coding instructions, architecture, experiments, PRDs, stage results.
- `reports/` — results, benchmark, stage B/Q results.
- `dataset/` — real corpora (gitignored): `images/` (2000 medical-prescription PNGs + `metadata.csv` blur labels), prescription sets, Bangladesh handwritten sets.
- `data/` — generated datasets + real mirror trio (`prescriptions/`).
- `runs/`, `data/` caches — gitignored.

## 4. Current delivered numbers (as of 2026-11-09)

- **Stage A:** M Test C **88.60%** vs B2 **73.06%** → **+15.54pp** delta (target ≥ +5 cleared); verifier AUC **0.938**.
- **Stage B (v3.1):** full corpus n=2000 → RETAKE 10.3% / REVIEW 20.35% / PASS 69.35%; sharp-labelled zero-false-RETAKE guarantee held; blurred-labelled zero-PASS guarantee held.
- **Stage B-Q:** retrained via real-ladder over enlarged real dataset (`--real-images dataset/images`) → **16 states / 48 params; exact parity** with fixed gate (reward **0.8076 vs 0.8076**, agreement **0.648**).

## 5. How to work forward (AI tips)

- Behavior changes → follow `docs/AI_CODING_INSTRUCTIONS.md` §4 (supervised procedure for thresholds; Q-gate retune; docs sync in the same change).
- Keep verdict vocabulary stable; no silent contract changes.
- Run `make test` (or `PYTHONPATH=src .venv/bin/python -m pytest tests -q`) and `.venv/bin/ruff check src tests` before declaring done.
- When in doubt: read `docs/ARCHITECTURE.md` §0–2 first (constraints), then `docs/AI_CODING_INSTRUCTIONS.md` §2–3.

## 6. Open threads / next steps

1. Post/finalize AI-267 delivery reply (done via issue #1 in `reyishreyas/smallai`).
2. Optional: Stage A dataset-size increase (`n_train`) and retrain; further real-corpus enrichment; deeper Q-gate analysis.
3. Cross-implementation parity for Go/Java/Rust MCP capture-gate contracts (tracked as openMF/mcp-mifosx#677).
4. MCP-side composed workflows beyond `run_document_intake` (e.g., `run_loan_intake`).
