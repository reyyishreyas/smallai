# SmallAI — Project Context & Handoff

> **Purpose of this file:** a single, AI-friendly entry point that explains what
> SmallAI is, where every piece lives, what the current state is, and how to
> work forward safely. It complements (not replaces)
> `docs/AI_CODING_INSTRUCTIONS.md` (process), `docs/ARCHITECTURE.md` (design)
> and `reports/STAGE_B_RESULTS.md` (measurements).

## 1. What SmallAI is

SmallAI is a lightweight, **non-LLM**, from-scratch (NumPy) capture-quality gate
for document photos. It answers one question — *"is this capture usable?"* — and
returns `RETAKE | REVIEW | PASS` together with the signals and the thresholds
that fired. It runs offline on CPU, with no model download, no GPU and no API
call.

Two stages:

| Stage | Scope | Status |
|---|---|---|
| **B** | Fixed offline capture-quality gate: real images → signals → `RETAKE \| REVIEW \| PASS` | Delivered (v3.1) — full 2000-image corpus: RETAKE **10.3%**, REVIEW **20.35%**, PASS **69.35%** |
| **B-Q** | Experimental tabular-Q variant of the capture gate (lightest RL: Q-table + bandit rewards) | Delivered — **16 states / 48 params**, **policy-identical** to the v3.1 fixed gate (**2000/2000 verdicts**; mean reward 0.8076 for both). `0.648` is label agreement with soft corpus labels, not Q-vs-fixed disagreement |

## 2. Hard invariants (never break — full list in `docs/AI_CODING_INSTRUCTIONS.md` §2)

1. No LLM / no neural net in the inference path (pure NumPy + Pillow IO only).
2. Deterministic (fixed seeds; pure NumPy; seeded RNG only).
3. Parameter budgets (Stage B: 0 learned params; Stage B-Q: ≤216 states / 48 params — observed 16/48).
4. Stable verdict contract: capture `RETAKE | REVIEW | PASS`; verification
   per-field `MATCH | REVIEW | MISMATCH | NO_SOURCE_DATA`, overall
   `VERIFIED | NEEDS_REVIEW | FAILED | NOTHING_TO_VERIFY`.
5. Offline / CPU-only; **~14 ms per image** measured end-to-end, with
   **< 10 ms** as the optimization target.

## 3. Key code locations

- `src/smallai/env/image_quality.py` — signal extraction + v3.1 thresholds.
- `src/smallai/stage_b.py` — fixed offline gate, sweep, real-corpus runner, `check_images`.
- `src/smallai/stage_b_q.py` — Q gate: quantization, samples, `QCaptureAgent`, `check_images_with_q`, `run_experiment`.
- `src/smallai/cli.py` — commands: stage-b / quality / qgate.
- `tests/` — image-quality gate (Stage B), Stage B-Q suites.
- `docs/` — architecture, AI coding instructions, this file.
- `reports/` — `STAGE_B_RESULTS.md` (full measured write-up).
- `dataset/` — real corpora (gitignored): `images/` (2000 medical-prescription PNGs + `metadata.csv` blur labels), prescription sets, handwritten sets.
- `data/` — generated datasets + real mirror trio (`prescriptions/`).
- `runs/`, `data/` caches — gitignored.

## 4. Current delivered numbers (as of 2026-11-14)

- **Stage B (v3.1):** full corpus n=2000 → RETAKE 10.3% / REVIEW 20.35% / PASS 69.35%; sharp-labelled zero-false-RETAKE guarantee held; blurred-labelled zero-PASS guarantee held.
- **Stage B-Q:** retrained via real-ladder over the enlarged real dataset (`--real-images dataset/images`) → **16 states / 48 params**, **policy-identical** to the fixed gate (**2000/2000 verdicts**, reward **0.8076** for both). `qgate --diagnose` shows **0** safety violations for either policy and **435** label disagreements concentrated in mixed-label images both gates PASS; **0.648** is label agreement, not policy disagreement.
- **Performance:** 13.8 ms/image measured over 50 corpus images; 2000 images in ~40 s with flat memory.
- **Tests:** 31 passing; `ruff check` and `ruff format --check` clean.

## 5. How to work forward (AI tips)

- Behavior changes → follow `docs/AI_CODING_INSTRUCTIONS.md` §4 (supervised procedure for thresholds; Q-gate retune; docs sync in the same change).
- Keep verdict vocabulary stable; no silent contract changes.
- Run `make check` (lint + tests) before declaring done; `make build` to verify packaging.
- When in doubt: read `docs/ARCHITECTURE.md` §0–2 first (constraints), then `docs/AI_CODING_INSTRUCTIONS.md` §2–3.

## 6. Open threads / next steps

1. Validate on a second, independent real corpus (handwritten document sets).
2. Collect more field captures to firm up the 0.08 readable-blur boundary and the dark thresholds.
3. Adjudicate the **435** mixed-label corpus images both gates forward as `PASS`;
   the Q gate issues the same verdicts as the fixed gate, so the remaining gap is
   label quality, not policy.
4. Profile toward **< 10 ms** per capture.
5. Cross-implementation parity (Kotlin / Go / Java / Rust) against golden
   vectors for the `RETAKE | REVIEW | PASS` contract.
