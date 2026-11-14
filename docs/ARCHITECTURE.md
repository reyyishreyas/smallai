# ARCHITECTURE — SmallAI capture-quality gate (AI-267)

*One decision-maker for one job: is this document capture usable?*
*Pure NumPy. No neural nets, no LLM, no CV/ML library.*

## 0. Read this first

A field officer photographs an ID or form with a phone — often
offline, often in bad light. Some captures are unusable: blurry,
glared, half out of frame. Catching them *at the camera*, with a
reason ("sharpness 0.031 < 0.08 → RETAKE"), saves the officer's time and a
reviewer's time later. This project builds that decision-maker:

- **Input:** five quality signals measured from the capture
- **Output:** `RETAKE` / `REVIEW` / `PASS`, with human-readable reasons
  and the exact gates that tripped
- **Where it runs:** offline, CPU only, ~14 ms/image

It is *not* a document-content model (it does not read what the ID
says). It is a capture-quality gate — the first stage of a longer
pipeline; OCR and content verification stay separate concerns, per the
PoC scope agreed with the SmallAI/Mifos team.

## 1. The constraints and what each one forces

| Ticket constraint (AI-267) | Rules out | What we built instead |
|---|---|---|
| From scratch, no black-box ML | sklearn / xgboost / torch / OpenCV in the core | hand-written signal math (pure NumPy bilinear resample, gradients, percentiles) |
| Strictly non-LLM, non-transformer, no neural net in the inference path | any learned model in the shipped path | fixed, supervised-calibrated thresholds (0 learned params) |
| < 10 ms CPU | heavyweight inference | grid-resampled vectorized NumPy ops (~14 ms/image including IO) |
| Deterministic | hidden randomness | fixed seeds, pure NumPy math, reproducible verdicts |
| Stable verdict contract | ad-hoc labels | `RETAKE \| REVIEW \| PASS` — the interface the Android/Kotlin port binds to |
| No dataset exists for real captures | naive threshold guessing | supervised calibration on a full 2000-image labelled corpus + field evidence |

## 2. The big picture

```
 ┌──────────────────────────────────────────────┐
 │ CAPTURE  (phone photo of a document)         │
 └──────────────────────┬───────────────────────┘
                        │
                        ▼
 ┌──────────────────────────────────────────────┐
 │ FIXED ANALYSIS GRID (env/image_quality.py)   │
 │ pure-NumPy bilinear resample, max side 512,  │
 │ aspect preserved → resolution-invariant      │
 └──────────────────────┬───────────────────────┘
                        │
                        ▼
 ┌──────────────────────────────────────────────┐
 │ SIGNALS  (5, all in [0,1])                   │
 │ sharpness  contrast  ink_ratio               │
 │ brightness exposure                          │
 └──────────────────────┬───────────────────────┘
                        │
                        ▼
 ┌──────────────────────────────────────────────┐
 │ decide(signals)  (stage_b.py)                │
 │ thresholds v3.1 → RETAKE | REVIEW | PASS     │
 │ + human-readable reasons (which gate tripped)│
 └──────────────────────────────────────────────┘

 experimental variant (stage_b_q.py):
 the same 5 signals, quantized → tabular Q-learning
 policy with the same RETAKE | REVIEW | PASS contract
```

## 3. The signals (`env/image_quality.py`)

- **Fixed analysis grid.** Every capture is resampled (pure-NumPy bilinear,
  max side 512, aspect preserved) before measuring, so the signals — and every
  threshold — transfer across capture devices and pixel sizes.
- **sharpness** — mean gradient magnitude *inside the ink neighbourhood*
  (1-px dilation of `luma < p90 − 0.25`): text-edge strength, the blur proxy.
  Measuring only where the content is avoids the background dilution that
  makes whole-frame mean gradient useless on paper-dominated documents.
- **contrast** — `bg − p25(ink pixels)`: luminance contrast measured in the
  ink region, ink-coverage independent (whole-frame `p99.5 − p0.5` only as a
  no-ink fallback).
- **ink_ratio** — fraction of pixels darker than `p90 − 0.25`: the "is there
  actually content" detector (blank page / photo of a wall → ~0).
- **brightness** — mean luma (feeds the too-dark gate).
- **exposure** — `|mean luma − 0.5|` (reporting aid).

## 4. The gate (`stage_b.py` → `decide()`)

- **RETAKE** — unusable: too dark (`brightness < 0.10`), no detectable content
  (`ink_ratio < 0.002`), or text edges genuinely destroyed (`sharpness < 0.08`).
- **REVIEW** — usable but borderline (`brightness < 0.25`, `contrast < 0.18`,
  `ink_ratio < 0.003`, or `sharpness < 0.13`).
- **PASS** — usable → continue.

The 0.08–0.13 sharpness zone is the **readable-blur band**: camera blur
degrades pages differently from a synthetic blur kernel, so a blurry-but-
human-readable capture is flagged for a human (REVIEW), not hard-rejected.
There is deliberately **no high-brightness gate** — white paper is the normal
background of a document capture; truly blank captures are caught by
`ink_ratio ≈ 0`. All thresholds are overridable per call.

## 5. Supervised calibration

Thresholds are calibrated against the full 2000-image Kaggle corpus
(`priyanshuyav13/synthetic-medical-prescription-ocr-dataset`): normalized
sharpness separates the metadata blur labels with **no class overlap**
(blurred max 0.092 vs sharp-label floor 0.208). The load-bearing guarantees
with the v3.1 cuts:

- **zero false rejects** on all 952 sharp-labelled images;
- **zero blurred-labelled captures ever reach PASS** — every one is stopped
  (RETAKE or REVIEW), so nothing unreadable is silently forwarded to OCR.

The 0.08 hard-RETAKE boundary is human-anchored (one field capture) and sits
safely above the blur corpus floor (0.072). Full procedure and evidence:
`reports/AI-267_STAGE_B_RESULTS.md`.

## 6. Stage B-Q (`stage_b_q.py`) — the experimental RL gate

- **State** — 4 signals quantized on hand-chosen bucket edges (≤ 216 states;
  observed 16); sharpness buckets include the v3.1 boundary.
- **Actions** — `RETAKE | REVIEW | PASS` (same contract).
- **Reward** — `1 − cost(action, truth)` with an asymmetric cost matrix
  (PASS-on-RETAKE = 2.0, worst; RETAKE-on-PASS = 1.5, mild).
- **Learning** — ε-greedy off-policy Q-learning, one-step terminal episodes
  (contextual bandit); deterministic seeds, pure NumPy.
- **Training data** — corpus soft blur labels + real-ladder field evidence
  (blurred variants of real captures encode the human-judgment zones).
- **Deployment** — greedy policy; unseen states generalize to the nearest
  trained state within **L1 bucket distance ≤ 1**, otherwise **fail-closed**
  (truly novel → RETAKE). Inference never mutates the table.

Result: **16 states / 48 parameters**, learn-parity with the supervised v3.1
gate (mean reward 0.8076 vs 0.8076 over the full 2000-image corpus), ~14 ms/
image inference, ~2 s full training on CPU.

## 7. File map

- `src/smallai/env/image_quality.py` — signal extraction + v3.1 thresholds.
- `src/smallai/stage_b.py` — fixed gate: `extract → decide`, degradation
  sweep (`run_sweep`), real-corpus runner (`run_real_corpus`), `check_images`.
- `src/smallai/stage_b_q.py` — Q gate: quantization, samples, `QCaptureAgent`,
  `check_images_with_q`, `run_experiment`.
- `src/smallai/cli.py` — `quality`, `stage-b`, `qgate` commands.
- `tests/test_image_quality_stage_b.py`, `tests/test_stage_b_q.py` — gate and
  Q-gate suites.
- `reports/AI-267_STAGE_B_RESULTS.md` — full measured write-up.
- `dataset/` — real corpora (gitignored).
- `.github/workflows/ci.yml` — lint + tests on push/PR (Python 3.10–3.13).
- `CONTRIBUTING.md`, `CHANGELOG.md`, `SECURITY.md`, `LICENSE` — project
  governance.

## 8. Running it

```bash
make setup                                   # venv + editable install
make check                                   # ruff + tests
make build                                   # sdist + wheel
python -m smallai quality --image <file>      # one capture
python -m smallai quality --image <dir> --json
python -m smallai stage-b --images-dir dataset/images
python -m smallai qgate                       # train + compare vs fixed gate
```

## 9. Honest limitations

- Blur labels are soft for 849/2000 corpus images (inconsistent flagging
  between metadata rows).
- The corpus brightness range is 0.85–1.15, so the extreme-dark thresholds
  come from the controlled exposure ladder, not the corpus.
- The 0.08 readable-blur boundary is human-anchored (one field capture);
  REVIEW is where such judgment calls should live. Policy-aware calibration is
  required before using the gate as a hard guardrail.
- Rotation ±7° is out of scope.
