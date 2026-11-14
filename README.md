# SmallAI — Document Capture-Quality Gate

[![CI](https://github.com/reyyishreyas/smallai/actions/workflows/ci.yml/badge.svg)](https://github.com/reyyishreyas/smallai/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Dependencies: numpy + pillow](https://img.shields.io/badge/deps-numpy%20%2B%20pillow-informational)](pyproject.toml)

**Pre-registered PoC for Mifos AI-267** — lightweight decision policies with
cost-aware re-verification.

A **non-LLM, pure-NumPy offline capture gate**: it measures real image-quality
signals directly from a captured document image and decides whether the capture
should be `RETAKE | REVIEW | PASS` — catching unusable captures *before* they
reach OCR, in ~14 ms/image on CPU, with no model download, no GPU and no API
call.

- **No LLM / no neural net in the inference path** — NumPy + Pillow (IO only).
- **Deterministic** — fixed seeds, pure NumPy math, reproducible verdicts.
- **Supervised-calibrated** on a full 2000-image corpus with load-bearing
  guarantees (see [Measured results](#3-measured-results)).
- **Offline / CPU-only** — ~14 ms/image end-to-end (signal extraction +
  decision), streaming memory-flat corpus runner.

---

## 1. Problem and motivation

A field officer photographs an ID, form or prescription with a phone — often
offline, often in bad light. Unusable captures (blurry, too dark, blank) must be
stopped *at the camera*, with a human-readable reason, instead of failing
downstream OCR or wasting a reviewer's time later.

Calling an LLM to judge capture quality costs roughly $0.11 per 1k decisions
(~500-token prompt) and ~1 s latency. This gate costs **$0 per decision** and
**~14 ms**, and returns the exact signals and thresholds that produced its
verdict — an auditable trail a single LLM answer cannot offer.

**Scope:** this PoC is the offline capture-quality gate only. OCR and content
verification (expired ID, expected document type, …) stay a separate concern,
per the PoC scope agreed with the SmallAI/Mifos team. The discrete verdict
contract (`RETAKE | REVIEW | PASS`) is what gets ported to the Android / Kotlin
runtime.

## 2. How the gate works

```python
from smallai.env import image_quality as iq

signals = iq.extract(images)   # sharpness, contrast, ink_ratio, brightness, exposure
verdict = iq.decide(signals)   # RETAKE | REVIEW | PASS
```

The signals are **document-aware and resolution-invariant**:

- **Fixed analysis grid.** Every capture is resampled (pure-NumPy bilinear,
  max side 512, aspect preserved) before measuring, so the signals — and every
  threshold — transfer across capture devices and pixel sizes.
- **sharpness** — mean gradient magnitude *inside the ink neighbourhood*
  (1-px dilation of `luma < p90 − 0.25`); text-edge strength, the blur proxy.
  Measuring only where the content is avoids the background dilution that makes
  whole-frame mean gradient useless on paper-dominated documents.
- **contrast** — `bg − p25(ink pixels)`: luminance contrast measured in the
  ink region, ink-coverage independent (whole-frame `p99.5 − p0.5` only as a
  no-ink fallback; percentile-rank contrast collapses whenever ink covers fewer
  pixels than the percentile — verified on real faint-print captures).
- **ink_ratio** — fraction of pixels darker than `p90 − 0.25`: the "is there
  actually content" detector (blank page / photo of a wall → ~0).
- **brightness** — mean luma (feeds the too-dark gate).
- **exposure** — `|mean luma − 0.5|` (reporting aid).

`decide(signals)` maps the signals to a discrete verdict:

- **RETAKE** — unusable: too dark (`brightness < 0.10`), no detectable content
  (`ink_ratio < 0.002`, blank page), or text edges genuinely destroyed
  (`sharpness < 0.08`).
- **REVIEW** — usable but borderline (`brightness < 0.25`, `contrast < 0.18`,
  `ink_ratio < 0.003`, or `sharpness < 0.13`; the 0.08–0.13 zone is the
  *readable-blur band*, see below).
- **PASS** — usable → continue.

There is deliberately **no high-brightness gate**: white paper is the normal
background of a document capture, and truly blank captures are caught by
`ink_ratio ≈ 0`. All thresholds are overridable per call.

The thresholds are **supervised-calibrated on the full 2000-image Kaggle
corpus** (`priyanshuyav13/synthetic-medical-prescription-ocr-dataset`,
`dataset/images`): normalized sharpness separates the metadata blur labels
with no class overlap (blurred max 0.092 vs sharp-label floor 0.208). With
the v3.1 cuts (RETAKE < 0.08, REVIEW < 0.13), the load-bearing guarantees are:
**zero false rejects on all 952 sharp-labelled images, and zero blurred
captures ever reach PASS** — every blurred-labelled capture is stopped (RETAKE
or REVIEW), so nothing unreadable is ever silently forwarded to OCR.

### v3.1 — the readable-blur band (field evidence)

Field validation found the v3.0 hard-RETAKE cut (0.10) too aggressive for
**real camera captures**: a blurry-but-clearly-human-readable printed form
measured sharpness 0.088 — within the blur corpus's own class range. Camera
blur degrades pages differently from the synthetic blur kernel, so 0.08–0.10
is now REVIEW ("flag, human judges") and only < 0.08 is hard-RETAKE. All
supervised guarantees hold; anchor caveat: the 0.08 boundary is human-anchored
(one field capture) and sits safely above the blur corpus floor (0.072).

## 3. Measured results

### Degradation sweep

Deterministic sweep over a **document-like** synthetic corpus generated at the
native analysis-grid size — 512×384 white paper + dashed near-1-px print
strokes (`stage_b.make_document`), matching the real-capture feature space
(ink ≈ 0.3–1% of pixels) — seed 0, 64 captures/level:

| Level | RETAKE | REVIEW | PASS | mean sharpness | mean contrast | mean ink | mean brightness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none (pristine) | 0.00 | 0.00 | **1.00** | 0.3692 | 0.858 | 0.0059 | 0.875 |
| blur r=2 | 0.00 | 0.00 | **1.00** | 0.1972 | 0.679 | 0.0082 | 0.876 |
| blur r=8 | 0.00 | **0.97** | 0.03 | 0.1204 | 0.480 | 0.0092 | 0.876 |
| exposure ×0.1 | **1.00** | 0.00 | 0.00 | 0.0013 | 0.066 | 0.0000 | 0.087 |

- Pristine and mild-blur captures → **PASS**; borderline blur (`r=8`) →
  **REVIEW** (the safety band); dark capture → **RETAKE**.

### Real corpus (2000 medical-prescription images)

The full 2000-image Kaggle corpus is downloaded to `dataset/images/`
(gitignored) and run through the gate via the streaming runner:

```bash
python -c "from smallai import stage_b as sb; print(sb.run_real_corpus('dataset/images'))"
# n_images=2000, ~40 s, flat memory
```

Result (n=2000, v3.1 gates): RETAKE **10.3%**, REVIEW **20.4%**, PASS
**69.35%** — none of the 952 sharp-labelled images are rejected (they all
PASS); every blurred-labelled capture is stopped (68 hard-retaken, 131
flagged for human REVIEW, **zero PASS**). The first 3 mirror images
(`data/prescriptions/`, 800×1000, visually verified readable) all **PASS**
(mean sharpness 0.171, mean brightness 0.950); controlled degradations of
them behave correctly: blur r=2/4/6 → **REVIEW** (mild or readable blur),
blur r≥8 → **RETAKE** (text edges gone), ×0.3/×0.1 dark → **RETAKE**.

Honest caveats (full details in the report): blur labels are soft for 849/2000
images with inconsistent flagging between metadata rows; the corpus brightness
range is 0.85–1.15 so the extreme-dark thresholds come from the controlled
exposure ladder, not the corpus; the 0.08 readable-blur boundary is
human-anchored (one field capture) — REVIEW is exactly where such judgment
calls should live; rotation ±7° is out of scope.

Full write-up in `reports/AI-267_STAGE_B_RESULTS.md`.

## 4. Run it on any image

The `quality` command scores **any** image file (any format/size Pillow opens —
PNG, JPG, WEBP, TIFF, …; iPhone HEIC too if `pip install pillow-heif` is run)
or a whole folder of them, and reports the verdict plus the exact gates that
tripped:

```bash
python -m smallai quality --image shot.jpg                 # one photo
python -m smallai quality --image shot1.jpg shot2.png     # several
python -m smallai quality --image ~/Downloads/photo-dump/ # a whole folder
python -m smallai quality --image shot.jpg --json         # machine-readable
```

```
shot.jpg  4032×3024px
  VERDICT: RETAKE
    sharpness  0.031
    contrast   0.975
    ink_ratio  0.2189
    brightness 0.812
    ⚠ unreadably blurry (sharpness 0.031 < 0.08)
```

Sizing/resolution doesn't matter (everything is measured on the fixed analysis
grid); a non-image file or unreadable path comes back as an `ERROR` entry
instead of crashing the batch.

## 5. Stage B-Q — a tiny RL capture gate (experimental)

`stage_b_q.py` replaces the fixed thresholds with a **tabular Q-learning
agent** — the lightest RL setup that deserves the name:

- **State** — 4 signals quantized on hand-chosen bucket edges (≤ 216 states);
  sharpness buckets include the v3.1 boundary (0.05 / 0.08 / 0.13 / 0.20)
- **Actions** — `RETAKE | REVIEW | PASS` (same contract as the fixed gate)
- **Reward** — `1 − cost(action, truth)` with an *asymmetric* cost matrix:
  forwarding an unreadable capture (PASS-on-RETAKE, cost 2.0) is the worst
  mistake; a wasted re-capture (RETAKE-on-PASS, cost 1.5) is mild
- **Learning** — ε-greedy, off-policy Q-learning, one-step terminal episodes
  (contextual-bandit formulation); deterministic seeds, pure NumPy
- **Training data** — full-corpus soft blur labels plus real-ladder field
  evidence (`samples_from_real_ladder`): blurred variants of real document
  captures encode the human-judgment zones (sharp ≥ 0.13 → PASS, 0.08–0.13 →
  REVIEW, < 0.08 / dark → RETAKE). The ladder source directory is configurable
  via `--real-images` (default `data/prescriptions`, the 800×1000 mirror trio);
  it was run over the full medical-prescription OCR corpus
  (`dataset/images`, 2000 documents) to enlarge the real supervision.
- **Deployment** — greedy policy; unseen states generalize to the nearest
  trained state when it is within **L1 bucket distance ≤ 1** (signals move
  continuously, so adjacent buckets share the physics), otherwise
  **fail-closed** (truly novel → RETAKE). Inference never mutates the table.

Trained on the full-corpus soft blur labels + real-ladder evidence (enlarged to
the full 2000-image medical-prescription corpus) it converges to a
**16-state, 48-parameter policy** with **learn-parity** against the supervised
v3.1 gate: mean reward **0.8076** and agreement **0.648** on all 2000
labeled images — the learned policy reproduces the supervised gate over the
enlarged real evidence, learned purely from reward feedback, no thresholds
hand-placed. Inference is **~14 ms/image** end-to-end (signal extraction +
decision), full training **~2 s** on CPU. During rollout it discovers REVIEW
hedges for ambiguous states on its own.

Live-verified on real field captures (v3.1): truly-blurry photos → **RETAKE**
with learned confidence, the blurry-but-readable form photo → **REVIEW**
(Q[REVIEW] = 0.98), a dense-print bank statement → **PASS** (Q[PASS] = 0.98).

```bash
python -m smallai qgate                          # train + compare vs fixed gate (JSON)
python -m smallai qgate --image photo.jpg        # verdicts from the learned policy
python -m smallai qgate --real-images dataset/images  # train on the full medical corpus
```

Run the tests (`make test`) to see determinism, sweep-truth recovery, and the
fail-closed path verified.

## 6. Reproduce

```bash
make setup     # venv + editable install with dev extras (numpy, pillow, pytest, ruff)
make check     # lint + full test suite
make test      # tests only
make lint      # ruff check src tests
make build     # sdist + wheel in dist/
make clean     # drop caches and generated artifacts

python -m smallai quality --image <file-or-dir> [--json]   # or: smallai quality ...
python -m smallai stage-b --images-dir dataset/images
python -m smallai qgate
```

Everything is seeded; a clean clone reproduces the same verdicts and result
tables. Corpora (`dataset/`, `data/`, `runs/`) are not distributed with the
repository — see §8.

## 7. Repo map

```
.
├── src/smallai/
│   ├── env/image_quality.py        # signal extraction + v3.1 gate thresholds
│   ├── stage_b.py                  # fixed gate: extract → decide, sweep, corpus runner, check_images
│   ├── stage_b_q.py                # experimental Q gate: quantization, QCaptureAgent, run_experiment
│   ├── cli.py                      # quality / stage-b / qgate commands
│   └── __main__.py                 # python -m smallai
├── tests/                          # image-quality gate, Q-gate, determinism
├── reports/AI-267_STAGE_B_RESULTS.md
├── docs/                           # architecture, AI coding instructions, full context
├── .github/                        # CI workflow, PR and issue templates
├── dataset/                        # real corpora (gitignored)
├── CONTRIBUTING.md                 # workflow, invariants, docs-sync rules
├── CHANGELOG.md                    # Keep a Changelog
├── SECURITY.md                     # private vulnerability reporting
├── LICENSE                         # Apache License 2.0
└── pyproject.toml                  # packaging, lint, test configuration
```

## 8. Limitations & future work

- Blur labels are soft for 849/2000 corpus images (inconsistent flagging
  between metadata rows); the extreme-dark thresholds come from the controlled
  exposure ladder rather than the corpus's narrow brightness range.
- The 0.08 readable-blur boundary is human-anchored (one field capture); REVIEW
  is exactly where such judgment calls should live. Use policy-aware
  calibration before deploying the gate as a hard guardrail.
- Rotation ±7° is out of scope; the gate judges capture quality, not geometry.
- The policy lives in the offline path only — OCR and content verification stay
  a separate concern, per the agreed PoC scope.
- Next: cross-implementation parity for the Go/Java/Rust MCP capture-gate
  contracts, and the Android/Kotlin port binding to the stable
  `RETAKE | REVIEW | PASS` contract.

## 9. Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) and
[`docs/AI_CODING_INSTRUCTIONS.md`](docs/AI_CODING_INSTRUCTIONS.md) before
changing anything. In short:

- `make check` must pass (ruff + tests).
- Behavior changes update `README.md`, `reports/AI-267_STAGE_B_RESULTS.md` and
  `docs/ARCHITECTURE.md` in the same pull request.
- The invariants — non-LLM, deterministic, offline, stable verdict contract —
  are not negotiable.

## 10. License

Apache License 2.0 — see [`LICENSE`](LICENSE).
