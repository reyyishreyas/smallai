# SmallAI

[![CI](https://github.com/reyyishreyas/smallai/actions/workflows/ci.yml/badge.svg)](https://github.com/reyyishreyas/smallai/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Dependencies: numpy + pillow](https://img.shields.io/badge/deps-numpy%20%2B%20pillow-informational)](pyproject.toml)

**A non-LLM, pure-NumPy capture-quality gate for document photos.**

SmallAI measures real image-quality signals directly from a captured document
and decides whether the capture is usable — `RETAKE | REVIEW | PASS` — **before**
it reaches OCR. It needs no model download, no GPU, no API call and no network.

- **No LLM, no neural network, no ML framework** — NumPy for the math, Pillow for
  image IO only.
- **Deterministic** — fixed seeds, pure NumPy, identical verdicts on re-run.
- **Offline and tiny** — ~14 ms per capture on CPU, streaming, flat memory.
- **Auditable** — every verdict reports the signals and the exact threshold that
  fired, not a prose opinion.
- **Portable contract** — a discrete three-way verdict, designed to be
  reimplemented in Kotlin/Go/Java/Rust and conformance-tested.

```text
RETAKE   unusable — ask for a new capture
REVIEW   usable but borderline — flag for a human
PASS     usable — continue to OCR
```

---

## 1. Why

Field officers photograph IDs, forms and prescriptions with a phone — often
offline, often in bad light. Unusable captures (blurry, too dark, blank) must be
stopped **at the camera**, with a readable reason, instead of failing downstream
OCR or wasting a reviewer's time.

| | LLM judge | SmallAI gate |
| --- | --- | --- |
| Cost per 1k decisions | ~$0.11 | **$0** |
| Latency per decision | ~1 s | **~14 ms** |
| Network required | Yes | **No** |
| Output | prose verdict | signals + thresholds that fired |
| Deterministic | No | **Yes** |
| Learned parameters | — | **0** |

## 2. How it works

```text
image ──► resample to fixed analysis grid (max side 512)
      ──► measure {sharpness, contrast, ink_ratio, brightness, exposure}
      ──► decide() ──► RETAKE | REVIEW | PASS
```

The signals are **document-aware and resolution-invariant**, so the same
thresholds work across capture devices and pixel sizes:

- **sharpness** — mean gradient magnitude *inside the ink neighbourhood* (1-px
  dilation of `luma < p90 − 0.25`). Measuring only where the content is avoids
  the background dilution that makes whole-frame mean gradient useless on
  paper-dominated documents.
- **contrast** — `bg − p25(ink pixels)`: contrast measured in the ink region, so
  it is independent of ink coverage. (Whole-frame `p99.5 − p0.5` is used only as
  a no-ink fallback; percentile-rank contrast collapses whenever ink covers
  fewer pixels than the percentile.)
- **ink_ratio** — fraction of pixels darker than `p90 − 0.25`: the "is there
  actually content" detector (blank page or photo of a wall → ~0).
- **brightness** — mean luma.
- **exposure** — `|mean luma − 0.5|`, a reporting aid.

### Decision thresholds

| Signal | RETAKE below | REVIEW below |
| --- | --- | --- |
| `brightness` | 0.10 | 0.25 |
| `ink_ratio` | 0.002 (blank) | 0.003 |
| `sharpness` | 0.08 (unreadable) | 0.13 (soft) |
| `contrast` | — | 0.18 |

There is deliberately **no high-brightness gate**: white paper is the normal
background of a document capture, and genuinely blank captures are caught by
`ink_ratio ≈ 0`. All thresholds are overridable per call.

### The readable-blur band (v3.1)

Camera blur on a physical page degrades differently from a synthetic blur
kernel. A field capture of a printed VAT form was plainly human-readable at
sharpness **0.088** — inside the blur corpus's own class range — so the
hard-`RETAKE` cut sits at **0.08** and the 0.08–0.13 zone is `REVIEW`: flag it,
let a human judge. The boundary is human-anchored and is the most fragile number
in the gate (see §8).

## 3. Results

### Degradation sweep

Deterministic synthetic corpus at the native analysis-grid size (document-like
white paper with near-1-px print strokes), seed 0, 64 captures per level:

| Level | RETAKE | REVIEW | PASS | mean sharpness | mean contrast |
| --- | --- | --- | --- | --- | --- |
| none (pristine) | 0.00 | 0.00 | **1.00** | 0.3692 | 0.858 |
| blur r=2 | 0.00 | 0.00 | **1.00** | 0.1972 | 0.679 |
| blur r=8 | 0.00 | **0.97** | 0.03 | 0.1204 | 0.480 |
| exposure ×0.1 | **1.00** | 0.00 | 0.00 | 0.0013 | 0.066 |

Pristine and mildly blurred captures `PASS`; borderline blur lands in the
`REVIEW` safety band; a dark capture is retaken.

### Real corpus (2,000 prescription images)

| Metric | Result |
| --- | --- |
| Verdicts | RETAKE **10.3%** · REVIEW **20.35%** · PASS **69.35%** |
| Sharp-labelled images (952) | **0 false rejects** — all `PASS` |
| Blurred-labelled images | **0 reach `PASS`** (68 `RETAKE`, 131 `REVIEW`) |
| Signal separation vs labels | blurred max sharpness 0.092 vs sharp-label floor 0.208 |

Two guarantees are load-bearing: **nothing unreadable is silently forwarded to
OCR**, and **nothing readable is rejected**.

### Performance

- **~14 ms per capture** end-to-end (signal extraction + decision), measured at
  13.8 ms/capture over 50 corpus images on CPU.
- Streaming corpus runner: 2,000 images in ~40 s with flat memory.
- **27 tests**, lint-clean, no runtime dependency beyond NumPy and Pillow.

## 4. Install

Requires Python 3.10+. Runtime dependencies are `numpy` and `pillow`.

```bash
git clone https://github.com/reyyishreyas/smallai.git
cd smallai
make setup            # venv + editable install with dev extras
```

Equivalent without `make`:

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

The package is not published to PyPI; install from source.

## 5. Usage

### Command line

```bash
smallai quality --image photo.jpg                # one capture
smallai quality --image a.jpg b.png d/           # several files or whole folders
smallai quality --image d/ --json                # machine-readable
```

`python -m smallai …` works identically. Any format Pillow can read is accepted
(PNG, JPEG, WEBP, TIFF, BMP …); iPhone HEIC works if `pillow-heif` is installed.
Unreadable paths come back as an `ERROR` entry instead of crashing a batch.

```console
$ smallai quality --image dataset/images/prescription_1004.png

prescription_1004.png  (dataset/images/prescription_1004.png)  710×466px
  VERDICT: RETAKE
    sharpness  0.079   (RETAKE < 0.08, REVIEW < 0.13)
    contrast   0.362   (REVIEW < 0.18)
    ink_ratio  0.0093   (RETAKE < 0.002, REVIEW < 0.003)
    brightness 0.908   (RETAKE < 0.10, REVIEW < 0.25)
    ⚠ unreadably blurry (sharpness 0.079 < 0.08)

summary: 1 image(s) — PASS 0, REVIEW 0, RETAKE 1  (0% pass)
```

```json
{
  "path": "dataset/images/prescription_1004.png",
  "size": [710, 466],
  "signals": {
    "sharpness": 0.07894,
    "contrast": 0.36177,
    "ink_ratio": 0.009277,
    "brightness": 0.907778,
    "exposure": 0.407778
  },
  "verdict": "RETAKE",
  "reasons": ["unreadably blurry (sharpness 0.079 < 0.08)"]
}
```

Other commands:

```bash
smallai stage-b --images-dir dataset/images [--n 500]   # verdict distribution over a corpus
smallai qgate [--real-images dataset/images]            # train + compare the Q variant
```

### Library

```python
import numpy as np
from smallai.env import image_quality as iq

images = np.zeros((1, 480, 640, 3), dtype=np.float32)   # RGB in [0, 1]

signals = iq.extract(images)                            # dict of arrays
verdicts = iq.decide_offline(images)                    # int verdict codes
print(iq.verdict_name(int(verdicts[0])))                # "RETAKE"
```

For per-file results with reasons, use `smallai.stage_b.check_images(paths)`.

## 6. Stage B-Q — a tiny RL gate (experimental)

`stage_b_q.py` replaces the fixed thresholds with a **tabular Q-learning agent**:
the lightest RL setup that deserves the name.

- **State** — four signals quantised on hand-chosen bucket edges (≤216 states);
  the sharpness buckets include the v3.1 boundaries (0.05 / 0.08 / 0.13 / 0.20).
- **Actions** — `RETAKE | REVIEW | PASS`, the same contract as the fixed gate.
- **Reward** — `1 − cost(action, truth)` with an asymmetric cost matrix:
  forwarding an unreadable capture (`PASS` on `RETAKE`, cost 2.0) is the worst
  mistake; a wasted re-capture (`RETAKE` on `PASS`, cost 1.5) is mild.
- **Learning** — ε-greedy, off-policy Q-learning over one-step terminal episodes;
  deterministic seeds, pure NumPy.
- **Deployment** — greedy policy. Unseen states generalise to the nearest trained
  state only within L1 bucket distance ≤ 1; anything further **fails closed** to
  `RETAKE`. Inference never mutates the table.

Trained on soft corpus labels plus real-ladder evidence from the full 2,000-image
corpus, it converges to a **16-state, 48-parameter** policy with learn-parity
against the fixed gate: **mean reward 0.8076 vs 0.8076**, per-image agreement
0.648. (The default 3-image ladder gives a 9-state, 27-parameter table.) Training
takes ~2 s on CPU.

This stage is **experimental** — it is a research comparison, not the shipped
policy.

## 7. Verify

```bash
make check     # ruff check + ruff format --check + 27 tests
make test      # tests only
make lint      # lint only
make build     # sdist + wheel in dist/
make clean     # drop caches and generated artifacts
```

Everything is seeded: a clean clone reproduces the same verdicts and tables.

## 8. Project structure

```text
src/smallai/
  env/image_quality.py   signal extraction + v3.1 thresholds
  stage_b.py             fixed gate: decide(), sweep, corpus runner, check_images
  stage_b_q.py           experimental Q gate: quantisation, agent, experiment
  cli.py                 quality / stage-b / qgate commands
  __main__.py            python -m smallai
tests/                   gate suite, Q-gate suite, determinism
docs/                    ARCHITECTURE, AI coding instructions, CONTEXT
reports/                 STAGE_B_RESULTS.md — measured write-up
.github/                 CI workflow, PR and issue templates
```

Corpora (`dataset/`, `data/`, `runs/`) are **not** distributed with the
repository.

## 9. Design constraints

1. **No LLM, transformer, or high-level ML library in the inference path.**
   NumPy math; Pillow for IO only.
2. **Deterministic.** Fixed seeds, pure NumPy, reproducible verdicts.
3. **Stable contract.** `RETAKE | REVIEW | PASS` — no silent vocabulary changes.
4. **Offline, CPU-only.** No network, GPU, or model download at inference.
5. **Small.** Stage B has 0 learned parameters; Stage B-Q stays ≤216 states.

## 10. Limitations

- Blur labels are soft for **849/2000** corpus images (inconsistent flagging
  between metadata rows), so the separation bound is softer than it looks.
- The corpus brightness range is **0.85–1.15**; extreme-dark thresholds come from
  a controlled exposure ladder rather than the corpus.
- The **0.08** readable-blur boundary is human-anchored on a single field
  capture. `REVIEW` exists precisely so such judgment calls stay with a human.
- Calibration rests on **one** synthetic-blur corpus; a second independent corpus
  has not been run yet.
- Rotation ±7° is out of scope — the gate judges capture quality, not geometry.
- The gate is a quality signal, not a security or content check.

## 11. Roadmap

- Validate on a second, independent real corpus (handwritten document sets).
- Field-capture set to firm up the 0.08 boundary and the dark thresholds.
- Analyse and reduce the Q-gate/fixed-gate disagreement (0.648).
- Profile toward **< 10 ms** per capture.
- Per-document-type policy calibration before hard-guardrail deployment.
- Kotlin port of the verdict contract, with golden-vector conformance tests, and
  parity across the Go/Java/Rust implementations.

## 12. Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) and
[`docs/AI_CODING_INSTRUCTIONS.md`](docs/AI_CODING_INSTRUCTIONS.md) first.
`make check` must pass; behavior changes update the docs in the same change; the
invariants in §9 are not negotiable.

Security reports: see [`SECURITY.md`](SECURITY.md) — please do not open a public
issue.

## 13. License

Apache License 2.0 — see [`LICENSE`](LICENSE).
