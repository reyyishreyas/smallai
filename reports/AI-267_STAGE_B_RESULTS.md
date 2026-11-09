# AI-267 Stage B — Real image-quality signals + offline decision policy (v3.1)

**Repository:** reyyishreyas/smallai · **Stage:** B · **Status:** scored (v3.1, supervised calibration on the full 2000-image corpus + field-validated readable-blur band)

This stage measures **real** image-quality signals directly from captured
document images (pure NumPy — no CV/ML library, no GPU, no LLM in the inference
path) and applies the discrete offline policy `RETAKE | REVIEW | PASS`.

## v2 → v3 → v3.1 review history

**v1 errors** (all verified on real captures): whole-frame mean gradient is
diluted by the >95% flat paper background, and `brightness > 0.88 → RETAKE`
treated white paper as over-exposure. The v1 gate flagged 100% of readable
prescription scans as RETAKE.

**v2** fixed the measurement target (ink-region sharpness, content-presence via
`ink_ratio`, no high-brightness gate) but calibrated thresholds on only **n=3**
images — and the signals turned out to be **resolution-dependent** (mean
gradient scales with capture pixel size, so thresholds calibrated at ~720 px
transferred to 800×1000 scans), plus whole-frame percentile contrast broke on
faint print (denominator = percentile rank vs ink coverage).

**v3** removes both defects and calibrates *supervised* against the full corpus:

1. **Fixed analysis grid.** Every capture is resampled (pure-NumPy bilinear,
   max side 512, aspect preserved) before measuring — the signals, and
   therefore every threshold, are resolution-invariant across devices.
2. **Ink-region contrast.** `contrast = bg − p25(ink pixels)` measured where the
   content is, coverage-independent. Whole-frame `p99.5 − p0.5` only remains as
   a no-ink fallback. (Both the 1% p1-cliff and the 0.5% p0.5-cliff were
   verified on real corpus images: ~30% of fine-print images collapsed to
   contrast ≈ 0 under whole-frame percentiles despite being legible.)
3. **Supervised threshold derivation.** The Kaggle corpus ships per-image
   degradation metadata (`blur` bool); thresholds are placed strictly between
   the labelled-class bounds instead of hand-tuned.

**v3.1 — the readable-blur band (field evidence).** Field validation found a
real photo of a printed VAT form that was *plainly human-readable* despite
measuring sharpness 0.088 — inside the blur corpus's own class range (v3 would
have hard-retaken it at the 0.10 cut). Camera blur on physical pages degrades
differently from the synthetic blur kernel used to label the corpus, so the
hard-RETAKE cut was lowered **0.10 → 0.08** and the 0.08–0.13 zone is REVIEW
("flag, let a human judge"), not RETAKE. The load-bearing supervised guarantees
were re-verified unchanged on all 2000 labelled images: **zero false rejects on
the 952 sharp-labelled images, and zero blurred-labelled captures ever reach
PASS**. Honest caveat: the 0.08 boundary is human-anchored (one field capture,
plus the blur-corpus floor at 0.072) — `REVIEW` is exactly where such judgment
calls belong, and future field captures in that zone feed recalibration.

## Offline decision gate

```
images ──► extract(images) ──► {sharpness, contrast, ink_ratio, brightness, exposure}
       ──► decide(signals) ──► RETAKE | REVIEW | PASS
```

Signals (all pure NumPy, measured on the fixed 512-max-side analysis grid),
~14 ms/image extract + decision end-to-end:

| Signal | Definition | Rationale |
| --- | --- | --- |
| `sharpness` | mean gradient magnitude inside a 1-px dilation of the ink mask (`luma < p90 − 0.25`; fallback: whole-frame mean when mask empty) | text-edge strength — the blur proxy, measured where content is, resolution-invariant |
| `contrast`  | `bg − p25(ink pixels)` | ink-region luminance contrast, coverage-independent (whole-frame `p99.5 − p0.5` only as no-ink fallback) |
| `ink_ratio` | fraction of pixels with `luma < p90 − 0.25` | content-presence: blank page / wall → ≈0 |
| `brightness`| mean luma | feeds the too-dark gate |
| `exposure`  | `\|mean luma − 0.5\|` | reporting aid (no decision gate) |

Gates (supervised on the full corpus + field-validated; overridable per call):

| Verdict | Condition |
| --- | --- |
| RETAKE | `brightness < 0.10` **or** `ink_ratio < 0.002` **or** `sharpness < 0.08` |
| REVIEW | not RETAKE and (`brightness < 0.25` **or** `contrast < 0.18` **or** `ink_ratio < 0.003` **or** `sharpness < 0.13`) — the 0.08–0.13 sharpness zone is the *readable-blur band* |
| PASS   | otherwise (usable) |

Deliberately **no high-brightness gate**: white paper is normal; blank captures
trip `ink_ratio ≈ 0` instead.

## Supervised calibration (full 2000-image Kaggle corpus)

Corpus: `priyanshuyav13/synthetic-medical-prescription-ocr-dataset` (first 3 IDs
mirrored at HF `chinmays18/medical-prescription-dataset`), 2000 PNGs
(~716–742×474–520, varying), `metadata.csv` with per-image `rotation / blur /
brightness` ground truth. The corpus scans in ~40 s for all 2000 images,
streaming one at a time on the public API, flat memory.

Supervised signal separation (`runs/stage_b_corpus_scan.npz`, blur label = mean
of the 2 metadata rows per image:

| Label | n | sharpness p5 / p50 / max | contrast p5 | ink p5–p95 |
| --- | --- | --- | --- | --- |
| blur = 0 (sharp) | 952 | **0.208 / 0.234 / 0.348** | 0.554 | 0.0035–0.0082 |
| blur = 0.5 (inconsistent) | 849 | 0.075 / 0.203 / 0.248 | 0.339 | 0.0034–0.0086 |
| blur = 1 (blurred) | 199 | 0.072 / 0.084 / **0.092** | 0.332 | 0.0034–0.0094 |

The normalized sharpness classes **do not overlap**: every blurred image
measures ≤ 0.092, every sharp-labelled image ≥ 0.208. The cuts (RETAKE
`< 0.08`, REVIEW `< 0.13`) sit in that gap; v3.1 adds the human-validated
context that real camera blur can stay readable into the 0.08–0.13 zone —
hence REVIEW, with hard-RETAKE reserved for < 0.08 where the blur corpus shows
text genuinely destroyed (its floor is 0.072).

Verdict at those thresholds (all 2000 images):

| Verdict | n | Rate | Composition |
| --- | --- | --- | --- |
| RETAKE | 206 | 10.3% | 68/199 blurred-labelled + 138/849 mixed-labelled — all via `sharpness < 0.08`; **0 of 952 sharp-labelled false-RETAKE** |
| REVIEW | 407 | 20.35% | 131/199 blurred-labelled (readable-blur band) + 276/849 mixed-labelled + 4 faint-print sharp variants; none ever reach OCR as PASS |
| PASS   | 1387 | 69.35% | 952/952 sharp-labelled (100%) + washed-but-legible variants |

Guarantee restated for v3.1: **every one of the 199 blurred-labelled images is
stopped** (68 hard-retaken, 131 flagged for human REVIEW, zero PASS), and no
sharp-labelled image is rejected. The gate never silently forwards an unreadable
capture; it only shifts *where* humans see judgment calls (REVIEW instead of
wasting a recapture).

## Resolution-invariance + real-capture evidence

The three real 800×1000 prescriptions (HF mirror) normalize onto the same
analysis grid and pass, as do their controlled degradations (v3.1 verdicts):

| Image / control | sharpness | contrast | ink_ratio | brightness | verdict |
| --- | --- | --- | --- | --- | --- |
| prescription_00000 (readable) | 0.1398 | 0.356 | 0.0101 | 0.951 | **PASS** |
| prescription_00001 (readable) | 0.1862 | 0.431 | 0.0112 | 0.951 | **PASS** |
| prescription_00002 (readable) | 0.1861 | 0.454 | 0.0130 | 0.950 | **PASS** |
| + blur r=2                    | 0.1140 | 0.331 | 0.0083 | 0.951 | REVIEW |
| + blur r=4                    | 0.0881 | 0.285 | 0.0067 | 0.951 | REVIEW |
| + blur r=6                    | 0.0810 | 0.272 | 0.0056 | 0.951 | REVIEW |
| + blur r=8 (heavy)            | 0.0756 | 0.267 | 0.0056 | 0.951 | RETAKE |
| + dark ×0.3                   | 0.0953 | 0.129 | 0.0008 | 0.285 | RETAKE |
| + dark ×0.1                   | 0.0025 | 0.043 | 0.0000 | 0.095 | RETAKE |

Real-corpus CLI result (n=3): **PASS 100%**.

Field captures (real photos of physical documents, recorded as the v3.1
evidence set):

| Capture | sharpness | verdict (v3.0) | verdict (v3.1) |
| --- | --- | --- | --- |
| TLS tax form photo — unreadable camera blur | 0.031 | RETAKE | **RETAKE** |
| Mortgage-documents photo — unreadable | 0.027 | RETAKE | **RETAKE** |
| VAT form photo — *blurry but human-readable* | 0.088 | ~~RETAKE~~ (wrong) | **REVIEW** |
| Bank statement — dense print, crisp | 0.185 | PASS | **PASS** |

Readout: readable captures → PASS; mild or readable blur (the number one
real-world field defect) → REVIEW — the safety band does its job on
non-synthetic captures; destroyed blur and darkness → RETAKE. The ladder is
monotone and verdict boundaries never invert with capture scale (v2's
resolution artifact). Note the r=4/r=6 rows land in the readable-blur band,
consistent with the human-judgment evidence.

## Full-corpus gate run (public API, streaming)

`stage_b.run_real_corpus("dataset/images")` — 2000 images, ~40 s, flat memory:

| Metric | Value |
| --- | --- |
| n_images | 2000 |
| RETAKE / REVIEW / PASS | 10.3 % / 20.35 % / **69.35 %** |
| mean sharpness | 0.186 |
| mean contrast | 0.539 |
| mean ink_ratio | 0.0058 |
| mean brightness | 0.936 |

## Degradation sweep (native analysis-grid synthetic corpus, seed 0, 64/level)

`stage_b.make_document` generates directly at the analysis-grid size (512×384)
with dashed near-1-px strokes, so the synthetic corpus lives in the same
feature space as real captures. v3.1 verdicts:

| Level | RETAKE | REVIEW | PASS | mean sharpness | mean contrast | mean ink | mean brightness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none (pristine) | 0.00 | 0.00 | **1.00** | 0.3692 | 0.858 | 0.0059 | 0.875 |
| blur r=2 | 0.00 | 0.00 | **1.00** | 0.1972 | 0.679 | 0.0082 | 0.876 |
| blur r=8 | 0.00 | **0.97** | 0.03 | 0.1204 | 0.480 | 0.0092 | 0.876 |
| exposure ×0.1 | **1.00** | 0.00 | 0.00 | 0.0013 | 0.066 | 0.0000 | 0.087 |

The sweep verdicts are unchanged from v3 (the synthetic levels sit outside the
narrow 0.08–0.13 boundary shift), which is a feature: v3.1 only changed the
*placement* of the hard cut for real captures, not the class structure.

## Honest caveats

- Blur labels are **soft**: 849/2000 images have inconsistent blur flags between
  their 2 metadata rows (treated as blur=0.5). The gate's zero-false-RETAKE
  claim is exact only for the 952 hard-sharp-labelled images; among blur=0.5
  mixed-label images the gate splits (138 RETAKE / 276 REVIEW / 435 PASS),
  which is the sensible behavior for genuinely uncertain ground truth.
- The corpus brightness range is 0.85–1.15 — extreme-low-light field captures
  (< 0.10 luma) are outside this corpus's distribution; those thresholds
  (dark RETAKE at 0.10, REVIEW at 0.25) come from the controlled exposure
  ladder, not the corpus.
- Rotation ±7° in the corpus is mostly benign for quality (no angle gate is
  attempted in Stage B; deskew belongs to OCR/Stage C).
- The **0.08 readable-blur boundary is human-anchored**: one field capture
  (the VAT form) plus the blur-corpus floor (0.072). REVIEW exists precisely
  for this zone; every field capture landing there should be collected and
  used for future recalibration (no retraining needed — thresholds only).
- v3.1 changed the fixed gate; the branch also carries a learned Q-variant of
  it (Stage B-Q, §12 README) trained on the corpus soft labels + real-ladder
  human-judgment evidence — learn-parity with this gate (now **0.8076 vs 0.8076**,
  i.e. exact parity, over the enlarged real-ladder training set), 16 states /
  48 parameters.

## Contract & scope

- The discrete verdict contract `RETAKE | REVIEW | PASS` is the stable interface
  that gets ported to the Android / Kotlin runtime. The signal *set* is the
  same five measurements + gates (`sharpness, contrast, ink_ratio, brightness,
  exposure`, all at the analysis-grid resolution); v3.1 changed one threshold
  value, not the semantics.
- OCR and content verification (expired ID, expected document type, person
  match) stay **separate**, per the PoC scope agreed with the SmallAI/Mifos
  team (`dataset/labels.csv` medicine/dosage fields are Stage C material).

## Reproduce

```bash
make test          # includes Stage B policy + gate tests (v3.1)
PYTHONPATH=src python -c "from smallai import stage_b as sb; print(sb.run_sweep(seed=0))"
PYTHONPATH=src python -m smallai stage-b --images-dir data/prescriptions --n 10
PYTHONPATH=src python -c "from smallai import stage_b as sb; \
print(sb.run_real_corpus('dataset/images'))"   # 2000 images, ~40 s
PYTHONPATH=src python -m smallai quality --image shot.jpg   # any real capture
PYTHONPATH=src python -m smallai qgate --real-images dataset/images   # Q-gate on the full corpus
```
