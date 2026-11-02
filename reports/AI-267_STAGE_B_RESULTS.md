# AI-267 Stage B — Real image-quality signals + offline decision policy (v2)

**Repository:** reyyishreyas/smallai · **Stage:** B · **Status:** scored (v2, document-aware recalibration)

Stage A scored a tabular Q-table over *synthetic* image-quality signals. Stage B
measures **real** image-quality signals directly from captured document images
(pure NumPy — no CV/ML library, no GPU, no LLM in the inference path) and applies
the discrete offline policy `RETAKE | REVIEW | PASS`.

## v2 — why the gate was recalibrated

The v1 gate (whole-frame mean gradient, `brightness > 0.88 → RETAKE`) flagged
**100% of the real prescription corpus as RETAKE** — including images that are
genuinely readable (dark handwriting on white paper). Two measurement errors:

1. **Background dilution.** A document frame is mostly flat paper, so the
   whole-frame mean gradient (v1 sharpness) is dominated by background ~0 and
   the text-edge evidence (the only real sharpness signal) washes out. Measured
   on the real corpus: readable text → mean gradient 0.009–0.034, i.e. below
   v1's REVIEW threshold — meaningless on documents.
2. **White paper ≠ over-exposure.** `brightness > 0.88 → RETAKE` treats the
   normal background of a document as a defect. Real scan mean luma is 0.95.

v2 therefore measures quality **where the content is** (document-aware feature
engineering) and uses an explicit **content-presence** detector instead of a
high-brightness gate.

## Offline decision gate

```
images ──► extract(images) ──► {sharpness, contrast, ink_ratio, brightness, exposure}
      ──► decide(signals) ──► RETAKE | REVIEW | PASS
```

Signals (all pure NumPy, per image):

| Signal | Definition | Rationale |
| --- | --- | --- |
| `sharpness` | mean gradient magnitude inside a 1-px dilation of the ink mask (`luma < p90 − 0.25`; fallback: whole-frame mean when mask empty) | text-edge strength — the blur proxy, measured where content is |
| `contrast`  | `p99.5 − p0.5` of luma | robust luminance range; robust to documents with <1% ink (p1 cliff fixed) |
| `ink_ratio` | fraction of pixels with `luma < p90 − 0.25` | content-presence: blank page / wall → ≈0 |
| `brightness`| mean luma | feeds the too-dark gate |
| `exposure`  | `\|mean luma − 0.5\|` | reporting aid (no decision gate) |

Gates (calibrated on the real corpus + controlled degradations; overridable):

| Verdict | Condition |
| --- | --- |
| RETAKE | `brightness < 0.10` **or** `contrast < 0.10` **or** `ink_ratio < 0.002` **or** `sharpness < 0.055` |
| REVIEW | not RETAKE and (`brightness < 0.25` **or** `contrast < 0.18` **or** `ink_ratio < 0.005` **or** `sharpness < 0.095`) |
| PASS   | otherwise (usable) |

Deliberately **no high-brightness gate**: white paper is normal; blank captures
trip `ink_ratio ≈ 0` instead.

## Calibration evidence (real prescription corpus + controlled degradations)

Real images: `chinmays18/medical-prescription-dataset` (Kaggle
`priyanshuyav13/synthetic-medical-prescription-ocr-dataset` mirror), 800×1000,
handwritten prescriptions — verified visually readable.

| Image / control | sharpness | contrast | ink_ratio | brightness | verdict |
| --- | --- | --- | --- | --- | --- |
| prescription_00000 (readable) | 0.1044 | 0.369 | 0.0105 | 0.951 | **PASS** |
| prescription_00001 (readable) | 0.1814 | 0.495 | 0.0115 | 0.951 | **PASS** |
| prescription_00002 (readable) | 0.1750 | 0.518 | 0.0135 | 0.950 | **PASS** |
| + blur r=2 (mild)             | 0.0809 | 0.350 | 0.0088 | 0.951 | REVIEW |
| + blur r=4                    | 0.0591 | 0.293 | 0.0073 | 0.951 | REVIEW |
| + blur r=6                    | 0.0532 | 0.278 | 0.0060 | 0.951 | RETAKE |
| + blur r=8 (heavy)            | 0.0488 | 0.273 | 0.0059 | 0.951 | RETAKE |
| + dark ×0.3                   | 0.0705 | 0.149 | 0.0014 | 0.285 | RETAKE |
| + dark ×0.1                   | 0.0033 | 0.050 | 0.0000 | 0.095 | RETAKE |

Real-corpus CLI result (n=3): **PASS 100%**, mean sharpness 0.154, mean
contrast 0.460, mean ink 0.0118, mean brightness 0.950.

The margin structure is monotone: readable → PASS (sharp ≥ 0.104), mild blur →
REVIEW (0.049–0.104 band), unreadable blur/dark → RETAKE (< 0.055 sharp or
< 0.10 contrast). Calibration sample is small (n=3 + degradation series), so
REVIEW acts as the safety band between the evidence-backed PASS and RETAKE
regions; thresholds are all overridable for field recalibration.

## Degradation sweep (synthetic document-like corpus, seed 0, 64/level)

`stage_b.make_document` synthesizes white-paper captures with dashed 1-px
handwriting strokes so the corpus lives in the same feature space as real
captures (ink ≈ 1–2% of pixels):

| Level | RETAKE | REVIEW | PASS | mean sharpness | mean contrast | mean ink | mean brightness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none (pristine) | 0.00 | 0.02 | **0.98** | 0.3473 | 0.768 | 0.0099 | 0.872 |
| blur r=2 | 0.00 | 0.00 | **1.00** | 0.1500 | 0.474 | 0.0156 | 0.871 |
| blur r=12 | 0.02 | **0.97** | 0.02 | 0.0866 | 0.269 | 0.0065 | 0.871 |
| exposure ×0.1 | **1.00** | 0.00 | 0.00 | 0.0020 | 0.078 | 0.0000 | 0.087 |

Readout:

- **Pristine document captures pass** (the one 0.98 REVIEW is a sparse-ink
  borderline inside the designed safety band).
- **Mild blur passes; blur r=12 → REVIEW** (borderline band), matching the real
  blur r=2–4 controls.
- **Dark captures → RETAKE** via both the brightness gate (0.087) and the
  contrast-collapse gate.

## Contract & scope

- The discrete verdict contract `RETAKE | REVIEW | PASS` is the stable interface
  that gets ported to the Android / Kotlin runtime. The signal *set* changed in
  v2 (`sharpness, contrast, ink_ratio, brightness, exposure`), so the port
  implements the same five measurements + gates; verdict semantics are unchanged.
- OCR and content verification (expired ID, expected document type, person match)
  stay **separate**, per the PoC scope agreed with the SmallAI/Mifos team.

## Reproduce

```bash
make test          # includes Stage B policy + gate tests (v2)
make data && make train && make eval   # journey-learning runs (unchanged)
PYTHONPATH=src python -c "from smallai import stage_b as sb; print(sb.run_sweep(seed=0))"
PYTHONPATH=src python -m smallai stage-b --images-dir data/prescriptions --n 10
```
