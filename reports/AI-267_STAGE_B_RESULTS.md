# AI-267 Stage B — Real image-quality signals + offline decision policy

**Repository:** reyyishreyas/smallai · **Stage:** B · **Status:** scored

Stage A scored a tabular Q-table over *synthetic* image-quality signals. Stage B
measures **real** image-quality signals directly from captured document images
(pure NumPy — no CV/ML library, no GPU, no LLM in the inference path) and applies
the discrete offline policy `RETAKE | REVIEW | PASS`.

## Offline decision gate

```
images ──► extract(images) ──► {sharpness, brightness, exposure} per image
      ──► decide(signals) ──► RETAKE | REVIEW | PASS
```

- `sharpness`  : mean gradient magnitude (blur proxy; low → blurry)
- `brightness` : mean luma
- `exposure`   : `|mean luma − 0.5|` (over/under exposure)

Gates (all real-signal-derived, overridable per call):

| Verdict | Condition |
| --- | --- |
| RETAKE | `sharpness < 0.02` **or** `exposure > 0.5` **or** `brightness < 0.12` **or** `brightness > 0.88` |
| REVIEW | not RETAKE and (`sharpness < 0.05` **or** `exposure > 0.3` **or** `brightness < 0.2` **or** `brightness > 0.8`) |
| PASS   | otherwise (usable) |

## Degradation sweep (seed 0, 64 captures per level)

Deterministic document-capture-like corpus (`stage_b.make_capture`), then a
controlled degradation sweep (`blur` box radius / `exposure` factor):

| Level | RETAKE | REVIEW | PASS | mean sharpness | mean brightness | mean exposure |
| --- | --- | --- | --- | --- | --- | --- |
| none (pristine) | 0.00 | 0.00 | **1.00** | 0.0534 | 0.447 | 0.053 |
| blur r=2 | 0.00 | 0.00 | **1.00** | 0.0514 | 0.447 | 0.053 |
| blur r=8 | 0.00 | **1.00** | 0.00 | 0.0325 | 0.441 | 0.059 |
| exposure ×0.1 | **1.00** | 0.00 | 0.00 | 0.0053 | 0.045 | 0.455 |

Readout:

- **Pristine captures pass** — the offline gate does not reject good images.
- **Mild blur still passes**, heavy blur (`r=8`) is flagged for review (the
  gradient-based sharpness proxy degrades smoothly, not all-or-nothing).
- **Dark captures are rejected** (brightness 0.045 → RETAKE) — the field-officer
  case where a photo taken in poor light would otherwise be wasted at back-office.

## Contract & scope

- The discrete verdict contract `RETAKE | REVIEW | PASS` is the stable interface
  that gets ported to the Android / Kotlin runtime.
- OCR and content verification (expired ID, expected document type, person match)
  stay **separate**, per the PoC scope agreed with the SmallAI/Mifos team.

## Reproduce

```bash
make test          # includes Stage B policy + gate tests
make data && make train && make eval   # journey-learning runs (unchanged)
PYTHONPATH=src python -c "from smallai import stage_b as sb; print(sb.run_sweep(seed=0))"
```
