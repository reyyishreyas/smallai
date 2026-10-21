# Stage A POC report - AI-267

*CLEF-inspired lightweight decision-policy POC.*

Measured results for the protocol frozen in `EXPERIMENT.md`. Every number below came out of one run of `python scripts/run_poc.py` on the machine listed under Environment; nothing is estimated, and the protocol was not touched after these results existed.

**Run status: scored.** Budget 1,000,000 episodes and 10,000 bootstrap resamples, both as pre-registered.

Total wall clock 65.0 s. Master seed `20261021`; every other seed derives from it by SHA-256, so the whole run reproduces from that one number.

## Environment

- Python 3.13.9
- numpy 2.5.3
- macOS-26.5.1-arm64-arm-64bit-Mach-O on arm64

Latency is measured on this machine only. EXPERIMENT.md section 12 makes the <10 ms decision-core criterion non-binding at table-lookup scale and rules out any Android claim in Stage A; on-device measurement is Stage B.

## Splits

Scenarios are partitioned before any noise is drawn, and no true signal vector appears in two splits. Train uses the blur-heavy field mixture, calibration and test the balanced one, so a one-shot policy cannot pass by learning "blurry means retake" alone.

| split | n | distribution | used for |
|---|---|---|---|
| train | 4,000 | skewed, blur-heavy | fitting all policies |
| calibration | 1,000 | balanced | fitting the confidence map only |
| test | 1,000 | balanced | scoring only, never tuned on |

Outcome classes in the calibration split (thin per-class support would make the isotonic map provisional): pass 345, retake 632, review 23.

Section 4 asks for exactly these counts, because isotonic regression needs adequate per-class support. Here the `review` class contributes 23 of 1000, which is thin: the confidence map rests on few observations wherever a decision turns on that outcome. The counts are reported rather than smoothed away.

## Policy field

| policy | description | role |
|---|---|---|
| Expert rules | deterministic expert; defines the labels | reference row, 100% by construction, not scored |
| Shortcut-CART | from-scratch CART, max depth 6 a priori, hyperparameters by CV on train only | graded baseline: one shot, no verification |
| NoRecheck-Q | Journey-Q with re-check actions removed | ablation: value of re-verification |
| ExpertOnly-Q | Journey-Q without flawed-trajectory replay | ablation: value of journey learning itself |
| Journey-Q | the full model | the deliverable |

Each ablation differs from Journey-Q in exactly one variable, and no hyperparameter of any policy was selected on the calibration or test splits.

## Training

Table entries are from the primary condition (sigma = 0.08). The plateau test is the pre-registered one: 100-episode block means over the final 20%, OLS slope, plateau iff the 95% CI contains zero.

| policy | episodes | plateau | slope [95% CI] | stored entries | visited states | coverage |
|---|---|---|---|---|---|---|
| Journey-Q (deliverable) | 200,000 | yes | -0.000005 [-0.000059, 0.000049] | 3,097 | 1,360 | 8.704% |
| NoRecheck-Q (ablation) | 250,000 | yes | 0.000018 [-0.000023, 0.000059] | 2,904 | 1,080 | 6.912% |
| ExpertOnly-Q (ablation) | 200,000 | yes | 0.000031 [-0.000007, 0.000068] | 2,079 | 837 | 5.357% |
| Shortcut-CART (baseline) | – | – | – | – | – | – |

Every TD policy declared plateau before its budget ran out, so none stopped on the episode cap.

Shortcut-CART is not TD-trained, which is why its row is empty above. Cross-validation on the train split alone chose min_samples_leaf=20, with 5-fold accuracies 0.980, 0.984, 0.983, 0.981, 0.990. Max depth stays 6 a priori.

Coverage is visited states over the 5^6 = 15,625 ceiling. EXPERIMENT.md section 3 asks for coverage over the states reachable under the sampling distribution, which has no closed form, so this figure is a documented lower bound on that quantity.

## Primary result

Journey-Q accuracy minus Shortcut-CART accuracy on the test split: **+0.300 percentage points**, with a paired bootstrap 95% CI of [-0.400, 1.000] over 10,000 resamples of test scenarios. Both policies are scored on the same 1,000 scenarios, so the comparison is paired.

McNemar's test on the 13 discordant pairs: chi-square 0.692, exact two-sided p = 0.5811.

The interval includes zero, so the difference is **inconclusive** whatever its sign.

EXPERIMENT.md section 2 requires that, with no mentor-supplied X, the measured delta and its interval are stated without a pass/fail claim. No X has been supplied, so none is made here.

## Hypotheses

H1 is the sole confirmatory hypothesis. H2-H4 are mechanism checks, reported with the same interval discipline but not gated on significance.

| hypothesis | claim | delta (pp) | 95% CI | p | reads as |
|---|---|---|---|---|---|
| H1 | Journey-Q accuracy > Shortcut-CART accuracy | 0.300 | [-0.400, 1.000] | 0.5811 | CI includes zero: inconclusive |
| H2 | Journey-Q accuracy >= NoRecheck-Q accuracy | 0.800 | [-0.200, 1.800] | 0.1516 | mechanism check, not gated on significance; CI printed for discipline |
| H3 | Journey-Q accuracy >= ExpertOnly-Q accuracy | 0.100 | [-0.700, 1.000] | 1.0000 | mechanism check, not gated on significance; CI printed for discipline |
| H4 | Journey-Q mean measurements/decision < 6 | – | – | – | measured mean 4.532; the tree reads all six signals by construction |

## Secondary metrics

All on the test split at the primary sigma. The expert row is not scored: its verdicts *are* the labels, so its accuracy is 100% by construction and its other columns do not exist.

| policy | accuracy % | delta vs Journey (pp) | mean measurements | review rate % | invalid rate % | mean latency (ms) |
|---|---|---|---|---|---|---|
| Expert rules (defines the labels) | 100.000 | – | – | – | – | – |
| Journey-Q (deliverable) | 98.000 | +0.000 | 4.532 | 1.300 | 0.400 | 0.0235 |
| NoRecheck-Q (ablation) | 97.200 | -0.800 | 4.519 | 1.600 | 7.700 | 0.0213 |
| ExpertOnly-Q (ablation) | 97.900 | -0.100 | 4.805 | 1.000 | 2.100 | 0.0237 |
| Shortcut-CART (baseline) | 97.700 | -0.300 | 6.000 | 0.000 | 8.700 | 0.0012 |

Invalid-verdict rate is the fraction of verdicts their own gathered evidence contradicts, under the section 3 rule: `pass` needs all six signals observed at pass level, `retake` an observed fail, `review` an observed suspect, and any terminal taken with no measurements is contradictory.

## Confidence calibration

The terminal margin is a score in reward units, not a probability. Section 9 fits an isotonic regression from margin to P(correct) on the calibration split and cross-checks it against temperature-scaled softmax fitted on the same split.

- isotonic ECE on test: **0.0110**
- temperature-scaled softmax ECE on test: **0.0052** (fitted tau = 0.305)
- mean absolute probability difference between the two: 0.0199

The two forms agree within the section 9 tolerances, so the isotonic map ships without a provisional flag.

The number shown to a user is the calibrated P(correct). The expert's hand-written confidence heuristic is internal only and never ships.

## Sensitivity to measurement noise

Sigma swept over [0.04, 0.08, 0.12]; sigma = 0.08 is the primary condition. The splits are identical across the sweep, so the only thing that changes is the noise.

| sigma | Journey-Q % | Shortcut-CART % | delta (pp) | 95% CI | mean measurements |
|---|---|---|---|---|---|
| 0.04 | 98.400 | 97.700 | +0.700 | [-0.100, 1.500] | 4.499 |
| 0.08 (primary) | 98.000 | 97.700 | +0.300 | [-0.400, 1.000] | 4.532 |
| 0.12 | 97.500 | 97.300 | +0.200 | [-0.700, 1.100] | 4.588 |

## Convergence artifacts

Section 7 requires the training curve and the state-visit histogram to ship so convergence is auditable. Shown here for Journey-Q at sigma = 0.08; the complete data is in `poc.json`.

### Training curve

| episode | mean block reward |
|---|---|
| 100 | 0.7075 |
| 16,700 | 0.7015 |
| 33,300 | 0.6905 |
| 49,900 | 0.6390 |
| 66,500 | 0.6905 |
| 83,100 | 0.6525 |
| 99,700 | 0.6560 |
| 116,300 | 0.6135 |
| 132,900 | 0.6810 |
| 149,500 | 0.6755 |
| 166,100 | 0.7370 |
| 182,700 | 0.6605 |
| 199,300 | 0.6575 |
| 200,000 | 0.7240 |

Final block mean: 0.6508.

### State-visit histogram

| visits | states |
|---|---|
| 1 | 348 |
| 2 | 138 |
| 3 | 66 |
| 4 | 41 |
| 5 | 22 |
| 6 | 19 |
| 7 | 17 |
| 8 | 21 |
| 9 | 14 |
| 10 | 25 |
| 11 | 14 |
| 12 | 14 |
| 13 | 23 |
| 14 | 22 |
| 15 | 15 |
| 16 | 19 |
| 17 | 6 |
| 18 | 10 |
| 19 | 10 |
| 20 | 11 |
| … | 266 further visit-count rows are in poc.json |

Journey-Q stored 3,097 values across 1,360 distinct states (8.704% of the 15,625 ceiling).

## Limitations

- **Synthetic Stage 0.** The six signals are sampled, not computed from images. This POC validates the decision logic, the learning mechanisms and calibration. It establishes nothing about real ID photographs. The Stage B spike, roughly 50 real captures against these distributions, gates the Android work.
- **A policy table, not a parametric model.** Reported as a stored-entry count. The ticket's 100K budget holds under either reading.
- **The `review` action is an open scope question with the mentor.** The binary fallback is pre-declared in EXPERIMENT.md section 11 and was not used here.
- **No pass/fail against X.** X has not been supplied, so the primary delta is reported with its interval and no threshold claim.

---

Produced by `python scripts/run_poc.py` from the frozen protocol in `EXPERIMENT.md`. Design constants live in `ARCHITECTURE.md`; measured results live here only.
