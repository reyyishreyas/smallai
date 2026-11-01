# Stage A.1 report - AI-267

Pre-registered round 2 (`EXPERIMENT_A1.md`). Stage A is untouched: its report, JSON and seeds are exactly as committed, and the numbers below are a separate, independently seeded round.

## Run parameters

| Parameter | Value |
| --- | --- |
| Master seeds | 20263030, 20263031, 20263032, 20263033, 20263034 |
| Test split size | 10,000 per seed |
| Train split size | 4,000 (skewed field mixture) |
| Calibration split size | 1,000 (balanced) |
| Episode budget per TD policy | 1,000,000 |
| Bootstrap resamples | 10,000 |
| Noise sweep | 0.04, 0.08, 0.12 (primary 0.08) |
| Wall clock | 402.0 s |

## Primary condition per master seed (sigma = 0.08)

| Seed | Journey-Q acc | Shortcut acc | Delta (pp) | Journey invalid | Shortcut invalid | Journey meas | Shortcut meas |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20263030 | 98.340 % | 97.870 % | +0.470 | 2.530 % | 7.310 % | 4.648 | 6.000 |
| 20263031 | 98.120 % | 98.240 % | -0.120 | 0.700 % | 8.150 % | 4.655 | 6.000 |
| 20263032 | 98.130 % | 97.910 % | +0.220 | 1.430 % | 7.990 % | 4.482 | 6.000 |
| 20263033 | 98.030 % | 97.730 % | +0.300 | 0.740 % | 7.890 % | 4.659 | 6.000 |
| 20263034 | 98.340 % | 98.360 % | -0.020 | 0.750 % | 8.030 % | 4.517 | 6.000 |

## H1 - Journey-Q accuracy > Shortcut-CART accuracy

The pre-registered primary summary is the per-seed paired delta: the 5 master seeds are independent replicates of the whole procedure, so that interval carries the training-draw noise as well as the evaluation noise.

- Seed-level mean delta: **+0.170 pp**, 95 % t-interval over 5 seeds [-0.127, +0.467] (sd 0.240, t = 1.59)
- Pooled paired bootstrap over 50,000 paired scenarios: **+0.170 pp**, 95 % CI [+0.070, +0.268]
- Pooled McNemar: journey-only 367, shortcut-only 282, p = 0.0010

**H1: NOT SUPPORTED.** Decision rule (EXPERIMENT_A1.md section 5): the seed-level interval must exclude zero on the positive side *and* the pooled McNemar must give p < 0.05.

## S1 (secondary) - evidence consistency

The invalid-verdict rate is the metric where the journey policy competes hardest. The paired indicator is *the shortcut is invalid and Journey-Q is not* versus the reverse.

- Pooled invalid rate: Journey-Q 1.230 % vs Shortcut-CART 7.874 %
- Paired difference (shortcut minus journey, positive is the journey policy's advantage): **+6.644 pp**, 95 % CI [+6.396, +6.890]
- McNemar on the paired indicator: 3769 scenarios where the shortcut is invalid and Journey-Q is not, 447 the other way round, p = < 1e-300 (underflows to zero)

## S2 (secondary) - measurement cost

- Pooled mean measurements per decision: Journey-Q 4.592 vs Shortcut-CART 6.000
- Paired difference (journey minus shortcut): **-1.408** measurements, 95 % CI [-1.423, -1.392]

## The pre-registered candidate reward

`WIDE_CONSISTENCY_REWARD`: the verdict term unchanged, the consistency term halved, and a suspect reading no longer refutes "retake". Trained on the same training stream and scored on the same noise stream as the frozen Journey-Q, so the only difference is the reward.

| Seed | Frozen Journey-Q | Candidate | Delta (pp) |
| --- | --- | --- | --- |
| 20263030 | 98.340 % | 98.180 % | -0.160 |
| 20263031 | 98.120 % | 98.150 % | +0.030 |
| 20263032 | 98.130 % | 98.100 % | -0.030 |
| 20263033 | 98.030 % | 98.340 % | +0.310 |
| 20263034 | 98.340 % | 98.360 % | +0.020 |

- Pooled paired bootstrap over 50,000 paired scenarios: **+0.034 pp**, 95 % CI [-0.062, +0.132]
- Pooled McNemar: candidate-only 312, frozen-only 295, p = 0.5161
- Seed-level mean delta: +0.034 pp, 95 % t-interval [-0.179, +0.247]
- Pooled consistency and cost: candidate invalid 2.182 % and 4.519 measurements, against the frozen reward's 1.230 % and 4.592

**Decision: the candidate is NOT ADOPTED.** Decision rule (EXPERIMENT_A1.md section 3): the pooled paired interval must exclude zero on the positive side *and* the pooled McNemar must give p < 0.05. The shipped default therefore stays the frozen reward.

## The Stage A split size, for continuity

The first 1,000 scenarios of each test split, scored exactly as Stage A scored its 1,000. Same sample size, same machinery, fresh data.

| Seed | Journey-Q acc | Shortcut acc | Delta (pp) | McNemar p |
| --- | --- | --- | --- | --- |
| 20263030 | 98.300 % | 98.500 % | -0.200 | 0.6875 |
| 20263031 | 98.100 % | 97.500 % | +0.600 | 0.2101 |
| 20263032 | 98.000 % | 98.300 % | -0.300 | 0.5488 |
| 20263033 | 98.100 % | 98.100 % | +0.000 | 1.0000 |
| 20263034 | 98.500 % | 98.400 % | +0.100 | 1.0000 |

- Pooled over the 5 subsets: **+0.040 pp**, 95 % CI [-0.260, +0.340], McNemar p = 0.8939
- Seed-level mean delta: +0.040 pp, 95 % t-interval [-0.395, +0.475]

## The noise sweep (all seeds)

The sweep is the protocol's sensitivity analysis, not a pre-registered hypothesis, so it is reported with intervals but no verdict.

| Sigma | Journey-Q acc | Shortcut acc | Delta (pp) | 95 % CI (pooled, 50,000 scenarios) | Seed-level mean [95 % t-interval] |
| --- | --- | --- | --- | --- | --- |
| 0.04 | 98.608 % | 98.100 % | +0.508 | [+0.406, +0.608] | +0.508 [+0.373, +0.643] |
| 0.08 | 98.192 % | 98.022 % | +0.170 | [+0.070, +0.268] | +0.170 [-0.127, +0.467] |
| 0.12 | 97.594 % | 97.554 % | +0.040 | [-0.090, +0.168] | +0.040 [-0.342, +0.422] |

The cost side by noise level:

| Sigma | Journey invalid | Shortcut invalid | Journey meas | Shortcut meas | Journey review | Shortcut review |
| --- | --- | --- | --- | --- | --- | --- |
| 0.04 | 0.246 % | 2.714 % | 4.498 | 6.000 | 1.040 % | 0.564 % |
| 0.08 | 1.230 % | 7.874 % | 4.592 | 6.000 | 0.960 % | 0.232 % |
| 0.12 | 2.164 % | 14.654 % | 4.687 | 6.000 | 1.294 % | 0.000 % |

## H2, H3 and H4 - the mechanism checks

H2-H4 are mechanism checks, not confirmatory hypotheses: the point estimate is reported with its interval, and no pass/fail claim is made on them.

| Seed | H2 Journey-Q vs NoRecheck-Q | H3 Journey-Q vs ExpertOnly-Q | H4 measurements |
| --- | --- | --- | --- |
| 20263030 | +1.240 pp [+0.950, +1.530] | +0.340 pp [+0.130, +0.550] | 4.648 (holds) |
| 20263031 | +1.890 pp [+1.500, +2.280] | -0.050 pp [-0.330, +0.220] | 4.655 (holds) |
| 20263032 | +0.500 pp [+0.270, +0.740] | +0.300 pp [+0.070, +0.540] | 4.482 (holds) |
| 20263033 | +0.120 pp [-0.140, +0.390] | -0.070 pp [-0.350, +0.220] | 4.659 (holds) |
| 20263034 | +1.500 pp [+1.140, +1.870] | +0.190 pp [-0.090, +0.460] | 4.517 (holds) |

The policy field, pooled over the 5 seeds at the primary sigma:

| Policy | Accuracy | Invalid | Mean measurements | Review rate |
| --- | --- | --- | --- | --- |
| journey_q | 98.192 % | 1.230 % | 4.592 | 0.960 % |
| no_recheck_q | 97.142 % | 6.132 % | 4.487 | 2.128 % |
| expert_only_q | 98.050 % | 1.008 % | 4.634 | 1.202 % |
| shortcut_cart | 98.022 % | 7.874 % | 6.000 | 0.232 % |

## Model size, coverage and calibration (primary sigma)

| Seed | Journey-Q entries | Visited states | Coverage | ECE isotonic | ECE softmax |
| --- | --- | --- | --- | --- | --- |
| 20263030 | 3,177 | 1,399 | 8.95 % | 0.01213 | 0.00439 |
| 20263031 | 3,391 | 1,481 | 9.48 % | 0.00765 | 0.00449 |
| 20263032 | 3,065 | 1,340 | 8.58 % | 0.00994 | 0.00579 |
| 20263033 | 3,160 | 1,393 | 8.92 % | 0.01912 | 0.00861 |
| 20263034 | 3,263 | 1,451 | 9.29 % | 0.01357 | 0.00247 |

## Integrity

- Stage A's protocol, code path and artifacts are unchanged; its run is re-executed after the code changes and diffed against the committed `reports/poc.json` (all 23,947 measured leaves identical apart from wall-clock fields and the recorded numpy version).
- No hyperparameter of any policy is selected on the calibration or test splits of this round.
- The development measurements that motivated this round were produced on development seeds under `scripts/dev/`, which never runs as part of the scored pipeline.

