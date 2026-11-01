# AI-267 Stage A.1 — Complete Measured Results

**CLEF-inspired lightweight decision-policy POC**
Repository: `reyyishreyas/smallai` · Stage: **A.1** (round 2) · Status: **scored**

Stage A.1 is the second, separately pre-registered round. Its pre-registration
(`EXPERIMENT_A1.md`) was written and committed **before** any training run, and it fixes what
the round measures, which seeds it uses and the rule that decides each claim.

**Stage A is untouched.** `reports/poc.md`, `reports/poc.json`, the Stage A seeds and the
Stage A results document (`AI-267_STAGE_A_RESULTS.md`) are exactly as they were; the numbers
below come from a different, independently seeded round.

Raw machine-readable source for every figure in this document: `reports/poc_a1.json`.
Narrative form of the same run: `reports/poc_a1.md`. Design constants: `ARCHITECTURE.md`.

---

## 0. Headline

**The graded accuracy claim is not established — and now we know that precisely instead of
vaguely.** Over five fresh master seeds and 50,000 paired test scenarios:

| summary | delta (Journey-Q − Shortcut-CART) | 95 % interval | excludes 0? |
|---|---|---|---|
| **Seed-level (pre-registered primary)** — 5 independent replicates of the whole procedure | **+0.170 pp** | **[−0.127, +0.467]** | no |
| Pooled scenario-level (evaluation noise only) — 50,000 paired scenarios | +0.170 pp | [+0.070, +0.268] | yes |

**H1: NOT SUPPORTED.** The pre-registered rule required *both* the seed-level interval to
exclude zero on the positive side *and* the pooled McNemar test to give p < 0.05. The
McNemar test passes (journey-only 367, shortcut-only 282, p = 0.0010); the seed-level
interval does not, so the pre-registered claim fails.

The two summaries disagree on purpose. The scenario-level bootstrap measures how well the
journey policy beats the tree **on these 50,000 captures**. The seed-level interval also
carries the *training draw* — how much the answer moves when the whole policy is retrained
from scratch (sd 0.240 pp across the five replicates). Since the ticket asks whether the
journey-learning *design* is better, the replicate-level question is the one that counts:
across five independent retrainings the advantage is not reliably positive.

**But the journey policy's real advantage is not on accuracy, and it is not in doubt:**

| result | Journey-Q | Shortcut-CART | effect |
|---|---|---|---|
| **Invalid-verdict rate** (the verdict contradicts its own gathered evidence) | **1.230 %** | **7.874 %** | **6.4× lower**, −6.644 pp [−6.890, −6.396], McNemar p < 1e-300 |
| **Mean measurements per decision** | **4.592** | 6.000 | **1.408 fewer** [1.392, 1.423] — 23.5 % fewer signal acquisitions |
| Accuracy (primary metric, σ = 0.08) | 98.192 % | 98.022 % | +0.170 pp, seed-level CI includes 0 |

**And the accuracy gain is real where the capture is clean.** The protocol's noise sweep
(now reported with intervals, which Stage A did not do) shows the journey learning's
accuracy advantage shrinking as measurement noise grows:

| σ | delta | pooled 95 % CI | seed-level 95 % t-interval |
|---|---|---|---|
| 0.04 | **+0.508 pp** | [+0.406, +0.608] | [+0.373, +0.643] — **both exclude 0** |
| 0.08 (primary) | +0.170 pp | [+0.070, +0.268] | [−0.127, +0.467] |
| 0.12 | +0.040 pp | [−0.090, +0.168] | [−0.342, +0.422] |

At σ = 0.04 the journey learning's accuracy gain is established at both levels (t = 10.5
over five seeds). At the protocol's primary σ = 0.08 it is marginal. At σ = 0.12 it is
gone. The ordering is monotone and the noise levels are pre-specified, so this is a
sensitivity result, not a post-hoc search.

**No pass/fail claim is made against the ticket's threshold.** The "X" in "journey learning
≥ X % over shortcut" (`ARCHITECTURE.md` §1) has still never been supplied, and per
`EXPERIMENT.md` §2 any X supplied now is post-hoc relative to the pre-registration.

---

## 1. Run facts

| field | value |
|---|---|
| Master seeds | `20263030`, `20263031`, `20263032`, `20263033`, `20263034` (5, all fresh) |
| Test split | **10,000 scenarios per seed** (50,000 paired scenarios pooled) |
| Train split | 4,000 (skewed field mixture) per seed |
| Calibration split | 1,000 (balanced) per seed |
| Episode budget | 1,000,000 per TD policy (pre-registered: 1,000,000) ✔ |
| Bootstrap resamples | 10,000 per delta (pre-registered: 10,000) ✔ |
| Noise sweep | σ ∈ {0.04, 0.08, 0.12}, primary 0.08 |
| Wall clock | **402.0 s** |
| Stage | `A.1` |

Every sub-seed (policy training seeds, per-scenario noise streams, bootstrap resamples)
derives from the master seed via SHA-256, so the whole round reproduces from those five
numbers.

**Why these seeds are fresh.** They are disjoint from the Stage A seed (20261021), from the
development *selection* seeds (20261030+) and from the development *confirmation* seeds
(20262030+). The last group exists because one development candidate looked like a win on
the selection seeds and turned out not to be (§10).

## 2. What changed relative to Stage A, and what did not

| item | Stage A | Stage A.1 |
|---|---|---|
| Test split size | 1,000 | **10,000** |
| Master seeds | 1 | **5 fresh** |
| Reward | frozen | frozen (unchanged — see §10 for why the candidate was not adopted) |
| Train/calibration sizes and mixtures | as specified | unchanged |
| Policy field (Shortcut-CART, Journey-Q, NoRecheck-Q, ExpertOnly-Q) | as specified | unchanged |
| Learning constants, episode budget, plateau test | §7 | unchanged |
| Calibration (isotonic + temperature, fitted on calibration only) | §9 | unchanged |
| Metrics, paired bootstrap, McNemar | §8 | unchanged, **plus** two pre-registered secondary analyses |
| Reporting | `reports/poc.{md,json}` | `reports/poc_a1.{md,json}` |

The test split was enlarged for a stated reason: on 1,000 paired scenarios at ~98 %
accuracy the one-sigma noise on a paired accuracy delta is about 0.36 pp, so a 1,000-scenario
split cannot resolve a sub-percentage-point effect in either direction. §9 shows what that
split size still cannot do, with the same data.

---

## 3. The primary condition, per master seed (σ = 0.08)

| Seed | Journey-Q acc | Shortcut acc | Delta | Journey invalid | Shortcut invalid | Journey meas | Shortcut meas |
|---|---|---|---|---|---|---|---|
| 20263030 | 98.340 % | 97.870 % | **+0.470** | 2.530 % | 7.310 % | 4.648 | 6.000 |
| 20263031 | 98.120 % | 98.240 % | **−0.120** | 0.700 % | 8.150 % | 4.655 | 6.000 |
| 20263032 | 98.130 % | 97.910 % | **+0.220** | 1.430 % | 7.990 % | 4.482 | 6.000 |
| 20263033 | 98.030 % | 97.730 % | **+0.300** | 0.740 % | 7.890 % | 4.659 | 6.000 |
| 20263034 | 98.340 % | 98.360 % | **−0.020** | 0.750 % | 8.030 % | 4.517 | 6.000 |

Two of the five replicates are negative. That is the whole story of the seed-level
interval: the per-replicate delta has sd 0.240 pp, so five replicates give a standard error
of 0.107 pp — an order of magnitude larger than the scenario-level evaluation noise.

## 4. H1 — Journey-Q accuracy > Shortcut-CART accuracy

**Pre-registered primary summary** (`EXPERIMENT_A1.md` §5): the per-seed paired delta,
summarised as the mean over the five master seeds with a paired t-interval over the seeds,
because the five seeds are independent replicates of the whole procedure and that interval
therefore carries the training-draw noise as well as the evaluation noise.

| quantity | value |
|---|---|
| Seed-level mean delta | **+0.170 pp** |
| Seed-level sd | 0.240 pp |
| Seed-level standard error | 0.107 pp |
| Seed-level 95 % t-interval (df = 4) | **[−0.127, +0.467]** |
| t-statistic | 1.59 |
| Pooled scenario-level delta (50,000 pairs) | +0.170 pp |
| Pooled paired-bootstrap 95 % CI | [+0.070, +0.268] |
| Pooled McNemar | journey-only 367, shortcut-only 282, p = 0.0010 |

**Verdict: H1 is NOT SUPPORTED.** The pre-registered rule requires the seed-level interval
to exclude zero on the positive side *and* the pooled McNemar p < 0.05. The second
condition holds; the first does not.

**Reading the two numbers together.** Pooled over 50,000 paired captures the journey policy
is ahead by 0.170 pp and the paired bootstrap says so. But those 50,000 captures were
scored with **five** policies, one per master seed, and the policies differ from each other
by more than the effect being measured. Re-running the procedure with a fresh training draw
moves the answer by ±0.24 pp. So the honest statement is:

> On a fixed set of captures the journey policy is a little more accurate than the
> one-shot tree; across independent retrainings of the whole policy, the evidence does not
> establish that it *reliably* is.

Both numbers are reported, and the pre-registration named in advance which one decides the
claim.

## 5. S1 (secondary, pre-registered) — evidence consistency

The invalid-verdict rate is the fraction of decisions whose verdict contradicts the
evidence the policy gathered for it (§3): a "pass" after an observed fail, a "retake" with
no observed fail, a "review" with nothing suspect in sight. It is the failure mode that
costs money on a capture gate — an unusable ID that gets uploaded anyway.

| quantity | Journey-Q | Shortcut-CART |
|---|---|---|
| Pooled invalid-verdict rate | **1.230 %** | **7.874 %** |
| Paired difference (shortcut − journey) | **+6.644 pp** | |
| 95 % paired-bootstrap CI | **[+6.396, +6.890]** | |
| Discordant scenarios | 3,769 shortcut-invalid-only | 447 journey-invalid-only |
| McNemar | p < 1e-300 (underflows) | |

**This is decisive.** The journey policy produces an evidence-contradicting verdict 6.4×
less often, on 50,000 paired scenarios, with a confidence interval nowhere near zero.

## 6. S2 (secondary, pre-registered) — measurement cost

| quantity | Journey-Q | Shortcut-CART |
|---|---|---|
| Mean measurements per decision | **4.592** | 6.000 |
| Paired difference (journey − shortcut) | **−1.408** | |
| 95 % paired-bootstrap CI | **[−1.423, −1.392]** | |

The tree reads all six signals by construction. The policy early-exits: it stops at the
first observed fail, and only re-measures when the evidence is thin. That is 23.5 % fewer
signal acquisitions per decision, which on device is the difference between one CV pass
and a second re-measurement.

## 7. The noise sweep

The sweep is the protocol's sensitivity analysis (§8), not a pre-registered hypothesis, so
it is reported with intervals and no verdict. Stage A reported the sweep as point
estimates; this round adds the intervals, because the sweep's shape is now the most
informative thing in the round.

### Accuracy

| σ | Journey-Q acc | Shortcut acc | Delta | 95 % CI (pooled, 50,000) | Seed-level mean [95 % t-interval] |
|---|---|---|---|---|---|
| 0.04 | 98.608 % | 98.100 % | **+0.508** | [+0.406, +0.608] | +0.508 [+0.373, +0.643] |
| 0.08 | 98.192 % | 98.022 % | +0.170 | [+0.070, +0.268] | +0.170 [−0.127, +0.467] |
| 0.12 | 97.594 % | 97.554 % | +0.040 | [−0.090, +0.168] | +0.040 [−0.342, +0.422] |

At σ = 0.04 the journey learning's accuracy gain is established at both the scenario level
and the replicate level (t = 10.5, McNemar p < 1e-300 on 459 vs 205 discordant
scenarios). It weakens as the sensor noise grows and is gone at σ = 0.12.

### Cost

| σ | Journey invalid | Shortcut invalid | Journey meas | Shortcut meas | Journey review | Shortcut review |
|---|---|---|---|---|---|---|
| 0.04 | 0.246 % | 2.714 % | 4.498 | 6.000 | 1.040 % | 0.564 % |
| 0.08 | 1.230 % | 7.874 % | 4.592 | 6.000 | 0.960 % | 0.232 % |
| 0.12 | 2.164 % | 14.654 % | 4.687 | 6.000 | 1.294 % | 0.000 % |

The cost side does not degrade with noise: the shortcut's invalid-verdict rate climbs from
2.7 % to 14.7 % as noise rises, while the journey policy stays between 0.2 % and 2.2 % and
keeps its measurement count at ~4.6 either way. **Where the noise is worst is exactly where
the journey design is worth most.**

## 8. H2, H3, H4 and the policy field

H2–H4 are mechanism checks, not confirmatory hypotheses: the point estimate is reported
with its interval and no pass/fail claim is made on them.

| Seed | H2 Journey-Q vs NoRecheck-Q | H3 Journey-Q vs ExpertOnly-Q | H4 measurements |
|---|---|---|---|
| 20263030 | +1.240 pp [+0.950, +1.530] | +0.340 pp [+0.130, +0.550] | 4.648 (holds) |
| 20263031 | +1.890 pp [+1.500, +2.280] | −0.050 pp [−0.330, +0.220] | 4.655 (holds) |
| 20263032 | +0.500 pp [+0.270, +0.740] | +0.300 pp [+0.070, +0.540] | 4.482 (holds) |
| 20263033 | +0.120 pp [−0.140, +0.390] | −0.070 pp [−0.350, +0.220] | 4.659 (holds) |
| 20263034 | +1.500 pp [+1.140, +1.870] | +0.190 pp [−0.090, +0.460] | 4.517 (holds) |

The policy field, pooled over the five seeds at σ = 0.08:

| Policy | Accuracy | Invalid | Mean measurements | Review rate |
|---|---|---|---|---|
| Journey-Q (deliverable) | 98.192 % | 1.230 % | 4.592 | 0.960 % |
| NoRecheck-Q (ablation) | 97.142 % | 6.132 % | 4.487 | 2.128 % |
| ExpertOnly-Q (ablation) | 98.050 % | 1.008 % | 4.634 | 1.202 % |
| Shortcut-CART (baseline) | 98.022 % | 7.874 % | 6.000 | 0.232 % |

Both halves of the design show up. Removing re-verification (NoRecheck-Q) costs 1.05 pp of
accuracy and multiplies the invalid-verdict rate by five; the journey policy's review rate
is four times the tree's, which is the price of the "check again before you reject"
behaviour. H4 holds in all five replicates (4.48–4.66 measurements, against the tree's 6).

## 9. The Stage A split size, for continuity

The first 1,000 scenarios of each seed's test split, scored exactly as Stage A scored its
1,000 — same sample size, same machinery, fresh data.

| Seed | Journey-Q acc | Shortcut acc | Delta | McNemar p |
|---|---|---|---|---|
| 20263030 | 98.300 % | 98.500 % | −0.200 | 0.6875 |
| 20263031 | 98.100 % | 97.500 % | +0.600 | 0.2101 |
| 20263032 | 98.000 % | 98.300 % | −0.300 | 0.5488 |
| 20263033 | 98.100 % | 98.100 % | +0.000 | 1.0000 |
| 20263034 | 98.500 % | 98.400 % | +0.100 | 1.0000 |

Pooled over the five 1,000-scenario subsets: **+0.040 pp**, 95 % CI **[−0.260, +0.340]**,
McNemar p = 0.8939. Seed-level mean +0.040 pp, t-interval [−0.395, +0.475].

**This is the continuity check, and it is the reason the round was needed.** At Stage A's
sample size the same procedure produces a delta of +0.040 pp with an interval three times
wider than the effect and a p-value of 0.89 — a coin flip, on five times less data, on
the same policies. Stage A's "+0.300 pp, CI [−0.400, 1.000], p = 0.5811" was one such draw.
Stage A.1's contribution is not a better number; it is a number with a known resolution.

## 10. The pre-registered candidate reward — tested and rejected

`WIDE_CONSISTENCY_REWARD` (`smallai/train.py`): the verdict term unchanged, the consistency
term halved, and a *suspect* reading no longer counts as refuting "retake" (a suspect
reading is the ambiguous case in which the latent value may sit below the fail threshold).

The motivation was specific: the policy's dominant residual error is retake-vs-review on
suspect evidence, and the frozen consistency term makes "retake" contradictory in exactly
that state, so the policy plays safe and gives up accuracy it already has evidence for.

The candidate was trained on the **same training stream** and scored on the **same noise
stream** as the frozen Journey-Q, so the reward is the only difference.

| Seed | Frozen Journey-Q | Candidate | Delta |
|---|---|---|---|
| 20263030 | 98.340 % | 98.180 % | −0.160 |
| 20263031 | 98.120 % | 98.150 % | +0.030 |
| 20263032 | 98.130 % | 98.100 % | −0.030 |
| 20263033 | 98.030 % | 98.340 % | +0.310 |
| 20263034 | 98.340 % | 98.360 % | +0.020 |

| quantity | value |
|---|---|
| Pooled delta (50,000 pairs) | **+0.034 pp** |
| Pooled 95 % CI | **[−0.062, +0.132]** |
| Pooled McNemar | candidate-only 312, frozen-only 295, p = 0.5161 |
| Seed-level mean | +0.034 pp, t-interval [−0.179, +0.247] |
| Candidate invalid-verdict rate | 2.182 % (frozen: 1.230 %) |
| Candidate mean measurements | 4.519 (frozen: 4.592) |

**Decision: NOT ADOPTED.** The pre-registered rule required the pooled interval to exclude
zero on the positive side *and* the pooled McNemar p < 0.05; neither holds. The shipped
default therefore stays the frozen Stage A reward, and the candidate's one clear effect is
to nearly double the invalid-verdict rate — the exact opposite of what it was designed to
fix.

**The winner's curse this round exists to catch.** The same candidate looked like a win on
16 development *selection* seeds (+0.375 ± 0.137 pp, t = 2.74) and did not reproduce on 16
*fresh* seeds (−0.100 ± 0.134 pp, t = −0.75). This round re-tested it on 50,000 paired
scenarios across five fresh master seeds and found nothing. A candidate that survives only
the seeds it was chosen on is not a candidate.

## 11. Everything we tried, and what each lever did

Every accuracy lever was tried on paired development runs and none survived. The table is
the evidence behind "the model is at its ceiling", not a summary of good intentions.

| lever | result | verdict |
|---|---|---|
| Richer state encoding (signal, threshold margin, order) | ceiling ≤ +0.18 pp | refuted twice |
| Check-order lever | ceiling +0.13 pp | refuted |
| Full-scan path (read everything, then decide) | worse | refuted |
| Learning knobs (lr, γ, ε, mix, budget, no-recheck) | −0.27 to −1.27 pp | refuted |
| Reward-shape sweep (one-sided, symmetric, broad) | null | refuted |
| Reward "broad consistency × 0.5" | +0.375 on selection seeds → −0.100 on fresh seeds | refuted (winner's curse) |
| Training longer (400 k / 1 M / 3 M episodes) | +0.2 pp, n.s. | refuted |
| Q-table ensembling | −0.19 / −0.20 / −0.08 pp | refuted |
| Posterior-decoded terminal action | collapses to 88.6 % | refuted |
| More training data (16 k vs 4 k) | +0.175 pp, n.s. | refuted |
| Balanced training prior (match the eval mixture) | −0.225 pp, n.s. | refuted |
| High-power measurement of the *existing* design (20,000 fresh scenarios) | journey 97.990 % vs tree 97.955 %, Δ +0.035 pp, CI [−0.140, +0.210], p = 0.7409 | the effect is ~0 |

Two of these deserve a note.

**The ceiling argument.** The policy is within ~0.2–0.4 pp of the analytic Bayes ceiling on
its own path (98.177 % under the zone-3 posterior, 98.323 % under the exact reading). Its
terminal choice matches the reward-optimal action on 997 of 1,000 dev scenarios, so the
remaining error is not a decision-rule bug — it is the irreducible ambiguity of a suspect
reading.

**The 20,000-scenario probe** is the single most important line in the table: it measured
the *shipped* design against the *shipped* baseline at high power before this round was
designed, and found the accuracy gain to be +0.035 pp. Stage A.1 confirms it: on 50,000
paired scenarios the delta is +0.170 pp, and across five retrainings it is not reliably
positive.

## 12. Model size, coverage and calibration (primary σ)

| Seed | Journey-Q entries | Visited states | Coverage (of 5⁶) | ECE isotonic | ECE temperature |
|---|---|---|---|---|---|
| 20263030 | 3,177 | 1,399 | 8.95 % | 0.01213 | 0.00439 |
| 20263031 | 3,391 | 1,481 | 9.48 % | 0.00765 | 0.00449 |
| 20263032 | 3,065 | 1,340 | 8.58 % | 0.00994 | 0.00579 |
| 20263033 | 3,160 | 1,393 | 8.92 % | 0.01912 | 0.00861 |
| 20263034 | 3,263 | 1,451 | 9.29 % | 0.01357 | 0.00247 |

The Q-table stays at ~3,100 entries against the 100 K budget, the calibration split stays
at 1,000 scenarios, and both calibration maps (isotonic and temperature) agree without
needing the provisional flag. Coverage is reported against the 5⁶ ceiling, which is a
documented *lower* bound (§7 of the protocol).

## 13. What this means for the ticket

1. **On the accuracy criterion, the answer is a measured no.** Journey learning does not
   reliably beat the one-shot tree on accuracy at σ = 0.08: +0.170 pp, seed-level 95 %
   interval [−0.127, +0.467]. If "X" is a percentage of accuracy, this design does not meet
   it at the protocol's primary noise level.
2. **At low noise it does.** At σ = 0.04 the gain is +0.508 pp with both intervals clear of
   zero. So the answer to "does journey learning help?" is *it depends on how noisy the
   capture is* — and on the field-officer use case (a phone camera capture in good light)
   the relevant σ may well be nearer 0.04 than 0.08.
3. **On the criteria that map to money, the answer is a decisive yes.** A 6.4× lower
   invalid-verdict rate (1.230 % vs 7.874 %, p < 1e-300) and 23.5 % fewer measurements
   (4.592 vs 6.000) are not marginal: they are the difference between "the app uploads an
   unusable ID anyway" and "the app catches it", and between one CV pass and two.
4. **The ticket's phrasing should change.** "Journey learning ≥ X% over shortcut" measured
   as accuracy cannot be met by this design at σ = 0.08. "Journey learning cuts
   evidence-contradicting verdicts by 6.4× and measurements by 23.5 % at equal accuracy" is
   a claim this design meets decisively and that maps onto the actual field problem.
5. **The model did not get better, and that is the finding.** Eleven levers, two
   independent confirmation rounds and a 20,000-scenario probe all agree: the shipped policy
   is at its Bayes ceiling on its own path. The remaining headroom is in the *evidence* —
   better signal extraction on device — not in the decision rule.

## 14. Integrity

- **Stage A is untouched.** Its protocol, code path, seeds and artifacts are unchanged. Its
  run was re-executed after this round's code changes and its output diffed leaf-by-leaf
  against the committed `reports/poc.json`: **23,947 measured leaves, identical apart from
  wall-clock fields (latency, elapsed) and the recorded numpy version.** The committed
  report was never overwritten — the re-run happens in a copy of the tree.
- **The pre-registration came first.** `EXPERIMENT_A1.md` was written and committed before
  any Stage A.1 training run, and it fixes the split size, the five seeds, the hypotheses,
  the secondary analyses, the decision rules and the stopping rule.
- **No hyperparameter of any policy is selected on the calibration or test splits** of this
  round. The candidate reward was scored against a pre-registered rule, not chosen by its
  result.
- **The scored pipeline was not modified while tuning.** All tuning experiments live under
  `scripts/dev/`, which never runs as part of the scored pipeline.
- **A bug was found and fixed, in the open.** `mcnemar` overflowed a float when a paired
  comparison had thousands of discordant pairs — which only happens at the enlarged split
  size. It is now computed in exact integer arithmetic, with a regression test, and the fix
  is bit-for-bit identical on the small counts Stage A produced (the Stage A reproduction
  above is the proof).
- **Tests:** 120 pass (110 pre-existing, 9 new for the reward spec, the split sizes and the
  new bootstrap helper; 1 added for the McNemar overflow).
- **The shipped model is unchanged.** The shipped artifact remains Stage A's Journey-Q
  table. Stage A.1's five replicates are written out for inspection only
  (`reports/journey_q_a1.json`); picking a replicate by its test accuracy would be
  selection on the test split.

## 15. Reproduction

```bash
# the scored round (five fresh master seeds, 10,000-scenario test splits)
python scripts/run_poc_a1.py

# Stage A, unchanged, in a copy of the tree, and the leaf-by-leaf diff
rsync -a --exclude .git --exclude .venv ./ /tmp/stage_a_repro/
(cd /tmp/stage_a_repro && python scripts/run_poc.py --out /tmp/stage_a_repro/reports)
python scripts/dev/compare_poc.py reports/poc.json /tmp/stage_a_repro/reports/poc.json
```

`reports/poc_a1.json` carries every number in this document, the per-seed condition
payloads (training reports, scores, calibration, the protocol's own H1–H4 results) and the
pooled aggregates.

---

*Stage A remains separately reported in `AI-267_STAGE_A_RESULTS.md`; the two rounds share
the policy but not the seeds, the split size or the analysis.*
