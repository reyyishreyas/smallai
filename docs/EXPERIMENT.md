# EXPERIMENT.md — pre-registered evaluation protocol (AI-267)

Framing: this is a **CLEF-inspired lightweight decision-policy
POC**. It does not reproduce Cloudflare Clef's architecture; it
applies the Clef ideas that matter offline here — typed actions,
evidence paths, calibrated confidence — to a tabular policy.

Status: PRE-REGISTRATION v2 (supersedes v1; no training run has
occurred). Committed to the repository **before** any training run
of the scored policies. Numbers ship as measured; the protocol
below does not change after results exist.

## 1. Scope: two stages

- **Stage A (this ticket):** decision-model POC on controlled,
  reproducible signal inputs. Validates the decision logic, the
  learning mechanisms, and calibration. It does NOT establish
  performance on real ID photographs.
- **Stage B (Android follow-up ticket):** signals extracted from
  real document images (CV metrics) fed into the same policy;
  measured on a representative image set and on-device. A half-day
  spike (~50 real captures; empirical signal distributions vs the
  synthetic ones) gates Stage B and retunes the synthetic
  distributions if they disagree.

## 2. Independence from X

The ticket references a threshold "X%" that has not been supplied.
This pre-registration is independent of X: X is supplied by the
project mentor as a pass/fail threshold applied to the measured
delta. If no X is supplied, the report states the measured delta
and effect size without a pass/fail claim. An X supplied after
results exist cannot be presented as pre-registered.

## 3. Environment and state semantics (exact)

Six signals in [0, 1]; thresholds per
`smallai/environment.py` (fail below FAIL, pass at/above PASS,
suspect between).

**Re-verification semantics.** A check returns a *fresh,
independent* noisy measurement of the same latent true signal
value (Gaussian, sigma = 0.08, clipped to [0, 1]) — a sensor
re-measurement. It is neither the cached value nor a new feature.
Noise is what creates the verification problem at all: with
noiseless signals a single reading reveals the truth and
sequential checking is worthless.

**State.** Per signal: `0` unchecked, `1` fail_final,
`2` suspect_pre, `3` suspect_post, `4` pass_final. A first
reading inside the suspect band becomes suspect_pre; any other
becomes fail_final / pass_final. A re-check from suspect_pre
yields a fresh reading -> fail_final / suspect_post / pass_final.

**Actions.** check(each unchecked) + re-check(suspect_pre only) +
terminal {pass, retake, review}. Re-check is illegal from every
non-suspect_pre state, so re-check loops are unrepresentable
(structurally, not by training).

**Terminal conditions (evidence consistency).**

- `pass`: consistent iff all six observed pass-level
- `retake`: consistent iff >= 1 observed fail
- `review`: consistent iff >= 1 observed suspect

Every terminal action is consistent or contradictory; a terminal
action taken with zero observations is contradictory.

**Bounds.** Maximum measurement actions per episode: 12 (6
initial + at most 6 re-checks). Maximum state count: 5^6 =
15,625; Q-values are stored sparsely; `poc.md` reports the
stored-entry count and visited-state coverage (visited states /
states reachable under the sampling distribution).

## 4. Splits (leakage-safe)

Splits partition the **underlying scenarios** (true signal
vectors) *before* any noise is drawn: noise is applied at
observation time within an episode, and no true scenario appears
in more than one split. All seeded (master seed fixed in
`scripts/run_poc.py`, reported in `poc.md`).

| split | n | distribution | used for |
|---|---|---|---|
| train | 4,000 | skewed (blur-heavy, field-realistic) | fitting all policies |
| calibration | 1,000 | balanced across defect causes | fitting the confidence map only |
| test | 1,000 | balanced across defect causes | scoring only, never tuned on |

`poc.md` reports outcome-class counts in the calibration split
(isotonic regression needs adequate per-class support; thin
support flags the map provisional).

## 5. Policy field (frozen)

| policy | description | role |
|---|---|---|
| Expert rules | deterministic expert; defines the labels | reference row (accuracy 100% by construction; not scored) |
| Shortcut-CART | from-scratch CART, max depth 6 a priori (one axis-aligned split per signal, so a path inspects all six); hyperparameters by cross-validation on the train split only | graded baseline: one-shot, no verification |
| NoRecheck-Q | Journey-Q with re-check actions removed | ablation: value of re-verification |
| ExpertOnly-Q | Journey-Q trained without flawed-trajectory replay (expert rollouts + on-policy epsilon deviations only; identical reward, budget, architecture) | ablation: value of journey learning itself |
| Journey-Q | the full model | the deliverable |

Each ablation differs from Journey-Q in exactly one variable. No
hyperparameter of any policy is selected on the calibration or
test splits.

## 6. Reward (frozen; part of the protocol)

Per episode:

- `+1.0` correct verdict; `-0.5` adjacent-wrong (review vs
  pass/retake); `-1.0` opposite-wrong (pass vs retake)
- `-0.05` per measurement action (checks and re-checks)
- `+0.10` evidence-consistent terminal; `-0.30` contradictory

**Bounds, derived:** max = `+1.0 + 0.10 - 0.05*1 = +1.05`
(correct retake/review after a single check); min =
`-1.0 - 0.30 - 0.05*12 = -1.90` (opposite verdict,
contradictory, all twelve measurements).

**No dominance claim.** Extra measurements cost `-0.05` each but
carry information value (a later check can prevent a wrong
verdict), so the accuracy-efficiency trade-off is empirical —
reported via H4 and the measurements column, not asserted. What
is guaranteed: episodes are finite (<= 12 measurements), all
reward terms are bounded, and the consistency bonus is strictly
smaller than the verdict and measurement terms combined, so no
trajectory can exceed +1.05 by farming intermediate terms.

## 7. Training budget

Maximum `10^6` episodes per TD policy (epsilon-greedy,
gamma = 0.99, lr = 0.3). Plateau criterion, declared before the
run: aggregate episode rewards into 100-episode block means over
the final 20% of the budget; fit an ordinary least-squares line
to those block means; plateau iff the 95% confidence interval of
the slope contains zero. If no plateau is
reached by `10^6` episodes, the pre-registered outcome is
**"did not converge within budget"**, reported as such — never a
silent stop. The training curve and the state-visit histogram
ship in `reports/poc.md` so convergence is auditable.

## 8. Metrics and uncertainty

**Primary:** Journey-Q accuracy minus Shortcut-CART accuracy on
the test split, percentage points, hard-label agreement with
expert verdicts.

**Uncertainty (mandatory for every accuracy delta reported):**
paired bootstrap 95% CI (10,000 resamples, seeded, resampling
test scenarios — all policies are scored on the same scenarios)
plus McNemar's test on discordant pairs. A delta whose CI
includes zero is reported as inconclusive, whatever its sign.

**Secondary (defined):**

- mean measurement actions per decision
- invalid-verdict rate (terminal action contradicts the gathered
  evidence, per section 3)
- review-rate = fraction of test scenarios whose terminal action
  is `review`; a review predicted on a pass-labeled scenario
  counts **both** as an accuracy error and in review-rate
- measured latency (`perf_counter`, mean ms per decision; dev
  machine only — see section 12)
- ECE (10 bins) and Brier score, both computed on the test split
- Q-table stored-entry count and visited-state coverage

**Sensitivity:** measurement noise sigma swept over
`{0.04, 0.08, 0.12}`; the sigma = 0.08 run is the primary one.

## 9. Confidence calibration (frozen)

Primary map: isotonic regression on the terminal-action margin
(`Q_best - Q_second`) -> P(action correct), fitted on the
calibration split. Cross-check: temperature-scaled
`softmax(Q/tau)`, `tau` fitted on the same split, reported as
ECE only. If the two forms disagree materially (ECE difference
> 0.03 or mean absolute probability difference > 0.05),
`poc.md` says so and the shipped map is flagged provisional.
The number shown to a user is the calibrated P(correct); the
expert's deterministic confidence heuristic is internal only and
never ships.

## 10. Hypotheses

H1 is the **sole confirmatory hypothesis** (the ticket's graded
criterion). H2–H4 are mechanism checks: reported with the same
CI discipline, not gated on significance.

- **H1 (primary, graded):** Journey-Q accuracy >
  Shortcut-CART accuracy.
- **H2:** Journey-Q accuracy >= NoRecheck-Q accuracy (value of
  re-verification).
- **H3:** Journey-Q accuracy >= ExpertOnly-Q accuracy (value of
  journey learning = learning from flawed trajectories, the
  ticket's central mechanism).
- **H4:** Journey-Q mean measurements/decision < 6 (early exit;
  the tree reads all six signals by construction).

## 11. Fallbacks (pre-declared)

- **Binary action space.** If the mentor rules the action space
  strictly binary (approve/reject, no review): one terminal
  action is removed; the ordinal partial-credit reward collapses
  to binary (`+1.0 / -1.0`, no `-0.5` adjacent term); the
  multi-step reasoning, journey learning, calibration, and every
  split above are unchanged.
- **X never supplied.** Report the measured delta and effect
  size with no pass/fail claim (section 2).

## 12. Reporting rules

Measured numbers ship as measured. The expert row is labeled as
defining the labels. The model is a **policy table**, reported
as a stored-entry count — not a parametric model; the ticket's
<100K budget is respected under either reading. Latency: the
decision core's <10 ms criterion is non-binding at table-lookup
scale; measurements in Stage A come from an Apple M4 dev machine
where Python interpreter overhead dominates, and **no
representative Android device is measured in Stage A** —
on-device measurement is Stage B. Naming in all artifacts:
"CLEF-inspired lightweight decision-policy POC".
