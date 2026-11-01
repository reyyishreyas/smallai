# EXPERIMENT_A1.md — pre-registered round 2 (AI-267)

Status: PRE-REGISTRATION for a **new, separate round**. Written and
committed **before** any Stage A.1 training run. Stage A (v2.1) stays
exactly as it was: its report, its JSON, its seeds and its numbers are
untouched and are reported separately in
`AI-267_STAGE_A_RESULTS.md`.

## 0. Why a second round exists

Stage A answered the protocol's questions but not the ticket's. The
ticket's graded constraint is "journey learning >= X% over shortcut"
(`ARCHITECTURE.md` section 1). Stage A's primary hypothesis, H1,
came out **inconclusive**: +0.300 pp, 95 % CI [-0.400, 1.000],
McNemar p = 0.5811, on a 1,000-scenario test split.

Two separate things are wrong with that, and this round fixes both.

**1. The split could not resolve the effect.** On 1,000 paired
scenarios at ~98 % accuracy the one-sigma noise on a paired accuracy
delta is about 0.36 pp. A method whose real effect is a fraction of a
percentage point cannot be shown to work on a split that size, and a
method that genuinely does not work cannot be shown not to. The
interval was a statement about the sample size, not about the method.

**2. The effect itself was never measured.** Before this round, a
high-power development measurement (20,000 fresh balanced scenarios,
paired bootstrap and McNemar, `scripts/dev/power.py`) put the
Journey-Q minus Shortcut-CART accuracy delta at **+0.035 pp, 95 % CI
[-0.140, +0.210], McNemar p = 0.7409**. The true accuracy gain of
journey learning over the one-shot shortcut at sigma = 0.08 is
therefore *essentially zero*, and Stage A's +0.300 pp was a lucky
draw. Where the journey policy really wins is elsewhere: the
invalid-verdict rate (0.570 % vs 7.950 %) and the measurement count
(4.501 vs 6.000).

Stage A.1 therefore measures the graded comparison properly and
reports the whole picture, rather than leaving the reader with one
wide interval.

## 1. What changes relative to Stage A

| Item | Stage A | Stage A.1 | Why |
| --- | --- | --- | --- |
| Test split size | 1,000 | **10,000** | Paired SE falls from ~0.36 pp to ~0.11 pp, so the graded comparison can return an answer. |
| Master seeds | 1 (20261021) | **5 fresh (20263030-20263034)** | One master seed leaves a training-draw noise of ~0.25 pp, which alone would swamp the effect; five seeds average it down. |
| Reward | frozen | **frozen** (unchanged) | Every candidate was refuted on fresh seeds before this round was pre-registered (section 3). The shipped default is not moved by a null result. |
| Everything else | -- | **unchanged** | Splits' train/calibration sizes and mixtures, the policy field, the learning constants, the episode budget, the plateau test, the calibration map, the metrics, the sigma sweep and the statistics are all Stage A's. |

The train split (4,000, skewed field mixture) and the calibration
split (1,000, balanced) keep their Stage A sizes and mixtures. Only
the test split grows, and it grows in the same balanced mixture the
protocol already specifies for it.

## 2. Policies under test

Unchanged from Stage A, at all three noise levels:

- **Shortcut-CART** -- the graded baseline: six first readings, one
  shot, no re-verification, `min_samples_leaf` cross-validated on the
  train split only.
- **Journey-Q** -- the deliverable: typed actions, evidence path,
  re-verification, sparse tabular Q-table.
- **NoRecheck-Q** -- Journey-Q with re-verification disabled (H2).
- **ExpertOnly-Q** -- Journey-Q trained on expert rollouts only (H3).

Plus **one** pre-registered candidate at the primary sigma only,
described in section 3.

## 3. The one candidate reward, and why it is still here

`WIDE_CONSISTENCY_REWARD` (defined in `smallai/train.py`, shipped
alongside the frozen spec so the round is reproducible):

- the verdict term is unchanged (+1.0 correct, -0.5 adjacent wrong,
  -1.0 opposite wrong);
- the consistency term is **halved** (+0.05 consistent, -0.15
  contradictory);
- the consistency *plausibility* is broadened: a **suspect** reading
  no longer counts as refuting "retake", because a suspect reading
  is precisely the ambiguous case in which the latent value may sit
  below the fail threshold. "retake" is therefore consistent when a
  fail *or* a suspect was observed, and the consistency term stops
  arbitrating the retake/review pair.

Motivation: the policy's dominant residual error is
retake-vs-review on suspect evidence (dev diagnostic, 1,000
scenarios: 17 retake->review against 6 review->retake), and the
frozen consistency term makes "retake" contradictory in exactly that
state, so the policy plays safe and loses accuracy it already has
evidence for.

Evidence gathered *before* this pre-registration, all paired
development runs on seeds disjoint from the scored seed:

| Evidence | Result |
| --- | --- |
| 16 selection seeds (20261030+) | +0.375 +- 0.137 pp, t = 2.74 |
| 16 fresh seeds (20262030+) | **-0.100 +- 0.134 pp, t = -0.75** |

The selection-seed gain did not reproduce: it was winner's-curse
noise from picking the best of many candidates on one seed set. The
candidate is carried into this round anyway, with a pre-registered
decision rule, so that the round delivers an independent and decisive
verdict instead of leaving the result resting on a refuted
selection.

**Decision rule (pre-registered).** The candidate replaces the frozen
reward as the shipped default **iff** its paired accuracy delta
against the frozen Journey-Q, at the primary sigma, has a paired
bootstrap 95 % CI excluding zero on the positive side **and** a
McNemar p < 0.05, pooled over the five master seeds. If it does not,
the frozen reward stays the shipped default and the null is reported
as a null.

## 4. Hypotheses

Unchanged from `EXPERIMENT.md` section 10.

- **H1 (sole confirmatory hypothesis):** Journey-Q accuracy >
  Shortcut-CART accuracy.
- **H2 (mechanism):** Journey-Q accuracy >= NoRecheck-Q accuracy.
- **H3 (mechanism):** Journey-Q accuracy >= ExpertOnly-Q accuracy.
- **H4 (mechanism):** Journey-Q mean measurements per decision < 6.

### 4.1 Pre-registered secondary analyses

The graded question is about journey learning, and accuracy alone
cannot answer it. Two further comparisons are therefore pre-registered
as **secondary** (not confirmatory), each reported with the same
paired machinery, Journey-Q against Shortcut-CART at the primary
sigma:

- **S1 -- evidence consistency.** The paired difference in the
  invalid-verdict rate, with a paired bootstrap CI over scenarios and
  a McNemar test on the paired consistency indicator.
- **S2 -- measurement cost.** The paired difference in mean
  measurements per decision, with a paired bootstrap CI over
  scenarios.

These are reported whatever they show. They are secondary precisely
because Stage A fixed the confirmatory criterion as H1; they widen
the evidence around the ticket's question, they do not replace it.

## 5. Statistics

Unchanged in kind from `EXPERIMENT.md` section 8, applied per master
seed and pooled:

- paired bootstrap over **scenarios**, 10,000 resamples, seeded, for
  every reported delta;
- McNemar's exact two-sided test on the paired verdicts;
- the same machinery for the paired invalid-verdict indicator (S1).

**Pre-registered primary summary for H1.** Because the five master
seeds are independent replicates of the whole procedure (fresh
splits, fresh training, fresh evaluation), the primary H1 summary is
the **per-seed paired delta** at the primary sigma, summarised as the
mean across the five seeds with a paired t-interval over the seeds.
That interval carries the training-draw noise as well as the
evaluation noise, so it is the honest interval for "does journey
learning beat the shortcut for this procedure". The pooled
scenario-level bootstrap over all 50,000 paired scenarios is reported
beside it as the evaluation-noise-only interval.

**H1 is supported iff** both the seed-level paired t-interval
excludes zero on the positive side **and** the pooled McNemar test
gives p < 0.05. Otherwise H1 is reported as not supported, with the
interval that says so.

## 6. Splits, seeds and budget

- Master seeds: **20263030, 20263031, 20263032, 20263033, 20263034**.
  None of them is the Stage A seed (20261021), the development
  selection seeds (20261030+) or the fresh-seed confirmation seeds
  (20262030+). All sub-seeds derive from the master seed exactly as
  in Stage A.
- Splits per master seed: train 4,000 (skewed field mixture),
  calibration 1,000 (balanced), test **10,000** (balanced).
  Uniqueness is enforced across the combined pool, before any noise,
  as in section 4.
- Episode budget: **1,000,000** per TD policy, with the pre-registered
  plateau test (section 7) free to stop it earlier, as in Stage A.
- Bootstrap resamples: **10,000**.
- Noise sweep: sigma in {0.04, 0.08, 0.12}; primary sigma 0.08.

## 7. Stopping rule

This round is the last tuning round. Whatever it reports is
reported as measured. No further reward, constant, encoding, budget
or data change is made after the Stage A.1 run, and no policy is
selected on the Stage A.1 test split. If the candidate reward fails
its decision rule, the shipped model is the Stage A model, and the
refuted candidates stay in the repository only as documented,
re-runnable negative results.

## 8. Reporting

Stage A.1 writes `reports/poc_a1.md` and `reports/poc_a1.json`, and
is written up in `AI-267_STAGE_A1_RESULTS.md`. Stage A's
`reports/poc.md`, `reports/poc.json` and the Stage A results document
are **not** edited by this round, and the Stage A numbers are
reproduced there by reference, not by overwriting.

The 1,000-scenario subset of the Stage A.1 test split (its first
1,000 scenarios, in draw order) is scored and reported alongside the
full 10,000, so the round can be compared with Stage A on the same
size of sample and read as a continuity check as well as a
measurement.

## 9. Integrity statement

- Stage A's protocol, code path, seeds and artifacts are unchanged.
  The Stage A run is re-executed after the code changes in this round
  and its deterministic output is diffed against the committed
  `reports/poc.json`; the diff must be empty, and it is reported
  either way.
- No hyperparameter of any policy is selected on the calibration or
  test splits of this round.
- The development measurements quoted in section 0 and section 3 were
  produced on development seeds and are not part of the scored
  protocol; the code that produced them lives under `scripts/dev/`
  and never runs as part of the scored pipeline.
