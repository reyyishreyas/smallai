# ARCHITECTURE — SmallAI decision-policy POC (AI-267)

*One decision-maker for one job: is this document capture usable?*
*CLEF-inspired lightweight decision-policy POC. Pure Python. No
neural nets.*

## 0. Read this first

A field officer photographs an ID or form with a phone — often
offline, often in bad light. Some captures are unusable: blurry,
glared, half out of frame. Catching them *at the camera*, with a
reason ("blur=0.20, retake"), saves the officer's time and a
reviewer's time later. This project builds that decision-maker:

- **Input:** six quality signals measured from the capture
- **Output:** `pass` / `retake` / `review`, with a calibrated
  confidence, human-readable reasons, and the exact sequence of
  checks that produced it
- **Where it runs:** offline, CPU only, milliseconds

It is *not* a document-content model (it does not read what the ID
says). It is a capture-quality gate — the first stage of a longer
pipeline. The Stage A / Stage B scope split is defined in
`EXPERIMENT.md` §1.

## 1. The constraints and what each one forces

| Ticket constraint (AI-267) | Rules out | What we built instead |
|---|---|---|
| From scratch, no black-box ML | sklearn / xgboost / torch in the core | hand-written Q-table + TD updates |
| Strictly non-LLM, non-transformer | any neural net | lookup table |
| < 100K parameters | big models | <=15,625 states, few legal actions each, stored sparsely |
| < 10ms CPU | heavyweight inference | dict lookups (~µs); the real latency budget lives in the CV stage (out of scope, Stage B) |
| Multi-step reasoning | single-shot classifiers | the check path *is* the reasoning chain |
| Journey learning >= X% over shortcut | one lone model, no baseline | pre-registered comparison vs a one-shot tree + two ablations |
| No dataset exists | supervised training on real captures | synthetic scenarios from documented distributions — explicitly allowed by the ticket |

## 2. The big picture

```
 ┌──────────────────────────────────────────────┐
 │ STAGE 0: perception  (Stage B: real CV)      │
 │ 6 signals in [0,1]:                          │
 │ blur glare brightness coverage skew resolution│
 └──────────────────────┬───────────────────────┘
                        │ noisy readings (sigma = 0.08)
                        ▼
 ┌──────────────────────────────────────────────┐
 │ STAGE 1: the policy (agent.py)               │
 │ state -> pick next action:                   │
 │ measure a signal / re-check / decide         │
 └──────────────────────┬───────────────────────┘
                        │ terminal action
                        ▼
 ┌──────────────────────────────────────────────┐
 │ DECISION                                     │
 │ {action, confidence, reasons, path}          │
 └──────────────────────────────────────────────┘

 learning loop (train.py, offline, once):
 expert + flawed behaviours -> trajectories
   -> rewards -> TD updates -> Q-table

 measuring stick (evaluate.py):
 journey model vs baselines on held-out data
```

Five Python files carry these boxes; the rest of this document
explains each one.

## 3. The signals (`environment.py`)

Each signal is a number in [0, 1], higher = better, with two
documented thresholds:

| signal | fail below | pass at/above | meaning |
|---|---|---|---|
| blur | 0.35 | 0.65 | 1 − blur strength |
| glare | 0.30 | 0.60 | absence of blown-out reflections |
| brightness | 0.35 | 0.60 | closeness to usable exposure |
| coverage | 0.55 | 0.80 | document fully in frame |
| skew | 0.40 | 0.70 | alignment to frame axes |
| resolution | 0.30 | 0.55 | usable pixel size |

Three zones: **fail** (below FAIL — unusable), **suspect** (in
between — a judgement call), **pass** (fine). The thresholds are
engineering defaults, stated as such; fitting them to real
captures is the Stage B spike's job.

Two facts shape everything else:

1. **Readings are noisy.** Every measurement = true value +
   Gaussian noise (sigma = 0.08), clipped to [0, 1]. A true 0.50
   brightness can read 0.58 once and 0.44 the next time. Noise is
   what makes *re-checking* worth anything: with perfect sensors,
   one reading settles everything and there is no sequential
   decision problem at all.
2. **Scenarios come from a defect-cause mixture.** A scenario is
   clean, has one named defect, or has multiple. Training data is
   blur-heavy (30% blur — field reality); test data is balanced
   across causes (~10% each). The skew exists so the one-shot
   baseline's shortcut ("blurry -> retake") becomes *measurable*
   on the balanced test set.

## 4. The expert (`solver.py`): labels without annotators

No labeled dataset exists, so ground truth comes from a
deterministic expert — the ticket explicitly allows this. The
expert:

1. checks signals in a fixed order — **blur -> brightness ->
   glare -> coverage -> skew -> resolution** (cheapest-to-fail
   first),
2. **exits early**: the first hard failure ends the episode with
   `retake`,
3. **re-checks** any reading that lands in the suspect band
   (at or above FAIL, below PASS) before believing it,
4. finishes: any remaining suspect -> `review`, else `pass`.

Every policy in this project emits the same output shape:

```python
Decision(
    action="retake",            # pass | retake | review
    confidence=0.74,            # calibrated P(correct), for agents
    reasons=("blur=0.20 failed (<0.35)",),   # the evidence
    path=("blur",),             # checks performed, in order
)
```

The expert is also the ticket's "heuristic rules" baseline, and
its verdicts *are* the labels. It scores 100% by construction —
a reference row, never a competitor.

## 5. The agent (`agent.py`): a lookup table that thinks in steps

### 5.1 State

Each of the six signals sits in one of five states:

| code | name | meaning |
|---|---|---|
| 0 | unchecked | not measured yet |
| 1 | fail_final | measured, decisively bad |
| 2 | suspect_pre | first reading in the suspect band — re-checkable |
| 3 | suspect_post | re-checked; still not decisive |
| 4 | pass_final | measured, fine |

A world-state is the 6-tuple -> 5^6 = **15,625 states maximum**,
stored sparsely (only visited ones).

### 5.2 Actions

- **check(s)** — measure signal `s` (only while unchecked)
- **re-check(s)** — measure again (only from `suspect_pre`; after
  it the signal is fail_final / suspect_post / pass_final, none of
  which permit another re-check — re-check loops are
  *unrepresentable*, not merely discouraged by training)
- **decide pass / retake / review** — legal at any time; ends the
  episode

At most 12 measurements per episode (6 initial + 6 re-checks).

### 5.3 The Q-table

The whole model is `dict[state][action] -> expected reward`. A
Q-value answers: "if I do this here, how well does the episode
end?" It starts at zero and is updated by experience (§6). At
decision time the agent walks greedily — best action at each
state, stop at a terminal — and the margin between the best and
second-best terminal Q-values is the raw confidence signal, later
mapped to a real probability (§7.3).

The model ships as JSON. The Android port consumes that file; a
table needs no inference runtime.

### 5.4 Walk-through — one capture, start to finish

Capture: true `blur = 0.21`, all other signals ~ 0.90.

1. Start: `(0,0,0,0,0,0)` — everything unchecked. Greedy action:
   `check(blur)`.
2. Reading: 0.20 (true 0.21 + noise). Below FAIL (0.35), not near
   a threshold -> `fail_final`. New state: `(1,0,0,0,0,0)`.
3. At this state `decide(retake)` dominates. Terminal.
4. Output: `retake`, path `("blur",)` — one measurement; the other
   five signals were never worth 0.05 of reward each. §8 scores
   this capture under every policy.

## 6. Learning (`train.py`)

### 6.1 Trajectories

A trajectory is one full trial: state -> action -> ... -> verdict.
Training mixes:

- **expert trajectories** — correct, early-exiting paths;
- **flawed trajectories** — from deliberately bad behaviours:
  random order, always-measure-everything, decide-first,
  never-recheck. Their *negative* rewards are the point: they
  teach which journeys end badly and why. This is "journey
  learning" — the model trains on failures, not just successes;
- **on-policy deviations** — the agent's own exploratory moves
  (see ε below).

### 6.2 Reward (frozen in `EXPERIMENT.md` §6)

| event | reward |
|---|---|
| correct verdict | +1.0 |
| adjacent-wrong (review vs pass/retake) | −0.5 |
| opposite-wrong (pass vs retake) | −1.0 |
| each measurement | −0.05 |
| evidence-consistent verdict | +0.10 |
| evidence-contradicting verdict | −0.30 |

"Consistent" has an exact definition: `pass` needs all six
observed pass-level; `retake` needs an observed fail; `review`
needs an observed suspect. Deciding with zero evidence is
contradictory. Everything is bounded — a trajectory scores in
[−1.90, +1.05] — so no intermediate term can be farmed.

Example, the blur capture of §5.4: `+1.0 (correct) − 0.05 (one
measurement) + 0.10 (consistent) = +1.05`, the maximum any
trajectory can earn.

### 6.3 The update rule, in plain words

TD(0): after each step, nudge the Q-value toward what actually
happened plus what the next state promises:

```
Q[s][a] <- Q[s][a] + lr * (immediate_reward + gamma * max Q[s'][.] - Q[s][a])
```

- **lr = 0.3** — take 30% of each new piece of evidence (fast
  enough for a table, slow enough to average out noise);
- **gamma = 0.99** — the future counts nearly as much as the
  present (episodes are short; the whole journey matters);
- **epsilon = 0.1** — one action in ten is random during training
  so unexplored options get tried.

Budget: at most 10^6 episodes, stopped only by a pre-registered
plateau test; failing to converge is itself a reportable result
(`EXPERIMENT.md` §7).

### 6.4 The four scored policies

| policy | trained how | isolates |
|---|---|---|
| **Journey-Q** | everything above | the deliverable |
| **NoRecheck-Q** | identical, minus re-check actions | the value of re-verification |
| **ExpertOnly-Q** | identical, minus flawed-trajectory replay | the value of journey learning itself |
| **Shortcut-CART** | from-scratch decision tree: six first-readings in, verdict out, one shot | what is lost without any sequential verification |

Each ablation differs from Journey-Q in exactly one variable.
The tree is the ticket's graded "shortcut baseline": it sees the
state and commits — no re-checks — and under the blur-heavy
training mix it learns to lean on blur first.

## 7. Evaluation (`evaluate.py`)

### 7.1 Splits (leakage-safe)

Scenarios (true signal vectors) are split **before** any noise:
train 4,000 (skewed) / calibration 1,000 (balanced) / test 1,000
(balanced). No scenario crosses splits; all seeded.

### 7.2 The graded question

Primary number: Journey-Q accuracy minus Shortcut-CART accuracy on
the test split, in percentage points, with a paired-bootstrap 95%
confidence interval and McNemar's test — both policies face the
same 1,000 scenarios, so the comparison is paired, and an
interval covering zero is reported as inconclusive. This is
pre-registered in `EXPERIMENT.md` before any training runs; the
X% threshold (if any) is supplied by the mentor, not chosen by us.

### 7.3 Confidence calibration

Raw Q-margins are scores, not probabilities: a "0.97" from the
table means nothing to a user until it is mapped. We fit an
**isotonic regression** — a monotone step map, fitted on the
calibration split — from margin to P(correct), and cross-check it
against temperature-scaled softmax. Reliability is reported as
ECE (expected calibration error: the average gap between stated
probability and observed frequency, in 10 bins) and Brier score.
The number the app shows is the calibrated one; the expert's
hand-written confidence heuristic stays internal.

## 8. The same capture, scored by every policy

Scenario: true brightness 0.50 (in the suspect band 0.35–0.60),
all other signals ~ 0.90. Expert verdict: **review** — it re-checks
the borderline reading, still suspect, then defers.

| policy | what it does here | verdict | outcome |
|---|---|---|---|
| Expert rules | brightness reads 0.58 (near PASS 0.60) -> re-check -> 0.44 -> still suspect -> checks the rest -> review | review | defines the label |
| Journey-Q | same suspicion-triggered re-check; terminal margin favours review | review | correct: `+1.0 − 0.05*k + 0.10` |
| NoRecheck-Q | first reading 0.58 looks near-pass; commits | pass | adjacent-wrong: `−0.5 − 0.05*k − 0.30` |
| Shortcut-CART | reads all six once; brightness 0.58 is its least informative feature | pass | mislabelled on the balanced test distribution |

This is the mechanism hypotheses H1–H3 bet on: noise puts
borderline readings on the wrong side of a threshold often enough
that verification pays for its −0.05. Whether it actually pays,
by how much, and with what confidence interval is *measured*, not
assumed.

## 9. Why a table, and not X

| alternative | why not (for this ticket) |
|---|---|
| decision tree (the ticket's other candidate) | one vector in, one label out — no check order, no early exit, no "you verified too late". Kept, but as the *baseline* |
| tiny MLP / CNN from scratch | needs real labeled images (PII-laden, nonexistent), >100K params, days not hours |
| LLM / transformer | forbidden by the ticket; 100ms+ and networked — the opposite of every constraint |
| rules only (the expert) | no learning, no journey, nothing to grade against a shortcut |
| tabular Q-learning | sequential verification is native; interpretable per decision; µs-scale; JSON artifact |

## 10. File map

| file | role | status |
|---|---|---|
| `smallai/environment.py` | signals, thresholds, noise, scenario sampling | built |
| `smallai/solver.py` | the expert (labels + rules baseline) | built |
| `smallai/agent.py` | state encoding, Q-table, decide(), JSON I/O | built |
| `smallai/train.py` | trajectories, rewards, TD learning, ablations | built |
| `smallai/evaluate.py` | calibration, metrics, uncertainty, tables | built |
| `scripts/run_poc.py` | one command: generate -> train -> evaluate -> `reports/poc.md` | built |
| `tests/` | gates per module | 110 gates |
| `EXPERIMENT.md` | the frozen pre-registered protocol | v2.1 |
| `ARCHITECTURE.md` | this document | — |

## 11. Running it

```bash
# any Python >= 3.10 with numpy and pytest
python -m pytest tests/ -q        # the gates
python scripts/run_poc.py         # full pipeline -> reports/poc.md

# a fast end-to-end check that writes nowhere near the scored
# numbers; the report labels itself a smoke run
python scripts/run_poc.py --budget 5000 --resamples 500 --out /tmp/smoke
```

The scored run takes the pre-registered budget of 10^6 episodes per
TD policy across the three noise conditions, and writes
`reports/poc.md`, `reports/poc.json` and one JSON Q-table per TD
policy.

## 12. Honest limitations

- **Synthetic Stage 0.** The signals are sampled, not computed
  from images. This POC validates the decision logic and the
  learning mechanisms, not capture CV. The Stage B spike (~50 real
  captures, real metrics) calibrates the distributions and gates
  the Android work.
- **Latency is trivially met at this layer.** Table lookups are
  microseconds; the real budget is the CV stage (out of scope).
  Stage A numbers come from a dev machine, not an Android device.
- **It is a policy table, not a parametric model.** Reported as a
  stored-entry count; the ticket's <100K budget holds under any
  reading.
- **The `review` action** is an open scope question with the
  mentor; the binary fallback is pre-declared in
  `EXPERIMENT.md` §11.

---

All numbers in this document are design constants (thresholds,
rewards, budgets) traceable to `EXPERIMENT.md`. No performance is
claimed here: the Stage A run has now been executed, and its
measured results live in `reports/poc.md` and nowhere else. Read
that report for the outcome, including whether the graded
comparison held up.
