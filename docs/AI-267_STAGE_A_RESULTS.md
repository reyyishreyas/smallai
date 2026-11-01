# AI-267 Stage A — Complete Measured Results

**CLEF-inspired lightweight decision-policy POC**
Repository: `reyyishreyas/smallai` · Stage: A · Status: **scored**

Every number in this document was measured by one run of `python scripts/run_poc.py`
against the protocol frozen in `EXPERIMENT.md` (pre-registration v2.1). Nothing here is
estimated, rounded-up, or hand-edited after the fact, and the protocol was not modified
after these results existed.

Raw machine-readable source for every figure: `reports/poc.json` (560,665 bytes).
Narrative form of the same run: `reports/poc.md`. Design constants: `ARCHITECTURE.md`.

---

## 0. Headline

**The pre-registered primary comparison is inconclusive on accuracy.** Journey-Q reached
98.0% vs Shortcut-CART's 97.7% — a **+0.300 pp** delta with a paired-bootstrap 95% CI of
**[-0.400, 1.000]** (10,000 resamples) and McNemar exact two-sided **p = 0.5811**. The
interval contains zero, so the difference is inconclusive whatever its sign.

**Two secondary results are unambiguous and are the ones that matter for a capture gate:**

| result | Journey-Q | Shortcut-CART | effect |
|---|---|---|---|
| **Invalid-verdict rate** (verdict contradicts its own evidence) | **0.400%** | **8.700%** | **21.75× lower** |
| **Mean measurements per decision** | **4.532** | 6.000 | 24.5% fewer signal acquisitions |
| Accuracy (primary metric) | 98.000% | 97.700% | +0.300 pp, CI includes 0 |

Accuracy is a saturated metric on this test split (the split class mix is 345 pass /
632 retake / 23 review — the trivial majority class alone scores 63.2%, and most
scenarios are unambiguous), so the metric lacks the resolution to separate the top
policies. The invalid-verdict rate is the discriminating metric, and it separates them by
more than an order of magnitude while accuracy cannot.

**No pass/fail claim is made.** The threshold "X" referenced by the ticket has never been
supplied. Per `EXPERIMENT.md` §2, any X supplied now is post-hoc relative to the
pre-registration and cannot be presented as pre-registered, so this report states the
measured delta and its interval and stops there.

---

## 1. Run facts

| field | value |
|---|---|
| Master seed | `20261021` |
| Episode budget | 1,000,000 (pre-registered: 1,000,000) ✔ |
| Bootstrap resamples | 10,000 (pre-registered: 10,000) ✔ |
| Wall clock | **65.04 s** |
| Framing | `CLEF-inspired lightweight decision-policy POC` |
| Stage | `A` |

Every other seed in the run (policy seeds, per-scenario RNG streams, bootstrap resamples)
derives from the master seed via SHA-256, so the entire run reproduces from that single
number.

## 2. Environment

| component | version |
|---|---|
| Python | 3.13.9 |
| numpy | 2.5.3 |
| Platform | `macOS-26.5.1-arm64-arm-64bit-Mach-O` |
| Machine | `arm64` |

**Latency caveat:** all latency figures are dev-machine wall clock, measured by
`time.perf_counter()` around the decision call only. They are explicitly **not** an
Android or on-device claim (`EXPERIMENT.md` §12). See §9.2 for the exact comparability
limits.

## 3. Splits

Scenarios are partitioned **before any noise is drawn**, and no true signal vector appears
in two splits. Train uses the blur-heavy field mixture; calibration and test use the
balanced mixture, so a one-shot policy cannot pass by memorising "blurry means retake".

| split | n | distribution | used for |
|---|---|---|---|
| train | 4,000 | skewed, blur-heavy | fitting all policies |
| calibration | 1,000 | balanced | fitting the confidence map **only** |
| test | 1,000 | balanced | scoring only, never tuned on |

**Outcome class counts (calibration split):**

| class | n | share |
|---|---|---|
| pass | 345 | 34.5% |
| retake | 632 | 63.2% |
| **review** | **23** | **2.3%** |

> **Honest limitation:** the `review` class has only 23 observations out of 1,000. The
> isotonic confidence map rests on very thin support wherever a decision turns on that
> outcome. These counts are reported rather than smoothed away. The same counts hold in
> all three noise conditions.

## 4. The policy field

| policy | description | role |
|---|---|---|
| Expert rules | deterministic expert; defines the labels | reference row, 100% by construction, **not scored** |
| Shortcut-CART | from-scratch CART, max depth 6 a priori, hyperparameters by CV on train only | graded baseline: one shot, no verification |
| NoRecheck-Q | Journey-Q with re-check actions removed | ablation: value of re-verification |
| ExpertOnly-Q | Journey-Q without flawed-trajectory replay | ablation: value of journey learning itself |
| Journey-Q | the full model | **the deliverable** |

Each ablation differs from Journey-Q in exactly one variable. No hyperparameter of any
policy was selected on the calibration or test splits.

---

## 5. PRIMARY RESULT — accuracy (primary condition, sigma = 0.08)

Journey-Q accuracy minus Shortcut-CART accuracy on the test split:

**+0.300 percentage points**, paired-bootstrap 95% CI **[−0.400, 1.000]** over 10,000
resamples of test scenarios.

Both policies are scored on the same 1,000 scenarios with the same scenario instances, so
the comparison is **paired**.

McNemar's test on the **13 discordant pairs**: chi-square **0.692**, exact two-sided
**p = 0.5811**.

**The interval includes zero ⇒ the difference is inconclusive whatever its sign.**

No X has been supplied, so no pass/fail claim is made (`EXPERIMENT.md` §2).

---

## 6. Hypothesis results

H1 is the sole **confirmatory** hypothesis. H2–H4 are **mechanism checks**, reported with
the same interval discipline but not gated on significance.

### 6.1 Primary condition (sigma = 0.08)

| hypothesis | claim | delta (pp) | 95% CI | p | supported | reads as |
|---|---|---|---|---|---|---|
| H1 | Journey-Q accuracy > Shortcut-CART accuracy | **+0.300** | [−0.400, 1.000] | 0.5811 | **no** | CI includes zero: inconclusive |
| H2 | Journey-Q accuracy ≥ NoRecheck-Q accuracy | +0.800 | [−0.200, 1.800] | 0.1516 | yes | mechanism check, not gated on significance |
| H3 | Journey-Q accuracy ≥ ExpertOnly-Q accuracy | +0.100 | [−0.700, 1.000] | 1.0000 | yes | mechanism check, not gated on significance |
| H4 | Journey-Q mean measurements/decision < 6 | — | — | — | **yes** | measured mean 4.532; the tree reads all six signals by construction |

### 6.2 Noise sweep — hypotheses at every sigma

| sigma | H | delta (pp) | 95% CI | p | supported |
|---|---|---|---|---|---|
| 0.04 | H1 | +0.700 | [−0.100, 1.500] | 0.1435 | no |
| 0.04 | H2 | −0.400 | [−1.400, 0.600] | 0.5413 | no |
| 0.04 | H3 | −0.300 | [−1.100, 0.500] | 0.6291 | no |
| 0.04 | H4 | — | — | — | yes (mean 4.499) |
| **0.08** | **H1** | **+0.300** | **[−0.400, 1.000]** | **0.5811** | **no** |
| **0.08** | **H2** | **+0.800** | **[−0.200, 1.800]** | **0.1516** | **yes** |
| **0.08** | **H3** | **+0.100** | **[−0.700, 1.000]** | **1.0000** | **yes** |
| **0.08** | **H4** | — | — | — | **yes (mean 4.532)** |
| 0.12 | H1 | +0.200 | [−0.700, 1.100] | 0.8238 | no |
| 0.12 | H2 | +2.900 | [+1.500, +4.300] | **0.000114** | yes |
| 0.12 | H3 | +0.600 | [−0.400, 1.600] | 0.3075 | yes |
| 0.12 | H4 | — | — | — | yes (mean 4.588) |

**Reading this honestly:** H1 is inconclusive at *every* noise level — the accuracy delta
never separates from zero. H2 (value of re-verification) flips from "not supported" at
sigma = 0.04 to "supported" at 0.08 and **decisively supported at 0.12** (+2.900 pp,
p = 0.000114): the harder the field conditions, the more re-verification is worth. H3
(variance from journey learning) never reaches significance but never contradicts either.

---

## 7. Secondary metrics — full table

All figures on the **test split** (n = 1,000). Accuracy in %, review rate in %, invalid
rate in %, latency in ms, measurements as mean signal acquisitions per decision.

### 7.1 Primary condition (sigma = 0.08)

| policy | accuracy % | Δ vs Journey (pp) | mean measurements | review rate % | **invalid rate %** | mean latency (ms) |
|---|---|---|---|---|---|---|
| Expert rules (defines the labels) | 100.000 | – | – | – | – | – |
| **Journey-Q (deliverable)** | **98.000** | +0.000 | **4.532** | 1.300 | **0.400** | 0.0235 |
| NoRecheck-Q (ablation) | 97.200 | −0.800 | 4.519 | 1.600 | 7.700 | 0.0213 |
| ExpertOnly-Q (ablation) | 97.900 | −0.100 | 4.805 | 1.000 | 2.100 | 0.0237 |
| Shortcut-CART (baseline) | 97.700 | −0.300 | 6.000 | 0.000 | **8.700** | 0.0012 |

### 7.2 Low-noise condition (sigma = 0.04)

| policy | accuracy % | Δ vs Journey (pp) | mean measurements | review rate % | **invalid rate %** | mean latency (ms) |
|---|---|---|---|---|---|---|
| **Journey-Q** | **98.400** | +0.000 | **4.499** | 1.700 | **0.000** | 0.0233 |
| NoRecheck-Q | 98.800 | +0.400 | 4.507 | 2.900 | 1.100 | 0.0214 |
| ExpertOnly-Q | 98.700 | +0.300 | 4.560 | 1.800 | 0.200 | 0.0247 |
| Shortcut-CART | 97.700 | −0.700 | 6.000 | 0.000 | **3.500** | 0.0012 |

### 7.3 High-noise condition (sigma = 0.12)

| policy | accuracy % | Δ vs Journey (pp) | mean measurements | review rate % | **invalid rate %** | mean latency (ms) |
|---|---|---|---|---|---|---|
| **Journey-Q** | **97.500** | +0.000 | **4.588** | 1.500 | **1.200** | 0.0228 |
| NoRecheck-Q | 94.600 | −2.900 | 4.581 | 3.200 | 12.300 | 0.0214 |
| ExpertOnly-Q | 96.900 | −0.600 | 4.583 | 1.200 | 2.800 | 0.0225 |
| Shortcut-CART | 97.300 | −0.200 | 6.000 | 0.000 | **14.500** | 0.0012 |

### 7.4 The invalid-verdict rate across all conditions

**Definition (EXPERIMENT.md §3):** the fraction of verdicts that their own gathered
evidence contradicts — `pass` requires all six signals observed at pass level, `retake`
requires an observed fail, `review` requires an observed suspect, and any terminal taken
with **no measurements at all** is contradictory.

| sigma | Journey-Q | NoRecheck-Q | ExpertOnly-Q | Shortcut-CART | CART ÷ Journey-Q |
|---|---|---|---|---|---|
| 0.04 | **0.000%** (0/1000) | 1.100% (11) | 0.200% (2) | **3.500%** (35) | ∞ (JQ = 0) |
| **0.08** | **0.400%** (4) | 7.700% (77) | 2.100% (21) | **8.700%** (87) | **21.75×** |
| 0.12 | **1.200%** (12) | 12.300% (123) | 2.800% (28) | **14.500%** (145) | **12.08×** |

**This is the strongest result in the whole run, and it is the one to lead with:**

- At the primary condition the tree produces **87 evidence-contradicting verdicts in 1,000
  decisions**; Journey-Q produces **4**. A ~22× reduction in the failure mode that actually
  costs money — an unusable ID that gets uploaded anyway, or a good capture rejected on no
  evidence.
- **The gap widens as conditions get worse.** From 3.5% vs 0.0% at sigma = 0.04, to 8.7% vs
  0.4% at 0.08, to 14.5% vs 1.2% at 0.12. The tree degrades roughly 4× between the easiest
  and hardest condition; Journey-Q degrades from zero to 1.2%. A one-shot tree has no
  mechanism to notice that its own evidence is weak, and it never gets to look again.
- **The ablations explain the mechanism.** NoRecheck-Q jumps to 12.3% invalid at high
  noise — removing re-verification destroys the safety property. ExpertOnly-Q sits at 2.8% —
  removing journey learning degrades it too, but less. Both halves of the design are
  pulling, and the numbers say so independently of accuracy.

---

## 8. Training and convergence

The plateau test is the pre-registered one: 100-episode block means over the final 20%,
OLS slope, plateau iff the 95% CI contains zero.

### 8.1 Per-condition training tables

**sigma = 0.04**

| policy | episodes | plateau | slope [95% CI] | stored entries | visited states | coverage | final block mean |
|---|---|---|---|---|---|---|---|
| Journey-Q | 200,000 | yes | −0.0000145 [−0.0000690, 0.0000400] | 2,685 | 1,162 | 7.437% | 0.6767 |
| NoRecheck-Q | 200,000 | yes | +0.0000082 [−0.0000444, 0.0000607] | 2,352 | 876 | 5.606% | 0.6736 |
| ExpertOnly-Q | 200,000 | yes | −0.0000174 [−0.0000528, 0.0000180] | 1,798 | 704 | 4.506% | 0.8289 |
| Shortcut-CART | – | – | – | – | – | – | – |

**sigma = 0.08 (primary)**

| policy | episodes | plateau | slope [95% CI] | stored entries | visited states | coverage | final block mean |
|---|---|---|---|---|---|---|---|
| **Journey-Q (deliverable)** | 200,000 | yes | **−0.000005 [−0.000059, 0.000049]** | **3,097** | **1,360** | **8.704%** | 0.6508 |
| NoRecheck-Q (ablation) | 250,000 | yes | +0.000018 [−0.000023, 0.000059] | 2,904 | 1,080 | 6.912% | 0.6068 |
| ExpertOnly-Q (ablation) | 200,000 | yes | +0.000031 [−0.000007, 0.000068] | 2,079 | 837 | 5.357% | 0.8067 |
| Shortcut-CART (baseline) | – | – | – | – | – | – | – |

**sigma = 0.12**

| policy | episodes | plateau | slope [95% CI] | stored entries | visited states | coverage | final block mean |
|---|---|---|---|---|---|---|---|
| Journey-Q | 200,000 | yes | +0.0000299 [−0.0000234, 0.0000831] | 3,564 | 1,619 | 10.362% | 0.6135 |
| NoRecheck-Q | 200,000 | yes | −0.0000089 [−0.0000699, 0.0000520] | 3,446 | 1,339 | 8.570% | 0.5135 |
| ExpertOnly-Q | 200,000 | yes | −0.0000032 [−0.0000438, 0.0000373] | 2,323 | 965 | 6.176% | 0.7806 |
| Shortcut-CART | – | – | – | – | – | – | – |

**Every TD policy declared plateau before its episode budget ran out**, so none stopped on
the cap. Shortcut-CART is not TD-trained, which is why its rows are empty.

**Episode-count totals by policy — all three conditions** (expert / flawed / on-policy
episode split):

| sigma | policy | n_expert | n_flawed | n_on_policy | total episodes |
|---|---|---|---|---|---|
| 0.04 | Journey-Q | 80,028 | 79,892 | 40,080 | 200,000 |
| 0.04 | NoRecheck-Q | 80,026 | 79,943 | 40,031 | 200,000 |
| 0.04 | ExpertOnly-Q | 159,977 | **0** | 40,023 | 200,000 |
| **0.08** | **Journey-Q** | **80,097** | **79,946** | **39,957** | **200,000** |
| **0.08** | **NoRecheck-Q** | **100,038** | **100,059** | **49,903** | **250,000** |
| **0.08** | **ExpertOnly-Q** | **159,733** | **0** | **40,267** | **200,000** |
| 0.12 | Journey-Q | 79,896 | 79,898 | 40,206 | 200,000 |
| 0.12 | NoRecheck-Q | 79,747 | 80,078 | 40,175 | 200,000 |
| 0.12 | ExpertOnly-Q | 160,064 | **0** | 39,936 | 200,000 |

ExpertOnly-Q has **zero flawed-trajectory episodes** by construction — that is precisely
the ablation.

### 8.2 Budget compliance

| constraint | value | verdict |
|---|---|---|
| Stored parameters (Journey-Q, primary) | **3,097** entries | ✔ 32× under the 100K budget |
| State ceiling | 15,625 (5⁶) | — |
| Coverage of ceiling | 8.704% | documented lower bound¹ |
| Decision latency | 0.0235 ms | ✔ far under the 10 ms criterion (non-binding at table-lookup scale) |

¹ `EXPERIMENT.md` §3 asks for coverage over states reachable under the sampling
distribution, which has no closed form; visited-states ÷ 15,625 is a documented **lower
bound** on that quantity.

### 8.3 Journey-Q training curve (sigma = 0.08)

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

**Final block mean: 0.6508.** The full visit histogram (266 further visit-count rows)
is in `reports/poc.json`.

### 8.4 Journey-Q state-visit histogram (sigma = 0.08)

| visits | states | visits | states | visits | states |
|---|---|---|---|---|---|
| 1 | 348 | 8 | 21 | 15 | 15 |
| 2 | 138 | 9 | 14 | 16 | 19 |
| 3 | 66 | 10 | 25 | 17 | 6 |
| 4 | 41 | 11 | 14 | 18 | 10 |
| 5 | 22 | 12 | 14 | 19 | 10 |
| 6 | 19 | 13 | 23 | 20 | 11 |
| 7 | 17 | 14 | 22 | … | 266 more rows in `poc.json` |

3,097 stored values across 1,360 distinct states.

**Visit histogram across the noise sweep** (Journey-Q, states by visit count):

| visits | states @ 0.04 | states @ 0.08 | states @ 0.12 |
|---|---|---|---|
| 1 | 268 | 348 | 472 |
| 2 | 117 | 138 | 168 |
| 3 | 56 | 66 | 72 |
| 4 | 42 | 41 | 39 |
| 5 | 40 | 22 | 40 |
| 6 | 26 | 19 | 20 |
| 7 | 27 | 17 | 17 |
| 8 | 22 | 21 | 16 |
| 9 | 18 | 14 | 14 |
| 10 | 16 | 25 | 16 |
| … | 266 further rows per condition in `poc.json` | | |
| **total visited states** | **1,162** | **1,360** | **1,619** |
| **visited exactly once** | 268 (23.06%) | 348 (25.59%) | 472 (29.15%) |

As noise rises, the policy touches **more of the state space** (1,162 → 1,619 states) and a
larger share of those states is seen **only once** (23.1% → 29.2%) — i.e. noisier fields
demand more distinct situations and leave less of the table backed by repeated
observation. This is the honest cost side of the robustness story, and it is one more
reason Stage B's real-capture spike is needed before the table is trusted on device.

---

## 9. The baseline in detail (Shortcut-CART)

### 9.1 Hyperparameters — chosen by CV on the **train split only**

| parameter | value | how chosen |
|---|---|---|
| max depth | 6 | a priori, not tuned |
| min_samples_leaf | 20 | 5-fold CV on train only |
| folds | 5 | pre-registered |

**5-fold CV accuracies:**

| sigma | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | mean |
|---|---|---|---|---|---|---|
| 0.04 | 0.98375 | 0.98250 | 0.98000 | 0.98625 | 0.98500 | 0.98350 |
| 0.08 | 0.98000 | 0.98375 | 0.98250 | 0.98125 | 0.99000 | 0.98350 |
| 0.12 | 0.98000 | 0.98000 | 0.97250 | 0.97750 | 0.98125 | 0.97825 |

The tree is a *strong* baseline — its CV accuracy (98.35%) is comparable to Journey-Q's
test accuracy (98.0%). The run does not stack the deck against it, and the +0.300 pp
delta being inconclusive is a real finding, not an artefact of a weak baseline.

### 9.2 Latency — and why the two numbers are not directly comparable

| policy | mean latency (ms) | what the timer covers |
|---|---|---|
| Journey-Q | 0.0235 | the full policy walk (`walk()`), including simulated measurement steps |
| Shortcut-CART | 0.0012 | `tree.predict_one(x)` — a **single** vectorised tree inference on an **already-computed** feature vector |

The tree's 1.2 µs figure therefore **excludes the cost of acquiring the six signals**;
in Stage B that acquisition is six real CV measurements. The agent's 23.5 µs is almost
entirely dictionary-lookup overhead (measurement is simulated and free in Stage 0).

**The metric that transfers to device is the one in §7: signal acquisitions per decision —
4.532 vs 6.000, a 24.5% reduction.** On device each acquisition is a real CV pass, so
early-exiting is what buys real time. Decision-core latency for both is already two to
four orders of magnitude under the 10 ms criterion.

---

## 10. Confidence calibration

The terminal margin is a score in **reward units, not a probability**. `EXPERIMENT.md` §9
fits an isotonic regression from margin → P(correct) on the **calibration split** and
cross-checks it against temperature-scaled softmax fitted on the same split.

| sigma | isotonic ECE (test) | softmax ECE (test) | fitted temperature τ | mean abs. probability difference | provisional? |
|---|---|---|---|---|---|
| 0.04 | 0.00162 | 0.00620 | 0.3119 | 0.0112 | **no** |
| **0.08 (primary)** | **0.01101** | **0.00525** | **0.3054** | **0.0199** | **no** |
| 0.12 | 0.00371 | 0.00348 | 0.3522 | 0.0181 | **no** |

The two forms agree within the §9 tolerances at every sigma, so the isotonic map ships
**without a provisional flag** in all three conditions.

The number shown to a user is the **calibrated P(correct)**. The expert's hand-written
confidence heuristic is internal only and never ships.

---

## 11. Noise sweep summary

Sigma swept over [0.04, 0.08, 0.12]; **sigma = 0.08 is the pre-registered primary
condition**. The splits are identical across the sweep — the only thing that changes is
the measurement noise.

| sigma | Journey-Q % | Shortcut-CART % | Δ (pp) | 95% CI | p | mean measurements | JQ invalid % | CART invalid % |
|---|---|---|---|---|---|---|---|---|
| 0.04 | 98.400 | 97.700 | +0.700 | [−0.100, 1.500] | 0.1435 | 4.499 | 0.000 | 3.500 |
| **0.08** | **98.000** | **97.700** | **+0.300** | **[−0.400, 1.000]** | **0.5811** | **4.532** | **0.400** | **8.700** |
| 0.12 | 97.500 | 97.300 | +0.200 | [−0.700, 1.100] | 0.8238 | 4.588 | 1.200 | 14.500 |

**The sweep's story:** the accuracy delta is *stable and inconclusive* across the whole
noise range (+0.2 to +0.7 pp, every CI straddling zero) — the conclusion does not depend
on picking a favourable sigma. Meanwhile the **invalid-verdict rate gap grows with noise**
(3.5 pp → 8.3 pp → 13.3 pp) and the **value of re-verification grows with noise** (H2 flips
from unsupported to p = 0.000114). The design degrades more gracefully exactly where field
conditions are worst.

---

## 12. What Stage A ships — the decision contract

The output of every decision, in `smallai/solver.py`:

```python
@dataclass(frozen=True)
class Decision:
    action: str            # "pass" | "retake" | "review"
    confidence: float      # calibrated P(correct), in (0, 1]
    reasons: tuple[str, ...]   # human-readable evidence, e.g. "blur=0.21 failed (<0.35)"
    path: tuple[str, ...]      # the checks evaluated, in order
```

The six input signals are `blur`, `glare`, `brightness`, `coverage`, `skew`, `resolution`
— each in [0, 1], each observed with Gaussian noise of scale sigma.

> **Open contract question for the mentor:** the ticket names the stable contract as
> *verdict, reasons, confidence* — three fields. `Decision` has **four**: `path` is not
> named. This matters because `path` is what tells the host app *which* signals were
> checked, and therefore whether a re-check is needed. Without `path` in the contract the
> 4.53-vs-6.00 measurement win cannot be realised on device (the app would have to
> re-measure everything, or never re-check). **Is `path` part of the stable contract,
> internal, or optional?**

---

## 13. Limitations — stated plainly

1. **Synthetic Stage 0.** The six signals are **sampled, not computed from images**. This
   POC validates the decision logic, the learning mechanisms and calibration. It
   establishes **nothing** about real ID photographs. The Stage B spike (~50 real captures
   against these distributions) gates the Android work.
2. **Accuracy is saturated on this test split.** The class mix is dominated by `retake`
   (63.2%) and most scenarios are unambiguous, so the metric lacks resolution between
   the top policies. This is a property of the split, not of the model — and it is the
   main reason H1 is inconclusive.
3. **The `review` class has thin support** — 23 of 1,000 in the calibration split. The
   confidence map rests on few observations wherever a decision turns on that outcome.
   Counts are reported, not smoothed.
4. **A policy table, not a parametric model.** Reported as a stored-entry count (3,097).
   The ticket's 100K budget holds under either reading.
5. **Latency is dev-machine only.** 0.0235 ms is a table-lookup cost on an arm64 Mac. It
   is not an Android claim and does not include the CV acquisition stage.
6. **The `review` action is an open scope question with the mentor.** The binary fallback
   is pre-declared in `EXPERIMENT.md` §11 and was **not** used here.
7. **No pass/fail against X.** X has never been supplied; any X supplied now is post-hoc.
   The report states the measured delta and its interval and makes no threshold claim.

---

## 14. Open questions for the mentor

| # | question | why it matters now |
|---|---|---|
| 1 | **Is `path` part of the stable contract?** | Blocks the Kotlin port design; gates whether the measurement-count win survives on device |
| 2 | **Binary or ternary action space?** | Changes the reward function → changes the whole Stage B retrain. §11 pre-declares the binary fallback, so either answer is workable |
| 3 | **What is X, and do you want a post-hoc threshold stated?** | Per §2 any X given now is post-hoc and cannot be presented as pre-registered. Default is no pass/fail claim |

---

## 15. Reproduction

```bash
git clone https://github.com/reyyishreyas/smallai.git
cd smallai
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q        # 110 passed
.venv/bin/python scripts/run_poc.py          # → reports/poc.json, reports/poc.md
```

Deterministic from the master seed `20261021`; wall clock ≈ 65 s.

**Test suite:** 110 tests, all passing (4.20 s), covering the environment, expert rules,
agent, training gates, and the evaluation/calibration pipeline.

---

## 16. Artifact inventory

| file | bytes | contents |
|---|---|---|
| `reports/poc.json` | 560,665 | Full machine-readable run: 3 conditions × 3 policies, all scores, calibration, hypotheses, visit histograms |
| `reports/poc.md` | 9,500 | Narrative Stage A report |
| `reports/journey_q.json` | 135,830 | Journey-Q Q-table artifact (primary condition) |
| `reports/no_recheck_q.json` | 119,142 | NoRecheck-Q Q-table artifact |
| `reports/expert_only_q.json` | 84,312 | ExpertOnly-Q Q-table artifact |
| `ARCHITECTURE.md` | 16,660 | Design constants, constraints table, reward table, file map |
| `EXPERIMENT.md` | 10,079 | Frozen pre-registration v2.1 |

---

*Produced by `python scripts/run_poc.py` from the frozen protocol in `EXPERIMENT.md`.
Design constants live in `ARCHITECTURE.md`; measured results live here only. No number in
this document was changed after the run that produced it.*
