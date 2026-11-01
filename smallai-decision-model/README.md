# SmallAI Decision Model

A **lightweight, non-LLM, interpretable decision model** built from scratch in
NumPy. It does not jump straight from input to answer: it predicts a chain of
intermediate checkable states first, scores each step with a built-in
**verifier**, and only then emits the final decision. It is trained on correct
*and* deliberately-wrong decision paths — **journey learning** (Mifos
AI-267 / PRD: `../SmallAI Decision Model PRD and Tech Plan.md`).

```
features ──► encoder ──► h ─┬─► step 1 (energy) ──► SURPLUS | MARGINAL | SHORTFALL
                            │        │
                            ├─► step 2 (environment) ─► LOW | MEDIUM | HIGH      (sees step 1)
                            │        │
                            ├─► step 3 (nav/comms) ───► RELIABLE | DEGRADED |
                            │                                          UNRELIABLE  (sees 1-2)
                            │        (fused decode: log p_step + λ·log validity,
                            │         the verifier re-decodes untrustworthy states)
                            ├─► verifier([h_v, state, k]) ─► validity ∈ [0,1] per step
                            └─► final head([h, s1, s2, s3]) ─► GO | GO_REDUCED_PAYLOAD |
                                                               HOLD | ABORT
```

Every decision returns a visible trace with a validity score per step:

```
Energy: SHORTFALL (valid 0.95) -> Environment: MEDIUM (valid 0.97)
      -> Nav: RELIABLE (valid 1.00) -> Decision: HOLD
```

## 1. Problem and motivation

Document-capture / mission-planner style decisions are tabular and naturally
sequential: intermediate states (feasibility, risk, reliability) are
checkable, and the final action genuinely depends on them. A 500-token LLM
call costs ~$0.11 per 1k decisions and ~1 s latency; this model costs ~$0 per
1M decisions at **0.26 ms** per decision on CPU — with an interpretable trace
the LLM cannot offer. No LLM, transformer, GPU or API is used anywhere.

## 2. Environment and oracle

**UAV pre-flight Go/No-Go**: 10 numeric inputs (battery %, wind, temperature,
GPS fix quality, link strength, payload kg, distance, sensor health, hour,
terrain difficulty). A deterministic Python oracle computes three intermediate
states from explicit formulas + thresholds, then maps the (s1, s2, s3) combo
to a final action through a rule table:

| Priority | Condition | Action |
| --- | --- | --- |
| 1 | risk HIGH or nav UNRELIABLE | ABORT |
| 2 | energy SHORTFALL | HOLD |
| 3 | any state degraded | GO_REDUCED_PAYLOAD |
| 4 | all comfortable | GO |

States interact (e.g. cold/extreme heat reduces effective battery; nighttime
GPS/sensor degradation) so features must genuinely be combined. The oracle
also defines the 27-combo **legality table** used for the format-consistency
penalty and the step-consistency metric.

## 3. Architecture and size

| Module | Shape | Params |
| --- | --- | --- |
| Encoder | Dense(10,32) → ReLU → Dense(32,32) → ReLU | 1,408 |
| Step head k | Dense(32+3k, 16) → ReLU → Dense(16, 3) | 3×(579/627/675) |
| Final head | Dense(32+9, 16) → ReLU → Dense(16, 4) | 740 |
| Verifier encoder | Dense(10,24) → ReLU → Dense(24,24) | 864 |
| Verifier head | Dense(24+4, 16) → ReLU → Dense(16, 1) | 481 |
| **Total** | | **5,374 parameters** (target: < 10K) |

~15.0K multiply-add FLOPs per decision, 10.4 KiB peak memory, pure NumPy
forward+backward.

## 4. How journey learning works

Each training example gets 1 valid trajectory + 1–2 corrupted ones, corrupted
by three styles: `random_flip` (random wrong state), `neighbour` (a plausible
adjacent-class mistake), and `shortcut` (final action ignores the reasoning).
A corruption strategy (`neighbour`) is **held out of training** so the
verifier must generalise to unseen corruption styles. Corrupted journeys are
the only training data where "this step is wrong" labels exist — they are
what makes a verifier learnable at all.

```
L = a·CE(final) + b·Σₖ CE(step_k) + c·BCE(verifier) + d·L_format
```

* `CE(final)`, `CE(step_k)` — correct chain on valid rows only.
* `BCE(verifier)` — **this is where incorrect paths are learned from**: the
  verifier sees each *presented* step state and must label it valid (1) or
  corrupted (0); it is trained on **every** journey row (label-0 rows are
  upweighted, `verifier_neg_weight: 8`, to counter low corruption counts).
  The verifier has its **own tiny encoder** (`Dense(10,24) → Dense(24,24)`),
  so the corrupted-path supervision shapes the verifier without competing
  with the decision encoder for capacity; and it trains to its own
  convergence budget (`verifier_epochs: 300`, batch 256).
* `L_format` — penalises probability mass on (steps, action) combos that
  violate the rule table; the gradient flows into the final head **and all
  three step heads** (derived through the outer-product structure, not a
  stop-gradient shortcut). It also acts on corrupted rows, giving the step
  heads ~2.5× more supervision than B2's valid rows alone.

Defaults: a = b = 1, c = 4, d = 0.2; decode weight λ = 1.0.

**Inference: predict → check → decide (PRD 5.2).** Each step k is decoded
with a fused score over candidate states c:

```
score(c) = log p_step(c | h, prev) + λ · log validity(c | h_v, k)
```

so a state the verifier deems unlikely *given the features* is re-decoded
before the final head sees it (λ = `eval.verifier_decode_weight`; λ = 0
recovers plain argmax decoding). Steps whose selected state still scores
below `verifier_threshold` are flagged low-confidence and fall back to HOLD.

## 5. From-scratch implementation notes

* `nn/layers.py`: `Dense`/`ReLU` with explicit cached-input backward passes
  (gradients *accumulate* because multiple loss terms flow through the shared
  encoder).
* `nn/losses.py`: log-sum-exp-stable softmax cross-entropy, sigmoid BCE with
  row weighting, and the L_format penalty with hand-derived gradients (incl.
  einsum contractions through `p1 ⊗ p2 ⊗ p3`).
* `nn/optim.py`: Adam and SGD-momentum, hand-written.
* **Numerical gradient check** (`tests/test_gradients.py`): finite differences
  on *every* parameter of a small model instance for both loss configurations
  — max relative error ~1e-7. This is the proof that the backprop is correct.
* A tiny-batch overfit test proves the model can drive CE → ~0.

## 6. Results

All numbers from `reports/results.md`, produced by the seeded end-to-end
pipeline (`make train && make eval`). Default configuration: **2,000 labeled
decision examples** / 5K val / 3×5K test vectors; every train example also
carries 1–2 corrupted journeys (≈ 5K training trajectories). 2,000 labels is
the deployment-realistic regime for a decision model like this — and, as the
sensitivity table below shows, exactly the regime where journey learning pays.

**Final decision accuracy (%)** — Test A clean / Test B noisy / Test C shifted:

| Model | A acc | A F1 | B acc | B F1 | C acc | C F1 |
| --- | --- | --- | --- | --- | --- | --- |
| **M (full)** | 87.2 | 86.2 | 86.6 | 85.6 | **88.6** | 86.7 |
| B2 (steps, no journey) | 70.2 | 66.4 | 70.7 | 67.0 | 73.1 | 67.1 |
| B1 (shortcut MLP) | 71.9 | 69.7 | 72.1 | 70.1 | 75.0 | 69.7 |
| B0 (hand heuristic) | 46.2 | 23.5 | 45.9 | 22.4 | 54.7 | 24.2 |

**Journey-learning delta on Test C (M − B2): +15.5 points — target ≥ +5
exceeded 3× over.** Stable across seeds (dataset regeneration + full
retraining every time):

| Seed | M Test C | B2 Test C | Δ (M − B2) | Verifier AUC |
| --- | --- | --- | --- | --- |
| 0 | 88.6 | 73.1 | **+15.5** | 0.938 |
| 1 | 89.2 | 73.7 | **+15.4** | 0.915 |
| 2 | 88.1 | 75.9 | **+12.2** | 0.937 |

Mean Δ = **+14.4 points**, worst-seed Δ = **+12.2**; verifier AUC > 0.9 on
every seed (including the held-out `neighbour` corruption style, 0.929) —
**both PRD §6.3 journey-learning gates clear.** M's step predictions are also
more consistent: step-consistency 97.8–98.6% on B2's ~86% (its final-head
decisions obey the rule table for its own steps), and per-step state accuracy
86–93% vs B2's 70–82%.

**Where the benefit comes from — both mechanisms are journey learning:**

1. **Corrupted journeys are free state-supervision.** M trains on ~2.5× as
   many rows as B2 (valid + corrupted counterfactuals), and `L_format` acts
   on all of them, so M's step heads and encoder learn the feature→state
   mapping and the rule table from far richer signal under the same
   budget — visible in the per-step accuracies above.
2. **Verifier-guided decode (predict → check → decide).** The verifier is
   trained *only* because corrupted journeys provide wrong-step labels. At
   inference it scores every candidate state per step and the decode
   maximises `log p_step + λ·log validity` (PRD 5.2). Ablating the decode
   (λ = 0) drops the Test C delta to +13.7/+12.4/+7.2 (seeds 0/1/2) — still
   above target, the verifier adds a further +1.5 to +3 points.

**Sensitivity to label count (seed 0, same budget):** the advantage is a
property of *scarce labels*, not a tuning artefact:

| n_train | M Test C | B2 Test C | Δ (M − B2) | Verifier AUC |
| --- | --- | --- | --- | --- |
| 2,000 | 88.6 | 73.1 | **+15.5** | 0.938 |
| 5,000 | 94.0 | 89.2 | +4.8 | 0.983 |
| 20,000 | 97.3 | 95.6 | +1.6 | 0.997 |
| 50,000 | 97.4 | 97.4 | +0.0 | 0.998 |

As valid labels rise, B2 alone eventually learns the whole mapping and the
final-accuracy gap converges to zero — while the verifier keeps getting
better (0.938 → 0.998) because corrupted rows keep growing. Journey
learning's final-accuracy value is concentrated where it matters: scarce,
expert-labeled decisions.

**Verifier quality** (guide for the corrupted-step-detection objective):

| Metric | Value |
| --- | --- |
| AUC (all held-out corrupted styles) | **0.938** |
| AUC on held-out-style `neighbour` corruption | 0.929 |
| Precision / Recall @ 0.5 | 0.948 / 0.934 |

Low-confidence flagging (HOLD fallback) rates: test A 8%, B 9%, C 16%.

## 7. Compute-cost comparison (vs an LLM alternative)

| Metric | SmallAI (M) | gpt-4o-mini proxy |
| --- | --- | --- |
| Parameters / tokens per decision | 5,374 (≈42 KiB float64) | ~500 tokens assumed |
| FLOPs per decision | 15.0K multiply-adds | (not modelled) |
| Latency p50 / p95 (CPU, single decision) | **0.248 / 0.257 ms** | 900 / 2,200 ms assumed |
| Throughput (vectorised batch 256) | ~258K decisions/s | — |
| Peak memory, single decision | 10.4 KiB | — |
| Cost per 1M decisions | $0 (local compute) | **$111.00** ($0.15/M input · $0.60/M output) |

Token assumption: ~420-token prompt with the feature table + ~80-token
structured chain-of-thought answer. Pricing reference: OpenAI published
pricing (gpt-4o-mini). The LLM is never called at inference; this is an
analytical proxy (B3).

## 8. Limitations and future work

* The benefit is concentrated in the scarce-label regime (>= +12 points at
  2K labels, ~0 at 50K). If an application can afford tens of thousands of
  labeled decisions, the shortcut model matches M's final accuracy — M still
  provides the verifier, consistency and trace benefits.
* The detection/decode mechanism was validated on a deterministic synthetic
  oracle; real decision logs would need calibrated validity scores (the
  threshold is fixed at 0.5) and a domain-realistic HOLD cost model.
* The verifier threshold gating trades accuracy for caution (HOLD prior is
  low); use policy-aware calibration before deploying it as a guardrail.
* Stretch goals (not done): from-scratch decision tree with rule extraction;
  tabular Q-learning on a sequential variant; a Streamlit trace viewer.

## 9. Reproduce

```bash
make setup     # venv + pip install -e ".[dev]"
make test      # 22 unit tests: oracle, gradient check, shapes, determinism
make data      # generate dataset (deterministic, seed from config)
make train     # trains M (journey), B2 (no journey), B1 (shortcut)
make eval      # metrics table + verifier + confusion matrices -> reports/
make benchmark # latency p50/p95 over 10K decisions, FLOPs, memory, LLM proxy
```

Everything is seeded (`--seed`, config `seed: 0`); a clean clone reproduces
the exact dataset, weights and result tables. Runs cache to `data/` and
`runs/`; use `--force` to regenerate.

```
smallai-decision-model/
├── configs/default.yaml     # every hyperparameter, no magic numbers in code
├── src/smallai/
│   ├── env/oracle.py        # deterministic solver, state/action spec, legality
│   ├── env/generate.py      # valid + corrupted trajectory generation (journeys)
│   ├── nn/layers.py         # Dense, ReLU (from scratch)
│   ├── nn/losses.py         # masked CE, weighted BCE, L_format + gradients
│   ├── nn/optim.py          # Adam / SGD-momentum
│   ├── model.py             # step-wise model with verifier + inference trace
│   ├── train.py             # journey-learning loop (Adam, early stop)
│   ├── evaluate.py          # accuracy/F1/steps/verifier/metrics harness
│   ├── baselines.py         # B0 heuristic, B1 shortcut MLP, B3 LLM proxy
│   ├── runs.py              # experiment orchestration + weight cache
│   └── benchmark.py         # latency, FLOPs, memory, cost proxy
├── tests/                   # gradient check, oracle, shapes, determinism
└── reports/                 # results.md/csv, loss curves, benchmark.md/json
```

## 9. Predict: arbitrary inputs

```bash
echo '[[85,4,18,0.9,0.85,0.2,10,0.95,10,0.3],
        [30,12,35,0.7,0.8,2,25,0.9,9,0.7]]' > /tmp/feats.json
python -m smallai predict --input /tmp/feats.json
```

Each row is one feature vector in the PRD order
`[battery_pct, wind_mps, temp_c, gps_fix_quality, link_signal, payload_kg,
distance_km, sensor_health, hour_of_day, terrain_difficulty]` and the output
is the full decision: predicted step states + names, final action, per-step
verifier validity, low-confidence flag, trace string, and final-probability
distribution.

```json
{
  "features": [85.0, 4.0, 18.0, 0.9, 0.85, 0.2, 10.0, 0.95, 10.0, 0.3],
  "states": [0, 0, 0],
  "state_names": ["SURPLUS", "LOW", "RELIABLE"],
  "action": 0,
  "action_name": "GO",
  "validities": [0.9999, 0.9999, 0.9999],
  "low_confidence": false,
  "trace": "Energy: SURPLUS (valid 1.00) -> Environment: LOW (valid 1.00) -> Nav: RELIABLE (valid 1.00) -> Decision: GO",
  "final_probs": [0.9977, 0.0022, 2.8e-06, 8.2e-06]
}
```
