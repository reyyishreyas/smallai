# AI-267 — Response: Build & Validation of the SmallAI Decision Model

**From:** SmallAI ML Engineering
**To:** Victor Romero (AI/ML Engineering lead) / MIFOS team
**Ticket:** AI-267 — [SmallAI] Build a lightweight, non-LLM Decision Model from scratch with multi-step reasoning capabilities
**Jira:** https://mifosforge.jira.com/browse/AI-267

---

## 1. Executive Summary

The SmallAI Decision Model has been built from scratch — pure NumPy, strictly non-LLM — and delivered with multi-step reasoning, journey learning, and self-verification. On the shifted evaluation set it surpassed the ticket's journey-learning acceptance target while remaining well inside the compute, latency, and cost budgets.

**Headline result:** Journey-learning delta on Test C — **+15.54 percentage points** (Model **M**: 88.60% vs. baseline **B2**: 73.06%) — against a ticket target of **>= +5 points**. Verifier AUC: **0.938** (gate: > 0.9 — cleared).

---

## 2. Objective — Requirement & Answer

**Requirement:** Develop a lightweight, compute-efficient, and interpretable Decision Model from scratch, explicitly avoiding LLM / transformer architectures, and capable of multi-step decision-making or intermediate state evaluation, inspired by reasoning concepts but implemented via traditional SmallAI paradigms (shallow NN, decision tree, Bayesian network, or lightweight tabular RL).

**Answer:** Delivered.

- **Lightweight / compute-efficient / interpretable:** 5,374 parameters, ~15K FLOPs, p95 latency **0.257 ms** on CPU.
- **Strictly non-LLM:** pure Python / NumPy from scratch; no black-box ML library in the core path; no LLM API in inference.
- **Multi-step decision-making:** 10 raw drone pre-flight features -> 3 intermediate reasoning states (energy, environmental risk, navigation/comms) -> final action, with intermediate verification.

---

## 3. Background & Context — Requirement & Answer

**Context:** Reasoning LLMs (DeepSeek-R1, OpenAI o1) show value in intermediate steps + RL ("journey learning"), but are compute-expensive and opaque.

**Answer:** The same ideas were adapted to small non-LLM architectures:

- Pure RL on small budgets (TinyZero-style) -> shallow MLP.
- Journey learning (learning from mistakes) -> corrupted + valid journeys.
- Intermediate verification steps -> dedicated verifier + fused decode.

---

## 4. Scope Delivered

| Scope item | Answer |
|---|---|
| Problem definition | Deterministic UAV pre-flight decision environment |
| From-scratch implementation | Pure NumPy (layers, losses, optim, model) |
| Multi-step logic | 3 intermediate states -> final decision |
| Journey learning integration | Train on valid + corrupted trajectories |
| Evaluation | Accuracy / latency / step validation / cost vs. baselines & LLM proxy |

---

## 5. Out of Scope — Maintained

- No LLM training / fine-tuning / distillation.
- No external LLM API in inference.
- No large-scale compute infrastructure.

---

## 6. Acceptance Criteria — Detailed Mapping

### AC1 — Architecture (from-scratch, non-LLM, < 100K params / symbolic)

- **Status:** Met.
- Pure Python / NumPy from scratch.
- Strictly non-LLM.
- **5,374 parameters** (< 100K and < 10K internal budget).

### AC2 — Multi-Step Reasoning (intermediate states before final output)

- **Status:** Met.
- Model outputs / utilizes intermediate decision states before the final action (structured analogue of chain-of-thought).

### AC3 — Journey Learning (learns from negative paths; accuracy >= X% over shortcut-learning baseline)

- **Status:** Met (with agreed X = +5).
- Pipeline explicitly includes and learns from negative / incorrect decision paths.
- **Journey delta on Test C: +15.54 pp** (M 88.60% vs. B2 73.06%) - **exceeds +5 target**.
- Verifier trained on corrupted journeys; fused decode `log p_step + lambda * log validity`.

### AC4 — Performance (< 10 ms per decision on standard CPU)

- **Status:** Met.
- **p95 latency: 0.257 ms**; ~15K FLOPs; 10.4 KiB.

### AC5 — Documentation (README with architecture, from-scratch details, cost comparison vs. LLM)

- **Status:** Met.
- README, architecture docs, benchmark report, results.

### AC6 — Code Review (core algorithm reviewed & approved by ML Engineering lead)

- **Status:** Pending - awaiting Victor's review / approval.

---

## 7. Design & Technical Approach

- **Environment:** deterministic UAV pre-flight oracle (10 features -> 3 states -> final action).
- **Model (M):** encoder -> step heads -> final head + verifier.
- **Verifier:** private encoder + head over hidden / candidate / step context.
- **Fused decode:** `log p_step(c) + lambda * log validity(c, k)`; argmax -> re-decoded step; final head -> action; low validity -> HOLD fallback.
- **Loss:** `a*CE(final) + b*sumCE(step) + c*BCE(verifier) + d*L_format`.

---

## 8. Evaluation & Results

- **Journey delta (Test C):** +15.54 pp (M 88.60% vs. B2 73.06%).
- **Verifier AUC:** 0.938 (precision 0.9475, recall 0.9336 @ 0.5).
- **Benchmark:** 5,374 params, p95 0.257 ms, ~15K FLOPs, 10.4 KiB; LLM proxy cost $111/M.

---

## 9. Documentation & Deliverables

- `README.md` (sections 4 / 6 / 9), `reports/results.md`, `reports/benchmark.md`, `AI-267-how-it-works.md`, `ARCHITECTURE.md`, `EXPERIMENT.md`.

---

## 10. Implementation Checklist

| Task | Status |
|---|---|
| 1 - Env / state / action space | Done |
| 2 - From-scratch architecture | Done |
| 3 - Journey learning training | Done |
| 4 - Eval scripts | Done |
| 5 - Documentation & comparison | Done |

---

## 11. Open Items

- AC6 code review approval (Victor).
- Final sign-off on acceptance framing.

---

## 12. References

- *Understanding Reasoning LLMs* - Sebastian Raschka.

---

## 13. Consolidated Deliverable Index

The following project documentation files were produced / updated as part of AI-267 and map onto the sections above:

- `smallai-decision-model/README.md` - architecture, from-scratch implementation, benchmark, predict CLI, results.
- `smallai-decision-model/reports/results.md` - accuracy / metrics tables.
- `smallai-decision-model/reports/benchmark.md` - latency / FLOPs / cost.
- `smallai-decision-model/reports/AI-267-how-it-works.md` - full workflow write-up.
- `ARCHITECTURE.md` - system architecture.
- `EXPERIMENT.md` - experiment record.
- `AI-267_STAGE_A_RESULTS.md`, `AI-267_STAGE_A1_RESULTS.md` - staged acceptance reports.
- `EXPERIMENT_A1.md` - Stage A.1 experiment record.
- Supporting artifacts: `reports/journey_q_a1.json`, `reports/poc_a1.json`, `reports/poc_a1.md` - machine-readable / supporting results.

---

**Status:** Acceptance (>= +5 pts journey delta) **cleared** - **Verdict:** Met.
