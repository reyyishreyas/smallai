# AI-267 — Response: answering all questions / acceptance criteria

**To:** Victor Romero (AI/ML Engineering lead) / MIFOS team
**Re:** [SmallAI] Build a lightweight, non-LLM Decision Model from scratch with multi-step reasoning capabilities — AI-267

---

## 1. Objective — Answered: Yes ✅

We developed a **lightweight, compute-efficient, and interpretable Decision Model from scratch**, explicitly avoiding LLM / transformer-based architectures. It is capable of **multi-step decision-making** via traditional SmallAI paradigms (shallow neural network + intermediate state evaluation).

## 2. Background & Context — Answered: Yes ✅

We adapted the core ideas from reasoning LLMs (DeepSeek-R1 / OpenAI o1, and Sebastian Raschka's write-up on reasoning models — *Understanding Reasoning LLMs*):

- multi-step evaluation,
- pure RL on small models (TinyZero-style),
- **journey learning** (learning from both correct and incorrect paths),
- intermediate verification steps,

…into a much smaller, non-LLM architecture.

## 3. Scope — Addressed ✅

| Scope item | Status |
|---|---|
| Problem definition (decision env / dataset / sequential process) | ✅ Deterministic UAV pre-flight Go/No-Go decision environment |
| From-scratch implementation (NumPy / base framework, no sklearn / xgboost black-box for core) | ✅ Pure NumPy from scratch |
| Multi-step logic (intermediate eval steps) | ✅ 3 intermediate states + final decision |
| Journey learning integration (train on optimal + suboptimal paths) | ✅ Valid + corrupted journeys |
| Evaluation (accuracy, latency, cost vs baseline / LLM proxy) | ✅ Full eval harness + benchmark |

## 4. Out of Scope — Respected ✅

- No training / fine-tuning / distilling of an LLM or transformer. ✅
- No external LLM API in the inference path. ✅
- No large-scale compute infrastructure (runs on CPU / minimal edge hardware). ✅

## 5. Acceptance Criteria — Status

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | **Architecture**: from-scratch, strictly non-LLM, < 100K params (or symbolic/rule-based) | ✅ | **5,374 parameters** (pure NumPy, from scratch) |
| 2 | **Multi-Step Reasoning**: intermediate states/steps before final output | ✅ | 3 intermediate states (energy, env risk, nav/comms) → final action; verifier-checked |
| 3 | **Journey Learning**: learns from negative paths; validation accuracy ≥ X% over shortcut-learning baseline | ✅ | **+15.54 pp** over B2 (M 88.60% vs B2 73.06%) on shifted Test C; **target ≥ +5 exceeded** (multi-seed mean +14.4) |
| 4 | **Performance**: inference < 10 ms per decision on standard CPU | ✅ | **p95 = 0.257 ms** |
| 5 | **Documentation**: README with architecture, from-scratch details, cost comparison vs LLM | ✅ | README + `reports/` + benchmark report |
| 6 | **Code Review**: core algorithm reviewed & approved by ML Engineering lead | ⬜ | **Pending — awaiting Victor's review/approval** |

## 6. Technical Notes & Suggestions — Addressed ✅

| Suggestion | Our implementation |
|---|---|
| Lightweight RL: Tabular Q / tiny DQN (< 3 layers) | We used a shallow NumPy MLP + journey learning + verifier (chosen after exhaustive experiments) |
| Interpretable ML: custom decision tree / RuleFit, explicit path | We used a small MLP with **named intermediate states** (interpretability via explicit state semantics, not tree extraction) |
| Probabilistic: Bayesian network for step-wise conditionals | We used **probabilistic decode**: `log p_step(c) + λ · log validity(c)` |
| Reward / Loss design: deterministic loss penalizing bad final decisions **and** invalid intermediate steps | Loss = `a·CE(final) + b·ΣCE(step) + c·BCE(verifier) + d·L_format` |
| Data generation: deterministic solver → synthetic trajectories (non-LLM distillation analog) | ✅ Deterministic oracle → valid + corrupted journey generation |

## 7. References — Answered: Yes ✅

- Primary: *Understanding Reasoning LLMs* — Sebastian Raschka.
- Key takeaways adapted: pure RL on small budgets, journey learning, intermediate verification.

## 8. Implementation Checklist — Status

| Task | Description | Status |
|---|---|---|
| Task 1 | Define decision environment, state space, action space | ✅ Done — deterministic UAV pre-flight oracle (10 features → 3 states → action) |
| Task 2 | Implement from-scratch model architecture (custom MLP / RL agent) | ✅ Done — pure NumPy MLP (encoder → step heads → final head) + private verifier encoder |
| Task 3 | Build training pipeline with journey learning (negative paths) | ✅ Done — corrupted journeys + fused decode + verifier BCE |
| Task 4 | Create evaluation scripts (accuracy, latency, intermediate step validation) | ✅ Done — full eval harness, verifier AUC, step validation, benchmark |
| Task 5 | Write documentation and performance comparison report | ✅ Done — README, reports, benchmark vs LLM proxy |

---

## Final numbers (to cite in the ticket)

- Journey-learning delta on Test C (M vs B2): **+15.54 pp** (M 88.60% vs B2 73.06%) — **exceeds the +5 pt target**.
- Verifier AUC: **0.938** (gate: > 0.9 — **cleared**).
- Model size: **5,374 params** (< 10K / < 100K).
- Latency: **p95 = 0.257 ms** (< 10 ms).
- FLOPs: **~15K**.

---

Want me to turn this into the actual **Jira comment** for AI-267, or adjust the tone/length before sending?
