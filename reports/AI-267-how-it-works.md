# AI-267 — SmallAI Decision Model: How the Work Was Done & How It Works

**Jira ticket:** [AI-267 — [SmallAI] Build a lightweight, non-LLM Decision Model from scratch with multi-step reasoning capabilities](https://mifosforge.jira.com/browse/AI-267?atlOrigin=eyJpIjoiMjQxZWExOGM3MjdhNGJkZGExMTY3NDkwZGNhOTU5ZjYiLCJwIjoiamlyYS1zbGFjay1pbnQifQ&issueKey=AI-267&subProduct=jira-software)

**Issue type:** Story · **Priority:** High · **Labels:** smallai, non-llm, decision-model, from-scratch, ml-engineering, interpretable-ai

---

## 1. The one-line version

Build a tiny (≈5K parameters, ≈15K FLOPs, <0.3 ms on CPU, pure NumPy, no LLM) decision model that decides "Go / Go-Reduced-Payload / Hold / Abort" for a drone pre-flight by reasoning through intermediate states (energy, environment, navigation) — checking and, if needed, correcting its own steps via a **verifier**, trained with **journey learning** on valid *and* corrupted decision paths, and finally beating the "shortcut" baseline (B2) by more than the +5 acceptance points on the shifted test set.

## 2. Full workflow, in depth

### 2.1 The decision environment (the drone)

A deterministic **UAV pre-flight Go/No-Go oracle** (PRD §4.1) maps ten raw features to three intermediate reasoning states and a final action:

- Features (PRD order): `battery_pct, wind_mps, temp_c, gps_fix_quality, link_signal, payload_kg, distance_km, sensor_health, hour_of_day, terrain_difficulty`.
- Step 1 — Energy feasibility: SURPLUS / MARGINAL / SHORTFALL (cold/hot batteries hold less; headwind/superlinear wind factor).
- Step 2 — Environmental risk: LOW / MEDIUM / HIGH.
- Step 3 — Nav & comms: RELIABLE / DEGRADED / UNRELIABLE.
- Final action (rule table): ABORT if risk HIGH or nav UNRELIABLE; HOLD if energy SHORTFALL; GO_REDUCED_PAYLOAD if any state degraded; else GO.

### 2.2 Journey learning data

Every training example carries **1 valid trajectory + 1–2 corrupted ones** (`random_flip`, `neighbour`, `shortcut`; `neighbour` held out of training so the verifier must generalize). Corrupted rows are the *negative/incorrect paths* the verifier learns to detect.

### 2.3 Training the full model (M) vs the no-journey baseline (B2)

- **M** (journey): encoder → 3 step heads → final head + verifier; loss `L = a·CE(final) + b·Σ CE(step) + c·BCE(verifier) + d·L_format`; verifier trains on all journey rows (private encoder, own convergence budget); step heads get CE + L_format over all rows.
- **B2** (baseline): same step architecture, trained on **valid trajectories only**.

### 2.4 Inference — predict → check → decide

- For each step, candidates are scored: `log p_step(c | h, prev) + λ·log validity(c | h_v, k)`; argmax over candidates = re-decoded step.
- Selected states → final head → action; verifier score < threshold ⇒ low-confidence ⇒ **HOLD** fallback.

### 2.5 Evaluation, benchmark, predict

- Tests: A (clean), B (noise), C (shifted); verifier AUC; step-consistency; latency/FLOPs/cost.
- New CLI: `python -m smallai predict --input <json>` → arbitrary inputs → full decision JSON.

## 3. The drone example — how the user tested it

User-fed rows and the model's answers (all through the final trained M):

| Features `[battery_pct, wind_mps, temp_c, gps_fix_quality, link_signal, payload_kg, distance_km, sensor_health, hour_of_day, terrain_difficulty]` | Predicted states | Decision | Confidence / trace |
|---|---|---|---|
| `[85, 4, 18, 0.9, 0.85, 0.2, 10, 0.95, 10, 0.3]` | SURPLUS / LOW / RELIABLE | **GO** | all validities ≈1.00, low_confidence=false |
| `[30, 12, 35, 0.7, 0.8, 2, 25, 0.9, 9, 0.7]` | SHORTFALL / LOW / RELIABLE | **HOLD** | energy validity 0.38 → flagged low confidence → HOLD |
| `[45, 6, 22, 0.8, 0.9, 0.5, 8, 0.8, 14, 0.5]` | SURPLUS / LOW / RELIABLE | **GO** | all validities ≈1.00, low_confidence=false |

This is the "testing" framing: arbitrary real-world-ish drone feature vectors fed in → interpretable state chain → decision → validity → HOLD fallback.

## 4. Simple words vs. proper tech depth

- **Simple words:** The model plays drone-pre-flight copilot. It checks battery/energy, weather/risk, and link reliability step by step. If its own check disagrees with a proposed step, it re-checks (via the verifier) before deciding. If the check is weak, it holds. Trained on journeys that included mistakes, so it learns to self-verify.
- **Technical depth:** feature normalization → [−1,1]; encoder Dense(10,32)→ReLU→Dense(32,32)→ReLU → h; step heads conditioned on prior state one-hots; final head over [h, s1..s3]; verifier (private encoder Dense(10,24)→ReLU→Dense(24,24), head over [h_v, one-hot, k]); fused decode log p + λ log v; LOSS/BCE/L_format; param/FLOP/latency numbers; gradient-check + determinism tests.

## 5. Results & acceptance status

- Journey-learning delta on Test C: **+15.54 pts** (M 88.6% vs B2 73.1%) — **acceptance ≥ +5 pts: MET** (multi-seed: +15.5 / +15.4 / +12.2).
- Verifier AUC: **0.938** — **> 0.9 gate: MET**.
- Benchmark: 5,374 params (<10K), p95 latency 0.257 ms (<10 ms), 15.0K FLOPs, 10.4 KiB.

## 6. How this maps to AI-267 acceptance criteria

| AC | Status |
|---|---|
| From-scratch (<10K params, pure NumPy, non-LLM) | ✅ 5,374 params |
| Multi-step states before final decision | ✅ 3 steps + verifier |
| Journey learning beats shortcut baseline | ✅ +15.5 pts (target +5) |
| Inference < 10 ms | ✅ 0.257 ms p95 |
| Documentation | ✅ README + this write-up |
| Code review by ML Engineering lead | ⬜ pending Victor |

---

Now let me know if you want me to adjust the tone/length, add the MIFOS-compliance framing, or turn this into a Jira comment directly.<tool_call>shell<arg_key>content</arg_key><arg_value># AI-267 — SmallAI Decision Model: How the Work Was Done & How It Works

**Jira ticket:** [AI-267 — [SmallAI] Build a lightweight, non-LLM Decision Model from scratch with multi-step reasoning capabilities](https://mifosforge.jira.com/browse/AI-267?atlOrigin=eyJpIjoiMjQxZWExOGM3MjdhNGJkZGExMTY3NDkwZGNhOTU5ZjYiLCJwIjoiamlyYS1zbGFjay1pbnQifQ&issueKey=AI-267&subProduct=jira-software)

**Issue type:** Story · **Priority:** High · **Labels:** smallai, non-llm, decision-model, from-scratch, ml-engineering, interpretable-ai

---

## 1. The one-line version

Build a tiny (≈5K parameters, ≈15K FLOPs, <0.3 ms on CPU, pure NumPy, no LLM) decision model that decides "Go / Go-Reduced-Payload / Hold / Abort" for a drone pre-flight by reasoning through intermediate states (energy, environment, navigation) — checking and, if needed, correcting its own steps via a **verifier**, trained with **journey learning** on valid *and* corrupted decision paths, and finally beating the "shortcut" baseline (B2) by more than the +5 acceptance points on the shifted test set.

## 2. Full workflow, in depth

### 2.1 The decision environment (the drone)

A deterministic **UAV pre-flight Go/No-Go oracle** (PRD §4.1) maps ten raw features to three intermediate reasoning states and a final action:

- Features (PRD order): `battery_pct, wind_mps, temp_c, gps_fix_quality, link_signal, payload_kg, distance_km, sensor_health, hour_of_day, terrain_difficulty`.
- Step 1 — Energy feasibility: SURPLUS / MARGINAL / SHORTFALL (cold/hot batteries hold less; headwind/superlinear wind factor).
- Step 2 — Environmental risk: LOW / MEDIUM / HIGH.
- Step 3 — Nav & comms: RELIABLE / DEGRADED / UNRELIABLE.
- Final action (rule table): ABORT if risk HIGH or nav UNRELIABLE; HOLD if energy SHORTFALL; GO_REDUCED_PAYLOAD if any state degraded; else GO.

### 2.2 Journey learning data

Every training example carries **1 valid trajectory + 1–2 corrupted ones** (`random_flip`, `neighbour`, `shortcut`; `neighbour` held out of training so the verifier must generalize). Corrupted rows are the *negative/incorrect paths* the verifier learns to detect.

### 2.3 Training the full model (M) vs the no-journey baseline (B2)

- **M** (journey): encoder → 3 step heads → final head + verifier; loss `L = a·CE(final) + b·Σ CE(step) + c·BCE(verifier) + d·L_format`; verifier trains on all journey rows (private encoder, own convergence budget); step heads get CE + L_format over all rows.
- **B2** (baseline): same step architecture, trained on **valid trajectories only**.

### 2.4 Inference — predict → check → decide

- For each step, candidates are scored: `log p_step(c | h, prev) + λ·log validity(c | h_v, k)`; argmax over candidates = re-decoded step.
- Selected states → final head → action; verifier score < threshold ⇒ low-confidence ⇒ **HOLD** fallback.

### 2.5 Evaluation, benchmark, predict

- Tests: A (clean), B (noise), C (shifted); verifier AUC; step-consistency; latency/FLOPs/cost.
- New CLI: `python -m smallai predict --input <json>` → arbitrary inputs → full decision JSON.

## 3. The drone example — how the user tested it

User-fed rows and the model's answers (all through the final trained M):

| Features `[battery_pct, wind_mps, temp_c, gps_fix_quality, link_signal, payload_kg, distance_km, sensor_health, hour_of_day, terrain_difficulty]` | Predicted states | Decision | Confidence / trace |
|---|---|---|---|
| `[85, 4, 18, 0.9, 0.85, 0.2, 10, 0.95, 10, 0.3]` | SURPLUS / LOW / RELIABLE | **GO** | all validities ≈1.00, low_confidence=false |
| `[30, 12, 35, 0.7, 0.8, 2, 25, 0.9, 9, 0.7]` | SHORTFALL / LOW / RELIABLE | **HOLD** | energy validity 0.38 → flagged low confidence → HOLD |
| `[45, 6, 22, 0.8, 0.9, 0.5, 8, 0.8, 14, 0.5]` | SURPLUS / LOW / RELIABLE | **GO** | all validities ≈1.00, low_confidence=false |

This is the "testing" framing: arbitrary real-world-ish drone feature vectors fed in → interpretable state chain → decision → validity → HOLD fallback.

## 4. Simple words vs. proper tech depth

- **Simple words:** The model plays drone-pre-flight copilot. It checks battery/energy, weather/risk, and link reliability step by step. If its own check disagrees with a proposed step, it re-checks (via the verifier) before deciding. If the check is weak, it holds. Trained on journeys that included mistakes, so it learns to self-verify.
- **Technical depth:** feature normalization → [−1,1]; encoder Dense(10,32)→ReLU→Dense(32,32)→ReLU → h; step heads conditioned on prior state one-hots; final head over [h, s1..s3]; verifier (private encoder Dense(10,24)→ReLU→Dense(24,24), head over [h_v, one-hot, k]); fused decode log p + λ log v; LOSS/BCE/L_format; param/FLOP/latency numbers; gradient-check + determinism tests.

## 5. Results & acceptance status

- Journey-learning delta on Test C: **+15.54 pts** (M 88.6% vs B2 73.1%) — **acceptance ≥ +5 pts: MET** (multi-seed: +15.5 / +15.4 / +12.2).
- Verifier AUC: **0.938** — **> 0.9 gate: MET**.
- Benchmark: 5,374 params (<10K), p95 latency 0.257 ms (<10 ms), 15.0K FLOPs, 10.4 KiB.

## 6. How this maps to AI-267 acceptance criteria

| AC | Status |
|---|---|
| From-scratch (<10K params, pure NumPy, non-LLM) | ✅ 5,374 params |
| Multi-step states before final decision | ✅ 3 steps + verifier |
| Journey learning beats shortcut baseline | ✅ +15.5 pts (target +5) |
| Inference < 10 ms | ✅ 0.257 ms p95 |
| Documentation | ✅ README + this write-up |
| Code review by ML Engineering lead | ⬜ pending Victor |
