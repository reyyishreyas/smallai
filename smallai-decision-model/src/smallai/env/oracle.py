"""Deterministic UAV pre-flight Go/No-Go oracle (PRD Task 1, section 4.1).

The oracle is the ground truth of the synthetic environment. It maps a raw
feature vector to three intermediate reasoning states (energy feasibility,
environmental risk, navigation/comms reliability) and a final decision, using
explicit formulas and thresholds plus a rule table. Every trajectory
therefore has a known correct answer, so valid and corrupted training paths
can be generated for free.

Feature order (10 features):
  battery_pct, wind_mps, temp_c, gps_fix_quality, link_signal, payload_kg,
  distance_km, sensor_health, hour_of_day, terrain_difficulty

Reasoning steps:
  1. Energy feasibility : SURPLUS / MARGINAL / SHORTFALL
  2. Environmental risk : LOW / MEDIUM / HIGH
  3. Nav & comms        : RELIABLE / DEGRADED / UNRELIABLE
  4. Final decision     : GO / GO_REDUCED_PAYLOAD / HOLD / ABORT
"""
from __future__ import annotations

import numpy as np

N_FEATURES = 10
FEATURE_NAMES = [
    "battery_pct", "wind_mps", "temp_c", "gps_fix_quality", "link_signal",
    "payload_kg", "distance_km", "sensor_health", "hour_of_day",
    "terrain_difficulty",
]
FEATURE_MIN = np.array([0.0, 0.0, -20.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
FEATURE_MAX = np.array([100.0, 30.0, 45.0, 1.0, 1.0, 5.0, 40.0, 1.0, 23.0, 1.0])

# Step state encodings
SURPLUS, MARGINAL, SHORTFALL = 0, 1, 2
LOW, MEDIUM, HIGH = 0, 1, 2
RELIABLE, DEGRADED, UNRELIABLE = 0, 1, 2

# Final actions
GO, GO_REDUCED, HOLD, ABORT = 0, 1, 2, 3

STEP_NAMES = ["energy", "environment", "navigation"]
STEP_STATE_NAMES = {
    0: ["SURPLUS", "MARGINAL", "SHORTFALL"],
    1: ["LOW", "MEDIUM", "HIGH"],
    2: ["RELIABLE", "DEGRADED", "UNRELIABLE"],
}
ACTION_NAMES = ["GO", "GO_REDUCED_PAYLOAD", "HOLD", "ABORT"]
N_STEPS = 3
N_STATES_PER_STEP = 3
N_ACTIONS = 4

# Oracle thresholds (set on empirical quantiles so every state/action class
# has enough support for learning and macro-F1)
ENERGY_SURPLUS_RATIO = 1.9    # ratio >= this -> SURPLUS
ENERGY_MARGINAL_RATIO = 1.0   # ratio in [1.0, 1.9) -> MARGINAL, below -> SHORTFALL
RISK_HIGH = 0.615             # risk >= this -> HIGH
RISK_LOW = 0.42               # risk < this -> LOW, between -> MEDIUM
NAV_RELIABLE = 0.465          # score >= this -> RELIABLE
NAV_DEGRADED = 0.375          # score in [0.375, 0.465) -> DEGRADED, below -> UNRELIABLE


def normalize(x: np.ndarray) -> np.ndarray:
    """Map raw features to roughly [-1, 1] for the network."""
    x = np.asarray(x, dtype=np.float64)
    return 2.0 * (x - FEATURE_MIN) / (FEATURE_MAX - FEATURE_MIN) - 1.0


def derived(x: np.ndarray) -> dict[str, np.ndarray]:
    """Intermediate physical quantities shared by the step formulas."""
    x = np.asarray(x, dtype=np.float64)
    battery = np.clip(x[:, 0], 0.0, 100.0) / 100.0
    wind = np.clip(x[:, 1] / 30.0, 0.0, 1.0)
    temp = x[:, 2]
    gps = np.clip(x[:, 3], 0.0, 1.0)
    link = np.clip(x[:, 4], 0.0, 1.0)
    payload = np.clip(x[:, 5] / 5.0, 0.0, 1.0)
    distance = np.clip(x[:, 6], 0.5, 40.0)
    sensor = np.clip(x[:, 7], 0.0, 1.0)
    hour = x[:, 8] % 24.0
    terrain = np.clip(x[:, 9], 0.0, 1.0)

    daylight = np.clip(np.sin(np.pi * (hour - 6.0) / 12.0), 0.0, 1.0)
    daylight = np.where((hour < 6.0) | (hour > 18.0), 0.0, daylight)
    # Cold or extreme heat hurts battery efficiency (feature interaction)
    temp_stress = np.clip(np.abs(temp - 15.0) / 60.0, 0.0, 1.0)
    return {
        "battery": battery, "wind": wind, "gps": gps, "link": link,
        "payload": payload, "distance": distance, "sensor": sensor,
        "hour": hour, "terrain": terrain, "daylight": daylight,
        "temp_stress": temp_stress,
    }


def energy_state(x: np.ndarray) -> np.ndarray:
    """Step 1: is required energy (distance, payload, wind) within battery?"""
    d = derived(x)
    # Cold/hot batteries hold less charge
    eff = 1.0 - 0.35 * d["temp_stress"]
    # Headwind makes rotors work harder (superlinear)
    wind_factor = 1.0 + 1.6 * d["wind"] * (0.4 + 0.6 * d["wind"])
    required = d["distance"] * (1.0 + 0.45 * d["payload"]) * wind_factor * (1.0 + 0.9 * d["terrain"])
    required *= 0.16  # scale to the battery's energy-unit budget
    available = 42.0 * d["battery"] * eff
    ratio = available / np.maximum(required, 1e-9)
    return np.where(ratio >= ENERGY_SURPLUS_RATIO, SURPLUS,
                    np.where(ratio >= ENERGY_MARGINAL_RATIO, MARGINAL, SHORTFALL))


def energy_ratio(x: np.ndarray) -> np.ndarray:
    """Diagnostic used by tests: how close is required energy to battery budget."""
    d = derived(x)
    eff = 1.0 - 0.35 * d["temp_stress"]
    wind_factor = 1.0 + 1.6 * d["wind"] * (0.4 + 0.6 * d["wind"])
    required = 0.16 * d["distance"] * (1.0 + 0.45 * d["payload"]) * wind_factor * (1.0 + 0.9 * d["terrain"])
    available = 42.0 * d["battery"] * eff
    return available / np.maximum(required, 1e-9)


def environment_state(x: np.ndarray) -> np.ndarray:
    """Step 2: wind and temperature combined risk."""
    d = derived(x)
    risk = np.clip(0.65 * d["wind"] + 0.35 * d["temp_stress"], 0.0, 1.0)
    return np.where(risk >= RISK_HIGH, HIGH, np.where(risk >= RISK_LOW, MEDIUM, LOW))


def navigation_state(x: np.ndarray) -> np.ndarray:
    """Step 3: GPS quality + link strength + sensor health."""
    d = derived(x)
    gps_eff = d["gps"] * (0.9 + 0.1 * d["daylight"])   # GPS slightly worse at night
    link_eff = d["link"] * (1.0 - 0.1 * d["payload"])  # heavy payload hurts the link
    sensor_eff = d["sensor"] * (0.85 + 0.15 * d["daylight"])  # night missions stress sensors
    score = 0.4 * gps_eff + 0.35 * link_eff + 0.25 * sensor_eff
    return np.where(score >= NAV_RELIABLE, RELIABLE,
                    np.where(score >= NAV_DEGRADED, DEGRADED, UNRELIABLE))


def final_action(s1: np.ndarray, s2: np.ndarray, s3: np.ndarray) -> np.ndarray:
    """Rule table mapping the three step states to the final action.

    Priority order:
      ABORT   if environmental risk is HIGH or nav is UNRELIABLE (safety first)
      HOLD    if energy is a SHORTFALL (wait to recharge / re-plan)
      GO_REDUCED if any condition is degraded (marginal energy, medium risk,
                 degraded nav)
      GO      otherwise
    """
    s1 = np.asarray(s1, dtype=np.int64)
    s2 = np.asarray(s2, dtype=np.int64)
    s3 = np.asarray(s3, dtype=np.int64)
    return np.where(
        (s2 == HIGH) | (s3 == UNRELIABLE), ABORT,
        np.where(
            s1 == SHORTFALL, HOLD,
            np.where(
                (s1 == MARGINAL) | (s2 == MEDIUM) | (s3 == DEGRADED),
                GO_REDUCED, GO,
            ),
        ),
    )


def solve(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Oracle solution for a batch of raw feature vectors.

    Returns (states, action): states has shape (N, 3) with one int per step,
    action has shape (N,)."""
    s1 = energy_state(x)
    s2 = environment_state(x)
    s3 = navigation_state(x)
    states = np.stack([s1, s2, s3], axis=1)
    return states.astype(np.int64), final_action(s1, s2, s3).astype(np.int64)


def legal_action_for_steps(states: np.ndarray) -> np.ndarray:
    """The (only) action consistent with the rule table for given step states.
    Used for the L_format consistency penalty and the step-consistency metric."""
    states = np.asarray(states, dtype=np.int64)
    return final_action(states[:, 0], states[:, 1], states[:, 2])


def build_legality_table() -> np.ndarray:
    """A (27, 4) boolean table: is each action legal for each step-state combo?
    Combo index = s1 * 9 + s2 * 3 + s3. Each combo has exactly one legal action."""
    table = np.zeros((N_STATES_PER_STEP ** N_STEPS, N_ACTIONS), dtype=bool)
    for c1 in range(3):
        for c2 in range(3):
            for c3 in range(3):
                idx = c1 * 9 + c2 * 3 + c3
                table[idx, final_action(np.array([c1]), np.array([c2]), np.array([c3]))[0]] = True
    return table


def trace_string(states: np.ndarray, action: int, validities: list[float]) -> str:
    """Human-readable decision trace, e.g.
    'Energy: MARGINAL (valid 0.97) -> Environment: HIGH (valid 0.93) -> ...'"""
    parts = []
    for k in range(N_STEPS):
        name = STEP_STATE_NAMES[k][int(states[k])]
        short = {"energy": "Energy", "environment": "Environment", "navigation": "Nav"}[STEP_NAMES[k]]
        parts.append(f"{short}: {name} (valid {validities[k]:.2f})")
    return " -> ".join(parts) + f" -> Decision: {ACTION_NAMES[int(action)]}"


def sample_any_combo(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random uniform step-state combination (N, 3). Used by corruption code."""
    return rng.integers(0, 3, size=(n, 3))
