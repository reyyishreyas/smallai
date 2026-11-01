"""Oracle unit tests (PRD Task 1): formulas, rule table, legality table,
determinism."""
from __future__ import annotations

import numpy as np

from smallai.env import oracle


def row(**kwargs) -> np.ndarray:
    """Build one feature vector by name, defaults chosen to be comfortable."""
    defaults = {
        "battery_pct": 90.0, "wind_mps": 5.0, "temp_c": 20.0,
        "gps_fix_quality": 0.9, "link_signal": 0.9, "payload_kg": 1.0,
        "distance_km": 5.0, "sensor_health": 0.9, "hour_of_day": 12.0,
        "terrain_difficulty": 0.2,
    }
    defaults.update(kwargs)
    return np.array([[defaults[n] for n in oracle.FEATURE_NAMES]])


def test_energy_shortfall_with_low_battery_and_long_distance():
    states, _action = oracle.solve(row(battery_pct=20.0, distance_km=30.0, payload_kg=4.0))
    assert states[0, 0] == oracle.SHORTFALL


def test_energy_surplus_with_full_battery_and_short_hop():
    states, _ = oracle.solve(row(battery_pct=100.0, distance_km=2.0, payload_kg=0.5))
    assert states[0, 0] == oracle.SURPLUS


def test_wind_and_cold_drive_environment_risk_high():
    states, _ = oracle.solve(row(wind_mps=25.0, temp_c=-5.0))
    assert states[0, 1] == oracle.HIGH


def test_calm_warm_conditions_keep_risk_low():
    states, _ = oracle.solve(row(wind_mps=3.0, temp_c=15.0))
    assert states[0, 1] == oracle.LOW


def test_poor_comms_make_navigation_unreliable():
    states, _ = oracle.solve(row(gps_fix_quality=0.1, link_signal=0.1, sensor_health=0.1))
    assert states[0, 2] == oracle.UNRELIABLE


def test_action_rule_table():
    # HIGH risk or UNRELIABLE nav -> ABORT regardless of energy
    _, a = oracle.solve(row(gps_fix_quality=0.05, link_signal=0.05, sensor_health=0.05,
                            battery_pct=10.0, distance_km=1.0))
    assert a[0] == oracle.ABORT
    # SHORTFALL energy, benign risk/nav -> HOLD
    _, a = oracle.solve(row(battery_pct=10.0, distance_km=30.0, payload_kg=4.0))
    assert a[0] == oracle.HOLD
    # comfortable everything -> GO
    _, a = oracle.solve(row())
    assert a[0] == oracle.GO
    # degraded nav but otherwise fine -> GO_REDUCED_PAYLOAD
    _, a = oracle.solve(row(gps_fix_quality=0.45, link_signal=0.45, sensor_health=0.45))
    assert a[0] == oracle.GO_REDUCED


def test_action_matches_legality_table():
    table = oracle.build_legality_table()
    rng = np.random.default_rng(0)
    combos = rng.integers(0, 3, size=(200, 3))
    combos = np.unique(combos, axis=0)
    actions = oracle.legal_action_for_steps(combos)
    for combo, act in zip(combos, actions):
        idx = combo[0] * 9 + combo[1] * 3 + combo[2]
        assert table[idx].tolist() == (np.arange(4) == act).tolist()
        assert table[idx].sum() == 1  # exactly one legal action per combo


def test_solve_is_deterministic():
    x = np.repeat(row(), 20, axis=0)
    x += np.linspace(0, 0.001, 20)[:, None]  # tiny perturbations, still deterministic
    s1, a1 = oracle.solve(x)
    s2, a2 = oracle.solve(x)
    assert np.array_equal(s1, s2) and np.array_equal(a1, a2)


def test_all_state_names_and_action_names():
    assert len(oracle.STEP_STATE_NAMES) == 3
    assert all(len(v) == 3 for v in oracle.STEP_STATE_NAMES.values())
    assert len(oracle.ACTION_NAMES) == 4
    assert oracle.ACTION_NAMES[oracle.GO] == "GO"


def test_trace_string_format():
    s = oracle.trace_string(np.array([1, 2, 0]), int(oracle.GO_REDUCED),
                            [0.97, 0.93, 0.99])
    assert s.startswith("Energy: MARGINAL (valid 0.97)")
    assert "Decision: GO_REDUCED_PAYLOAD" in s
