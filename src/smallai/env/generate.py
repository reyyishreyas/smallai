"""Valid and corrupted trajectory generation (PRD Task 1, sections 4.2-4.3).

Because the oracle is deterministic, every trajectory has a known correct
answer. A "journey" is either valid (all steps and the final action match the
oracle) or corrupted by one of several mistake styles:

  random_flip : a step state (or the final action) is replaced by a random
                different state
  neighbour   : a "plausible mistake" - a step state is replaced by an
                adjacent class (e.g. MEDIUM instead of HIGH)
  shortcut    : the final action ignores the reasoning steps entirely (a lazy
                answer sampled from a biased distribution)

The training table stores both valid rows and corrupted rows so the verifier
can learn from incorrect trajectories (journey learning).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import oracle

STRATEGY_VALID = 0
STRATEGY_RANDOM_FLIP = 1
STRATEGY_NEIGHBOUR = 2
STRATEGY_SHORTCUT = 3
STRATEGY_NAMES = ["valid", "random_flip", "neighbour", "shortcut"]
_CORRUPTION_CODES = {
    "random_flip": STRATEGY_RANDOM_FLIP,
    "neighbour": STRATEGY_NEIGHBOUR,
    "shortcut": STRATEGY_SHORTCUT,
}
ALL_CORRUPTION_CODES = [STRATEGY_RANDOM_FLIP, STRATEGY_NEIGHBOUR, STRATEGY_SHORTCUT]

# Biased proposal used by the "shortcut" corruption: the answer comes from
# habit, without reasoning about the steps.
_SHORTCUT_PROPOSAL = np.array([0.55, 0.30, 0.10, 0.05])  # GO, GO_REDUCED, HOLD, ABORT


@dataclass
class JourneyTable:
    """Flat training table: one row per trajectory (valid or corrupted)."""
    x: np.ndarray            # raw features (M, 10)
    states: np.ndarray       # presented step states (M, 3)
    action: np.ndarray       # presented final action (M,)
    is_valid: np.ndarray     # (M,) 1.0 valid, 0.0 corrupted
    matches: np.ndarray      # (M, 3) 1.0 where presented state == oracle state
    first_bad: np.ndarray    # (M,) first corrupted step 0..2, 3 = only final bad, -1 = valid
    strategy: np.ndarray     # (M,) strategy code


@dataclass
class Dataset:
    """Generated dataset bundle (train / val / testA / testB / testC / verifier)."""
    train: JourneyTable
    val_x: np.ndarray
    val_states: np.ndarray
    val_action: np.ndarray
    test_a_x: np.ndarray
    test_a_states: np.ndarray
    test_a_action: np.ndarray
    test_b_x: np.ndarray
    test_b_states: np.ndarray
    test_b_action: np.ndarray
    test_c_x: np.ndarray
    test_c_states: np.ndarray
    test_c_action: np.ndarray
    verifier_x: np.ndarray          # raw features of verifier rows
    verifier_states: np.ndarray     # presented states (rows, 3)
    verifier_matches: np.ndarray    # per-step validity labels (rows, 3)
    verifier_strategy: np.ndarray   # (rows,)


def sample_features(n: int, rng: np.random.Generator, mode: str,
                    cfg_data=None) -> np.ndarray:
    """Sample raw features. mode: clean | noisy | shifted."""
    width = oracle.FEATURE_MAX - oracle.FEATURE_MIN
    x = oracle.FEATURE_MIN + rng.random((n, oracle.N_FEATURES)) * width
    if mode == "noisy":
        sigma = np.array([cfg_data.noise_sigma[name] for name in oracle.FEATURE_NAMES])
        x = x + rng.normal(0.0, 1.0, size=x.shape) * sigma
        x[:, 8] = x[:, 8] % 24.0
        x = np.clip(x, oracle.FEATURE_MIN, oracle.FEATURE_MAX)
    elif mode == "shifted":
        shift = cfg_data.shift
        for j, name in enumerate(oracle.FEATURE_NAMES):
            if name not in shift:
                continue
            if name in ("distance_km",):       # multiplicative feature
                x[:, j] = x[:, j] * float(shift[name])
            else:
                x[:, j] = x[:, j] + float(shift[name])
        x = np.clip(x, oracle.FEATURE_MIN, oracle.FEATURE_MAX)
    return x


def _pick_action_excluding(true_action: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample an action from the biased proposal, excluding the true action."""
    n = true_action.shape[0]
    probs = np.tile(_SHORTCUT_PROPOSAL, (n, 1))
    probs[np.arange(n), true_action] = 0.0
    probs = probs / probs.sum(axis=1, keepdims=True)
    return np.array([rng.choice(4, p=probs[i]) for i in range(n)])


def corrupt(true_states: np.ndarray, true_action: np.ndarray,
            rng: np.random.Generator, strategy: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Corrupt a batch of trajectories.

    Returns (states, action, first_bad):
      states     (N, 3) presented (possibly corrupted) step states
      action     (N,)   presented final action
      first_bad  (N,)   0..2 = first corrupted step, 3 = only final corrupted
    """
    n = true_states.shape[0]
    states = true_states.copy()
    action = true_action.copy()
    first_bad = np.full(n, -1)

    if strategy == STRATEGY_RANDOM_FLIP:
        slot = rng.integers(0, 4, size=n)
        for s in range(3):
            sel = slot == s
            if not sel.any():
                continue
            # force a different state: +1 or +2 mod 3
            bump = rng.integers(1, 3, size=int(sel.sum()))
            states[sel, s] = (true_states[sel, s] + bump) % 3
            first_bad[sel] = s
        fin = slot == 3
        if fin.any():
            action[fin] = (true_action[fin] + rng.integers(1, 4, size=int(fin.sum()))) % 4
            first_bad[fin] = 3
    elif strategy == STRATEGY_NEIGHBOUR:
        # plausible mistake: move a step state by +-1 within its valid range
        slot = rng.integers(0, 3, size=n)
        for s in range(3):
            sel = slot == s
            if not sel.any():
                continue
            t = true_states[sel, s]
            down = t > 0
            up = t < 2
            into = np.zeros(t.shape, dtype=np.int64)
            both = down & up
            into[both] = rng.choice(np.array([-1, 1]), size=int(both.sum()))
            into[down & ~both] = -1
            into[up & ~both] = 1
            states[sel, s] = t + into
            first_bad[sel] = s
    elif strategy == STRATEGY_SHORTCUT:
        # steps stay true, but the final action ignores them (habit answer)
        action = _pick_action_excluding(true_action, rng)
        first_bad = np.full(n, 3)
    else:
        raise ValueError(f"unknown strategy {strategy}")

    # If a step was corrupted, optionally present the rule-table outcome of the
    # corrupted combo (a "consistent but wrong" final action).
    step_corrupted = first_bad < 3
    if step_corrupted.any():
        combo_action = oracle.legal_action_for_steps(states)
        recompute = step_corrupted & (rng.random(n) < 0.5)
        action[recompute] = combo_action[recompute]

    return states, action, first_bad


def _corrupt_group(true_states, true_action, codes, rng):
    """Apply per-row corruption codes to a batch. Returns corrupted copies."""
    out_s = true_states.copy()
    out_a = true_action.copy()
    out_f = np.full(true_states.shape[0], -1)
    for code in ALL_CORRUPTION_CODES:
        grp = codes == code
        if not grp.any():
            continue
        s_c, a_c, f_c = corrupt(true_states[grp], true_action[grp], rng, code)
        out_s[grp] = s_c
        out_a[grp] = a_c
        out_f[grp] = f_c
    return out_s, out_a, out_f


def _journey_table(n_examples: int, cfg_data, rng: np.random.Generator) -> JourneyTable:
    """Training table: 1 valid row + 1..corrupted_max corrupted rows per example."""
    x = sample_features(n_examples, rng, "clean")
    true_states, true_action = oracle.solve(x)

    names = list(cfg_data.corruption_probs.keys())
    probs = np.array([cfg_data.corruption_probs[s] for s in names], dtype=np.float64)
    prob_sum = probs.sum()
    probs = probs / (prob_sum if prob_sum > 0 else 1.0)
    # Hold out one corruption strategy from training so the verifier has to
    # generalise to an unseen corruption style.
    holdout = getattr(cfg_data, "holdout_strategy", "")
    if holdout in names:
        probs[names.index(holdout)] = 0.0
        probs = probs / probs.sum()

    rows = {
        "x": [x],
        "states": [true_states],
        "action": [true_action],
        "is_valid": [np.ones(n_examples)],
        "matches": [np.ones((n_examples, 3))],
        "first_bad": [np.full(n_examples, -1)],
        "strategy": [np.full(n_examples, STRATEGY_VALID)],
    }

    counts = rng.integers(1, cfg_data.corrupted_max + 1, size=n_examples)
    for count in (1, 2):
        sel = counts >= count
        n_sel = int(sel.sum())
        if n_sel == 0:
            continue
        codes = np.array([_CORRUPTION_CODES[names[i]]
                          for i in rng.choice(len(names), size=n_sel, p=probs)])
        c_states, c_action, c_first = _corrupt_group(
            true_states[sel], true_action[sel], codes, rng)
        rows["x"].append(x[sel])
        rows["states"].append(c_states)
        rows["action"].append(c_action)
        rows["is_valid"].append(np.zeros(n_sel))
        rows["matches"].append((c_states == true_states[sel]).astype(np.float64))
        rows["first_bad"].append(c_first)
        rows["strategy"].append(codes)

    return JourneyTable(
        x=np.concatenate(rows["x"]),
        states=np.concatenate(rows["states"]).astype(np.int64),
        action=np.concatenate(rows["action"]).astype(np.int64),
        is_valid=np.concatenate(rows["is_valid"]),
        matches=np.concatenate(rows["matches"]),
        first_bad=np.concatenate(rows["first_bad"]).astype(np.int64),
        strategy=np.concatenate([np.atleast_1d(a) for a in rows["strategy"]]),
    )


def verifier_rows(x: np.ndarray, true_states: np.ndarray, true_action: np.ndarray,
                  rng: np.random.Generator, n_corrupt: int) -> tuple[np.ndarray, np.ndarray,
                                                                     np.ndarray, np.ndarray]:
    """Verifier evaluation rows: corrupted trajectories built from `x` rows
    (plus their valid counterparts come from the caller if needed)."""
    n_corrupt = min(n_corrupt, x.shape[0])
    idx = rng.permutation(x.shape[0])[:n_corrupt]
    codes = rng.choice(len(ALL_CORRUPTION_CODES), size=n_corrupt)
    c_states, _c_action, _c_first = _corrupt_group(
        true_states[idx], true_action[idx],
        np.array(ALL_CORRUPTION_CODES)[codes], rng)
    matches = (c_states == true_states[idx]).astype(np.float64)
    return x[idx], c_states, matches, codes + 1


def generate(cfg_data, seed: int) -> Dataset:
    """Generate the full dataset bundle (deterministic given seed)."""
    rng = np.random.default_rng(seed)

    table = _journey_table(cfg_data.n_train, cfg_data, rng)

    val_x = sample_features(cfg_data.n_val, rng, "clean")
    val_states, val_action = oracle.solve(val_x)

    a_x = sample_features(cfg_data.n_test_clean, rng, "clean")
    a_states, a_action = oracle.solve(a_x)

    b_x = sample_features(cfg_data.n_test_clean, rng, "noisy", cfg_data)
    b_states, b_action = oracle.solve(b_x)

    c_x = sample_features(cfg_data.n_test_clean, rng, "shifted", cfg_data)
    c_states, c_action = oracle.solve(c_x)

    vx, vs, vm, vstrat = verifier_rows(a_x, a_states, a_action, rng, cfg_data.n_verifier_eval)

    return Dataset(
        train=table,
        val_x=val_x, val_states=val_states, val_action=val_action,
        test_a_x=a_x, test_a_states=a_states, test_a_action=a_action,
        test_b_x=b_x, test_b_states=b_states, test_b_action=b_action,
        test_c_x=c_x, test_c_states=c_states, test_c_action=c_action,
        verifier_x=vx, verifier_states=vs, verifier_matches=vm,
        verifier_strategy=vstrat,
    )


def save_dataset(ds: Dataset, path: Path) -> None:
    """Serialise all arrays into a single compressed npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for key in ("x", "states", "action", "is_valid", "matches", "first_bad", "strategy"):
        flat[f"train_{key}"] = getattr(ds.train, key)
    for name, arr in ds.__dict__.items():
        if name != "train":
            flat[name] = arr
    np.savez_compressed(path, **flat)


def load_dataset(path: Path) -> Dataset:
    z = np.load(path)
    train = JourneyTable(
        x=z["train_x"], states=z["train_states"], action=z["train_action"],
        is_valid=z["train_is_valid"], matches=z["train_matches"],
        first_bad=z["train_first_bad"], strategy=z["train_strategy"])
    return Dataset(
        train=train,
        val_x=z["val_x"], val_states=z["val_states"], val_action=z["val_action"],
        test_a_x=z["test_a_x"], test_a_states=z["test_a_states"],
        test_a_action=z["test_a_action"],
        test_b_x=z["test_b_x"], test_b_states=z["test_b_states"],
        test_b_action=z["test_b_action"],
        test_c_x=z["test_c_x"], test_c_states=z["test_c_states"],
        test_c_action=z["test_c_action"],
        verifier_x=z["verifier_x"], verifier_states=z["verifier_states"],
        verifier_matches=z["verifier_matches"], verifier_strategy=z["verifier_strategy"],
    )
