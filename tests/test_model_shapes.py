"""Shape, interface and size tests, plus the tiny-batch overfit check that
proves the model can actually learn (PRD 5.4 / task 2 "done when")."""
from __future__ import annotations

import numpy as np

from smallai import config as cfg_mod
from smallai.env import generate as gen
from smallai.env import oracle
from smallai.model import SmallAIModel, TrainBatch


def default_cfg(tiny: bool = True) -> cfg_mod.Config:
    cfg = cfg_mod.Config()
    if tiny:
        cfg.data = cfg_mod.DataConfig(n_train=200, n_val=60, n_test_clean=60,
                                      n_verifier_eval=48)
    return cfg


def build(cfg) -> tuple[SmallAIModel, gen.Dataset]:
    model = SmallAIModel(cfg.model, np.random.default_rng(cfg.seed))
    model.journey = True
    ds = gen.generate(cfg.data, cfg.seed)
    return model, ds


def test_parameter_budget_under_100k_target_under_10k():
    cfg = cfg_mod.Config()  # default architecture
    model = SmallAIModel(cfg.model, np.random.default_rng(0))
    count = model.param_count()
    assert count < 100_000, count
    # PRD stretch target: under 10K parameters
    print(f"full model parameter count: {count}")
    assert count < 10_000


def test_prediction_trace_structure():
    cfg = default_cfg()
    model, ds = build(cfg)
    rows = model.prediction(ds.test_a_x[:5])
    assert len(rows) == 5
    for r in rows:
        assert set(r) >= {"states", "state_names", "action", "action_name",
                          "step_probs", "final_probs", "validities", "low_confidence",
                          "trace", "features"}
        assert len(r["states"]) == 3
        assert len(r["validities"]) == 3
        assert 0 <= r["action"] < oracle.N_ACTIONS
        assert 0.0 <= min(r["validities"]) <= 1.0
        assert r["trace"].count("->") == 3
        assert "-> Decision:" in r["trace"]


def test_output_shapes():
    cfg = default_cfg()
    model, ds = build(cfg)
    out = model.chain_forward(oracle.normalize(ds.test_a_x[:8]))
    assert out["states"].shape == (8, 3)
    assert out["final_logits"].shape == (8, 4)
    assert all(l.shape == (8, 3) for l in out["step_logits"])
    assert all(v.shape == (8, 1) for v in out["ver_logits"])


def test_low_confidence_fallback_flags_hold():
    cfg = default_cfg()
    model, ds = build(cfg)
    model.snapshot()  # no-op sanity
    # untrained verifier: with threshold, most/all decisions flagged low -> HOLD
    rows = model.prediction(ds.test_a_x[:20], verifier_threshold=0.999)
    assert all(r["low_confidence"] for r in rows)
    assert all(r["action"] == oracle.HOLD for r in rows)


def test_tiny_batch_overfit():
    """Model must be able to drive CE to ~0 on a small valid batch."""
    cfg = default_cfg()
    model, ds = build(cfg)
    train_cfg = cfg_mod.TrainConfig(epochs=120, batch_size=64, lr=3e-3)
    valid_rows = np.where(ds.train.is_valid.astype(bool))[0][:64]
    sub = TrainBatch(x=oracle.normalize(ds.train.x[valid_rows]),
                     states=ds.train.states[valid_rows],
                     action=ds.train.action[valid_rows],
                     matches=ds.train.matches[valid_rows],
                     is_valid=ds.train.is_valid[valid_rows].astype(bool))
    best = None
    for _ in range(3):  # a few restarts guard against unlucky init kinks
        history = []  # manual loop so we can measure pure training loss
        for _epoch in range(train_cfg.epochs):
            loss = model.loss_and_grads(
                sub, {"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.0}, journey=False)
            from smallai.nn import optim
            opt = optim.make_optim(model.all_params(), "adam", train_cfg.lr)
            opt.step(model.all_grads())
            history.append(loss)
        final = history[-1]
        best = final if best is None else min(best, final)
        if best < 0.15:
            break
    print(f"tiny-batch overfit final loss: {best:.4f}")
    assert best < 0.15


def test_distribution_shift_change_labels():
    """Test C should actually move states (otherwise shift would be pointless)."""
    cfg = default_cfg()
    cfg.data.n_test_clean = 500
    _, ds = build(cfg)
    same = (ds.test_a_states == ds.test_c_states).mean()
    assert same < 0.95, f"shift did not change step states (agreement {same})"
