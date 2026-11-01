"""Numerical gradient check (PRD 5.4): finite differences on every parameter
of a small instance of the full model — the proof that the hand-written
backprop is correct."""
from __future__ import annotations

import numpy as np
import pytest

from smallai import config as cfg_mod
from smallai.env import generate as gen
from smallai.model import SmallAIModel, TrainBatch


def tiny_cfg() -> cfg_mod.Config:
    cfg = cfg_mod.Config()
    cfg.model = cfg_mod.ModelConfig(hidden_enc=6, hidden_head=5, hidden_verifier=4)
    cfg.data.n_train = 64
    cfg.data.n_val = 16
    cfg.data.n_test_clean = 16
    cfg.data.n_verifier_eval = 16
    return cfg


def make_batch(n: int, seed: int, cfg: cfg_mod.Config):
    ds = gen.generate(cfg.data, seed)
    table = ds.train
    rows = np.concatenate([
        np.where(table.is_valid.astype(bool))[0][:n // 2],
        np.where(~table.is_valid.astype(bool))[0][:n - n // 2],
    ])[:n]
    return TrainBatch(
        x=gen.oracle.normalize(table.x[rows]),
        states=table.states[rows],
        action=table.action[rows],
        matches=table.matches[rows],
        is_valid=table.is_valid[rows].astype(bool),
    )


EPS = 1e-6


@pytest.mark.parametrize("journey", [False, True])
def test_gradient_check_all_params(journey: bool):
    cfg = tiny_cfg()
    model = SmallAIModel(cfg.model, np.random.default_rng(3))
    model.journey = journey
    batch = make_batch(10, seed=0, cfg=cfg)
    weights = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.2}

    model.loss_and_grads(batch, weights)
    grad_list = [g.copy() for g in model.all_grads()]  # snapshot before probing
    param_list = model.all_params()
    assert len(param_list) == len(grad_list) and len(param_list) > 0

    def loss_value() -> float:
        loss = model.loss_and_grads(batch, weights)
        model.zero_grads()  # keep grad buffers clean between probes
        return loss

    worst = 0.0
    for p, g in zip(param_list, grad_list):
        for it in np.ndindex(p.shape):
            orig = p[it]
            p[it] = orig + EPS
            f_plus = loss_value()
            p[it] = orig - EPS
            f_minus = loss_value()
            p[it] = orig
            numeric = (f_plus - f_minus) / (2 * EPS)
            analytic = g[it]
            denom = max(abs(numeric), abs(analytic), 1e-12)
            rel = abs(numeric - analytic) / denom
            worst = max(worst, rel)
            assert rel < 1e-5, (
                f"param {it}: analytic {analytic:.3e} vs numeric {numeric:.3e} "
                f"(rel err {rel:.2e}, journey={journey})")
    assert worst < 1e-5
