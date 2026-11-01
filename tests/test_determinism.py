"""Determinism tests (PRD "definition of done"): identical seeds must give
identical data and identical trained weights."""
from __future__ import annotations

import numpy as np

from smallai import config as cfg_mod
from smallai.env import generate as gen
from smallai.model import SmallAIModel
from smallai.train import train_model


def cfg() -> cfg_mod.Config:
    c = cfg_mod.Config()
    c.data = cfg_mod.DataConfig(n_train=150, n_val=40, n_test_clean=40,
                                n_verifier_eval=36)
    c.train = cfg_mod.TrainConfig(epochs=3, batch_size=64)
    return c


def test_dataset_generation_is_deterministic():
    a = gen.generate(cfg().data, seed=5)
    b = gen.generate(cfg().data, seed=5)
    for name in ("val_x", "val_states", "val_action", "test_a_x", "test_b_x"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    t1, t2 = a.train, b.train
    for name in ("x", "states", "action", "is_valid", "matches", "first_bad", "strategy"):
        assert np.array_equal(getattr(t1, name), getattr(t2, name)), name


def test_dataset_generation_changes_with_seed():
    a = gen.generate(cfg().data, seed=1)
    b = gen.generate(cfg().data, seed=2)
    assert not np.array_equal(a.test_a_x, b.test_a_x)


def test_training_is_deterministic():
    c = cfg()
    c.train.epochs = 5
    ds_a = gen.generate(c.data, seed=c.seed)
    ds_b = gen.generate(c.data, seed=c.seed)

    m1 = SmallAIModel(c.model, np.random.default_rng(c.seed))
    m2 = SmallAIModel(c.model, np.random.default_rng(c.seed))
    train_model(m1, ds_a.train, ds_a.val_x, ds_a.val_states, ds_a.val_action,
                c.train, seed=c.seed)
    train_model(m2, ds_b.train, ds_b.val_x, ds_b.val_states, ds_b.val_action,
                c.train, seed=c.seed)
    for p1, p2 in zip(m1.all_params(), m2.all_params()):
        assert np.array_equal(p1, p2)


def test_verifier_training_is_deterministic():
    from smallai.train import train_verifier
    c = cfg()
    c.train = cfg_mod.TrainConfig(epochs=3, batch_size=64,
                                  verifier_epochs=20, verifier_batch_size=64)
    ds_a = gen.generate(c.data, seed=c.seed)
    ds_b = gen.generate(c.data, seed=c.seed)
    m1 = SmallAIModel(c.model, np.random.default_rng(c.seed))
    m2 = SmallAIModel(c.model, np.random.default_rng(c.seed))
    train_verifier(m1, ds_a.train, c.train, seed=c.seed)
    train_verifier(m2, ds_b.train, c.train, seed=c.seed)
    for p1, p2 in zip(m1.verifier_encoder.params() + m1.verifier_head.params(),
                      m2.verifier_encoder.params() + m2.verifier_head.params()):
        assert np.array_equal(p1, p2)
