"""Experiment orchestration: generate/load datasets, train a model kind,
persist weights + history to runs/ (PRD section 9)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as cfg_mod
from .baselines import ShortcutMLP
from .env import generate as gen
from .model import SmallAIModel
from .train import History, train_model, train_verifier

MODEL_KINDS = ["M", "B2", "B1"]


def data_path(root: Path) -> Path:
    return Path(root) / "data" / "dataset.npz"


def load_or_generate_dataset(cfg: cfg_mod.Config, root: Path, logger=None) -> gen.Dataset:
    path = data_path(root)
    if path.exists():
        if logger:
            logger(f"loading cached dataset from {path}")
        return gen.load_dataset(path)
    if logger:
        logger("generating dataset ...")
    ds = gen.generate(cfg.data, cfg.seed)
    gen.save_dataset(ds, path)
    if logger:
        logger(f"saved dataset to {path}")
    return ds


def build_model(cfg: cfg_mod.Config, kind: str):
    rng = np.random.default_rng(cfg.seed + 100 + {"M": 0, "B2": 1, "B1": 2}[kind])
    if kind == "B1":
        model = ShortcutMLP(cfg.model, rng)
        model.journey = False  # shortcut has no verifier
        return model
    model = SmallAIModel(cfg.model, rng)
    model.journey = kind != "B2"
    model.decode_weight = cfg.eval.verifier_decode_weight
    return model


def run_kind(cfg: cfg_mod.Config, root: Path, kind: str,
             dataset: gen.Dataset | None = None, logger=None):
    """Train (or load) one model kind and return it."""
    run_dir = Path(root) / "runs" / kind
    weights = run_dir / "model.npz"
    model = build_model(cfg, kind)

    if weights.exists():
        if logger:
            logger(f"{kind}: loading trained weights from {weights}")
        _load_weights(model, weights)
        return model

    ds = dataset or load_or_generate_dataset(cfg, root, logger)
    if logger:
        if kind == "B1":
            logger(f"{kind}: training shortcut MLP (encoder -> action) ... "
                   f"params={model.param_count()}")
            model.fit(ds.train.x[ds.train.is_valid.astype(bool)],
                      ds.train.action[ds.train.is_valid.astype(bool)],
                      cfg.train, cfg.seed)
        else:
            n_valid = int(ds.train.is_valid.sum())
            logger(f"{kind}: training on {n_valid + int((~ds.train.is_valid.astype(bool)).sum())}"
                   f" trajectory rows ({n_valid} valid, "
                   f"{int((~ds.train.is_valid.astype(bool)).sum())} corrupted) ... "
                   f"params={model.param_count()}")
            history = train_model(model, ds.train, ds.val_x, ds.val_states,
                                  ds.val_action, cfg.train, cfg.seed, logger=logger)
            if kind == "M":
                train_verifier(model, ds.train, cfg.train, cfg.seed, logger=logger)
            _save_run(run_dir, model, history, cfg, kind)
            return model
    _save_run(run_dir, model, History(), cfg, kind)
    return model


def _save_run(run_dir: Path, model, history: History, cfg, kind: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "model.npz",
                        **{f"p{i}": p for i, p in enumerate(model.snapshot())})
    (run_dir / "history.json").write_text(json.dumps(history.epochs, indent=2))
    (run_dir / "meta.json").write_text(json.dumps({
        "kind": kind, "param_count": model.param_count(),
        "seed": cfg.seed, "journey_learning": kind != "B2",
    }, indent=2))


def _load_weights(model, path: Path) -> None:
    z = np.load(path)
    keys = sorted(z.files, key=lambda k: int(k[1:]))
    for p, k in zip(model.all_params(), keys):
        p[...] = z[k]


def load_or_train(cfg: cfg_mod.Config, root: Path, kind: str, logger=None):
    ds = load_or_generate_dataset(cfg, root, logger)
    return run_kind(cfg, root, kind, dataset=ds, logger=logger)
