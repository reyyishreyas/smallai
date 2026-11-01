"""Journey-learning training loop (PRD Task 3, section 5.3).

Trains the step model on valid rows (final + step cross-entropy) and — with
journey learning enabled — on corrupted rows as well (verifier BCE +
L_format). Includes early stopping on validation accuracy with best-weight
snapshotting, all seeded for full reproducibility.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import config as cfg_mod
from .env import generate as gen
from .env import oracle
from .model import SmallAIModel, TrainBatch
from .nn import losses, optim


@dataclass
class History:
    """Per-epoch training record."""
    epochs: list[dict] = field(default_factory=list)

    def add(self, **entry: Any) -> None:
        self.epochs.append(entry)

    @property
    def losses(self) -> list[float]:
        return [e["train_loss"] for e in self.epochs]

    @property
    def val_accs(self) -> list[float]:
        return [e["val_acc"] for e in self.epochs]


def train_model(model: SmallAIModel, table: gen.JourneyTable,
                val_x: np.ndarray, val_states: np.ndarray,
                val_action: np.ndarray, cfg_train: cfg_mod.TrainConfig,
                seed: int, logger: Callable[[str], None] | None = None) -> History:
    """Train with Adam + early stopping on validation accuracy. `table` holds
    all journey rows; when model.journey is False only valid rows are used,
    which yields the B2 (steps, no journey learning) setup."""
    rng = np.random.default_rng(seed + 1)
    opt = optim.make_optim(model.all_params(), "adam", cfg_train.lr)
    history = History()
    batch_size = cfg_train.batch_size

    rows = np.arange(table.x.shape[0])
    if not model.journey:
        rows = rows[table.is_valid.astype(bool)]

    best_val = -1.0
    best_epoch = -1
    best_snap = model.snapshot()
    stale = 0

    for epoch in range(cfg_train.epochs):
        perm = rng.permutation(rows.shape[0])
        losses_epoch = []
        for start in range(0, rows.shape[0], batch_size):
            idx = rows[perm[start:start + batch_size]]
            batch = TrainBatch(
                x=oracle.normalize(table.x[idx]),
                states=table.states[idx],
                action=table.action[idx],
                matches=table.matches[idx],
                is_valid=table.is_valid[idx].astype(bool),
            )
            loss = model.loss_and_grads(batch, {
                "a": cfg_train.weight_a,
                "b": cfg_train.weight_b,
                "c": cfg_train.weight_c,
                "d": cfg_train.weight_d,
                "verifier_neg_weight": cfg_train.verifier_neg_weight,
            })
            opt.step(model.all_grads())
            losses_epoch.append(loss)
        train_loss = float(np.mean(losses_epoch))

        pred_states, pred_action = model.predict_chain(val_x)
        val_acc = float((pred_action == val_action).mean())
        val_step_acc = float((pred_states == val_states).mean())
        history.add(epoch=epoch, train_loss=train_loss, val_acc=val_acc,
                    val_step_acc=val_step_acc)
        if logger:
            logger(f"epoch {epoch:3d}: loss {train_loss:.4f} "
                   f"val_acc {val_acc:.4f} val_step_acc {val_step_acc:.4f}")

        if val_acc > best_val + 1e-12:
            best_val, best_epoch, stale = val_acc, epoch, 0
            best_snap = model.snapshot()
        else:
            stale += 1
            if stale >= cfg_train.early_stop_patience:
                if logger:
                    logger(f"early stop at epoch {epoch} (best epoch {best_epoch})")
                break

    model.restore(best_snap)
    if logger:
        logger(f"best val acc {best_val:.4f} at epoch {best_epoch}")
    return history


def train_verifier(model: SmallAIModel, table: gen.JourneyTable,
                   cfg_train: cfg_mod.TrainConfig, seed: int,
                   logger: Callable[[str], None] | None = None) -> float:
    """Train the verifier head + its own encoder on **all** journey rows.

    Per PRD 5.3 the BCE(verifier) loss applies to every row: presented steps
    matching the oracle are labelled 1, corrupted ones 0. The verifier has a
    private encoder, so this pass is independent of (and cannot disturb) the
    decision heads trained by `train_model`. It runs to a fixed epoch budget
    (`verifier_epochs`) with its own batch size; deterministic per seed.
    Returns the final mean BCE loss."""
    rng = np.random.default_rng(seed + 40)
    params = model.verifier_encoder.params() + model.verifier_head.params()
    opt = optim.make_optim(params, "adam", cfg_train.lr)
    rows = np.arange(table.x.shape[0])          # every row carries validity labels
    bs = cfg_train.verifier_batch_size
    loss = float("nan")
    for epoch in range(cfg_train.verifier_epochs):
        perm = rng.permutation(rows.shape[0])
        losses_epoch = []
        for start in range(0, rows.shape[0], bs):
            idx = rows[perm[start:start + bs]]
            model.zero_grads()
            x = oracle.normalize(table.x[idx])
            h_v = model.verifier_encoder.forward(x)
            d_hv = np.zeros_like(h_v)
            total = 0.0
            for k in range(oracle.N_STEPS):
                inp = model._verifier_input(h_v, table.states[idx, k], k)
                logits = model.verifier_head.forward(inp)
                bce, dv = losses.bce_with_logits(
                    logits, table.matches[idx, k],
                    neg_row_weight=cfg_train.verifier_neg_weight)
                total += bce
                model.verifier_head.forward(inp)   # restore caches
                grad_in = model.verifier_head.backward(cfg_train.weight_c * dv)
                d_hv += grad_in[:, :model.hv]
            model.verifier_encoder.backward(d_hv)
            opt.step(model.verifier_encoder.grads() + model.verifier_head.grads())
            losses_epoch.append(total)
            loss = float(np.mean(losses_epoch))
        if logger and (epoch % 25 == 0 or epoch == cfg_train.verifier_epochs - 1):
            logger(f"verifier epoch {epoch:3d}: bce {loss:.4f}")
    return loss
