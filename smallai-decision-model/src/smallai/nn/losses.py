"""Losses with hand-derived gradients: masked cross-entropy, sigmoid+BCE,
the L_format consistency penalty, and a plain softmax backward helper."""
from __future__ import annotations

import numpy as np


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax (log-sum-exp trick)."""
    zmax = logits.max(axis=1, keepdims=True)
    shifted = logits - zmax
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def softmax(logits: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(logits))


def softmax_xent(logits: np.ndarray, targets: np.ndarray,
                 mask: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Mean softmax cross-entropy, optionally masked to a subset of rows.

    Returns (loss, d_logits); the gradient is already normalised and zero on
    masked rows."""
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    n = logits.shape[0]
    if mask is None:
        mask = np.ones(n, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    count = max(int(mask.sum()), 1)

    ls = log_softmax(logits)
    rows = np.arange(n)
    loss = -(mask * ls[rows, targets]).sum() / count
    probs = np.exp(ls)
    d_logits = np.zeros_like(logits)
    onehot = np.zeros_like(logits)
    onehot[rows, targets] = 1.0
    scale = (mask / count).astype(np.float64)
    d_logits = scale[:, None] * (probs - onehot)
    return float(loss), d_logits


def softmax_backward_from_probs(probs: np.ndarray, d_probs: np.ndarray) -> np.ndarray:
    """Backward through softmax given pre-scaled dL/dp -> dL/dlogits."""
    s = (d_probs * probs).sum(axis=1, keepdims=True)
    return probs * (d_probs - s)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def bce_with_logits(logits: np.ndarray, targets: np.ndarray,
                    mask: np.ndarray | None = None,
                    eps: float = 1e-7,
                    neg_row_weight: float = 1.0) -> tuple[float, np.ndarray]:
    """Weighted binary cross-entropy over masked rows; gradient dL/dlogits.

    `neg_row_weight` upweights rows whose target is 0 (e.g. corrupted steps,
    which are rare relative to valid ones); the loss is a weighted mean so the
    gradient stays correctly normalised."""
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64).reshape(logits.shape)
    n = logits.shape[0]
    if mask is None:
        mask = np.ones(n, dtype=bool)
    mask = np.asarray(mask, dtype=bool)

    p = sigmoid(logits)
    p = np.clip(p, eps, 1.0 - eps)
    # Work on the raveled 1-D forms throughout: broadcasting a (N,) mask
    # against a (N, 1) tensor silently produces an (N, N) matrix.
    loss_vec = -(targets.ravel() * np.log(p).ravel()
                 + (1.0 - targets.ravel()) * np.log1p(-p).ravel())
    row_w = mask.ravel().astype(np.float64) * np.where(
        targets.ravel() == 0.0, neg_row_weight, 1.0)
    total_w = max(row_w.sum(), 1e-12)
    loss = (row_w * loss_vec).sum() / total_w
    # per-row gradient scale, already normalised by the weighted row count
    d_logits = ((row_w / total_w).reshape(logits.shape)) * (p - targets)
    return float(loss), d_logits


def format_penalty(final_probs: np.ndarray,
                   step_probs: list[np.ndarray],
                   legality: np.ndarray,
                   weight: float) -> tuple[float, np.ndarray, list[np.ndarray]]:
    """L_format: penalise probability mass on illegal step-action combinations.

    `legality` is the (27, 4) table built by oracle.build_legality_table(); the
    combo index is s1*9 + s2*3 + s3. With p_combo = p1 ⊗ p2 ⊗ p3 and
    E_c = P(final action is the legal action of combo c), the loss is

        L = weight * mean_i [ 1 - Σ_c p_combo_ic * E_ic ]

    Unlike a naive stop-gradient version, the gradient is derived through the
    full outer product, so it flows into the final head AND all three step
    heads — pulling the step distributions themselves towards combinations
    whose rule-table outcome matches the final head.
    """
    p1, p2, p3 = step_probs
    n = p1.shape[0]
    combo = (p1[:, :, None, None] * p2[:, None, :, None] * p3[:, None, None, :])
    combo = combo.reshape(n, legality.shape[0])          # (N, 27), = p1 ⊗ p2 ⊗ p3
    E = (legality @ final_probs.T).T                     # (N, 27): prob legal(c) matches action
    consistency = (combo * E).sum(axis=1)
    loss = float(weight * np.mean(1.0 - consistency))

    scale = weight / n
    # final head: dconsistency/dp_a = Σ_c p_combo_c * A_ca
    M = combo @ legality                                  # (N, 4)
    d_probs_final = -scale * M
    grad_logits_final = softmax_backward_from_probs(final_probs, d_probs_final)

    # step heads: dconsist/dp_k comes from multi-linearity of the outer product
    Er = E.reshape(n, 3, 3, 3)                            # (n, c1, c2, c3)
    g1 = np.einsum('ndxy,nx,ny->nd', Er, p2, p3)
    g2 = np.einsum('ndxy,nd,ny->nx', Er, p1, p3)
    g3 = np.einsum('ndxy,nd,nx->ny', Er, p1, p2)
    grad_logits_steps = [
        softmax_backward_from_probs(p_k, -scale * g_k)
        for p_k, g_k in zip(step_probs, (g1, g2, g3))
    ]
    return loss, grad_logits_final, grad_logits_steps
