"""Optimisation with hand-written update rules: Adam and SGD with momentum."""
from __future__ import annotations

import numpy as np


class Adam:
    """Standard Adam over a flat list of parameter arrays (layer-owned)."""

    def __init__(self, params: list[np.ndarray], lr: float = 3e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        b1t = 1.0 - self.beta1 ** self.t
        b2t = 1.0 - self.beta2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * np.square(g)
            m_hat = m / b1t
            v_hat = v / b2t
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def state_dict(self) -> dict:
        return {"t": self.t, "m": [m.copy() for m in self.m], "v": [v.copy() for v in self.v]}


class SGDMomentum:
    """SGD with classical momentum."""

    def __init__(self, params: list[np.ndarray], lr: float = 1e-2, momentum: float = 0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.vel = [np.zeros_like(p) for p in params]

    def step(self, grads: list[np.ndarray]) -> None:
        for p, g, vel in zip(self.params, grads, self.vel):
            vel *= self.momentum
            vel += g
            p -= self.lr * vel


def make_optim(params: list[np.ndarray], name: str, lr: float):
    if name == "adam":
        return Adam(params, lr=lr)
    if name == "sgd":
        return SGDMomentum(params, lr=lr)
    raise ValueError(f"unknown optimiser {name!r}")
