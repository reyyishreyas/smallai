"""From-scratch neural network layers with explicit forward and backward passes.
No autograd, no PyTorch, no sklearn — plain NumPy (PRD section 5.4)."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class Dense:
    """Fully connected layer: y = x @ W + b, with cached-input backward pass."""

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator):
        bound = np.sqrt(2.0 / n_in)  # He init (most layers feed a ReLU)
        self.W = rng.normal(0.0, bound, size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None

    def zero_grads(self) -> None:
        self.dW.fill(0.0)
        self.db.fill(0.0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        # Accumulate: multiple loss terms can flow through the same layer.
        self.dW += self._x.T @ grad_out
        self.db += grad_out.sum(axis=0)
        return grad_out @ self.W.T

    def params(self) -> list[np.ndarray]:
        return [self.W, self.b]

    def grads(self) -> list[np.ndarray]:
        return [self.dW, self.db]


class ReLU:
    """Rectified linear unit with mask-cache backward pass."""

    def __init__(self) -> None:
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self._mask

    def params(self) -> list[np.ndarray]:
        return []

    def grads(self) -> list[np.ndarray]:
        return []


class Stack:
    """A sequential stack of layers sharing a single forward/backward chain."""

    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad = grad_out
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for layer in self.layers:
            out.extend(layer.params())
        return out

    def grads(self) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for layer in self.layers:
            out.extend(layer.grads())
        return out

    def __iter__(self) -> Iterator:
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)
