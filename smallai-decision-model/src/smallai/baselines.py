"""Baselines (PRD Task 4, section 6.1).

  B0 Heuristic      : hand-written threshold rules on the raw features, no learning
  B1 Shortcut MLP   : same encoder, features straight to the final action,
                      trained on valid data only (no steps, no verifier)
  B2 Steps-no-journey: full step architecture trained on valid data only
                      (implemented in runs.run_kind with journey=False)
  B3 LLM proxy      : analytical cost estimate only — no API in the loop

B1 vs B2 vs M isolates the effect of step-wise reasoning and then of journey
learning.
"""
from __future__ import annotations

import numpy as np

from . import config as cfg_mod
from .env import oracle
from .nn import layers as nn_layers
from .nn import losses, optim


class Heuristic:
    """B0: hand-written commonsense rules over raw features. No learning."""

    journey = False  # no verifier

    def predict_chain(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        battery, wind, temp = x[:, 0], x[:, 1], x[:, 2]
        gps, link, payload = x[:, 3], x[:, 4], x[:, 5]
        dist, sensor, _hour, terrain = x[:, 6], x[:, 7], x[:, 8], x[:, 9]

        fatal = (
            (battery < 18.0) | (wind > 16.0) | (gps < 0.30) | (sensor < 0.25)
            | (link < 0.25) | (temp < -12.0) | (temp > 42.0)
        )
        hold = (battery < 42.0) | (wind > 10.0)
        reduced = (payload > 2.5) | (dist > 24.0) | (link < 0.55) | (terrain > 0.6)

        action = np.where(fatal, oracle.ABORT,
                          np.where(hold, oracle.HOLD,
                                   np.where(reduced, oracle.GO_REDUCED, oracle.GO)))
        states = np.repeat(action[:, None], 3, axis=1)  # placeholder: no real steps
        return states, action.astype(np.int64)


class ShortcutMLP:
    """B1: encoder + final action head only. Trained on valid data only."""

    def __init__(self, cfg_model, rng: np.random.Generator):
        d = oracle.N_FEATURES
        H = cfg_model.hidden_enc
        hh = cfg_model.hidden_head
        self.H = H
        self.journey = False  # no verifier in the shortcut model
        self.encoder = nn_layers.Stack([nn_layers.Dense(d, H, rng), nn_layers.ReLU(),
                                        nn_layers.Dense(H, H, rng), nn_layers.ReLU()])
        self.head = nn_layers.Stack([nn_layers.Dense(H, hh, rng), nn_layers.ReLU(),
                                     nn_layers.Dense(hh, oracle.N_ACTIONS, rng)])

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.head.forward(self.encoder.forward(x))

    def predict_chain(self, x_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = self.logits(oracle.normalize(x_raw))
        act = out.argmax(axis=1)
        return np.zeros((x_raw.shape[0], 3), dtype=np.int64), act

    def fit(self, x: np.ndarray, action: np.ndarray, cfg_train, seed: int) -> list[float]:
        """Minimal CE-only training with Adam (B1 sees valid data only)."""
        rng = np.random.default_rng(seed + 7)
        opt = optim.make_optim(self.all_params(), "adam", cfg_train.lr)
        xn = oracle.normalize(x)
        losses_hist = []
        for _ in range(cfg_train.epochs):
            perm = rng.permutation(x.shape[0])
            ep = []
            for start in range(0, x.shape[0], cfg_train.batch_size):
                idx = perm[start:start + cfg_train.batch_size]
                for stack in (self.encoder, self.head):
                    for lay in stack:
                        if isinstance(lay, nn_layers.Dense):
                            lay.zero_grads()
                loss, d_logits = losses.softmax_xent(self.logits(xn[idx]), action[idx])
                g_h = self.head.backward(d_logits)   # grad wrt head input (= h)
                self.encoder.backward(g_h)
                opt.step(self.all_grads())
                ep.append(loss)
            losses_hist.append(float(np.mean(ep)))
        return losses_hist

    def all_params(self) -> list[np.ndarray]:
        return self.encoder.params() + self.head.params()

    def param_count(self) -> int:
        return int(sum(p.size for p in self.all_params()))

    def all_grads(self) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for lay in list(self.encoder) + list(self.head):
            out.extend(lay.grads())
        return out

    def snapshot(self) -> list[np.ndarray]:
        return [p.copy() for p in self.all_params()]

    def restore(self, snap: list[np.ndarray]) -> None:
        for p, s in zip(self.all_params(), snap):
            p[...] = s


def llm_proxy_estimate(cfg_llm: cfg_mod.LLMProxyConfig, n_decisions: int = 1_000_000) -> dict:
    """B3: analytical cost estimate for an LLM solving the same task. The LLM
    is never used in the inference path; the numbers come from token-count
    assumptions and published pricing."""
    per_call_in = cfg_llm.prompt_tokens * cfg_llm.usd_per_m_input / 1e6
    per_call_out = cfg_llm.completion_tokens * cfg_llm.usd_per_m_output / 1e6
    per_call = per_call_in + per_call_out
    return {
        "model": cfg_llm.model_name,
        "tokens_per_decision": cfg_llm.prompt_tokens + cfg_llm.completion_tokens,
        "usd_per_1000_decisions": per_call * 1000,
        "usd_per_1m_decisions": per_call * n_decisions,
        "latency_ms_p50": cfg_llm.latency_ms_p50,
        "latency_ms_p95": cfg_llm.latency_ms_p95,
    }
