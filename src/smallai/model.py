"""Step-wise decision model with a verifier head (PRD section 5).

Architecture (pure NumPy, ~5.4K parameters):
  Encoder:   Dense(d,32) -> ReLU -> Dense(32,32) -> ReLU              -> h
  Step 1:    Dense(h,16) -> ReLU -> Dense(16,3) -> softmax
  Step 2:    Dense([h, s1], 16) -> ReLU -> Dense(16,3) -> softmax     (sequential)
  Step 3:    Dense([h, s1, s2], 16) -> ReLU -> Dense(16,3) -> softmax
  Final:     Dense([h, s1, s2, s3], 16) -> ReLU -> Dense(16,4) -> softmax
  Verifier:  own tiny path  Dense(d,vh) -> ReLU -> Dense(vh,vh)
             -> Dense([h_v, s_onehot, k], 16) -> ReLU -> Dense(16,1) -> sigmoid

Step k sees the previous steps' states, so the chain is genuinely sequential.
The verifier learns from both correct and deliberately wrong trajectories
(journey learning) — corrupted journeys are the only place where "this step is
wrong" labels exist — and scores the validity of each proposed step state.

Inference (PRD 5.2 `predict -> check -> decide`): each step is decoded with a
fused score, log p(step | h, prev) + lam * log validity(state | h_v, k), so a
proposed state that the verifier deems unlikely gets re-decoded before the
final head sees it. lam = eval.verifier_decode_weight. Steps the verifier still
scores below `verifier_threshold` flag the decision as low confidence and fall
back to HOLD.

All gradients are hand-derived (see loss_and_grads) and verified by a
numerical gradient check in tests/test_gradients.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import oracle
from .nn import losses
from .nn.layers import Dense, ReLU, Stack


@dataclass
class TrainBatch:
    """One batch of presented trajectories (valid or corrupted)."""
    x: np.ndarray            # normalised features (N, d)
    states: np.ndarray       # presented step states (N, 3)
    action: np.ndarray       # presented final action (N,)
    matches: np.ndarray      # (N, 3) per-step validity labels
    is_valid: np.ndarray     # (N,) bool: True where the presented path matches the oracle


class SmallAIModel:
    """The full step-wise model with verifier head.

    `self.journey` controls whether verifier + L_format participate (on for
    model M; off for B2, which is trained on valid trajectories only)."""

    def __init__(self, cfg_model, rng: np.random.Generator):
        d = oracle.N_FEATURES
        H = cfg_model.hidden_enc
        hh = cfg_model.hidden_head
        hv = cfg_model.hidden_verifier
        self.cfg = cfg_model
        self.H = H
        self.hv = hv
        self.journey = True
        # default inference weight of the verifier in the fused step decode;
        # overridden by eval.verifier_decode_weight via runs.build_model
        self.decode_weight = 1.0

        self.encoder = Stack([Dense(d, H, rng), ReLU(), Dense(H, H, rng), ReLU()])
        self.step_heads = []
        for k in range(oracle.N_STEPS):
            self.step_heads.append(Stack([
                Dense(H + 3 * k, hh, rng),
                ReLU(),
                Dense(hh, oracle.N_STATES_PER_STEP, rng),
            ]))
        self.final_head = Stack([Dense(H + 9, hh, rng), ReLU(), Dense(hh, oracle.N_ACTIONS, rng)])
        # verifier: its own tiny encoder so the corrupted-journey supervision
        # shapes the verifier without competing with the decision encoder, and
        # a head over [verifier features, one-hot of the proposed step, k]
        self.verifier_encoder = Stack([Dense(d, hv, rng), ReLU(), Dense(hv, hv, rng)])
        self.verifier_head = Stack([Dense(hv + 4, 16, rng), ReLU(), Dense(16, 1, rng)])
        self.legality = oracle.build_legality_table()

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _onehot(states: np.ndarray, n: int = oracle.N_STATES_PER_STEP) -> np.ndarray:
        out = np.zeros((states.shape[0], n))
        out[np.arange(states.shape[0]), states] = 1.0
        return out

    def _verifier_input(self, h_v: np.ndarray, states_col: np.ndarray, k: int) -> np.ndarray:
        """Verifier input: [verifier features, one-hot proposed state, k/2]."""
        kcol = np.full((h_v.shape[0], 1), float(k) / (oracle.N_STEPS - 1))
        return np.concatenate([h_v, self._onehot(states_col), kcol], axis=1)

    def presented_forward(self, x: np.ndarray, states: np.ndarray) -> dict:
        """Teacher-forced forward: the presented states (true for valid rows,
        corrupted for invalid rows) feed later heads."""
        h = self.encoder.forward(x)
        out: dict = {"h": h}
        step_logits, step_probs = [], []
        for k in range(oracle.N_STEPS):
            inp = np.concatenate([h] + [self._onehot(states[:, j]) for j in range(k)], axis=1)
            logits = self.step_heads[k].forward(inp)
            step_logits.append(logits)
            step_probs.append(losses.softmax(logits))
        final_inp = np.concatenate([h] + [self._onehot(states[:, j]) for j in range(3)], axis=1)
        final_logits = self.final_head.forward(final_inp)
        h_v = self.verifier_encoder.forward(x)
        ver_inputs = [self._verifier_input(h_v, states[:, k], k)
                      for k in range(oracle.N_STEPS)]
        ver_logits = [self.verifier_head.forward(ver_inputs[k])
                      for k in range(oracle.N_STEPS)]
        out.update(step_logits=step_logits, step_probs=step_probs,
                   final_logits=final_logits,
                   final_probs=losses.softmax(final_logits),
                   ver_logits=ver_logits, ver_inputs=ver_inputs)
        return out

    # ------------------------------------------------------------------ losses
    def loss_and_grads(self, batch: TrainBatch, w: dict[str, float],
                       journey: bool | None = None) -> float:
        """Combined journey-learning loss (PRD 5.3):

            L = a*CE(final) + b*sum_k CE(step_k) + c*BCE(verifier) + d*L_format

        * CE(final) and CE(step_k) apply to valid rows only.
        * BCE(verifier) applies to every row: presented steps that match the
          oracle are labelled 1, corrupted ones 0 — this is where incorrect
          trajectories are learned from.
        * L_format keeps the final action consistent with the rule table for
          the currently predicted steps (stop-gradient through step softmax).

        With journey=False only the final + step CE are computed (B2 setup)."""
        if journey is None:
            journey = self.journey
        batch_size = batch.x.shape[0]
        H = self.H
        self.zero_grads()
        fwd = self.presented_forward(batch.x, batch.states)
        valid = batch.is_valid.astype(bool)

        def back_through(stack: Stack, d_rows: np.ndarray,
                         mask: np.ndarray | None) -> None:
            """Scatter a head gradient over all rows, push through the head and
            the shared encoder (accumulating into the encoder's grads)."""
            if mask is not None and d_rows.shape[0] != batch_size:
                full = np.zeros((batch_size, d_rows.shape[1]))
                full[mask] = d_rows
                d_rows = full
            grad_in = stack.backward(d_rows)
            self.encoder.backward(grad_in[:, :H])

        # --- final cross-entropy on valid rows
        ce_final, d_final = losses.softmax_xent(
            fwd["final_logits"][valid], batch.action[valid])
        back_through(self.final_head, w["a"] * d_final, valid)

        # --- step cross-entropy on valid rows (teacher forced)
        ce_steps = []
        for k in range(oracle.N_STEPS):
            ce_k, d_k = losses.softmax_xent(
                fwd["step_logits"][k][valid], batch.states[valid, k])
            ce_steps.append(ce_k)
            back_through(self.step_heads[k], w["b"] * d_k, valid)

        total = float(w["a"] * ce_final + w["b"] * sum(ce_steps))

        if journey:
            # --- verifier BCE on every row (corrupted steps labelled 0).
            # The verifier has its own encoder; the BCE gradient stops at the
            # verifier representation (no competition with the decision path).
            # One shared head is forwarded once per step k, so re-replay the
            # corresponding input before each backward to restore caches.
            bce_total = 0.0
            for k in range(oracle.N_STEPS):
                bce_k, dv_k = losses.bce_with_logits(
                    fwd["ver_logits"][k], batch.matches[:, k],
                    neg_row_weight=w.get("verifier_neg_weight", 1.0))
                bce_total += bce_k
                self.verifier_head.forward(fwd["ver_inputs"][k])
                grad_in = self.verifier_head.backward(w["c"] * dv_k)
                self.verifier_encoder.backward(grad_in[:, :self.hv])
            total += w["c"] * bce_total

            # --- L_format on every row: keeps the final action consistent
            # with the rule table for the currently predicted steps; gradient
            # flows into the final and all three step heads.
            fmt, d_fmt_final, d_fmt_steps = losses.format_penalty(
                fwd["final_probs"], fwd["step_probs"], self.legality, w["d"])
            total += fmt
            back_through(self.final_head, d_fmt_final, mask=None)
            for k in range(oracle.N_STEPS):
                back_through(self.step_heads[k], d_fmt_steps[k], mask=None)
        return float(total)

    # ------------------------------------------------------------ autoregressive
    def chain_forward(self, x: np.ndarray, decode_weight: float | None = None) -> dict:
        """Inference-time forward: each step is fed its own fused decode (no
        teacher forcing), fully vectorised.

        With journey learning, a step is not just the step-head's argmax: every
        candidate state is scored by the verifier (which saw corrupted paths in
        training, so it knows which (feature, state) pairs are impossible) and
        the decode maximises log p_step + lam * log validity; lam defaults to
        `self.decode_weight`. B2 (journey=False) decodes by plain argmax."""
        dw = self.decode_weight if decode_weight is None else decode_weight
        h = self.encoder.forward(x)
        step_logits = []
        states_pred = np.zeros((x.shape[0], oracle.N_STEPS), dtype=np.int64)
        if self.journey:
            h_v = self.verifier_encoder.forward(x)
            # candidate validity scores per step: (N, n_states, n_steps)
            v_cand = np.zeros((x.shape[0], oracle.N_STATES_PER_STEP, oracle.N_STEPS))
            ver_logits = []
            for k in range(oracle.N_STEPS):
                for c in range(oracle.N_STATES_PER_STEP):
                    inp = self._verifier_input(
                        h_v, np.full(x.shape[0], c, dtype=np.int64), k)
                    v_cand[:, c, k] = losses.sigmoid(self.verifier_head.forward(inp)[:, 0])
        else:
            ver_logits = []
        for k in range(oracle.N_STEPS):
            inp = np.concatenate(
                [h] + [self._onehot(states_pred[:, j]) for j in range(k)], axis=1)
            logits = self.step_heads[k].forward(inp)
            step_logits.append(logits)
            if self.journey:
                lp = losses.log_softmax(logits)
                logv = np.log(np.clip(v_cand[:, :, k], 1e-9, 1.0))
                fused = lp + dw * logv
                states_pred[:, k] = fused.argmax(axis=1)
                ver_logits.append(self.verifier_head.forward(
                    self._verifier_input(h_v, states_pred[:, k], k)))
            else:
                states_pred[:, k] = logits.argmax(axis=1)
        final_inp = np.concatenate(
            [h] + [self._onehot(states_pred[:, j]) for j in range(3)], axis=1)
        final_logits = self.final_head.forward(final_inp)
        return {"h": h, "states": states_pred, "step_logits": step_logits,
                "final_logits": final_logits, "ver_logits": ver_logits}

    def predict_chain(self, x_raw: np.ndarray,
                      decode_weight: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised autoregressive prediction for evaluation:
        returns (predicted_states (N,3), predicted_action (N,))."""
        out = self.chain_forward(oracle.normalize(x_raw), decode_weight)
        return out["states"], out["final_logits"].argmax(axis=1)

    def prediction(self, x_raw: np.ndarray,
                   verifier_threshold: float | None = None,
                   decode_weight: float | None = None) -> list[dict]:
        """Decision + interpretable trace for each raw feature row.

        Every decision comes with the chain of intermediate states, the verifier
        validity score of the *selected* state per step, and the fused decode
        (`decode_weight`, PRD 5.2: predict -> check -> decide). Steps the
        verifier still scores below `verifier_threshold` (if the verifier is
        trained) flag the decision as low confidence and fall back to HOLD."""
        out = self.chain_forward(oracle.normalize(x_raw), decode_weight)
        states = out["states"]
        action = out["final_logits"].argmax(axis=1)
        rows: list[dict] = []
        for i in range(x_raw.shape[0]):
            if self.journey:
                validities = [float(losses.sigmoid(out["ver_logits"][k][i, 0]))
                              for k in range(oracle.N_STEPS)]
                low_conf = verifier_threshold is not None and min(validities) < verifier_threshold
                final = int(oracle.HOLD) if low_conf else int(action[i])
            else:
                validities = [float("nan")] * oracle.N_STEPS
                low_conf = False
                final = int(action[i])
            rows.append({
                "states": [int(states[i, k]) for k in range(oracle.N_STEPS)],
                "state_names": [oracle.STEP_STATE_NAMES[k][int(states[i, k])]
                                for k in range(oracle.N_STEPS)],
                "action": final,
                "action_name": oracle.ACTION_NAMES[final],
                "step_probs": [losses.softmax(out["step_logits"][k][i][None])[0].tolist() for k in range(oracle.N_STEPS)],
                "final_probs": losses.softmax(out["final_logits"][i][None])[0].tolist(),
                "validities": validities,
                "low_confidence": bool(low_conf),
                "trace": oracle.trace_string(states[i], final, validities),
                "features": [float(v) for v in x_raw[i]],
            })
        return rows

    def verifier_scores(self, x_raw: np.ndarray, presented_states: np.ndarray):
        """Verifier validity score (0-1) for each presented step state. Used to
        evaluate verifier quality on held-out corrupted trajectories."""
        out = self.presented_forward(oracle.normalize(x_raw), presented_states)
        return [losses.sigmoid(out["ver_logits"][k][:, 0]) for k in range(oracle.N_STEPS)]

    # ---------------------------------------------------------------- params
    def stacks(self) -> list[Stack]:
        return [self.encoder, self.final_head, self.verifier_encoder,
                self.verifier_head] + list(self.step_heads)

    def all_params(self) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for s in self.stacks():
            out.extend(s.params())
        return out

    def all_grads(self) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for s in self.stacks():
            out.extend(s.grads())
        return out

    def zero_grads(self) -> None:
        for stack in self.stacks():
            for layer in stack:
                if isinstance(layer, Dense):
                    layer.zero_grads()

    def param_count(self) -> int:
        return int(sum(p.size for p in self.all_params()))

    def snapshot(self) -> list[np.ndarray]:
        return [p.copy() for p in self.all_params()]

    def restore(self, snap: list[np.ndarray]) -> None:
        for p, s in zip(self.all_params(), snap):
            p[...] = s
