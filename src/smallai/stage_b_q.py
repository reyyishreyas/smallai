"""Stage B-Q (AI-267, experimental): a tiny tabular Q-learning capture gate.

This branch replaces the v3 *fixed* thresholds with a **learned** policy — the
lightest weight RL setup that still deserves the name:

  State     s = quantize(signal row) on 5 hand-chosen bucket edges per signal
            (sharpness 5 × ink 4 × brightness 4 × contrast 3 = 240 possible
            states; in practice far fewer are ever observed)
  Actions   a ∈ {RETAKE, REVIEW, PASS}  (the same verdict contract as Stage B)
  Reward    1 − cost(action, expert verdict), with an *asymmetric* cost matrix:
            forwarding an unusable capture (PASS on RETAKE-truth) is the
            costliest mistake; burdening a good capture with a re-take is the
            mildest error.
  Episode   single-step and terminal (the gate makes one decision) → this is a
            contextual-bandit / one-step Q-learning formulation.
  Agent     off-policy tabular Q-learning, ε-greedy exploration (seeded RNG),
            update Q[s, a] += α (r − Q[s, a]); γ = 0 (no next state).
            Fail-closed at deployment: a state never seen in training is
            rejected (RETAKE) — never PASS without training evidence.

Why this counts as RL: an agent chooses actions in an environment, receives
receives scalar rewards, and improves its value estimates online. Why it is
still tiny: no gradients, no neural network, ~hundreds of float parameters —
the whole table is smaller than a single convolution layer, and the greedy
policy it converges to is fully auditable row by row.

Expert labels (the reward source):
  corpus soft blur labels → RETAKE if lab == 1, REVIEW if lab == 0.5,
  PASS if lab == 0  (labels are soft for 849/2000 images — see the Stage B
  report); the degradation sweep carries its own level→verdict ground truth.

Everything is deterministic (fixed seeds) and pure NumPy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import stage_b as sb
from .env import image_quality as iq

RETAKE, REVIEW, PASS = iq.RETAKE, iq.REVIEW, iq.PASS
DECISION_NAMES = iq.DECISION_NAMES

# Bucket edges per signal (same information the v3 gate thresholds used, but
# finer — the agent is free to discover where the real cut lines are).
# A value in [edge[i-1], edge[i]) lands in bucket i; below edge[0] -> 0.
# NOTE on ink_ratio: the v3 gate has NO upper-ink cut (dense print is normal
# for many documents), so ink gets only the "no content / sparse / has ink"
# buckets — a coarse upper bucket would create unseen-state dead zones for
# real-world dense-print documents (bank statements, forms).
STATE_EDGES: dict[str, tuple[float, ...]] = {
    "sharpness": (0.05, 0.10, 0.13, 0.20),
    "ink_ratio": (0.002, 0.003),
    "brightness": (0.10, 0.25, 0.60),
    "contrast": (0.18, 0.35),
}
_STATE_KEYS = ("sharpness", "ink_ratio", "brightness", "contrast")

# cost[action][expert] — how much a wrong decision costs the pipeline.
# Rows: the action taken. Columns: the expert (ground-truth) verdict.
COST = np.array([
    # expert:   RETAKE  REVIEW  PASS
    [0.0,       0.5,    1.5],   # action RETAKE (wasted re-capture on a good shot)
    [0.75,      0.0,    0.5],   # action REVIEW (extra human round)
    [2.0,       0.5,    0.0],   # action PASS (forwarding an unreadable capture!)
])
REWARD = 1.0 - COST  # reward[a, e]


def quantize_signals(signals: dict[str, np.ndarray]) -> list[tuple[int, ...]]:
    """Quantize a batch of signal rows into discrete state tuples."""
    buckets = {}
    for k in _STATE_KEYS:
        v = np.asarray(signals[k], dtype=np.float64).reshape(-1)
        buckets[k] = np.digitize(v, STATE_EDGES[k])
    n = len(next(iter(buckets.values())))
    return [tuple(int(buckets[k][i]) for k in _STATE_KEYS) for i in range(n)]


def samples_from_sweep(n_docs: int = 24, seed: int = 0
                       ) -> tuple[list[tuple[int, ...]], list[int]]:
    """Build (state, expert) samples from the degradation sweep corpus.

    Ground truth comes from the sweep level itself: pristine and mild blur are
    PASS, border blur r=8 is REVIEW, darkness ×0.1 is RETAKE. Deterministic.
    """
    rng = np.random.default_rng(seed)
    levels: tuple[tuple[str, float, int], ...] = (
        ("none", 0.0, PASS),
        ("blur", 2.0, PASS),
        ("blur", 8.0, REVIEW),
        ("exposure", 0.1, RETAKE),
    )
    states, experts = [], []
    for _ in range(n_docs):
        base = sb.make_document(rng)[None]
        for kind, lvl, verdict in levels:
            img = sb.corrupt(base, kind, lvl, rng)
            sig = iq.extract(img)
            states.append(quantize_signals(sig)[0])
            experts.append(verdict)
    return states, experts


def samples_from_corpus(npz_path: str | Path = "runs/stage_b_corpus_scan.npz"
                        ) -> tuple[list[tuple[int, ...]], list[int]]:
    """(state, expert) samples from the full-corpus scan cache.

    Requires `runs/stage_b_corpus_scan.npz` (see the Stage B report for how it
    was produced: one scan row per corpus image, with its soft blur label).
    Soft label → expert verdict: 0 → PASS, 0.5 → REVIEW, 1 → RETAKE.
    """
    d = np.load(npz_path, allow_pickle=True)
    sig = {k: d[k] for k in ("sharpness", "contrast", "ink_ratio", "brightness")}
    lab = np.asarray(d["lab"], dtype=np.float64)
    experts = [RETAKE if l == 1 else (REVIEW if l == 0.5 else PASS) for l in lab]
    return quantize_signals(sig), experts


class QCaptureAgent:
    """Tabular Q-learning agent for the one-step capture decision."""

    def __init__(self, alpha: float = 0.5, eps: float = 0.25, seed: int = 0):
        self.alpha = alpha
        self.eps = eps
        self.rng = np.random.default_rng(seed)
        self.q: dict[tuple[int, ...], np.ndarray] = {}

    # -- environment interaction ------------------------------------------------
    def _ensure(self, s: tuple[int, ...]) -> np.ndarray:
        if s not in self.q:
            self.q[s] = np.zeros(len(DECISION_NAMES))
        return self.q[s]

    def act(self, s: tuple[int, ...], greedy: bool = False) -> int:
        """ε-greedy action choice (uniform random exploration, seeded)."""
        row = self._ensure(s)
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(len(DECISION_NAMES)))
        return int(np.argmax(row))  # deterministic tie-break

    # -- learning ----------------------------------------------------------------
    def learn(self, states: list[tuple[int, ...]], experts: list[int],
              epochs: int = 40) -> dict:
        """Off-policy Q-learning over terminal one-step episodes.

        Every step: a = act(s); r = REWARD[a, e]; Q[s,a] += α(r − Q[s,a]).
        With γ=0 and a deterministic environment the values converge to the
        immediate expected reward of each (state, action) pair.
        """
        for _ in range(epochs):
            for s, e in zip(states, experts):
                a = self.act(s)
                r = REWARD[a, e]
                row = self._ensure(s)
                row[a] += self.alpha * (r - row[a])
        return self.stats()

    # -- deployment ----------------------------------------------------------------
    def resolve_state(self, s: tuple[int, ...]) -> tuple[tuple[int, ...], int] | tuple[None, None]:
        """Map a possibly-unseen state onto training evidence.

        Returns (resolved_state, hamming_distance) of the nearest trained
        state when it is within distance 1 of the query (signals move
        continuously, so adjacent buckets share the physics); returns
        (None, None) when the state is novel enough to fail closed.
        """
        if s in self.q:
            return s, 0
        best, best_d = None, None
        for ts in self.q:
            d = sum((a != b) for a, b in zip(s, ts))
            if best_d is None or d < best_d:
                best, best_d = ts, d
        if best is not None and best_d <= 1:
            return best, best_d
        return None, None

    def predict(self, states: list[tuple[int, ...]]) -> list[int]:
        """Greedy (deployment) decisions — no exploration.

        Unseen states generalize to the nearest trained bucket when adjacent;
        otherwise they fail closed (RETAKE — never PASS without evidence).
        Inference never mutates the table.
        """
        out = []
        for s in states:
            resolved, _ = self.resolve_state(s)
            if resolved is None:
                out.append(RETAKE)
            else:
                out.append(int(np.argmax(self.q[resolved])))
        return out

    def stats(self) -> dict:
        return {
            "n_states": len(self.q),
            "n_params": sum(row.size for row in self.q.values()),
            "max_possible_states": int(np.prod(
                [len(ed) + 1 for ed in STATE_EDGES.values()])),
        }


def evaluate(actions: list[int], experts: list[int]) -> dict:
    """Reward / agreement / confusion of a decision sequence vs experts.

    REWARD lies in [-1, 1] already (reward = 1 − cost), so the mean reward is
    directly comparable across policies.
    """
    a, e = np.asarray(actions), np.asarray(experts)
    return {
        "mean_reward": float(REWARD[a, e].mean()),
        "mean_cost": float(COST[a, e].mean()),
        "agreement": float((a == e).mean()),
        "confusion": np.array([[int(((a == i) & (e == j)).sum())
                                for j in range(3)] for i in range(3)]).tolist(),
    }


def check_images_with_q(paths: list[str | Path],
                        npz_path: str | Path = "runs/stage_b_corpus_scan.npz",
                        seed: int = 0) -> list[dict]:
    """Score arbitrary image files with the *learned* Q-policy (demo path).

    Uses `stage_b.check_images` for the per-image signals, then maps each
    image's quantized state through the greedy Q-policy trained on the corpus
    labels (falls back to sweep-truth when the npz cache is absent). Returns
    the same records as check_images, plus:
      state       the quantized bucket tuple (sharp, ink, bright, contrast)
      verdict_q   the policy's decision
      q_values    learned Q-values per action [RETAKE, REVIEW, PASS]
      verdict_fixed  the fixed v3 gate's verdict for side-by-side comparison
    """
    results = sb.check_images(paths)
    if Path(npz_path).exists():
        st, ex = samples_from_corpus(npz_path)
    else:
        st, ex = samples_from_sweep()
    agent = QCaptureAgent(seed=seed)
    agent.learn(st, ex)

    for r in results:
        if "signals" not in r:
            continue
        s = {k: np.array([r["signals"][k]]) for k in _STATE_KEYS}
        state = quantize_signals(s)[0]
        resolved, dist = agent.resolve_state(state)
        if resolved is None:
            # fail-closed: a state never observed during training is rejected —
            # never PASS what the policy has no value estimates for
            qv = np.zeros(len(DECISION_NAMES))
            r.setdefault("reasons", []).append(
                "unseen state — fail-closed default (no training evidence)")
            r["state_used"] = None
        else:
            qv = agent.q[resolved]
            r["state_used"] = list(resolved)
            if dist > 0:
                r.setdefault("reasons", []).append(
                    f"state generalized {dist} bucket(s) away from a trained state")
        r["state"] = list(state)
        r["verdict_q"] = DECISION_NAMES[int(np.argmax(qv))]
        r["q_values"] = [round(float(x), 3) for x in qv]
        r["verdict_fixed"] = iq.verdict_name(
            int(iq.decide({k: np.array([r["signals"][k]]) for k in _STATE_KEYS})[0]))
    return results


def run_experiment(npz_path: str | Path = "runs/stage_b_corpus_scan.npz",
                   seed: int = 0) -> dict:
    """Train the Q gate and compare it with the fixed v3 thresholds.

    Uses the full-corpus scan when the npz cache exists (2000 supervised
    samples, soft-label experts); otherwise falls back to the deterministic
    degradation sweep. Returns both policies scored against the SAME labels,
    so the comparison is apples-to-apples.
    """
    p = Path(npz_path)
    if p.exists():
        states, experts = samples_from_corpus(p)
        source = "corpus (2000 images, soft blur labels)"

        # fixed v3 gate on the same rows, same label mapping
        d = np.load(p, allow_pickle=True)
        sig = {k: d[k] for k in _STATE_KEYS}
    else:
        from collections import defaultdict

        states, experts = samples_from_sweep()
        source = "sweep (synthetic, level-truth)"
        rows: dict[str, list[float]] = defaultdict(list)
        rng = np.random.default_rng(0)
        levels = (("none", 0.0), ("blur", 2.0), ("blur", 8.0), ("exposure", 0.1))
        for _ in range(24):
            base = sb.make_document(rng)[None]
            for kind, lvl in levels:  # keep in sync with samples_from_sweep truth
                s_row = iq.extract(sb.corrupt(base, kind, lvl, rng))
                for k in _STATE_KEYS:
                    rows[k].append(float(s_row[k][0]))
        sig = {k: np.array(v) for k, v in rows.items()}

    agent = QCaptureAgent(seed=seed)
    agent.learn(states, experts)
    q_actions = agent.predict(states)
    fixed_actions = [int(x) for x in iq.decide(sig)]

    return {
        "source": source,
        "n_samples": len(states),
        "q_gate": {**agent.stats(), **evaluate(q_actions, experts)},
        "fixed_gate": evaluate(fixed_actions, experts),
        "verdict_names": list(DECISION_NAMES),
    }
