"""Posterior-decoded terminal for AI-267 (development only).

The idea
--------
The frozen reward (EXPERIMENT.md section 6) gives every terminal
action a value that is an *affine function of the label
distribution*:

    Q(s, a) = sum_l P(l | s) * verdict_reward(a, l)
              + consistency_bonus(a, s)

The verdict part is a known 3x3 matrix over the labels
(``pass``/``review``/``retake``) and the consistency part is a
*known* function of the action and the evidence state. So the three
terminal Q-values are three linear equations in the three unknown
label probabilities -- they can be solved for. The Q-table therefore
carries the label posterior, and the table has so far been read only
as a *ranking* of terminals.

The ranking is not accuracy-aligned: the reward pays -0.5 for an
adjacent-wrong verdict and -1.0 for an opposite-wrong one, and adds
+0.10 / -0.30 for evidence consistency. The reward-optimal terminal
sits about 0.5 pp below the accuracy-optimal one (see
``scripts/dev/reward.py``).

Decoding the posterior and taking its arg-max restores
accuracy-alignment *without* changing the reward, the learning rule,
the state encoding or any frozen constant -- and without using the
test split, the calibration split or the generative model.

Two variants are measured:

- ``swap``: the policy walks exactly as before; only the final
  verdict is replaced by the decoded arg-max. The measurement path
  is untouched, so this isolates the decision rule.
- ``decoded``: the decoded posterior also decides *when* to stop.

Nothing here touches the scored protocol or the scored test split.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from smallai.agent import (  # noqa: E402
    TERMINALS,
    QAgent,
    State,
    legal_actions,
    transition,
)
from smallai.environment import SIGNALS, Scenario, observe  # noqa: E402
from smallai.evaluate import derive_seed, is_consistent, levels_of  # noqa: E402
from smallai.solver import Decision  # noqa: E402
from smallai.train import (  # noqa: E402
    MIX,
    consistency_bonus,
    make_splits,
    oracle_label,
    train_td,
    verdict_reward,
)

DEV_SEED = 20261030
AGENT_NAME = "decode_probe"

#: Label order used by the decode matrix.
LABELS = ("pass", "review", "retake")

#: ``M[a, l] = verdict_reward(a, l)`` for the frozen reward.
MATRIX = np.array(
    [[verdict_reward(action, label) for label in LABELS] for action in TERMINALS],
    dtype=float,
)
INVERSE = np.linalg.inv(MATRIX)


def decode(agent: QAgent, state: State) -> np.ndarray:
    """Recover the label posterior implied by the terminal Q-values.

    Parameters
    ----------
    agent : QAgent
        The trained policy.
    state : State
        The terminal state.

    Returns
    -------
    numpy.ndarray
        Unnormalised label probabilities in :data:`LABELS` order.
    """
    values = agent.q.get(state, {})
    q = np.array([values.get(action, 0.0) for action in TERMINALS])
    bonus = np.array([consistency_bonus(action, state) for action in TERMINALS])
    return INVERSE @ (q - bonus)


def _argmax(posterior: np.ndarray) -> str:
    """Pick the label with the largest decoded probability.

    Parameters
    ----------
    posterior : numpy.ndarray
        Decoded label probabilities.

    Returns
    -------
    str
        The chosen label.
    """
    clipped = np.clip(posterior, 0.0, None)
    total = clipped.sum()
    if total <= 0.0:
        return LABELS[int(np.argmax(posterior))]
    normalised = clipped / total
    return LABELS[int(np.argmax(normalised))]


def walk_frozen(agent: QAgent, scenario: Scenario, rng: random.Random):
    """Replay the frozen policy and return its terminal decision.

    Parameters
    ----------
    agent : QAgent
        The trained policy.
    scenario : Scenario
        The capture to judge.
    rng : random.Random
        Seeded noise source.

    Returns
    -------
    tuple
        The terminal action, the terminal state and the measurement count.
    """
    state: State = (0,) * len(SIGNALS)
    measurements = 0
    while True:
        action = agent.act(state, rng, epsilon=0.0)
        if action in TERMINALS:
            return action, state, measurements
        measurements += 1
        reading = observe(scenario, action.partition(":")[2], rng)
        state = transition(state, action, reading)


def walk_decoded(agent: QAgent, scenario: Scenario, rng: random.Random):
    """Replay the policy but decode the terminal verdict.

    Parameters
    ----------
    agent : QAgent
        The trained policy.
    scenario : Scenario
        The capture to judge.
    rng : random.Random
        Seeded noise source.

    Returns
    -------
    tuple
        The terminal action, the terminal state and the measurement count.
    """
    state: State = (0,) * len(SIGNALS)
    measurements = 0
    while True:
        legal = legal_actions(state, agent.allow_recheck)
        values = agent.q.get(state, {})
        best = max(values.get(a, 0.0) for a in legal)
        tied = [a for a in legal if values.get(a, 0.0) >= best - 1e-12]
        measurements_tied = [a for a in tied if a not in TERMINALS]
        if measurements_tied:
            action = min(
                measurements_tied,
                key=lambda a: SIGNALS.index(a.partition(":")[2]),
            )
        else:
            action = _argmax(decode(agent, state))
        if action in TERMINALS:
            return action, state, measurements
        measurements += 1
        reading = observe(scenario, action.partition(":")[2], rng)
        state = transition(state, action, reading)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEV_SEED)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budget", type=int, default=200_000)
    args = parser.parse_args(argv)

    print(
        f"{'master seed':<12}{'frozen %':>10}{'swap %':>10}{'decoded %':>11}"
        f"{'swap inval%':>13}{'dec inval%':>12}"
        f"{'frozen meas':>13}{'swap meas':>11}{'dec meas':>11}"
    )
    print("-" * 93)
    for master in [args.seed + offset for offset in range(args.seeds)]:
        splits = make_splits(master)
        scenarios = splits["test"]
        labels = [oracle_label(s) for s in scenarios]
        agent = QAgent(name=AGENT_NAME)
        train_td(
            agent,
            splits["train"],
            seed=derive_seed(master, "train", AGENT_NAME),
            budget=args.budget,
            mix=MIX,
        )

        started = time.perf_counter()
        hits = {"frozen": 0, "swap": 0, "decoded": 0}
        invalid = {"frozen": 0, "swap": 0, "decoded": 0}
        meas = {"frozen": 0, "swap": 0, "decoded": 0}
        for index, (scenario, label) in enumerate(zip(scenarios, labels)):
            rng = random.Random(derive_seed(master, AGENT_NAME, str(index)))
            frozen_action, frozen_state, frozen_meas = walk_frozen(
                agent, scenario, rng
            )
            hits["frozen"] += frozen_action == label
            invalid["frozen"] += not is_consistent(
                frozen_action, levels_of(frozen_state)
            )
            meas["frozen"] += frozen_meas

            swap_action = _argmax(decode(agent, frozen_state))
            hits["swap"] += swap_action == label
            invalid["swap"] += not is_consistent(
                swap_action, levels_of(frozen_state)
            )
            meas["swap"] += frozen_meas

            rng2 = random.Random(derive_seed(master, AGENT_NAME, str(index)))
            decoded_action, decoded_state, decoded_meas = walk_decoded(
                agent, scenario, rng2
            )
            hits["decoded"] += decoded_action == label
            invalid["decoded"] += not is_consistent(
                decoded_action, levels_of(decoded_state)
            )
            meas["decoded"] += decoded_meas

        n = len(scenarios)
        print(
            f"{master:<12}"
            f"{100 * hits['frozen'] / n:>10.3f}"
            f"{100 * hits['swap'] / n:>10.3f}"
            f"{100 * hits['decoded'] / n:>11.3f}"
            f"{100 * invalid['swap'] / n:>13.3f}"
            f"{100 * invalid['decoded'] / n:>12.3f}"
            f"{meas['frozen'] / n:>13.3f}"
            f"{meas['swap'] / n:>11.3f}"
            f"{meas['decoded'] / n:>11.3f}"
        )
        print(f"             ({time.perf_counter() - started:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
