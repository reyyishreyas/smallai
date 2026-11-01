"""Reward-vs-accuracy diagnostic for AI-267.

Question
--------
The learned policy reaches 97.3-97.6 % accuracy while the Bayes
ceiling on its own path and its own encoding is 98.18 %. The missing
0.88 pp sits in the *terminal decision*. This script splits that gap
into two very different causes:

1. **Under-trained table.** The Q-values have not converged, so the
   greedy terminal is not even the reward-optimal one.
2. **Misaligned reward.** The table has converged on the frozen
   reward, but that reward's verdict/consistency terms do not point
   at the accuracy-optimal terminal.

Method
------
For every scenario on the development split, the policy's own
measurement path is replayed. The exact posterior over the oracle
label given the observations is then computed (see
:mod:`scripts.dev.ceiling`), and three terminal choices are compared
at the very same state:

- ``policy`` -- what the Q-table actually picks;
- ``accuracy`` -- ``argmax_l P(label = l | observations)``;
- ``reward`` -- ``argmax_a [ sum_l P(l) * verdict_reward(a, l)
  + consistency_bonus(a, state) ]``.

Reporting the *expected* reward of each choice under the same
posterior separates the two causes cleanly: if the policy's choice
scores below the reward-optimal choice, the table is under-trained;
if it scores at or above it, the reward itself is the misaligned
part.

Development-time analysis only. It never touches the scored test
split.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ceiling as C  # noqa: E402

from smallai.agent import QAgent  # noqa: E402
from smallai.evaluate import derive_seed  # noqa: E402
from smallai.train import (  # noqa: E402
    MIX,
    REWARD_MEASUREMENT,
    consistency_bonus,
    make_splits,
    oracle_label,
    train_td,
    verdict_reward,
)

DEV_SEED = 20261030
TERMINALS = ("review", "retake", "pass")


def _expected_reward(
    action: str, posterior: dict[str, float], state: tuple[int, ...]
) -> float:
    """Expected frozen verdict + consistency reward of a terminal action.

    Parameters
    ----------
    action : str
        The terminal action.
    posterior : dict
        Label to probability.
    state : tuple of int
        The evidence gathered when the action was taken.

    Returns
    -------
    float
        The expected reward.
    """
    verdict = sum(
        prob * verdict_reward(action, label)
        for label, prob in posterior.items()
    )
    return verdict + consistency_bonus(action, state)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEV_SEED)
    parser.add_argument("--sigma", type=float, default=C.NOISE_SIGMA)
    parser.add_argument("--budget", type=int, default=200_000)
    args = parser.parse_args(argv)

    splits = make_splits(args.seed)
    scenarios = splits["test"]
    labels = [oracle_label(s) for s in scenarios]
    hypotheses = C._hypotheses()

    agent = QAgent(name="journey_q")
    train_td(
        agent,
        splits["train"],
        seed=derive_seed(args.seed, "train", "journey_q"),
        budget=args.budget,
        mix=MIX,
    )

    rep = next(
        r for r in C._make_representations() if r.name == "zone3 (current)"
    )
    triggers = {
        name: frozenset(
            index
            for index, zone in enumerate(rep.signals[name].zone)
            if zone == "suspect"
        )
        for name in C.SIGNALS
    }
    tables = C._build_tables(rep, hypotheses, args.sigma, triggers=triggers)

    hits = Counter()
    policy_vs_reward = Counter()
    policy_vs_accuracy = Counter()
    reward_value = {name: 0.0 for name in ("policy", "accuracy", "reward")}
    expected_reward_gap = 0.0
    measurements = 0

    for index, (scenario, label) in enumerate(zip(scenarios, labels)):
        rng = random.Random(derive_seed(args.seed, "journey_q", str(index)))
        observation, policy_action, state = C._policy_path(
            agent, scenario, rng, args.sigma
        )
        measurements += observation.measurements
        posterior = C._posterior(
            observation, rep, hypotheses, tables, args.sigma, triggers
        )

        accuracy_action = max(
            posterior.items(), key=lambda kv: (kv[1], kv[0])
        )[0]
        reward_action = max(
            TERMINALS,
            key=lambda a: (_expected_reward(a, posterior, state), a),
        )

        for name, action in (
            ("policy", policy_action),
            ("accuracy", accuracy_action),
            ("reward", reward_action),
        ):
            if action == label:
                hits[name] += 1
            reward_value[name] += _expected_reward(
                action, posterior, state
            ) + REWARD_MEASUREMENT * observation.measurements

        policy_vs_reward[(policy_action, reward_action)] += 1
        policy_vs_accuracy[(policy_action, accuracy_action)] += 1
        expected_reward_gap += _expected_reward(
            reward_action, posterior, state
        ) - _expected_reward(policy_action, posterior, state)

    n = len(scenarios)
    print("=" * 78)
    print("REWARD VS ACCURACY DIAGNOSTIC")
    print(f"dev test split, {n:,} scenarios, budget={args.budget:,}")
    print("=" * 78)
    print()
    print("accuracy of each terminal choice at the policy's own state:")
    for name in ("policy", "accuracy", "reward"):
        print(f"  {name:<10}{100 * hits[name] / n:>8.3f} %")
    print()
    print("expected frozen episode reward of each choice:")
    for name in ("policy", "accuracy", "reward"):
        print(f"  {name:<10}{reward_value[name] / n:>12.5f}")
    print()
    print(
        "expected reward forgone by the policy's own terminal "
        f"vs the reward-optimal terminal: {expected_reward_gap / n:+.5f}"
    )
    print(f"mean measurements: {measurements / n:.3f}")
    print()
    print("policy terminal vs reward-optimal terminal:")
    for (policy_action, reward_action), count in sorted(
        policy_vs_reward.items(), key=lambda kv: -kv[1]
    ):
        flag = "  <-- policy is not reward-optimal" if policy_action != reward_action else ""
        print(f"  {policy_action:<8}-> {reward_action:<8}{count:>5}{flag}")
    print()
    print("policy terminal vs accuracy-optimal terminal:")
    for (policy_action, accuracy_action), count in sorted(
        policy_vs_accuracy.items(), key=lambda kv: -kv[1]
    ):
        flag = "  <-- policy is not accuracy-optimal" if policy_action != accuracy_action else ""
        print(f"  {policy_action:<8}-> {accuracy_action:<8}{count:>5}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
