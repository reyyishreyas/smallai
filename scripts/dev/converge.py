"""Convergence probe for AI-267 (development only, NOT the scored protocol).

Why
---
``smallai.train`` declares a plateau at the *first* check:
``MIN_EPISODES = 200_000`` and ``CHECK_INTERVAL = 50_000``, so the
pre-registered slope test fires at episode 200,000 on every run and
the protocol's 10^6-episode budget is never spent. The scored Stage A
run confirms it (``episodes = 200000``, ``plateau = true``,
``checked_at = 200000``).

That makes a real question open: is the Q-table actually converged at
200,000 episodes, or is the plateau test stopping a run that would
still improve? The learning-knob sweep could not answer it, because
every configuration was stopped at the same place.

This script trains with the burn-in pushed past the budget (so the
plateau test cannot fire) and compares against the frozen behaviour
on the very same development seeds, with the training and scoring
noise streams held identical so the comparison is paired.

Nothing here touches the scored protocol or the scored test split.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from smallai.agent import QAgent  # noqa: E402
from smallai.evaluate import derive_seed, score_agent  # noqa: E402
from smallai.train import MIX, make_splits, oracle_label, train_td  # noqa: E402

DEV_SEED = 20261030
AGENT_NAME = "converge_probe"


@dataclass(frozen=True)
class Variant:
    """One training-length configuration.

    Attributes
    ----------
    name : str
        Label.
    budget : int
        Episode cap.
    min_episodes : int
        Burn-in before the plateau test may fire.
    """

    name: str
    budget: int
    min_episodes: int


def run_variant(variant: Variant, seed: int) -> dict:
    """Train and score one configuration on a development split.

    Parameters
    ----------
    variant : Variant
        The configuration to run.
    seed : int
        Development master seed.

    Returns
    -------
    dict
        The measured metrics.
    """
    started = time.perf_counter()
    splits = make_splits(seed)
    labels = [oracle_label(s) for s in splits["test"]]
    agent = QAgent(name=AGENT_NAME)
    report = train_td(
        agent,
        splits["train"],
        seed=derive_seed(seed, "train", AGENT_NAME),
        budget=variant.budget,
        mix=MIX,
        min_episodes=variant.min_episodes,
    )
    score = score_agent(
        agent,
        splits["test"],
        labels,
        derive_seed(seed, "score", AGENT_NAME),
    )
    return {
        "accuracy": score.accuracy,
        "invalid_rate": score.invalid_rate,
        "mean_measurements": score.mean_measurements,
        "stored_entries": report.stored_entries,
        "visited_states": report.visited_states,
        "episodes": report.episodes,
        "plateau": report.plateau,
        "final_block_mean": report.final_block_mean,
        "seconds": time.perf_counter() - started,
    }


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEV_SEED)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args(argv)

    seeds = [args.seed + offset for offset in range(args.seeds)]

    variants = [
        Variant("frozen (stop at 200k)", 1_000_000, 200_000),
        Variant("burn-in 400k", 1_000_000, 400_000),
        Variant("burn-in 1M (no plateau)", 1_000_000, 1_000_000),
        Variant("budget 3M (no plateau)", 3_000_000, 3_000_000),
    ]

    header = (
        f"{'variant':<26}"
        f"{'acc% mean':>10}{'sd':>7}{'inval%':>9}{'meas':>8}"
        f"{'entries':>9}{'eps':>10}{'final reward':>15}"
        f"{'per-seed accuracy %':>40}"
    )
    print(header)
    print("-" * len(header))
    baseline: list[float] | None = None
    for variant in variants:
        rows = [run_variant(variant, seed) for seed in seeds]
        accuracies = [row["accuracy"] for row in rows]
        mean = sum(accuracies) / len(accuracies)
        variance = (
            sum((a - mean) ** 2 for a in accuracies) / (len(accuracies) - 1)
            if len(accuracies) > 1
            else 0.0
        )
        invalid = sum(row["invalid_rate"] for row in rows) / len(rows)
        meas = sum(row["mean_measurements"] for row in rows) / len(rows)
        entries = sum(row["stored_entries"] for row in rows) / len(rows)
        episodes = sum(row["episodes"] for row in rows) / len(rows)
        reward = sum(row["final_block_mean"] for row in rows) / len(rows)
        detail = " ".join(f"{100 * a:>7.3f}" for a in accuracies)
        print(
            f"{variant.name:<26}"
            f"{100 * mean:>10.3f}{100 * variance ** 0.5:>7.3f}"
            f"{100 * invalid:>9.3f}{meas:>8.3f}{entries:>9,.0f}"
            f"{episodes:>10,.0f}{reward:>15.5f}{detail:>40}"
        )
        if baseline is None:
            baseline = accuracies
        else:
            deltas = [
                100 * (a - b) for a, b in zip(accuracies, baseline)
            ]
            print(
                f"{'  paired delta (pp)':<26}"
                f"{sum(deltas) / len(deltas):>+10.3f}"
                f"{'':>7}{'':>9}{'':>8}{'':>9}{'':>10}{'':>15}"
                f"{' '.join(f'{d:>+7.3f}' for d in deltas):>40}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
