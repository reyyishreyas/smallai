"""Learning-knob sweep for AI-267 (development only, NOT the scored protocol).

Why
---
The reward sweep (``scripts/dev/reward_sweep.py``) showed the frozen
*reward* leaves roughly 0.4-0.7 pp of accuracy on the table. This
sweep asks the complementary question: are the frozen *learning
constants* (EXPERIMENT.md section 7: lr 0.3, gamma 0.99, epsilon 0.1,
budget 10^6, mix 40/40/20) also leaving accuracy on the table?

Every knob is varied one at a time from the frozen baseline, on
several development seeds, with the training and scoring noise
streams held identical across configurations so the comparison is
paired.

Nothing here touches the scored protocol or the scored test split.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dev.tune import Config, DEV_SEED, _mix_for  # noqa: E402

from smallai.agent import QAgent  # noqa: E402
from smallai.evaluate import derive_seed, score_agent  # noqa: E402
from smallai.train import make_splits, oracle_label, train_td  # noqa: E402

AGENT_NAME = "knob_probe"


def run_config(config: Config, seed: int) -> dict:
    """Train and score one configuration on a development split.

    Parameters
    ----------
    config : Config
        The configuration to run.
    seed : int
        Development master seed.

    Returns
    -------
    dict
        The measured metrics.
    """
    import smallai.environment as env

    previous = env.NOISE_SIGMA
    env.NOISE_SIGMA = config.sigma
    started = time.perf_counter()
    try:
        splits = make_splits(seed)
        labels = [oracle_label(s) for s in splits["test"]]
        agent = QAgent(
            lr=config.lr,
            gamma=config.gamma,
            allow_recheck=config.allow_recheck,
            name=AGENT_NAME,
        )
        report = train_td(
            agent,
            splits["train"],
            seed=derive_seed(seed, "train", AGENT_NAME),
            budget=config.budget,
            mix=_mix_for(config.mix),
            epsilon=config.epsilon,
        )
        score = score_agent(
            agent,
            splits["test"],
            labels,
            derive_seed(seed, "score", AGENT_NAME),
        )
    finally:
        env.NOISE_SIGMA = previous
    return {
        "accuracy": score.accuracy,
        "invalid_rate": score.invalid_rate,
        "mean_measurements": score.mean_measurements,
        "stored_entries": report.stored_entries,
        "episodes": report.episodes,
        "plateau": report.plateau,
        "seconds": time.perf_counter() - started,
    }


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEV_SEED)
    parser.add_argument("--budget", type=int, default=200_000)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args(argv)

    seeds = [args.seed + offset for offset in range(args.seeds)]
    base = Config(budget=args.budget)

    grid: list[tuple[str, Config]] = [("baseline", base)]
    for lr in (0.1, 0.2, 0.5, 0.7):
        grid.append((f"lr={lr}", replace(base, lr=lr)))
    for gamma in (0.95, 0.98, 1.0):
        grid.append((f"gamma={gamma}", replace(base, gamma=gamma)))
    for epsilon in (0.05, 0.2, 0.3):
        grid.append((f"epsilon={epsilon}", replace(base, epsilon=epsilon)))
    grid.append(("mix=expert_only", replace(base, mix="expert_only")))
    grid.append(("budget=500k", replace(base, budget=500_000)))
    grid.append(("no recheck", replace(base, allow_recheck=False)))

    header = (
        f"{'config':<20}"
        f"{'acc% mean':>10}{'sd':>7}{'inval%':>9}{'meas':>8}"
        f"{'entries':>10}{'per-seed accuracy %':>40}"
    )
    print(header)
    print("-" * len(header))
    baseline_accuracies: list[float] | None = None
    for name, config in grid:
        rows = [run_config(config, seed) for seed in seeds]
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
        detail = " ".join(f"{100 * a:>7.3f}" for a in accuracies)
        print(
            f"{name:<20}"
            f"{100 * mean:>10.3f}{100 * variance ** 0.5:>7.3f}"
            f"{100 * invalid:>9.3f}{meas:>8.3f}{entries:>10,.0f}"
            f"{detail:>40}"
        )
        if name == "baseline":
            baseline_accuracies = accuracies
        elif baseline_accuracies is not None:
            deltas = [
                100 * (a - b) for a, b in zip(accuracies, baseline_accuracies)
            ]
            print(
                f"{'  paired delta vs baseline (pp)':<20}"
                f"{sum(deltas) / len(deltas):>10.3f}"
                f"{'':>7}{'':>9}{'':>8}{'':>10}"
                f"{' '.join(f'{d:>7.3f}' for d in deltas):>40}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
