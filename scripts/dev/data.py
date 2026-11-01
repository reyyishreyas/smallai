"""Training-data probe for AI-267 (development only).

Why
---
Every accuracy lever tried so far has come back null or negative:
richer state encoding, check order, reward shape, learning
constants, training length, Q-table ensembling. The one ingredient
that has never been varied is the *training data itself*.

Section 4 of the protocol fixes the train split at 4,000 scenarios
drawn from the skewed, field-realistic mixture (blur 30 %, every
other defect 5 %), while calibration and test are drawn from the
balanced mixture (blur 10 %, every other defect 10 %). The policy is
therefore fitted under a different defect prior than the one it is
scored under, and it sees only 4,000 distinct captures.

Both halves of that are worth a measurement:

- *Quantity*: 4,000 distinct scenarios over 200,000 episodes is fifty
  passes over a small pool, so a larger pool may generalise better.
- *Prior*: the skewed prior changes the posterior of "is this
  suspect signal really a defect", which is exactly the
  review-vs-retake decision the policy gets wrong most often.

This script trains the frozen policy on four training sets and scores
each on the *same* test split with the *same* score seed, so the
comparison is paired and only the training data varies.

Nothing here touches the scored protocol or the scored test split.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from smallai.agent import QAgent  # noqa: E402
from smallai.environment import (  # noqa: E402
    SIGNALS,
    TEST_WEIGHTS,
    TRAIN_WEIGHTS,
    sample_scenario,
)
from smallai.evaluate import derive_seed, score_agent  # noqa: E402
from smallai.train import MIX, oracle_label, train_td  # noqa: E402

DEV_SEED = 20261030
AGENT_NAME = "data_probe"
TEST_N = 1_000


@dataclass(frozen=True)
class Variant:
    """One training-data configuration.

    Attributes
    ----------
    name : str
        Label.
    train_n : int
        Distinct training captures.
    balanced : bool
        Whether the training captures come from the balanced mixture.
    """

    name: str
    train_n: int
    balanced: bool


def make_splits(
    seed: int, train_n: int, balanced: bool
) -> dict[str, list]:
    """Draw splits with a configurable training set.

    Mirrors :func:`smallai.train.make_splits` -- uniqueness is
    enforced across the combined pool, and the test split always uses
    the balanced mixture.

    Parameters
    ----------
    seed : int
        Master seed.
    train_n : int
        Number of distinct training captures.
    balanced : bool
        Whether the training captures use the balanced mixture.

    Returns
    -------
    dict
        The three splits.
    """
    rng = random.Random(seed)
    seen: set[tuple[float, ...]] = set()
    splits: dict[str, list] = {"train": [], "calibration": [], "test": []}
    plan = (
        ("train", train_n, TEST_WEIGHTS if balanced else TRAIN_WEIGHTS),
        ("calibration", 1_000, TEST_WEIGHTS),
        ("test", TEST_N, TEST_WEIGHTS),
    )
    for name, count, weights in plan:
        drawn: list = []
        while len(drawn) < count:
            scenario = sample_scenario(rng, weights)
            key = tuple(round(scenario.signals[s], 12) for s in SIGNALS)
            if key in seen:
                continue
            seen.add(key)
            drawn.append(scenario)
        splits[name] = drawn
    return splits


def run_variant(variant: Variant, seed: int, budget: int) -> dict:
    """Train and score one training-data configuration.

    Parameters
    ----------
    variant : Variant
        The configuration to run.
    seed : int
        Development master seed.
    budget : int
        Episode cap.

    Returns
    -------
    dict
        The measured metrics.
    """
    splits = make_splits(seed, variant.train_n, variant.balanced)
    labels = [oracle_label(s) for s in splits["test"]]
    agent = QAgent(name=AGENT_NAME)
    report = train_td(
        agent,
        splits["train"],
        seed=derive_seed(seed, "train", AGENT_NAME),
        budget=budget,
        mix=MIX,
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
    }


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEV_SEED)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--budget", type=int, default=200_000)
    args = parser.parse_args(argv)

    seeds = [args.seed + offset for offset in range(args.seeds)]
    variants = [
        Variant("frozen (4k skewed)", 4_000, False),
        Variant("4k balanced", 4_000, True),
        Variant("16k skewed", 16_000, False),
        Variant("16k balanced", 16_000, True),
    ]

    header = (
        f"{'variant':<22}{'acc% mean':>10}{'sd':>7}{'inval%':>9}"
        f"{'meas':>8}{'entries':>9}{'paired delta vs frozen (pp)':>34}"
    )
    print(header)
    print("-" * len(header))
    baseline: list[float] | None = None
    for variant in variants:
        started = time.perf_counter()
        rows = [run_variant(variant, seed, args.budget) for seed in seeds]
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
        if baseline is None:
            delta_text = ""
        else:
            deltas = [
                100 * (a - b) for a, b in zip(accuracies, baseline)
            ]
            delta_mean = sum(deltas) / len(deltas)
            delta_var = sum((d - delta_mean) ** 2 for d in deltas) / (
                len(deltas) - 1
            )
            delta_se = delta_var ** 0.5 / len(deltas) ** 0.5
            t_stat = delta_mean / delta_se if delta_se > 0 else 0.0
            delta_text = (
                f"{delta_mean:+8.3f}+-{delta_se:<5.3f}(t={t_stat:>6.2f})"
            )
        print(
            f"{variant.name:<22}{100 * mean:>10.3f}"
            f"{100 * variance ** 0.5:>7.3f}{100 * invalid:>9.3f}"
            f"{meas:>8.3f}{entries:>9,.0f}{delta_text:>34}"
            f"  ({time.perf_counter() - started:.0f} s)"
        )
        if baseline is None:
            baseline = accuracies
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
