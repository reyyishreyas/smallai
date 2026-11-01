"""Reward-shape sweep for AI-267 (development only, NOT the scored protocol).

Why
---
The reward diagnostic (``scripts/dev/reward.py``) showed that the
learned table has converged on the frozen reward, but that the frozen
reward's own optimum sits about 0.5 pp below the accuracy optimum.
The frozen reward has three parts:

- the **verdict term** -- +1.0 correct, -0.5 adjacent wrong, -1.0
  opposite wrong. Asymmetric, so it is *not* a monotone function of
  "right": with ``P(pass)=0.40, P(review)=0.35, P(retake)=0.25`` the
  frozen reward prefers ``review`` over ``pass`` even though accuracy
  prefers ``pass``. A symmetric verdict term (``-x`` for any wrong
  verdict) makes the expected reward monotone in ``P(correct)`` and
  therefore accuracy-aligned.
- the **consistency term** -- +0.10 when the verdict matches the
  observed evidence, -0.30 when it contradicts it. This is a hard
  0.40 swing that can override the evidence: a signal whose reading
  landed in the suspect band is often *truly* below the fail
  threshold, so the accuracy-optimal verdict there is ``retake`` while
  the consistency term forbids it. A *one-sided* consistency term
  (0 when consistent, -0.30 when contradictory) keeps the pressure to
  avoid contradictions without ever pulling the decision.
- the **measurement cost** -- -0.05 per acquisition.

This sweep measures each part end to end: train a fresh agent per
variant on development seeds, then score it on the development test
split. Every variant is a *candidate for a new, pre-registered round*.
None of them touches the frozen protocol, the scored pipeline or the
scored test split.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import smallai.train as train_mod  # noqa: E402

from smallai.agent import QAgent  # noqa: E402
from smallai.agent import FAIL_FINAL, PASS_FINAL, SUSPECT_POST, SUSPECT_PRE  # noqa: E402
from smallai.evaluate import derive_seed, score_agent  # noqa: E402
from smallai.train import (  # noqa: E402
    MIX,
    REWARD_CORRECT,
    make_splits,
    oracle_label,
    train_td,
)

DEV_SEED = 20261030

ORIGINAL_VERDICT = train_mod.verdict_reward
ORIGINAL_CONSISTENCY = train_mod.consistency_bonus
ORIGINAL_COST = train_mod.REWARD_MEASUREMENT

AGENT_NAME = "reward_probe"


@dataclass(frozen=True)
class Variant:
    """One candidate reward shape.

    Attributes
    ----------
    name : str
        Label.
    adjacent : float
        Reward for an adjacent-wrong verdict.
    opposite : float
        Reward for an opposite-wrong verdict.
    consistency : float
        Multiplier on the consistency term.
    shape : str
        ``"frozen"`` (+1 consistent / -3 contradictory),
        ``"one_sided"`` (0 consistent / -3 contradictory) or
        ``"symmetric"`` (+1 consistent / -1 contradictory).
    cost : float
        Reward per measurement.
    """

    name: str
    adjacent: float
    opposite: float
    consistency: float = 1.0
    shape: str = "frozen"
    cost: float = -0.05


def _verdict_factory(variant: Variant):
    """Build a verdict function for one variant.

    Parameters
    ----------
    variant : Variant
        The candidate shape.

    Returns
    -------
    callable
        A drop-in replacement for ``verdict_reward``.
    """

    def verdict_reward(action: str, label: str) -> float:
        """Reward for one terminal verdict.

        Parameters
        ----------
        action : str
            The terminal action.
        label : str
            The oracle verdict.

        Returns
        -------
        float
            The verdict reward.
        """
        if action == label:
            return REWARD_CORRECT
        if {action, label} == {"pass", "retake"}:
            return variant.opposite
        return variant.adjacent

    return verdict_reward


def _consistency_factory(variant: Variant):
    """Build a consistency function for one variant.

    Parameters
    ----------
    variant : Variant
        The candidate shape.

    Returns
    -------
    callable
        A drop-in replacement for ``consistency_bonus``.
    """
    if variant.shape == "frozen":
        def shape(action: str, state: tuple[int, ...]) -> float:
            """Frozen consistency shape.

            Parameters
            ----------
            action : str
                The terminal action.
            state : tuple of int
                The evidence gathered.

            Returns
            -------
            float
                +1 when consistent, -3 when contradictory.
            """
            return 1.0 if ORIGINAL_CONSISTENCY(action, state) > 0 else -3.0

    elif variant.shape == "one_sided":
        def shape(action: str, state: tuple[int, ...]) -> float:
            """One-sided consistency shape.

            Parameters
            ----------
            action : str
                The terminal action.
            state : tuple of int
                The evidence gathered.

            Returns
            -------
            float
                0 when consistent, -3 when contradictory.
            """
            return 0.0 if ORIGINAL_CONSISTENCY(action, state) > 0 else -3.0

    elif variant.shape == "symmetric":
        def shape(action: str, state: tuple[int, ...]) -> float:
            """Symmetric consistency shape.

            Parameters
            ----------
            action : str
                The terminal action.
            state : tuple of int
                The evidence gathered.

            Returns
            -------
            float
                +1 when consistent, -1 when contradictory.
            """
            return 1.0 if ORIGINAL_CONSISTENCY(action, state) > 0 else -1.0

    elif variant.shape == "broad":
        def shape(action: str, state: tuple[int, ...]) -> float:
            """Broad consistency shape: a suspect does not refute retake.

            The frozen shape calls ``retake`` contradictory unless a
            *fail* was observed. But a suspect reading is exactly the
            ambiguous case: the latent value may sit below the fail
            threshold, so "retake" is a plausible reading of that
            evidence, not a contradiction of it. Under this shape
            ``retake`` is consistent whenever a fail *or* a suspect
            was observed, and the consistency term stops arbitrating
            the retake/review pair -- the verdict term does, and the
            verdict term prefers the more likely label.

            Parameters
            ----------
            action : str
                The terminal action.
            state : tuple of int
                The evidence gathered.

            Returns
            -------
            float
                +1 when plausible, -3 when contradictory.
            """
            has_fail = FAIL_FINAL in state
            has_suspect = SUSPECT_PRE in state or SUSPECT_POST in state
            if action == "pass":
                plausible = all(code == PASS_FINAL for code in state)
            elif action == "retake":
                plausible = has_fail or has_suspect
            else:
                plausible = has_suspect
            return 1.0 if plausible else -3.0

    else:
        raise ValueError(f"Unknown consistency shape {variant.shape!r}")

    def consistency_bonus(action: str, state: tuple[int, ...]) -> float:
        """Scaled consistency term.

        Parameters
        ----------
        action : str
            The terminal action.
        state : tuple of int
            The evidence gathered.

        Returns
        -------
        float
            The scaled consistency reward.
        """
        return variant.consistency * shape(action, state)

    return consistency_bonus


def run_variant(variant: Variant, seed: int, budget: int) -> dict:
    """Train and score one reward variant on the development split.

    Parameters
    ----------
    variant : Variant
        The candidate reward shape.
    seed : int
        Development master seed.
    budget : int
        Episode cap.

    Returns
    -------
    dict
        The measured metrics.
    """
    # The agent name is deliberately *not* variant-specific: the
    # training and scoring noise streams are then identical across
    # variants, so the comparison is paired on the evaluation noise.
    train_mod.verdict_reward = _verdict_factory(variant)
    train_mod.consistency_bonus = _consistency_factory(variant)
    train_mod.REWARD_MEASUREMENT = variant.cost
    started = time.perf_counter()
    try:
        splits = make_splits(seed)
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
    finally:
        train_mod.verdict_reward = ORIGINAL_VERDICT
        train_mod.consistency_bonus = ORIGINAL_CONSISTENCY
        train_mod.REWARD_MEASUREMENT = ORIGINAL_COST
    return {
        "name": variant.name,
        "accuracy": score.accuracy,
        "invalid_rate": score.invalid_rate,
        "mean_measurements": score.mean_measurements,
        "review_rate": score.review_rate,
        "stored_entries": report.stored_entries,
        "episodes": report.episodes,
        "plateau": report.plateau,
        "final_block_mean": report.final_block_mean,
        "seconds": time.perf_counter() - started,
    }


def default_variants() -> list[Variant]:
    """The candidate reward shapes to scan.

    Returns
    -------
    list
        The variants.
    """
    return [
        Variant("frozen (Stage A)", -0.5, -1.0),
        Variant("broad cons", -0.5, -1.0, shape="broad"),
        Variant("broad cons x0.5", -0.5, -1.0, consistency=0.5, shape="broad"),
        Variant("sym -0.5 + broad", -0.5, -0.5, shape="broad"),
        Variant("sym -0.5 + broad x0.5", -0.5, -0.5, consistency=0.5, shape="broad"),
        Variant("sym -0.25 + broad", -0.25, -0.25, shape="broad"),
        Variant("sym -0.5 + frozen", -0.5, -0.5),
        Variant("sym -0.5 + frozen x0.5", -0.5, -0.5, consistency=0.5),
    ]


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEV_SEED)
    parser.add_argument("--budget", type=int, default=200_000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated variant names to run.",
    )
    args = parser.parse_args(argv)

    seeds = [args.seed + offset for offset in range(args.seeds)]
    variants = default_variants()
    if args.only:
        wanted = {name.strip() for name in args.only.split(",")}
        variants = [v for v in variants if v.name in wanted]

    header = (
        f"{'variant':<28}"
        f"{'acc% mean':>10}{'sd':>7}{'inval%':>9}{'meas':>8}"
        f"{'paired delta vs frozen (pp)':>30}"
        f"{'per-seed accuracy %':>40}"
    )
    print(header)
    print("-" * len(header))
    baseline_accuracies: list[float] | None = None
    for variant in variants:
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
        detail = " ".join(f"{100 * a:>7.3f}" for a in accuracies)
        if baseline_accuracies is None:
            delta_text = ""
        else:
            deltas = [
                100 * (a - b)
                for a, b in zip(accuracies, baseline_accuracies)
            ]
            delta_mean = sum(deltas) / len(deltas)
            delta_var = sum((d - delta_mean) ** 2 for d in deltas) / (
                len(deltas) - 1
            )
            delta_se = delta_var ** 0.5 / len(deltas) ** 0.5
            t_stat = delta_mean / delta_se if delta_se > 0 else 0.0
            delta_text = (
                f"{delta_mean:+8.3f}+-{delta_se:<5.3f}(t={t_stat:>5.2f})"
                f" {' '.join(f'{d:>+6.2f}' for d in deltas)}"
            )
        print(
            f"{variant.name:<28}"
            f"{100 * mean:>10.3f}{100 * variance ** 0.5:>7.3f}"
            f"{100 * invalid:>9.3f}{meas:>8.3f}"
            f"{delta_text:>30}{detail:>40}"
        )
        if baseline_accuracies is None:
            baseline_accuracies = accuracies
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
