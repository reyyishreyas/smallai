"""Seed-ensemble probe for AI-267 (development only, NOT the scored protocol).

Why
---
The one large, reproducible effect left after the reward, knob,
encoding and length sweeps is *training-seed variance*: trained on
the very same splits, the same code and the same budget, Journey-Q
lands anywhere between roughly 96.9 % and 98.9 % accuracy purely
because of the seed that drives its training episodes. The scored
Stage A run is a single draw from that distribution, so half of the
time it is a poor draw and no amount of protocol cleverness shows up
in the report.

Every member of an ensemble learns the *same* MDP and therefore the
same optimal value function Q*, so the members are independent noisy
estimates of one quantity. Averaging them is plain estimator
variance reduction -- the same idea as a deep ensemble in supervised
learning -- and it needs no change to the reward, the learning rule,
the state encoding or any frozen constant.

This script trains K members that differ only in their training
seed, then scores:

- each member alone (the spread of the single-agent distribution);
- the Q-averaged ensemble;
- the Q-averaged ensemble with unvisited entries counted as zero.

Everything is scored with one fixed score seed and one shared agent
name, so the per-scenario noise stream is identical for every
policy and the comparison is paired.

Nothing here touches the scored protocol or the scored test split.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from smallai.agent import TERMINALS, QAgent  # noqa: E402
from smallai.evaluate import derive_seed, mcnemar, score_agent  # noqa: E402
from smallai.train import MIX, make_splits, oracle_label, train_td  # noqa: E402

DEV_SEED = 20261030
AGENT_NAME = "ensemble_probe"


def _averaged_table(
    members: list[QAgent], fill_zero: bool
) -> dict[tuple[int, ...], dict[str, float]]:
    """Average the members' Q-tables.

    Parameters
    ----------
    members : list of QAgent
        The independently trained members.
    fill_zero : bool
        When True, a member that never visited an action counts as
        an estimate of 0.0 instead of being skipped.

    Returns
    -------
    dict
        ``state -> {action -> averaged value}``.
    """
    table: dict[tuple[int, ...], dict[str, float]] = {}
    for member in members:
        for state, actions in member.q.items():
            target = table.setdefault(state, {})
            for action, value in actions.items():
                target.setdefault(action, []).append(value)  # type: ignore[arg-type]
    averaged: dict[tuple[int, ...], dict[str, float]] = {}
    for state, actions in table.items():
        row: dict[str, float] = {}
        for action, values in actions.items():
            if fill_zero:
                values = list(values) + [0.0] * (len(members) - len(values))
            row[action] = sum(values) / len(values)
        averaged[state] = row
    return averaged


def _ensemble_agent(
    members: list[QAgent], name: str, fill_zero: bool
) -> QAgent:
    """Build an agent whose table is the average of the members'.

    Parameters
    ----------
    members : list of QAgent
        The independently trained members.
    name : str
        The ensemble's label.
    fill_zero : bool
        Whether unvisited entries count as zero.

    Returns
    -------
    QAgent
        The ensemble, sharing ``QAgent``'s decision rule exactly.
    """
    ensemble = QAgent(name=name)
    ensemble.q = _averaged_table(members, fill_zero)
    return ensemble


def _train_member(seed: int, index: int, budget: int) -> QAgent:
    """Train one ensemble member on the development train split.

    Parameters
    ----------
    seed : int
        Development master seed (fixes the splits).
    index : int
        The member index, which selects its training seed.
    budget : int
        Episode cap.

    Returns
    -------
    QAgent
        The trained member.
    """
    splits = make_splits(seed)
    member = QAgent(name=AGENT_NAME)
    train_td(
        member,
        splits["train"],
        seed=derive_seed(seed, "train", f"{AGENT_NAME}_{index}"),
        budget=budget,
        mix=MIX,
    )
    return member


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEV_SEED)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--budget", type=int, default=200_000)
    args = parser.parse_args(argv)

    for master in [args.seed + offset for offset in range(args.seeds)]:
        splits = make_splits(master)
        test_labels = [oracle_label(s) for s in splits["test"]]
        cal_labels = [oracle_label(s) for s in splits["calibration"]]
        score_seed = derive_seed(master, "score", AGENT_NAME)

        started = time.perf_counter()
        members = [
            _train_member(master, index, args.budget)
            for index in range(args.members)
        ]

        member_scores = [
            score_agent(member, splits["test"], test_labels, score_seed)
            for member in members
        ]
        accuracies = [100 * score.accuracy for score in member_scores]
        invalid = [100 * score.invalid_rate for score in member_scores]
        meas = [score.mean_measurements for score in member_scores]

        print(f"=== master seed {master} ===")
        print(
            "  members:  "
            + "  ".join(f"{a:.1f}" for a in accuracies)
            + f"   | mean {statistics.mean(accuracies):.3f}"
            f"  median {statistics.median(accuracies):.3f}"
            f"  min {min(accuracies):.3f}"
            f"  max {max(accuracies):.3f}"
            f"  sd {statistics.pstdev(accuracies):.3f}"
        )
        print(
            "  invalid %:"
            + "  ".join(f"{v:.1f}" for v in invalid)
            + f"   | mean {statistics.mean(invalid):.3f}"
        )
        print(
            "  measure: "
            + "  ".join(f"{v:.2f}" for v in meas)
            + f"   | mean {statistics.mean(meas):.3f}"
        )

        for label, fill_zero in (("ensemble (mean of estimates)", False),
                                 ("ensemble (zero-filled)", True)):
            ensemble = _ensemble_agent(members, AGENT_NAME, fill_zero)
            score = score_agent(
                ensemble, splits["test"], test_labels, score_seed
            )
            cal = score_agent(
                ensemble, splits["calibration"], cal_labels, score_seed
            )
            delta = 100 * (score.accuracy - statistics.mean(
                [s.accuracy for s in member_scores]
            ))
            best = max(member_scores, key=lambda s: s.accuracy)
            worst = min(member_scores, key=lambda s: s.accuracy)
            against_best = mcnemar(best.correct, score.correct)
            against_worst = mcnemar(worst.correct, score.correct)
            print(
                f"  {label:<28} acc {100 * score.accuracy:.3f} "
                f"({delta:+.3f} pp vs member mean)"
                f"  invalid {100 * score.invalid_rate:.3f}%"
                f"  meas {score.mean_measurements:.3f}"
                f"  cal {100 * cal.accuracy:.3f}%"
                f"  entries {ensemble.stored_entries:,}"
            )
            print(
                f"      vs best member : member-only {against_best.first_only}, "
                f"ensemble-only {against_best.second_only} "
                f"(p={against_best.p_value:.4f})"
            )
            print(
                f"      vs worst member: member-only {against_worst.first_only}, "
                f"ensemble-only {against_worst.second_only} "
                f"(p={against_worst.p_value:.4f})"
            )
        print(f"  ({time.perf_counter() - started:.1f} s)")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
