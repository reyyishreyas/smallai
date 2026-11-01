"""Statistical-power probe for AI-267 (development only).

The question
------------
The ticket's graded criterion is "journey learning >= X% over the
shortcut" (ARCHITECTURE.md section 1, EXPERIMENT.md section 10, H1).
Stage A measured the Journey-Q minus Shortcut-CART delta on the
protocol's 1,000-scenario test split as +0.300 pp with a 95 % CI of
[-0.400, 1.000] -- inconclusive.

That interval is a *power* statement, not a verdict on the method:
the two policies disagree on only a handful of scenarios, so a
1,000-scenario split cannot resolve a sub-percentage-point effect no
matter how good the policy is.

This probe measures the same delta with the same policies, the same
splits and the same noise model, but on a large freshly drawn
evaluation set, so the estimate converges on the true effect and the
interval says something about the method instead of about the sample
size.

Everything is fitted on the development train split; the evaluation
set is drawn from the balanced test mixture with its own seed and is
never used for fitting or selection.

Nothing here touches the scored protocol or the scored test split.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from smallai.agent import QAgent  # noqa: E402
from smallai.environment import TEST_WEIGHTS, sample_scenario  # noqa: E402
from smallai.evaluate import (  # noqa: E402
    derive_seed,
    mcnemar,
    paired_bootstrap_delta,
    score_agent,
    score_tree,
)
from smallai.train import (  # noqa: E402
    MIX,
    cross_validated_tree,
    make_splits,
    make_tree_dataset,
    oracle_label,
    train_td,
)

DEV_SEED = 20261030
AGENT_NAME = "power_probe"


def draw_evaluation_set(seed: int, count: int):
    """Draw a fresh balanced evaluation set.

    Parameters
    ----------
    seed : int
        Seed for the draw.
    count : int
        Number of scenarios.

    Returns
    -------
    tuple
        The scenarios and their oracle labels.
    """
    rng = random.Random(seed)
    scenarios = [sample_scenario(rng, TEST_WEIGHTS) for _ in range(count)]
    labels = [oracle_label(scenario) for scenario in scenarios]
    return scenarios, labels


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEV_SEED)
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--budget", type=int, default=200_000)
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args(argv)

    splits = make_splits(args.seed)
    train = splits["train"]
    train_labels = [oracle_label(s) for s in train]

    started = time.perf_counter()
    agent = QAgent(name=AGENT_NAME)
    report = train_td(
        agent,
        train,
        seed=derive_seed(args.seed, "train", AGENT_NAME),
        budget=args.budget,
        mix=MIX,
    )
    x_train, y_train = make_tree_dataset(
        train,
        train_labels,
        random.Random(derive_seed(args.seed, "tree_fit", "0.08")),
    )
    tree, leaf, fold_scores = cross_validated_tree(
        x_train,
        y_train,
        seed=derive_seed(args.seed, "tree_cv", "0.08"),
    )
    print(
        f"trained journey_q ({report.episodes:,} episodes, "
        f"{report.stored_entries:,} entries) and shortcut_cart "
        f"(leaf={leaf}, folds={[round(s, 3) for s in fold_scores]}) "
        f"in {time.perf_counter() - started:.1f} s"
    )

    scenarios, labels = draw_evaluation_set(
        derive_seed(args.seed, "eval", str(args.n)), args.n
    )
    agent_score = score_agent(
        agent,
        scenarios,
        labels,
        derive_seed(args.seed, "score", AGENT_NAME),
    )
    x_eval, _ = make_tree_dataset(
        scenarios,
        labels,
        random.Random(derive_seed(args.seed, "tree_test", "0.08")),
    )
    tree_score = score_tree(tree, x_eval, scenarios, labels)

    print()
    print(f"evaluation set: {args.n:,} fresh balanced scenarios")
    print(f"  journey_q     accuracy {100 * agent_score.accuracy:.3f} %")
    print(f"  shortcut_cart accuracy {100 * tree_score.accuracy:.3f} %")
    print(
        f"  journey_q     invalid {100 * agent_score.invalid_rate:.3f} %"
        f"   measurements {agent_score.mean_measurements:.3f}"
    )
    print(
        f"  shortcut_cart invalid {100 * tree_score.invalid_rate:.3f} %"
        f"   measurements {tree_score.mean_measurements:.3f}"
    )

    delta = paired_bootstrap_delta(
        agent_score.correct,
        tree_score.correct,
        seed=derive_seed(args.seed, "bootstrap", str(args.n)),
        resamples=args.resamples,
    )
    test = mcnemar(agent_score.correct, tree_score.correct)
    print()
    print(
        f"  delta (journey - shortcut): {delta.delta_pp:+.3f} pp"
        f"   95% CI [{delta.ci_low_pp:+.3f}, {delta.ci_high_pp:+.3f}]"
        f"   excludes zero: {delta.excludes_zero}"
    )
    print(
        f"  McNemar: journey-only {test.second_only}, "
        f"shortcut-only {test.first_only}, p={test.p_value:.4f}"
    )
    print()
    print(
        "  for reference, the same comparison on the protocol's"
        " 1,000-scenario test split is only worth about +/-"
        f"{100 * (1.96 * (0.978 * 0.022 / 1000) ** 0.5):.3f} pp"
        " of one-sigma noise"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
