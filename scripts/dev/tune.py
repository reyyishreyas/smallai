"""Development harness for AI-267 tuning (NOT the scored protocol).

Discipline
----------
This harness exists to *develop* the policy. It deliberately never
reads the scored protocol's test split:

- Development uses its own master seed (``--seed``, default
  ``DEV_SEED``), which generates its own train / calibration / test
  scenarios via :func:`smallai.train.make_splits`.
- Every knob below is chosen by looking **only** at the development
  split.
- The scored run (``scripts/run_poc.py``, master seed 20261021) is
  never re-used for a tuning decision, and the reported Stage A
  result in ``reports/poc.md`` stays exactly as it is.

A tuning round that ends up claiming anything has to be re-confirmed
on a *fresh* master seed with brand-new splits; see
``scripts/dev/confirm.py``.

Usage
-----
    python scripts/dev/tune.py --set lr=0.5 gamma=0.95
    python scripts/dev/tune.py --sweep epsilon
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from smallai.agent import QAgent
from smallai.evaluate import derive_seed, score_agent, score_tree
from smallai.train import (
    MIX,
    MIX_EXPERT_ONLY,
    cross_validated_tree,
    make_splits,
    make_tree_dataset,
    oracle_label,
    train_td,
)

#: Master seed reserved for development. Deliberately different from
#: the scored seed (20261021) so no development decision can be
#: contaminated by the scored test split.
DEV_SEED = 20261030


@dataclass
class Config:
    """One training configuration.

    Attributes
    ----------
    lr : float
        TD learning rate.
    gamma : float
        Discount factor.
    epsilon : float
        Exploration rate during training.
    budget : int
        Episode cap for training.
    mix : str
        Which trajectory mix: ``"standard"`` (40/40/20) or
        ``"expert_only"`` (80/0/20).
    allow_recheck : bool
        False builds the NoRecheck-Q ablation.
    sigma : float
        Measurement noise scale to train and score at.
    name : str
        Label for the run.
    """

    lr: float = 0.3
    gamma: float = 0.99
    epsilon: float = 0.1
    budget: int = 200_000
    mix: str = "standard"
    allow_recheck: bool = True
    sigma: float = 0.08
    name: str = "baseline"

    def to_overrides(self) -> dict:
        """Config as a plain dict for JSON output."""
        return asdict(self)


@dataclass
class Result:
    """Metrics for one configuration.

    Attributes
    ----------
    name : str
        Run label.
    accuracy : float
        Test-split accuracy of the agent.
    tree_accuracy : float
        Test-split accuracy of the matched CART baseline.
    delta_pp : float
        Agent minus tree, in percentage points.
    invalid_rate : float
        Fraction of evidence-contradicting verdicts.
    mean_measurements : float
        Mean signal acquisitions per decision.
    review_rate : float
        Fraction of ``review`` verdicts.
    stored_entries : int
        Model size in stored (state, action) values.
    visited_states : int
        Distinct states visited in training.
    episodes : int
        Episodes actually run.
    plateau : bool
        Whether the plateau test fired.
    seconds : float
        Wall clock for the whole run.
    """

    name: str
    accuracy: float
    tree_accuracy: float
    delta_pp: float
    invalid_rate: float
    mean_measurements: float
    review_rate: float
    stored_entries: int
    visited_states: int
    episodes: int
    plateau: bool
    seconds: float
    extra: dict = field(default_factory=dict)


def _mix_for(name: str) -> dict[str, float]:
    """Map a mix name to the share dictionary."""
    if name == "standard":
        return MIX
    if name == "expert_only":
        return MIX_EXPERT_ONLY
    raise ValueError(f"Unknown mix {name!r}")


def _set_sigma(sigma: float):
    """Patch the environment noise scale for this run."""
    import smallai.environment as env

    previous = env.NOISE_SIGMA
    env.NOISE_SIGMA = sigma
    return previous


def run(config: Config, seed: int = DEV_SEED) -> Result:
    """Train one configuration and score it on the dev test split.

    Parameters
    ----------
    config : Config
        The configuration to run.
    seed : int
        Development master seed; generates the splits.

    Returns
    -------
    Result
        The measured metrics.
    """
    started = time.perf_counter()
    previous = _set_sigma(config.sigma)
    try:
        splits = make_splits(seed)
        labels = {
            name: [oracle_label(s) for s in scenarios]
            for name, scenarios in splits.items()
        }
        agent = QAgent(
            lr=config.lr,
            gamma=config.gamma,
            allow_recheck=config.allow_recheck,
            name=config.name,
        )
        report = train_td(
            agent,
            splits["train"],
            seed=derive_seed(seed, "train", config.name),
            budget=config.budget,
            mix=_mix_for(config.mix),
            epsilon=config.epsilon,
        )
        score = score_agent(
            agent,
            splits["test"],
            labels["test"],
            derive_seed(seed, "score", config.name),
        )
        # Matched baseline: same splits, same noise, same tree recipe.
        x_train, y_train = make_tree_dataset(
            splits["train"],
            labels["train"],
            random.Random(derive_seed(seed, "tree_fit", config.name)),
        )
        tree, leaf, cv = cross_validated_tree(
            x_train, y_train, derive_seed(seed, "tree_cv", config.name)
        )
        x_test, _ = make_tree_dataset(
            splits["test"],
            labels["test"],
            random.Random(derive_seed(seed, "tree_test", config.name)),
        )
        tree_score = score_tree(
            tree,
            x_test,
            splits["test"],
            labels["test"],
        )
    finally:
        import smallai.environment as env

        env.NOISE_SIGMA = previous

    return Result(
        name=config.name,
        accuracy=score.accuracy,
        tree_accuracy=tree_score.accuracy,
        delta_pp=100.0 * (score.accuracy - tree_score.accuracy),
        invalid_rate=score.invalid_rate,
        mean_measurements=score.mean_measurements,
        review_rate=score.review_rate,
        stored_entries=report.stored_entries,
        visited_states=report.visited_states,
        episodes=report.episodes,
        plateau=report.plateau,
        seconds=time.perf_counter() - started,
        extra={
            "tree_invalid_rate": tree_score.invalid_rate,
            "tree_mean_measurements": tree_score.mean_measurements,
            "tree_leaf": leaf,
            "tree_cv": cv,
            "final_block_mean": report.final_block_mean,
            "slope": report.slope,
            "ci_low": report.ci_low,
            "ci_high": report.ci_high,
            "n_expert": report.n_expert,
            "n_flawed": report.n_flawed,
            "n_on_policy": report.n_on_policy,
        },
    )


def format_row(result: Result) -> str:
    """Render one result as a fixed-width table row."""
    return (
        f"{result.name:<28}"
        f"{100 * result.accuracy:>8.3f}"
        f"{100 * result.tree_accuracy:>9.3f}"
        f"{result.delta_pp:>9.3f}"
        f"{100 * result.invalid_rate:>9.3f}"
        f"{result.mean_measurements:>8.3f}"
        f"{100 * result.review_rate:>8.2f}"
        f"{result.stored_entries:>8,}"
        f"{result.episodes:>9,}"
        f"{result.seconds:>8.1f}"
    )


HEADER = (
    f"{'run':<28}"
    f"{'acc%':>8}"
    f"{'tree%':>9}"
    f"{'delta':>9}"
    f"{'inval%':>9}"
    f"{'meas':>8}"
    f"{'rev%':>8}"
    f"{'entries':>8}"
    f"{'eps':>9}"
    f"{'sec':>8}"
)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEV_SEED)
    parser.add_argument("--budget", type=int, default=200_000)
    parser.add_argument("--sigma", type=float, default=0.08)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config field (repeatable).",
    )
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args(argv)

    config = Config(budget=args.budget, sigma=args.sigma, name="baseline")
    for pair in args.set:
        key, _, raw = pair.partition("=")
        key = key.strip()
        if not hasattr(config, key):
            raise SystemExit(f"Unknown config field {key!r}")
        current = getattr(config, key)
        if isinstance(current, bool):
            value = raw.strip().lower() in ("1", "true", "yes")
        elif isinstance(current, int):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        else:
            value = raw.strip()
        setattr(config, key, value)

    result = run(config, seed=args.seed)
    print(HEADER)
    print(format_row(result))
    if args.json:
        payload = {
            "config": config.to_overrides(),
            "seed": args.seed,
            "result": asdict(result),
        }
        Path(args.json).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
