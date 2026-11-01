"""Diagnostics for AI-267 policy development (NOT the scored protocol).

Answers one decisive question:

    When the policy gets a verdict wrong, is the fault the *learned
    table* (it chose a worse terminal than the best one available at
    that state) or the *state representation* (the state cannot tell
    two different situations apart, so the best terminal was also
    wrong)?

The two have completely different fixes:

- table fault -> improve learning, reward shaping, exploration
- representation fault -> richer observation, or accept the ceiling

The state codes here are produced by ``smallai.agent.transition`` --
the exact function the policy uses -- so the analysed states and the
policy's states are the same objects. The posterior P(label | state)
is estimated from a large independent sample, not from the same
scenarios the policy is scored on.

Everything here runs on the development seed's own splits and never
touches the scored protocol's test split.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from smallai.agent import (
    TERMINALS,
    UNCHECKED,
    QAgent,
    State,
    transition,
)
from smallai.environment import (
    FAIL,
    PASS,
    SIGNALS,
    Scenario,
    observe,
    sample_scenario,
)
from smallai.evaluate import derive_seed, score_agent
from smallai.solver import ORDER
from smallai.train import MIX, make_splits, oracle_label, train_td

DEV_SEED = 20261030

#: Sentinel for a signal the expert never measured (it exits early on
#: the first hard failure), so signatures stay fixed-length.
NOT_MEASURED = -1.0


def walk_expert_path(
    scenario: Scenario, rng: random.Random
) -> tuple[State, tuple[float, ...]]:
    """Walk the expert's path through the agent's own transition fn.

    Parameters
    ----------
    scenario : Scenario
        The capture to walk.
    rng : random.Random
        Seeded noise source.

    Returns
    -------
    tuple
        ``(final_state_codes, readings)`` where readings holds one
        float per signal and :data:`NOT_MEASURED` for signals the
        expert never reached.
    """
    state: State = (UNCHECKED,) * len(SIGNALS)
    readings: dict[str, float] = {}
    for name in ORDER:
        action = f"check:{name}"
        reading = observe(scenario, name, rng)
        readings[name] = reading
        state = transition(state, action, reading)
        index = SIGNALS.index(name)
        if state[index] == 2:  # SUSPECT_PRE -> the expert re-checks
            second = observe(scenario, name, rng)
            readings[name] = second
            state = transition(state, f"recheck:{name}", second)
    signature = tuple(readings.get(name, NOT_MEASURED) for name in SIGNALS)
    return state, signature


def estimate_posterior(
    seed: int, n: int = 60_000
) -> dict[State, Counter]:
    """Estimate P(oracle label | 5-code state) from a large sample.

    Scenarios are drawn from the balanced (test-time) mixture, which
    is the distribution a deployed policy faces. For each one the
    expert's path is walked and the resulting state is recorded
    against the oracle label.

    Parameters
    ----------
    seed : int
        Seeded source for both the scenarios and the noise.
    n : int
        Sample size.

    Returns
    -------
    dict
        State to a Counter of labels.
    """
    rng = random.Random(seed)
    from smallai.environment import TEST_WEIGHTS

    table: dict[State, Counter] = defaultdict(Counter)
    for _ in range(n):
        scenario = sample_scenario(rng, TEST_WEIGHTS)
        label = oracle_label(scenario)
        state, _ = walk_expert_path(scenario, rng)
        table[state][label] += 1
    return dict(table)


def bayes_ceiling(table: dict[State, Counter], n: int) -> float:
    """Best achievable accuracy given only the 5-code state.

    Parameters
    ----------
    table : dict
        State to label counts, from :func:`estimate_posterior`.
    n : int
        Total samples the table was built from.

    Returns
    -------
    float
        The ceiling, as a fraction.
    """
    if not table:
        return 0.0
    return sum(max(c.values()) for c in table.values()) / n


def analyse(
    agent: QAgent,
    scenarios: list[Scenario],
    labels: list[str],
    posterior: dict[State, Counter],
    seed: int,
    name: str,
) -> dict:
    """Classify every wrong verdict on the dev test split.

    Parameters
    ----------
    agent : QAgent
        The trained policy.
    scenarios : list of Scenario
        The dev test captures.
    labels : list of str
        Their oracle labels.
    posterior : dict
        State to label counts, from :func:`estimate_posterior`.
    seed : int
        Seeded scoring seed.
    name : str
        Policy name, used to derive the per-scenario seed.

    Returns
    -------
    dict
        The counts, the confusion table and the achieved ceiling.
    """
    score = score_agent(agent, scenarios, labels, seed)

    rng = random.Random(seed)
    confusion: Counter = Counter()
    table_fault = 0
    representation_fault = 0
    unseen_state = 0
    error_causes: Counter = Counter()
    correct_by_cause: Counter = Counter()
    total_by_cause: Counter = Counter()
    error_measurements: list[int] = []
    # Accumulate the Bayes-optimal correctness on the policy's own
    # paths, which is the ceiling this policy could have reached.
    best_correct = 0.0

    for index, (scenario, label) in enumerate(zip(scenarios, labels)):
        policy_rng = random.Random(derive_seed(seed, name, str(index)))
        state: State = (UNCHECKED,) * len(SIGNALS)
        measurements = 0
        while True:
            action = agent.act(state, policy_rng, epsilon=0.0)
            if action in TERMINALS:
                break
            reading = observe(scenario, action.partition(":")[2], policy_rng)
            measurements += 1
            state = transition(state, action, reading)

        counts = posterior.get(state)
        if counts:
            best_correct += max(counts.values()) / sum(counts.values())
        else:
            best_correct += 0.0

        total_by_cause[scenario.cause] += 1
        confusion[(action, label)] += 1
        if action == label:
            correct_by_cause[scenario.cause] += 1
            continue

        if not counts:
            unseen_state += 1
            representation_fault += 1
            continue
        best_label = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        if action != best_label:
            table_fault += 1
        else:
            representation_fault += 1
        error_causes[scenario.cause] += 1
        error_measurements.append(measurements)

    n = len(scenarios)
    return {
        "accuracy": score.accuracy,
        "invalid_rate": score.invalid_rate,
        "mean_measurements": score.mean_measurements,
        "review_rate": score.review_rate,
        "total_errors": table_fault + representation_fault,
        "table_fault": table_fault,
        "representation_fault": representation_fault,
        "unseen_state": unseen_state,
        "bayes_ceiling_on_policy_paths": best_correct / n,
        "confusion": {
            f"{a}->{l}": c for (a, l), c in sorted(confusion.items())
        },
        "errors_by_cause": dict(error_causes),
        "accuracy_by_cause": {
            cause: correct_by_cause[cause] / total_by_cause[cause]
            for cause in sorted(total_by_cause)
        },
        "counts_by_cause": dict(total_by_cause),
        "mean_measurements_on_errors": (
            sum(error_measurements) / len(error_measurements)
            if error_measurements
            else 0.0
        ),
    }


def print_posterior(posterior: dict[State, Counter], n: int) -> None:
    """Print the state -> label posterior, biggest states first.

    Parameters
    ----------
    posterior : dict
        State to label counts.
    n : int
        Total sample size.
    """
    print("  P(label | 5-code state), states by sample count:")
    for state, counts in sorted(
        posterior.items(), key=lambda kv: -sum(kv[1].values())
    ):
        total = sum(counts.values())
        parts = ", ".join(
            f"{label} {100 * c / total:.2f}%" for label, c in counts.most_common()
        )
        codes = "".join(str(code) for code in state)
        print(
            f"    [{codes:<6}] n={total:>7,}  "
            f"{parts}"
        )


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEV_SEED)
    parser.add_argument("--budget", type=int, default=200_000)
    parser.add_argument("--posterior-samples", type=int, default=60_000)
    args = parser.parse_args(argv)

    splits = make_splits(args.seed)
    labels = {
        name: [oracle_label(s) for s in scenarios]
        for name, scenarios in splits.items()
    }

    agent = QAgent(name="journey_q")
    train_td(
        agent,
        splits["train"],
        seed=derive_seed(args.seed, "train", "journey_q"),
        budget=args.budget,
        mix=MIX,
    )

    print("=" * 74)
    print("STATE POSTERIOR (60k independent balanced samples)")
    print("=" * 74)
    posterior = estimate_posterior(
        derive_seed(args.seed, "posterior"), args.posterior_samples
    )
    print_posterior(posterior, args.posterior_samples)
    print()
    print(
        f"  Bayes ceiling given ONLY the 5-code state: "
        f"{100 * bayes_ceiling(posterior, args.posterior_samples):.3f} %"
    )
    print(
        f"  distinct states: {len(posterior):,}   "
        f"state-space ceiling 15,625"
    )

    print()
    print("=" * 74)
    print("ERROR ANALYSIS (dev test split, 1,000 scenarios)")
    print("=" * 74)
    errors = analyse(
        agent,
        splits["test"],
        labels["test"],
        posterior,
        derive_seed(args.seed, "score", "journey_q"),
        "journey_q",
    )
    print(f"  accuracy                          {100 * errors['accuracy']:>8.3f} %")
    print(f"  invalid rate                      {100 * errors['invalid_rate']:>8.3f} %")
    print(f"  mean measurements                 {errors['mean_measurements']:>8.3f}")
    print(f"  review rate                       {100 * errors['review_rate']:>8.3f} %")
    print(
        f"  Bayes ceiling on the same paths    "
        f"{100 * errors['bayes_ceiling_on_policy_paths']:>8.3f} %"
    )
    print(f"  total errors                      {errors['total_errors']:>8,}")
    print(
        f"    table fault (better learning)    {errors['table_fault']:>8,}"
    )
    print(
        f"    representation fault (ceiling)   {errors['representation_fault']:>8,}"
        f"   (unseen states: {errors['unseen_state']:,})"
    )
    print()
    print("  confusion (agent action -> true label):")
    for key, count in errors["confusion"].items():
        print(f"    {key:<26} {count:>6,}")
    print()
    print("  accuracy by scenario cause:")
    for cause, acc in errors["accuracy_by_cause"].items():
        count = errors["counts_by_cause"][cause]
        print(f"    {cause:<12} {100 * acc:>7.2f} %   (n={count:,})")
    print()
    print("  errors by scenario cause:")
    for cause, count in sorted(
        errors["errors_by_cause"].items(), key=lambda kv: -kv[1]
    ):
        print(f"    {cause:<12} {count:>6,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
