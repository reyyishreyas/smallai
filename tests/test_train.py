"""Gates for training: splits, equivalence, rewards, plateau, CART."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from smallai.agent import QAgent, check_action
from smallai.environment import SIGNALS, Scenario
from smallai.solver import ExpertPolicy
from smallai.train import (
    FLAWS,
    MIX,
    MIX_EXPERT_ONLY,
    ShortcutTree,
    Step,
    consistency_bonus,
    cross_validated_tree,
    episode_reward,
    expert_action,
    make_flawed_policy,
    make_splits,
    make_tree_dataset,
    oracle_label,
    plateau_test,
    rollout,
    train_td,
    verdict_reward,
    visit_histogram,
)

START = tuple(0 for _ in SIGNALS)


def scenario(**overrides: float) -> Scenario:
    signals = {name: 0.95 for name in SIGNALS}
    signals.update(overrides)
    return Scenario(signals=signals, cause="custom")


# --- splits (EXPERIMENT.md section 4) ---


def test_splits_have_pre_registered_sizes() -> None:
    splits = make_splits(seed=1)
    assert len(splits["train"]) == 4000
    assert len(splits["calibration"]) == 1000
    assert len(splits["test"]) == 1000


def test_no_scenario_crosses_splits() -> None:
    """Leakage safety: true signal vectors are unique globally."""
    splits = make_splits(seed=2)
    seen: set[tuple] = set()
    for name, scenarios in splits.items():
        for s in scenarios:
            key = tuple(round(s.signals[n], 12) for n in SIGNALS)
            assert key not in seen, f"{s.cause} leaked into {name}"
            seen.add(key)


def test_splits_are_deterministic_under_seed() -> None:
    a = make_splits(seed=3)
    b = make_splits(seed=3)
    for name in a:
        assert [s.signals for s in a[name]] == [s.signals for s in b[name]]


# --- expert equivalence: solver vs agent-space rule ---


def test_expert_action_matches_solver_under_same_noise() -> None:
    """The state-space expert consumes noise identically.

    Same scenario + same rng seed must give the same verdict and
    the same checked-signal order as ExpertPolicy, or trajectory
    labels would not be the solver's labels.
    """

    def policy(state, _rng):
        return expert_action(state)

    for seed in range(20):
        s = scenario(brightness=0.50 if seed % 2 else 0.95)
        solver_decision = ExpertPolicy(noisy=True).decide(
            s, random.Random(seed)
        )
        steps = rollout(policy, s, random.Random(seed))
        assert steps[-1].action == solver_decision.action
        checks = [
            step.action.partition(":")[2]
            for step in steps
            if not step.done and step.action.startswith("check:")
        ]
        assert tuple(checks) == solver_decision.path


def test_expert_action_rechecks_suspects_immediately() -> None:
    """A re-check, when taken, follows its check on the next step."""

    def policy(state, _rng):
        return expert_action(state)

    steps = rollout(policy, scenario(brightness=0.50), random.Random(0))
    actions = [step.action for step in steps]
    for i, action in enumerate(actions):
        if action.startswith("recheck:"):
            name = action.partition(":")[2]
            assert actions[i - 1] == f"check:{name}"


def test_expert_action_without_recheck_believes_first_reading() -> None:
    """NoRecheck world: a suspect reading defers, never rechecks."""
    state = (2, 4, 4, 4, 4, 4)  # blur suspect_pre, rest pass
    assert expert_action(state, allow_recheck=False) != ("recheck:blur")
    # with nothing left to check, suspect_pre defers to review
    assert expert_action(state, allow_recheck=False) == "review"


def test_oracle_label_is_deterministic() -> None:
    s = scenario(brightness=0.50)
    assert oracle_label(s) == oracle_label(s)


def test_oracle_labels_capture_the_three_verdicts() -> None:
    assert oracle_label(scenario()) == "pass"
    assert oracle_label(scenario(blur=0.10)) == "retake"
    assert oracle_label(scenario(brightness=0.50)) == "review"


# --- flawed behaviours ---


def test_decide_first_terminates_with_zero_evidence() -> None:
    rng = random.Random(0)
    policy = make_flawed_policy("decide_first", rng)
    steps = rollout(policy, scenario(), random.Random(0))
    assert len(steps) == 1
    assert steps[0].done


def test_never_recheck_never_rechecks() -> None:
    rng = random.Random(1)
    policy = make_flawed_policy("never_recheck", rng)
    for seed in range(10):
        steps = rollout(policy, scenario(brightness=0.50), random.Random(seed))
        assert all(not step.action.startswith("recheck") for step in steps)


def test_measure_all_checks_every_signal_even_on_hard_fail() -> None:
    rng = random.Random(2)
    policy = make_flawed_policy("measure_all", rng)
    steps = rollout(policy, scenario(blur=0.05), random.Random(0))
    checks = [
        step.action for step in steps if step.action.startswith("check:")
    ]
    assert len(checks) == 6
    assert steps[-1].action == "retake"


def test_random_order_is_a_permutation_and_legal() -> None:
    rng = random.Random(3)
    policy = make_flawed_policy("random_order", rng)
    steps = rollout(policy, scenario(), random.Random(0))
    checks = [
        step.action.partition(":")[2]
        for step in steps
        if step.action.startswith("check:")
    ]
    assert sorted(checks) == sorted(SIGNALS)


def test_unknown_flaw_raises() -> None:
    with pytest.raises(ValueError):
        make_flawed_policy("sloppy", random.Random(0))


# --- the frozen reward (EXPERIMENT.md section 6) ---


def test_verdict_terms() -> None:
    assert verdict_reward("pass", "pass") == 1.0
    assert verdict_reward("review", "pass") == -0.5
    assert verdict_reward("retake", "review") == -0.5
    assert verdict_reward("pass", "retake") == -1.0


def test_consistency_terms() -> None:
    fail_state = (1, 4, 4, 4, 4, 4)
    assert consistency_bonus("retake", fail_state) == 0.10
    assert consistency_bonus("pass", fail_state) == -0.30
    suspect_state = (3, 4, 4, 4, 4, 4)
    assert consistency_bonus("review", suspect_state) == 0.10
    assert consistency_bonus("review", START) == -0.30
    all_pass = (4, 4, 4, 4, 4, 4)
    assert consistency_bonus("pass", all_pass) == 0.10


def test_maximum_trajectory_scores_one_point_zero_five() -> None:
    """ARCHITECTURE 6.2 example: +1.0 - 0.05 + 0.10 = +1.05."""
    trajectory = [
        Step(START, check_action("blur"), 0.20, (1, 0, 0, 0, 0, 0), False),
        Step((1, 0, 0, 0, 0, 0), "retake", None, (1, 0, 0, 0, 0, 0), True),
    ]
    assert episode_reward(trajectory, "retake") == pytest.approx(1.05)


def test_minimum_trajectory_scores_minus_one_point_nine() -> None:
    """Opposite + contradictory + all twelve measurements."""
    trajectory = []
    state = list(START)
    # six checks, every reading suspect_post
    for name in SIGNALS:
        taken = tuple(state)
        state[SIGNALS.index(name)] = 3
        trajectory.append(
            Step(taken, check_action(name), 0.50, tuple(state), False)
        )
    # six re-checks, still suspect
    for name in SIGNALS:
        trajectory.append(
            Step(tuple(state), f"recheck:{name}", 0.50, tuple(state), False)
        )
    # a fail sneaks into the evidence: "pass" is now contradictory
    state[0] = 1
    final = tuple(state)
    trajectory.append(Step(final, "pass", None, final, True))
    measurements = sum(1 for s in trajectory if not s.done)
    assert measurements == 12
    # opposite (pass vs label retake) + contradictory + 12 * -0.05
    assert episode_reward(trajectory, "retake") == pytest.approx(
        -1.0 - 0.30 - 0.60
    )


def test_live_rollout_rewards_stay_in_bounds() -> None:
    """No episode can escape the pre-registered envelope."""
    for seed in range(30):
        s = scenario(brightness=0.50 if seed % 3 == 0 else 0.95)
        label = oracle_label(s)
        policy = make_flawed_policy(
            FLAWS[seed % len(FLAWS)], random.Random(seed)
        )
        total = episode_reward(rollout(policy, s, random.Random(seed)), label)
        assert -1.90 - 1e-9 <= total <= 1.05 + 1e-9


# --- plateau test (EXPERIMENT.md section 7) ---


def test_flat_reward_series_declares_plateau() -> None:
    rng = random.Random(0)
    blocks = [0.7 + rng.gauss(0, 0.01) for _ in range(200)]
    assert plateau_test(blocks).plateau is True


def test_strong_upward_trend_is_not_plateau() -> None:
    blocks = [0.1 + 0.005 * i for i in range(200)]
    assert plateau_test(blocks).plateau is False


# --- the training loop, smoke scale ---


def test_train_td_smoke_runs_and_reports() -> None:
    splits = make_splits(seed=5)
    train = splits["train"][:40]
    agent = QAgent()
    report = train_td(
        agent,
        train,
        seed=7,
        budget=600,
        min_episodes=200,
        check_interval=200,
    )
    assert report.episodes <= 600
    assert (
        report.n_expert + report.n_flawed + report.n_on_policy
        == report.episodes
    )
    assert report.stored_entries > 0
    assert abs(sum(MIX.values()) - 1.0) < 1e-9


def test_expert_only_mix_trains_without_flaws() -> None:
    train = make_splits(seed=6)["train"][:30]
    report = train_td(
        QAgent(),
        train,
        seed=8,
        budget=300,
        min_episodes=300,
        check_interval=300,
        mix=MIX_EXPERT_ONLY,
    )
    assert report.n_flawed == 0
    assert report.n_expert + report.n_on_policy == report.episodes


def test_no_recheck_agent_learns_no_recheck_actions() -> None:
    train = make_splits(seed=6)["train"][:30]
    agent = QAgent(allow_recheck=False, name="no_recheck_q")
    train_td(
        agent, train, seed=9, budget=300, min_episodes=300, check_interval=300
    )
    assert all(
        not action.startswith("recheck")
        for actions in agent.q.values()
        for action in actions
    )


def test_bad_mix_raises() -> None:
    with pytest.raises(ValueError):
        train_td(
            QAgent(),
            [scenario()],
            seed=0,
            budget=1,
            mix={"expert": 0.5, "flawed": 0.2, "on_policy": 0.2},
        )


# --- the shortcut baseline (from-scratch CART) ---


def separable_dataset(
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    x = np.array([[rng.random() for _ in SIGNALS] for _ in range(n)])
    y = np.array(
        ["retake" if row[0] < 0.5 else "pass" for row in x],
        dtype=object,
    )
    return x, y


def test_tree_learns_a_trivial_axis_split() -> None:
    x, y = separable_dataset(400, seed=0)
    tree = ShortcutTree().fit(x, y)
    assert float((tree.predict(x) == y).mean()) == 1.0


def test_tree_respects_max_depth() -> None:
    rng = random.Random(1)
    x = np.array([[rng.random() for _ in SIGNALS] for _ in range(200)])
    y = np.array(
        [rng.choice(["pass", "retake", "review"]) for _ in range(200)],
        dtype=object,
    )
    tree = ShortcutTree(max_depth=2).fit(x, y)

    def depth(node) -> int:
        if node.label is not None:
            return 0
        return 1 + max(depth(node.left), depth(node.right))

    assert depth(tree._root) <= 2


def test_cv_picks_from_the_grid_only() -> None:
    x, y = separable_dataset(300, seed=2)
    tree, chosen, scores = cross_validated_tree(
        x, y, seed=3, grid=(1, 5, 20), folds=4
    )
    assert chosen in (1, 5, 20)
    assert len(scores) == 4
    # refitted on all data: near-perfect on a separable set
    assert float((tree.predict(x) == y).mean()) > 0.99


def test_tree_dataset_shape() -> None:
    splits = make_splits(seed=4)
    train = splits["train"][:10]
    labels = [oracle_label(s) for s in train]
    x, y = make_tree_dataset(train, labels, random.Random(0))
    assert x.shape == (10, 6)
    assert list(y) == labels


# --- the two artifacts EXPERIMENT.md section 7 requires to ship ---


def test_training_report_carries_the_curve() -> None:
    """Section 7 ships the training curve, so it must exist."""
    train = make_splits(seed=6)["train"][:30]
    report = train_td(
        QAgent(),
        train,
        seed=10,
        budget=500,
        min_episodes=500,
        check_interval=500,
    )
    assert len(report.block_means) == 500 // 100
    assert all(isinstance(mean, float) for mean in report.block_means)


def test_visit_histogram_partitions_the_visited_states() -> None:
    """Every visited state falls in exactly one bucket."""
    train = make_splits(seed=6)["train"][:30]
    report = train_td(
        QAgent(),
        train,
        seed=11,
        budget=300,
        min_episodes=300,
        check_interval=300,
    )
    assert sum(report.visit_histogram.values()) == report.visited_states
    assert all(count > 0 for count in report.visit_histogram)


def test_visit_histogram_groups_states_by_visit_count() -> None:
    visits = {
        START: 3,
        (1,) + START[:5]: 3,
        (2,) + START[:5]: 1,
    }
    assert visit_histogram(visits) == {1: 1, 3: 2}


def test_report_serializes_without_json_complaints() -> None:
    """The new fields must stay JSON-friendly, keys included."""
    train = make_splits(seed=6)["train"][:20]
    report = train_td(
        QAgent(),
        train,
        seed=12,
        budget=200,
        min_episodes=200,
        check_interval=200,
    )
    payload = report.to_dict()
    assert all(isinstance(k, str) for k in payload["visit_histogram"])
    assert payload["block_means"] == list(report.block_means)
    json.dumps(payload)
