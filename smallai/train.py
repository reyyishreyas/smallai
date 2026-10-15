"""Learning: trajectories, rewards, TD training, ablations (AI-267).

Implements ARCHITECTURE.md section 6 against the frozen protocol
(EXPERIMENT.md sections 4-7): leakage-safe scenario splits, the
expert and deliberately flawed behaviour policies expressed in the
agent's state space, the frozen reward decomposed into per-step
rewards for TD(0), the epsilon-greedy training loop with the
pre-registered plateau test, and the two TD ablations (NoRecheck-Q,
ExpertOnly-Q) plus the from-scratch CART shortcut baseline.

Engineering defaults, documented rather than fitted (the protocol
freezes only lr, gamma, epsilon, budget and the reward): the
trajectory mix is 40% expert, 40% flawed, 20% on-policy per
episode; ExpertOnly-Q redistributes the flawed share to expert
(80/0/20). The plateau test (EXPERIMENT.md section 7: 100-episode
block means over the final 20%, OLS slope, 95% CI containing zero)
is checked every ``CHECK_INTERVAL`` episodes after a
``MIN_EPISODES`` burn-in, each time on the final 20% of episodes
run so far; at the full budget this coincides with the
pre-registered wording. The CI uses the normal 1.96 critical value
-- with >= 200 blocks the t-difference is negligible.

Labels are the expert's verdicts in *oracle* mode (noisy=False):
a deterministic function of the true signal vector. That is what
makes the expert's 100%-by-construction reference row meaningful
under fresh measurement noise, and what every policy's accuracy is
measured against.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from smallai.agent import (
    FAIL_FINAL,
    PASS_FINAL,
    SUSPECT_POST,
    SUSPECT_PRE,
    TERMINALS,
    UNCHECKED,
    QAgent,
    State,
    recheck_action,
    transition,
)
from smallai.environment import (
    SIGNALS,
    TEST_WEIGHTS,
    TRAIN_WEIGHTS,
    Scenario,
    observe,
    sample_scenario,
)
from smallai.solver import ORDER, ExpertPolicy

# Split sizes (EXPERIMENT.md section 4).
TRAIN_N = 4000
CALIBRATION_N = 1000
TEST_N = 1000

# Plateau-test plumbing (operationalization of section 7).
BLOCK = 100
FINAL_FRACTION = 0.2
CHECK_INTERVAL = 50_000
MIN_EPISODES = 200_000
Z_95 = 1.96

# Trajectory mix per episode (engineering default, see module
# docstring). ExpertOnly-Q passes MIX_EXPERT_ONLY instead.
MIX: dict[str, float] = {
    "expert": 0.4,
    "flawed": 0.4,
    "on_policy": 0.2,
}
MIX_EXPERT_ONLY: dict[str, float] = {
    "expert": 0.8,
    "flawed": 0.0,
    "on_policy": 0.2,
}

# The deliberately bad behaviours of ARCHITECTURE.md section 6.1.
FLAWS: tuple[str, ...] = (
    "random_order",
    "measure_all",
    "decide_first",
    "never_recheck",
)

# Reward terms, frozen in EXPERIMENT.md section 6.
REWARD_CORRECT = 1.0
REWARD_ADJACENT_WRONG = -0.5
REWARD_OPPOSITE_WRONG = -1.0
REWARD_MEASUREMENT = -0.05
REWARD_CONSISTENT = 0.10
REWARD_CONTRADICTORY = -0.30

Policy = Callable[[State, random.Random], str]


@dataclass(frozen=True)
class Step:
    """One transition of a trajectory.

    Attributes
    ----------
    state : tuple of int
        The state the action was taken in.
    action : str
        The action taken.
    reading : float or None
        The noisy reading it produced; None for terminals.
    next_state : tuple of int
        The resulting state (equals ``state`` for terminals, whose
        evidence is the state itself).
    done : bool
        Whether the action ended the episode.
    """

    state: State
    action: str
    reading: float | None
    next_state: State
    done: bool


@dataclass(frozen=True)
class PlateauResult:
    """Outcome of the pre-registered slope test.

    Attributes
    ----------
    slope : float
        OLS slope of block mean reward vs block index.
    ci_low : float
        Lower end of the 95% CI (normal approximation).
    ci_high : float
        Upper end of the 95% CI.
    plateau : bool
        True iff the CI contains zero.
    blocks : int
        Number of block means the test used.
    """

    slope: float
    ci_low: float
    ci_high: float
    plateau: bool
    blocks: int


@dataclass(frozen=True)
class TrainingReport:
    """What one training run produced (ships in reports/poc.md).

    Attributes
    ----------
    name : str
        The trained agent's label.
    seed : int
        The run's master seed.
    budget : int
        The episode cap that applied.
    episodes : int
        Episodes actually run.
    plateau : bool
        Whether the slope test ever declared plateau.
    checked_at : int or None
        Episode index of the plateau check that stopped the run.
    slope : float or None
        Slope at the stopping (or final) check.
    ci_low : float or None
        CI lower end at that check.
    ci_high : float or None
        CI upper end at that check.
    n_expert : int
        Expert-trajectory episodes.
    n_flawed : int
        Flawed-trajectory episodes.
    n_on_policy : int
        Epsilon-greedy on-policy episodes.
    stored_entries : int
        Q-table size after training.
    visited_states : int
        Distinct states touched.
    final_block_mean : float
        Mean reward of the final 20% of complete blocks.
    """

    name: str
    seed: int
    budget: int
    episodes: int
    plateau: bool
    checked_at: int | None
    slope: float | None
    ci_low: float | None
    ci_high: float | None
    n_expert: int
    n_flawed: int
    n_on_policy: int
    stored_entries: int
    visited_states: int
    final_block_mean: float

    def to_dict(self) -> dict:
        """The report as a JSON-friendly dict."""
        return {
            "name": self.name,
            "seed": self.seed,
            "budget": self.budget,
            "episodes": self.episodes,
            "plateau": self.plateau,
            "checked_at": self.checked_at,
            "slope": self.slope,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_expert": self.n_expert,
            "n_flawed": self.n_flawed,
            "n_on_policy": self.n_on_policy,
            "stored_entries": self.stored_entries,
            "visited_states": self.visited_states,
            "final_block_mean": self.final_block_mean,
        }


def make_splits(seed: int) -> dict[str, list[Scenario]]:
    """Draw the leakage-safe train / calibration / test splits.

    Scenarios (true signal vectors) are partitioned *before* any
    noise (EXPERIMENT.md section 4): uniqueness is enforced across
    the combined 6,000 scenarios, so no true scenario can appear in
    two splits. Train uses the skewed field mixture, calibration and
    test the balanced one.

    Parameters
    ----------
    seed : int
        Master seed; identical seeds give identical splits.

    Returns
    -------
    dict
        Split name to list of scenarios: 4,000 train (skewed),
        1,000 calibration and 1,000 test (both balanced).
    """
    rng = random.Random(seed)
    seen: set[tuple[float, ...]] = set()
    splits: dict[str, list[Scenario]] = {
        "train": [],
        "calibration": [],
        "test": [],
    }
    plan = (
        ("train", TRAIN_N, TRAIN_WEIGHTS),
        ("calibration", CALIBRATION_N, TEST_WEIGHTS),
        ("test", TEST_N, TEST_WEIGHTS),
    )
    for name, count, weights in plan:
        drawn: list[Scenario] = []
        while len(drawn) < count:
            scenario = sample_scenario(
                rng, weights if weights is not None else TEST_WEIGHTS
            )
            key = tuple(round(scenario.signals[s], 12) for s in SIGNALS)
            if key in seen:
                continue
            seen.add(key)
            drawn.append(scenario)
        splits[name] = drawn
    return splits


def oracle_label(scenario: Scenario) -> str:
    """The expert's verdict on true values: the ground-truth label.

    Parameters
    ----------
    scenario : Scenario
        The capture to label.

    Returns
    -------
    str
        ``"pass"``, ``"retake"`` or ``"review"``.
    """
    return ExpertPolicy(noisy=False).decide(scenario, None).action


def expert_action(state: State, allow_recheck: bool = True) -> str:
    """The expert's rule, in the agent's state space.

    Mirrors :class:`smallai.solver.ExpertPolicy` exactly: exit on
    any fail, re-check a suspect immediately, check the next signal
    in cheapest-to-fail order, and finish suspect -> review, else
    pass. Equivalence is pinned by a test that runs both against
    the same seeded noise.

    Parameters
    ----------
    state : tuple of int
        Current per-signal codes.
    allow_recheck : bool
        Whether re-checks exist. False serves the NoRecheck-Q
        ablation: a first suspect reading is believed (it stays
        suspect_pre) and defers to review at the end.

    Returns
    -------
    str
        The expert's next action.
    """
    if FAIL_FINAL in state:
        return "retake"
    if allow_recheck:
        for name in ORDER:
            if state[SIGNALS.index(name)] == SUSPECT_PRE:
                return recheck_action(name)
    for name in ORDER:
        if state[SIGNALS.index(name)] == UNCHECKED:
            return f"check:{name}"
    if SUSPECT_PRE in state or SUSPECT_POST in state:
        return "review"
    return "pass"


def make_flawed_policy(
    flaw: str, rng: random.Random, allow_recheck: bool = True
) -> Policy:
    """Build one of the four deliberately bad behaviours.

    Each flaw deviates from the expert in exactly one way:
    ``random_order`` permutes the check order, ``measure_all``
    never exits early, ``decide_first`` picks a terminal before any
    measurement (zero evidence, contradictory by section 3),
    ``never_recheck`` believes first readings.

    Parameters
    ----------
    flaw : str
        One of :data:`FLAWS`.
    rng : random.Random
        Seeded randomness (the permutation, or the decide-first
        verdict).
    allow_recheck : bool
        Whether re-check actions exist at all. False serves the
        NoRecheck-Q ablation: its training data must contain no
        re-checks either, or it would learn transitions it can
        never take at decision time.

    Returns
    -------
    callable
        A ``policy(state, rng) -> action`` behaviour.

    Raises
    ------
    ValueError
        If the flaw name is unknown.
    """
    if flaw == "decide_first":
        verdicts = list(TERMINALS)

        def decide_first(state: State, r: random.Random) -> str:
            return r.choice(verdicts)

        return decide_first

    order = list(ORDER)
    if flaw == "random_order":
        rng.shuffle(order)
    elif flaw not in ("measure_all", "never_recheck"):
        raise ValueError(f"Unknown flaw {flaw!r}")

    def behave(state: State, r: random.Random) -> str:
        if flaw == "never_recheck" or not allow_recheck:
            if FAIL_FINAL in state:
                return "retake"
            for name in order:
                if state[SIGNALS.index(name)] == UNCHECKED:
                    return f"check:{name}"
        elif flaw == "measure_all":
            # No early exit: re-check and measure everything first.
            for name in order:
                if state[SIGNALS.index(name)] == SUSPECT_PRE:
                    return recheck_action(name)
            for name in order:
                if state[SIGNALS.index(name)] == UNCHECKED:
                    return f"check:{name}"
            if FAIL_FINAL in state:
                return "retake"
        else:  # random_order: expert logic, permuted order
            if FAIL_FINAL in state:
                return "retake"
            for name in order:
                if state[SIGNALS.index(name)] == SUSPECT_PRE:
                    return recheck_action(name)
            for name in order:
                if state[SIGNALS.index(name)] == UNCHECKED:
                    return f"check:{name}"
        if SUSPECT_PRE in state or SUSPECT_POST in state:
            return "review"
        return "pass"

    return behave


def rollout(
    behavior: Policy, scenario: Scenario, rng: random.Random
) -> list[Step]:
    """Run one trajectory: behaviour against the noisy environment.

    Every action is applied through
    :func:`smallai.agent.transition`, which rejects illegal ones --
    a behaviour bug fails loudly instead of corrupting training.
    Structurally bounded at 12 measurements.

    Parameters
    ----------
    behavior : callable
        ``policy(state, rng) -> action``.
    scenario : Scenario
        The capture the episode runs on (fresh noise per draw).
    rng : random.Random
        Seeded randomness for readings and behaviour.

    Returns
    -------
    list of Step
        The transitions, terminal step last.

    Raises
    ------
    RuntimeError
        If a behaviour somehow exceeds 12 measurements.
    """
    state: State = (UNCHECKED,) * len(SIGNALS)
    steps: list[Step] = []
    measurements = 0
    while True:
        action = behavior(state, rng)
        if action in TERMINALS:
            steps.append(Step(state, action, None, state, True))
            return steps
        reading = observe(scenario, action.partition(":")[2], rng)
        measurements += 1
        if measurements > 12:
            raise RuntimeError("trajectory exceeded 12 measurements")
        next_state = transition(state, action, reading)
        steps.append(Step(state, action, reading, next_state, False))
        state = next_state


def verdict_reward(action: str, label: str) -> float:
    """The frozen verdict term (EXPERIMENT.md section 6).

    Parameters
    ----------
    action : str
        The policy's terminal action.
    label : str
        The expert's oracle verdict.

    Returns
    -------
    float
        +1.0 correct, -1.0 opposite (pass vs retake), -0.5
        adjacent (review vs pass/retake).
    """
    if action == label:
        return REWARD_CORRECT
    if {action, label} == {"pass", "retake"}:
        return REWARD_OPPOSITE_WRONG
    return REWARD_ADJACENT_WRONG


def consistency_bonus(action: str, state: State) -> float:
    """The frozen evidence-consistency term (section 6).

    ``retake`` is consistent iff some observed fail exists,
    ``review`` iff some observed suspect, ``pass`` iff all six
    signals are observed pass-level. Deciding with zero or thin
    evidence is contradictory.

    Parameters
    ----------
    action : str
        The terminal action.
    state : tuple of int
        The evidence gathered when it was taken.

    Returns
    -------
    float
        +0.10 consistent, -0.30 contradictory.
    """
    has_fail = FAIL_FINAL in state
    has_suspect = SUSPECT_PRE in state or SUSPECT_POST in state
    all_pass = all(code == PASS_FINAL for code in state)
    if action == "retake" and has_fail:
        return REWARD_CONSISTENT
    if action == "review" and has_suspect:
        return REWARD_CONSISTENT
    if action == "pass" and all_pass:
        return REWARD_CONSISTENT
    return REWARD_CONTRADICTORY


def episode_reward(trajectory: list[Step], label: str) -> float:
    """The frozen per-episode total (section 6, bounds -1.90..+1.05).

    Parameters
    ----------
    trajectory : list of Step
        One full episode.
    label : str
        The expert's oracle verdict for its scenario.

    Returns
    -------
    float
        verdict term + consistency term - 0.05 per measurement.
    """
    terminal = trajectory[-1]
    measurements = sum(1 for step in trajectory if not step.done)
    return (
        verdict_reward(terminal.action, label)
        + consistency_bonus(terminal.action, terminal.state)
        + REWARD_MEASUREMENT * measurements
    )


def plateau_test(
    block_means: list[float], final_fraction: float = FINAL_FRACTION
) -> PlateauResult:
    """The pre-registered slope test (EXPERIMENT.md section 7).

    Fits OLS to the final ``final_fraction`` of 100-episode block
    means and declares plateau iff the 95% CI of the slope
    contains zero (normal critical value; block counts here are in
    the hundreds, where t and normal agree to two decimals).

    Parameters
    ----------
    block_means : list of float
        Complete 100-episode block means, oldest first.
    final_fraction : float
        Fraction of blocks (from the end) the test uses.

    Returns
    -------
    PlateauResult
        Slope, CI, verdict and the block count used.
    """
    n_used = max(3, int(len(block_means) * final_fraction))
    used = block_means[-n_used:]
    n = len(used)
    if n < 3:
        return PlateauResult(math.nan, math.nan, math.nan, False, n)
    x = np.arange(n, dtype=float)
    y = np.asarray(used, dtype=float)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    s_xx = float(((x - x_mean) ** 2).sum())
    slope = float(((x - x_mean) * (y - y_mean)).sum()) / s_xx
    intercept = y_mean - slope * x_mean
    residuals = y - (intercept + slope * x)
    s_squared = float((residuals**2).sum()) / (n - 2)
    se = math.sqrt(s_squared / s_xx)
    ci_low = slope - Z_95 * se
    ci_high = slope + Z_95 * se
    return PlateauResult(
        slope=slope,
        ci_low=ci_low,
        ci_high=ci_high,
        plateau=ci_low <= 0.0 <= ci_high,
        blocks=n,
    )


def train_td(
    agent: QAgent,
    scenarios: list[Scenario],
    *,
    seed: int,
    budget: int = 1_000_000,
    mix: dict[str, float] = MIX,
    check_interval: int = CHECK_INTERVAL,
    min_episodes: int = MIN_EPISODES,
    epsilon: float = 0.1,
) -> TrainingReport:
    """Train one TD agent by epsilon-greedy Q-learning (section 7).

    Each episode draws a training scenario (with replacement),
    rolls out one behaviour from the mix, and applies TD(0)
    updates step by step with the frozen reward decomposed as
    ``-0.05`` per measurement and the verdict + consistency terms
    on the terminal step. The plateau test is evaluated every
    ``check_interval`` episodes past ``min_episodes`` and stops the
    run early when it passes; otherwise the budget stops it.

    Parameters
    ----------
    agent : QAgent
        The agent to train in place (``allow_recheck=False`` builds
        NoRecheck-Q; the same loop serves ExpertOnly-Q via
        ``mix=MIX_EXPERT_ONLY``).
    scenarios : list of Scenario
        The train split.
    seed : int
        Master seed for scenario choice, behaviours and epsilon.
    budget : int
        Episode cap (10^6 in the scored run).
    mix : dict
        Shares of expert / flawed / on_policy episodes.
    check_interval : int
        Episodes between plateau checks.
    min_episodes : int
        Burn-in before the first check.
    epsilon : float
        On-policy exploration rate (0.1 per section 7).

    Returns
    -------
    TrainingReport
        Episode counts, plateau outcome and table size.

    Raises
    ------
    ValueError
        If the mix shares do not sum to 1.0.
    """
    if abs(sum(mix.values()) - 1.0) > 1e-9:
        raise ValueError(f"mix shares must sum to 1.0, got {mix}")
    rng = random.Random(seed)
    labels = [oracle_label(s) for s in scenarios]
    allow_recheck = agent.allow_recheck

    def expert_policy(state: State, _r: random.Random) -> str:
        return expert_action(state, allow_recheck)

    counts = {"expert": 0, "flawed": 0, "on_policy": 0}
    block_means: list[float] = []
    block_rewards: list[float] = []
    total_reward = 0.0
    plateau = False
    checked_at: int | None = None
    last = PlateauResult(math.nan, math.nan, math.nan, False, 0)

    for episode in range(1, budget + 1):
        u = rng.random()
        if u < mix["expert"]:
            behavior = expert_policy
            counts["expert"] += 1
        elif u < mix["expert"] + mix["flawed"]:
            behavior = make_flawed_policy(
                FLAWS[rng.randrange(len(FLAWS))],
                rng,
                allow_recheck,
            )
            counts["flawed"] += 1
        else:
            counts["on_policy"] += 1

            def behavior(state: State, r: random.Random) -> str:
                return agent.act(state, r, epsilon)

        index = rng.randrange(len(scenarios))
        trajectory = rollout(behavior, scenarios[index], rng)
        label = labels[index]
        reward_total = episode_reward(trajectory, label)
        total_reward += reward_total
        for step in trajectory:
            if step.done:
                reward = verdict_reward(step.action, label)
                reward += consistency_bonus(step.action, step.state)
            else:
                reward = REWARD_MEASUREMENT
            agent.update(
                step.state,
                step.action,
                reward,
                step.next_state,
                step.done,
            )
        block_rewards.append(reward_total)
        if len(block_rewards) == BLOCK:
            block_means.append(sum(block_rewards) / len(block_rewards))
            block_rewards.clear()
        if (
            episode >= min_episodes
            and episode % check_interval == 0
            and len(block_means) >= 3
        ):
            last = plateau_test(block_means)
            if last.plateau:
                plateau = True
                checked_at = episode
                break

    n_final = max(1, int(len(block_means) * FINAL_FRACTION))
    tail = block_means[-n_final:] or [total_reward]
    return TrainingReport(
        name=agent.name,
        seed=seed,
        budget=budget,
        episodes=episode,
        plateau=plateau,
        checked_at=checked_at,
        slope=None if math.isnan(last.slope) else last.slope,
        ci_low=None if math.isnan(last.ci_low) else last.ci_low,
        ci_high=None if math.isnan(last.ci_high) else last.ci_high,
        n_expert=counts["expert"],
        n_flawed=counts["flawed"],
        n_on_policy=counts["on_policy"],
        stored_entries=agent.stored_entries,
        visited_states=agent.visited_states,
        final_block_mean=sum(tail) / len(tail),
    )


# --- the from-scratch shortcut baseline (CART) ---


@dataclass
class _Node:
    """One CART node: an internal split or a leaf verdict."""

    feature: int | None = None
    threshold: float | None = None
    left: _Node | None = None
    right: _Node | None = None
    label: str | None = None


class ShortcutTree:
    """From-scratch CART: six first-readings in, verdict out.

    The ticket's graded shortcut baseline (EXPERIMENT.md section
    5): one shot, no re-checks, max depth 6 a priori (one
    axis-aligned split per signal). Gini splits, majority leaves,
    ``min_samples_leaf`` chosen by cross-validation on the train
    split only.

    Parameters
    ----------
    max_depth : int
        Depth cap (6 a priori).
    min_samples_leaf : int
        Minimum samples in a leaf.
    """

    def __init__(self, max_depth: int = 6, min_samples_leaf: int = 1) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.classes_: list[str] = []
        self._root: _Node | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> ShortcutTree:
        """Fit the tree on first-readings and oracle labels.

        Parameters
        ----------
        x : numpy.ndarray
            Shape (n, 6): one noisy first-reading per signal.
        y : numpy.ndarray
            Shape (n,): the label strings.

        Returns
        -------
        ShortcutTree
            self, for chaining.
        """
        self.classes_ = sorted(set(y.tolist()))
        self._root = self._build(x, y, depth=0)
        return self

    def _build(self, x: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        """Recursively partition until pure, capped or too small."""
        labels, counts = np.unique(y, return_counts=True)
        majority = str(labels[int(counts.argmax())])
        if (
            depth >= self.max_depth
            or len(labels) == 1
            or len(y) < 2 * self.min_samples_leaf
        ):
            return _Node(label=majority)
        best = self._best_split(x, y)
        if best is None:
            return _Node(label=majority)
        feature, threshold = best
        left_mask = x[:, feature] <= threshold
        right_mask = ~left_mask
        if (
            int(left_mask.sum()) < self.min_samples_leaf
            or int(right_mask.sum()) < self.min_samples_leaf
        ):
            return _Node(label=majority)
        return _Node(
            feature=feature,
            threshold=threshold,
            left=self._build(x[left_mask], y[left_mask], depth + 1),
            right=self._build(x[right_mask], y[right_mask], depth + 1),
        )

    def _best_split(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[int, float] | None:
        """Gini-optimal (feature, threshold), or None if none."""
        n = len(y)
        best_weighted = math.inf
        best_split: tuple[int, float] | None = None
        class_index = {c: i for i, c in enumerate(self.classes_)}
        codes = np.array([class_index[v] for v in y])
        k = len(self.classes_)
        for feature in range(x.shape[1]):
            order = np.argsort(x[:, feature], kind="stable")
            sorted_x = x[order, feature]
            sorted_codes = codes[order]
            left_counts = np.zeros(k, dtype=np.int64)
            right_counts = np.bincount(sorted_codes, minlength=k)
            for i in range(n - 1):
                left_counts[sorted_codes[i]] += 1
                right_counts[sorted_codes[i]] -= 1
                if sorted_x[i] == sorted_x[i + 1]:
                    continue
                n_left = i + 1
                n_right = n - n_left
                if (
                    n_left < self.min_samples_leaf
                    or n_right < self.min_samples_leaf
                ):
                    continue
                gini_left = 1.0 - float(((left_counts / n_left) ** 2).sum())
                gini_right = 1.0 - float(((right_counts / n_right) ** 2).sum())
                weighted = (n_left * gini_left + n_right * gini_right) / n
                if weighted < best_weighted - 1e-12:
                    best_weighted = weighted
                    best_split = (
                        feature,
                        float((sorted_x[i] + sorted_x[i + 1]) / 2),
                    )
        return best_split

    def predict_one(self, row: np.ndarray) -> str:
        """The tree's verdict for one feature row.

        Parameters
        ----------
        row : numpy.ndarray
            Shape (6,): the six first-readings.

        Returns
        -------
        str
            ``"pass"``, ``"retake"`` or ``"review"``.
        """
        if self._root is None:
            raise ValueError("tree is not fitted")
        node = self._root
        while node.label is None:
            assert node.feature is not None
            assert node.threshold is not None
            assert node.left is not None
            assert node.right is not None
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.label

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Verdicts for a batch of rows.

        Parameters
        ----------
        x : numpy.ndarray
            Shape (n, 6).

        Returns
        -------
        numpy.ndarray
            Shape (n,), dtype object: the verdict strings.
        """
        return np.array([self.predict_one(row) for row in x], dtype=object)


def make_tree_dataset(
    scenarios: list[Scenario],
    labels: list[str],
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """One noisy first-reading per signal per scenario.

    This is the tree's whole worldview: six readings, one shot, no
    re-checks (EXPERIMENT.md section 5).

    Parameters
    ----------
    scenarios : list of Scenario
        The captures.
    labels : list of str
        Their oracle labels.
    rng : random.Random
        Seeded noise source.

    Returns
    -------
    tuple of numpy.ndarray
        ``x`` shape (n, 6) and ``y`` shape (n,).
    """
    x = np.array(
        [[observe(s, name, rng) for name in SIGNALS] for s in scenarios]
    )
    return x, np.array(labels, dtype=object)


def cross_validated_tree(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    grid: tuple[int, ...] = (1, 5, 20),
    folds: int = 5,
) -> tuple[ShortcutTree, int, list[float]]:
    """Pick ``min_samples_leaf`` by CV on the train split only.

    No hyperparameter of any policy may be selected on the
    calibration or test splits (EXPERIMENT.md section 5); this
    function is only ever handed train data.

    Parameters
    ----------
    x : numpy.ndarray
        Shape (n, 6) first-readings.
    y : numpy.ndarray
        Shape (n,) labels.
    seed : int
        Seeded fold shuffle.
    grid : tuple of int
        Candidate leaf sizes.
    folds : int
        Number of folds.

    Returns
    -------
    tuple
        ``(tree, chosen_leaf, fold_accuracies)``; the tree is
        refitted on all of ``x``.
    """
    rng = random.Random(seed)
    indices = list(range(len(y)))
    rng.shuffle(indices)
    fold_sizes = [len(indices) // folds] * folds
    for i in range(len(indices) % folds):
        fold_sizes[i] += 1
    chunks: list[list[int]] = []
    start = 0
    for size in fold_sizes:
        chunks.append(indices[start : start + size])
        start += size

    best_leaf = grid[0]
    best_score = -1.0
    best_scores: list[float] = []
    for leaf in grid:
        scores: list[float] = []
        for holdout in chunks:
            holdout_set = set(holdout)
            train_idx = [i for i in indices if i not in holdout_set]
            model = ShortcutTree(min_samples_leaf=leaf)
            model.fit(x[train_idx], y[train_idx])
            predicted = model.predict(x[holdout])
            accuracy = float((predicted == y[holdout]).mean())
            scores.append(accuracy)
        mean_score = sum(scores) / len(scores)
        if mean_score > best_score + 1e-12:
            best_score = mean_score
            best_leaf = leaf
            best_scores = scores
    tree = ShortcutTree(min_samples_leaf=best_leaf).fit(x, y)
    return tree, best_leaf, best_scores
