"""Evaluation: calibration, metrics and uncertainty (AI-267 Stage A).

Implements EXPERIMENT.md sections 8-10 against the frozen protocol.
Every function here is a pure measurement: no policy is ever tuned
on the calibration or test splits, and no number is produced that
the caller did not supply data for.

Scoring a Q agent goes through :func:`walk`, a greedy replay that
mirrors :meth:`smallai.agent.QAgent.decide` but keeps two things
``decide`` discards:

- the three terminal Q-values, which section 9's temperature-scaled
  ``softmax(Q/tau)`` cross-check needs (the margin alone is only a
  difference, and a difference cannot be softmaxed);
- the final state codes, which section 3's evidence-consistency
  rule needs (whether the verdict matches what was actually
  measured).

A gate pins ``walk`` to ``decide`` under the same seed, so the two
can never drift apart silently.

Confidence is a *score* until it is calibrated. Section 9 fits an
isotonic regression from terminal margin to P(correct) on the
calibration split and cross-checks it with temperature-scaled
softmax; if the two disagree materially the shipped map is flagged
provisional rather than quietly preferred.
"""

from __future__ import annotations

import bisect
import hashlib
import math
import random
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from smallai import environment
from smallai.agent import (
    MAX_STATES,
    TERMINALS,
    UNCHECKED,
    QAgent,
    State,
    transition,
)
from smallai.environment import SIGNALS, Scenario, observe
from smallai.train import ShortcutTree

#: Section 8: every reported accuracy delta carries this CI.
BOOTSTRAP_RESAMPLES: int = 10_000

#: Section 8: ECE is defined on 10 equal-width confidence bins.
ECE_BINS: int = 10

#: Section 3: at most 6 checks plus 6 re-checks per episode.
MAX_MEASUREMENTS: int = 12

#: Section 3: the evidence level a state code records. suspect_pre
#: and suspect_post are both "suspect" -- the difference is whether
#: the signal has been re-checked, which does not change the
#: consistency rule.
LEVEL_BY_CODE: dict[int, str] = {
    1: "fail",
    2: "suspect",
    3: "suspect",
    4: "pass",
}


def derive_seed(master: int, *parts: str) -> int:
    """Derive a reproducible sub-seed from the master seed.

    A single master seed cannot serve every consumer (splits, each
    policy's training, each scoring rng, the bootstrap) without
    correlating them, and reusing one seed everywhere would make
    two policies share noise draws. This hashes the master seed
    with a label instead, so every stream is independent yet
    recoverable from ``(master, label)`` alone.

    Parameters
    ----------
    master : int
        The run's master seed (EXPERIMENT.md section 4).
    *parts : str
        Labels naming the stream, e.g. ``"train"``, ``"journey_q"``.

    Returns
    -------
    int
        A 64-bit seed, stable across runs and platforms.
    """
    payload = f"{master}:".encode() + ":".join(parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@contextmanager
def noise_sigma(sigma: float) -> Iterator[None]:
    """Run a block at a given sensor-noise level (section 8).

    ``observe`` reads the environment's module-level
    :data:`smallai.environment.NOISE_SIGMA` at call time, so the
    sensitivity sweep sets it for the duration of one complete
    train-and-score cycle and restores it afterwards. Nothing else
    in the package holds a cached copy.

    Parameters
    ----------
    sigma : float
        The Gaussian standard deviation to measure under.

    Yields
    ------
    None
        Control, with the noise level set.
    """
    previous = environment.NOISE_SIGMA
    environment.NOISE_SIGMA = sigma
    try:
        yield
    finally:
        environment.NOISE_SIGMA = previous


@dataclass(frozen=True)
class Walk:
    """One scored decision: the verdict plus what section 8 needs.

    Attributes
    ----------
    action : str
        The terminal action taken.
    margin : float
        ``Q_best - Q_second`` over the terminals, in reward units.
    terminal_q : tuple of float
        The three terminal Q-values, ordered as
        :data:`smallai.agent.TERMINALS`. Never a probability.
    state : tuple of int
        The final state codes, i.e. the evidence the verdict rests
        on.
    measurements : int
        Measurement actions taken before deciding.
    """

    action: str
    margin: float
    terminal_q: tuple[float, ...]
    state: State
    measurements: int


def walk(agent: QAgent, scenario: Scenario, rng: random.Random) -> Walk:
    """Run one greedy decision and keep what ``decide`` discards.

    Same policy, same noise source, same structural bound as
    :meth:`smallai.agent.QAgent.decide`; this variant additionally
    returns the terminal Q-vector (needed to softmax) and the final
    state (needed to test evidence consistency).

    Parameters
    ----------
    agent : QAgent
        The trained policy to replay.
    scenario : Scenario
        The capture to judge.
    rng : random.Random
        Seeded noise source for the measurements.

    Returns
    -------
    Walk
        The verdict, its evidence and its confidence signals.

    Raises
    ------
    RuntimeError
        If the walk exceeds 12 measurements, which the transition
        rules make unreachable; a guard, like ``decide``'s.
    """
    state: State = (UNCHECKED,) * len(SIGNALS)
    measurements = 0
    while True:
        action = agent.act(state, rng, epsilon=0.0)
        if action in TERMINALS:
            values = agent.q.get(state, {})
            return Walk(
                action=action,
                margin=agent.terminal_margin(state),
                terminal_q=tuple(values.get(name, 0.0) for name in TERMINALS),
                state=state,
                measurements=measurements,
            )
        measurements += 1
        if measurements > MAX_MEASUREMENTS:
            raise RuntimeError("walk exceeded 12 measurements")
        reading = observe(scenario, action.partition(":")[2], rng)
        state = transition(state, action, reading)


def levels_of(state: Sequence[int]) -> list[str]:
    """The evidence levels a final state records.

    Parameters
    ----------
    state : sequence of int
        Per-signal state codes.

    Returns
    -------
    list of str
        One ``"fail"`` / ``"suspect"`` / ``"pass"`` per measured
        signal; unchecked signals contribute nothing, so a decision
        taken with no measurements yields an empty list.
    """
    return [LEVEL_BY_CODE[code] for code in state if code != UNCHECKED]


def is_consistent(action: str, levels: Sequence[str]) -> bool:
    """Evidence consistency, exactly as frozen in section 3.

    ``pass`` is consistent only when all six signals were observed
    and all read pass-level: a capture cannot be certified on
    partial evidence. ``retake`` needs one observed fail, ``review``
    one observed suspect. A terminal taken with zero observations
    is contradictory under every action, which the six-signal
    requirement on ``pass`` subsumes.

    Parameters
    ----------
    action : str
        The terminal action.
    levels : sequence of str
        Evidence levels of the measured signals.

    Returns
    -------
    bool
        True iff the verdict is supported by the evidence.
    """
    if action == "pass":
        return len(levels) == len(SIGNALS) and all(
            level == "pass" for level in levels
        )
    if action == "retake":
        return "fail" in levels
    if action == "review":
        return "suspect" in levels
    raise ValueError(f"Unknown terminal action {action!r}")


@dataclass
class PolicyScore:
    """One policy's raw outcome on one split.

    Attributes
    ----------
    name : str
        The policy's label.
    labels : list of str
        The oracle verdict per scenario.
    actions : list of str
        The policy's verdict per scenario.
    correct : list of bool
        Verdict equal to the oracle label.
    measurements : list of int
        Measurement actions per decision.
    consistent : list of bool
        Evidence-consistent verdict (section 3).
    margins : list of float or None
        Terminal margins; None for the tree, which has no Q-table
        and so no margin to calibrate.
    terminal_q : list of tuple or None
        Terminal Q-vectors; None for the tree, for the same reason.
    latency_ms : list of float
        Measured wall-clock cost of each decision (section 12:
        dev machine only, never an Android claim).
    """

    name: str
    labels: list[str]
    actions: list[str]
    correct: list[bool]
    measurements: list[int]
    consistent: list[bool]
    margins: list[float] | None
    terminal_q: list[tuple[float, ...]] | None
    latency_ms: list[float]

    @property
    def accuracy(self) -> float:
        """Fraction of verdicts equal to the oracle label."""
        return accuracy(self.correct)

    @property
    def mean_measurements(self) -> float:
        """Mean measurement actions per decision (section 8)."""
        return sum(self.measurements) / len(self.measurements)

    @property
    def review_rate(self) -> float:
        """Fraction of scenarios that ended in ``review``."""
        return sum(1 for a in self.actions if a == "review") / len(
            self.actions
        )

    @property
    def invalid_rate(self) -> float:
        """Fraction of verdicts their own evidence contradicts."""
        return sum(1 for ok in self.consistent if not ok) / len(
            self.consistent
        )

    @property
    def mean_latency_ms(self) -> float:
        """Mean measured milliseconds per decision."""
        return sum(self.latency_ms) / len(self.latency_ms)


def score_agent(
    agent: QAgent,
    scenarios: Sequence[Scenario],
    labels: Sequence[str],
    seed: int,
) -> PolicyScore:
    """Score one Q agent on a split.

    Each scenario gets its own rng, derived from ``seed`` and the
    scenario index, so the run is reproducible and no two policies
    are handed the same noise stream. The scenario itself -- the
    true signal vector -- is identical for every policy, which is
    what makes the section 8 comparison paired.

    Parameters
    ----------
    agent : QAgent
        The trained policy.
    scenarios : sequence of Scenario
        The split to score on.
    labels : sequence of str
        The oracle verdict per scenario, same order.
    seed : int
        Master seed for the per-scenario noise streams.

    Returns
    -------
    PolicyScore
        Verdicts, evidence and measured latency.

    Raises
    ------
    ValueError
        If ``scenarios`` and ``labels`` differ in length.
    """
    if len(scenarios) != len(labels):
        raise ValueError("scenarios and labels must be the same length")
    score = PolicyScore(
        name=agent.name,
        labels=list(labels),
        actions=[],
        correct=[],
        measurements=[],
        consistent=[],
        margins=[],
        terminal_q=[],
        latency_ms=[],
    )
    for index, (scenario, label) in enumerate(zip(scenarios, labels)):
        rng = random.Random(derive_seed(seed, agent.name, str(index)))
        start = time.perf_counter()
        step = walk(agent, scenario, rng)
        score.latency_ms.append((time.perf_counter() - start) * 1000.0)
        score.actions.append(step.action)
        score.correct.append(step.action == label)
        score.measurements.append(step.measurements)
        score.consistent.append(
            is_consistent(step.action, levels_of(step.state))
        )
        score.margins.append(step.margin)
        score.terminal_q.append(step.terminal_q)
    return score


def score_tree(
    tree: ShortcutTree,
    x: object,
    scenarios: Sequence[Scenario],
    labels: Sequence[str],
    name: str = "shortcut_cart",
) -> PolicyScore:
    """Score the one-shot tree: six readings in, verdict out.

    The tree observes every signal exactly once, so its evidence is
    always complete and its measurement count is always six. It has
    no Q-table, hence no terminal margin: ``margins`` and
    ``terminal_q`` stay None and the tree is reported uncalibrated
    (section 9 calibrates the Q agent's margin, which the tree does
    not produce).

    Parameters
    ----------
    tree : ShortcutTree
        The fitted tree.
    x : numpy.ndarray
        Shape (n, 6): the same noisy first-readings the tree was
        given, so evidence and verdict refer to one measurement
        set.
    scenarios : sequence of Scenario
        The captures, for classifying each reading's level.
    labels : sequence of str
        The oracle verdict per scenario.
    name : str
        The policy's label.

    Returns
    -------
    PolicyScore
        Verdicts and evidence, with margins left empty.
    """
    score = PolicyScore(
        name=name,
        labels=list(labels),
        actions=[],
        correct=[],
        measurements=[],
        consistent=[],
        margins=None,
        terminal_q=None,
        latency_ms=[],
    )
    for index, (scenario, label) in enumerate(zip(scenarios, labels)):
        start = time.perf_counter()
        action = tree.predict_one(x[index])
        score.latency_ms.append((time.perf_counter() - start) * 1000.0)
        levels = [
            scenario.level(signal_name, float(x[index][position]))
            for position, signal_name in enumerate(SIGNALS)
        ]
        score.actions.append(action)
        score.correct.append(action == label)
        score.measurements.append(len(SIGNALS))
        score.consistent.append(is_consistent(action, levels))
    return score


def accuracy(correct: Sequence[bool]) -> float:
    """Fraction of correct verdicts.

    Parameters
    ----------
    correct : sequence of bool
        One entry per scored scenario.

    Returns
    -------
    float
        Accuracy in ``[0, 1]``.
    """
    return sum(1 for ok in correct if ok) / len(correct)


def ece(
    confidences: Sequence[float],
    correct: Sequence[bool],
    bins: int = ECE_BINS,
) -> float:
    """Expected calibration error over equal-width bins.

    The average gap between stated confidence and observed
    frequency, weighted by how many decisions land in each bin
    (EXPERIMENT.md section 8).

    Parameters
    ----------
    confidences : sequence of float
        Stated P(correct), in ``[0, 1]``.
    correct : sequence of bool
        Whether each decision was right.
    bins : int
        Number of equal-width bins over ``[0, 1]``.

    Returns
    -------
    float
        ECE in ``[0, 1]``; 0 is perfect.
    """
    total = len(confidences)
    counts = [0] * bins
    conf_sums = [0.0] * bins
    correct_sums = [0.0] * bins
    for confidence, ok in zip(confidences, correct):
        index = min(int(confidence * bins), bins - 1)
        counts[index] += 1
        conf_sums[index] += confidence
        correct_sums[index] += 1.0 if ok else 0.0
    error = 0.0
    for index in range(bins):
        if counts[index] == 0:
            continue
        mean_confidence = conf_sums[index] / counts[index]
        mean_accuracy = correct_sums[index] / counts[index]
        error += (counts[index] / total) * abs(mean_accuracy - mean_confidence)
    return error


def brier(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Mean squared gap between stated probability and outcome.

    Parameters
    ----------
    confidences : sequence of float
        Stated P(correct).
    correct : sequence of bool
        Whether each decision was right.

    Returns
    -------
    float
        The Brier score, a mean over decisions; lower is better,
        0 is perfect.
    """
    return sum(
        (confidence - (1.0 if ok else 0.0)) ** 2
        for confidence, ok in zip(confidences, correct)
    ) / len(confidences)


@dataclass(frozen=True)
class McNemarResult:
    """McNemar's test on two policies' paired verdicts.

    Attributes
    ----------
    first_only : int
        Scenarios the first policy got right and the second wrong.
    second_only : int
        Scenarios the second got right and the first wrong.
    chi_square : float
        ``(b - c)^2 / (b + c)`` over the discordant pairs; 0 when
        there are none.
    p_value : float
        Two-sided exact binomial p-value on the discordant pairs.
    """

    first_only: int
    second_only: int
    chi_square: float
    p_value: float


def mcnemar(first: Sequence[bool], second: Sequence[bool]) -> McNemarResult:
    """McNemar's test, exact, on discordant pairs.

    Only scenarios the two policies disagree about carry
    information, so the test conditions on those. The p-value is
    the two-sided exact binomial at p = 0.5, which is the right
    choice here: discordant counts in this experiment run to the
    tens, where the chi-square approximation is coarse.

    Parameters
    ----------
    first : sequence of bool
        Correctness of the first policy, per scenario.
    second : sequence of bool
        Correctness of the second, same order.

    Returns
    -------
    McNemarResult
        Discordant counts, chi-square and exact p-value.

    Raises
    ------
    ValueError
        If the two sequences differ in length.
    """
    if len(first) != len(second):
        raise ValueError("paired sequences must be the same length")
    first_only = sum(1 for a, b in zip(first, second) if a and not b)
    second_only = sum(1 for a, b in zip(first, second) if b and not a)
    total = first_only + second_only
    if total == 0:
        return McNemarResult(0, 0, 0.0, 1.0)
    chi_square = (first_only - second_only) ** 2 / total
    tail = sum(
        math.comb(total, i) for i in range(min(first_only, second_only) + 1)
    )
    # Exact integer arithmetic, one division at the end. The binomial
    # counts do not fit in a float once there are thousands of
    # discordant pairs -- which is what an enlarged test split
    # produces -- so converting them to one would overflow. Dividing
    # the two integers directly keeps the value exact and gives the
    # same answer as the float path for the small totals.
    p_value = min(1.0, 2 * tail / (2**total))
    return McNemarResult(first_only, second_only, chi_square, p_value)


@dataclass(frozen=True)
class BootstrapDelta:
    """A paired accuracy delta with its section 8 confidence interval.

    Attributes
    ----------
    delta_pp : float
        ``first - second`` accuracy, in percentage points.
    ci_low_pp : float
        Lower end of the 95% percentile CI, in points.
    ci_high_pp : float
        Upper end of the 95% percentile CI, in points.
    resamples : int
        Bootstrap resamples drawn.
    """

    delta_pp: float
    ci_low_pp: float
    ci_high_pp: float
    resamples: int

    @property
    def excludes_zero(self) -> bool:
        """True iff the interval does not contain zero."""
        return self.ci_low_pp > 0.0 or self.ci_high_pp < 0.0


def paired_bootstrap_delta(
    first: Sequence[bool],
    second: Sequence[bool],
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> BootstrapDelta:
    """Paired bootstrap CI for an accuracy delta (section 8).

    Resamples *scenarios*, not policies: both policies' verdicts
    travel together, so the interval describes the delta and not
    two independent accuracies. Resampling is chunked to keep peak
    memory flat regardless of the resample count.

    Parameters
    ----------
    first : sequence of bool
        Correctness of the first policy.
    second : sequence of bool
        Correctness of the second, same order.
    seed : int
        Seeded resampling.
    resamples : int
        Number of bootstrap resamples (10,000 per section 8).

    Returns
    -------
    BootstrapDelta
        The observed delta and its percentile CI, in points.

    Raises
    ------
    ValueError
        If the two sequences differ in length.
    """
    if len(first) != len(second):
        raise ValueError("paired sequences must be the same length")
    a = np.asarray([1.0 if ok else 0.0 for ok in first])
    b = np.asarray([1.0 if ok else 0.0 for ok in second])
    n = len(a)
    delta = float((a.mean() - b.mean()) * 100.0)
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples)
    # Chunked so peak memory stays flat no matter how many
    # resamples are asked for: 250 x 1,000 indices is 2 MB.
    chunk = 250
    for start in range(0, resamples, chunk):
        size = min(chunk, resamples - start)
        index = rng.integers(0, n, size=(size, n))
        estimates[start : start + size] = (a[index] - b[index]).mean(
            axis=1
        ) * 100.0
    low, high = np.percentile(estimates, (2.5, 97.5))
    return BootstrapDelta(
        delta_pp=delta,
        ci_low_pp=float(low),
        ci_high_pp=float(high),
        resamples=resamples,
    )


def paired_bootstrap_mean_delta(
    first: Sequence[float],
    second: Sequence[float],
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> BootstrapDelta:
    """Paired bootstrap CI for the difference of two paired means.

    The same estimator as :func:`paired_bootstrap_delta`, but for a
    continuous paired quantity -- the measurement count, for which the
    correctness encoding of that function does not apply. Scenarios
    are resampled, and both policies' values travel together, so the
    interval describes the difference and not two independent means.

    Parameters
    ----------
    first : sequence of float
        The first policy's per-scenario values.
    second : sequence of float
        The second policy's per-scenario values, same order.
    seed : int
        Seeded resampling.
    resamples : int
        Number of bootstrap resamples (10,000 per section 8).

    Returns
    -------
    BootstrapDelta
        The observed difference and its percentile CI, in points.

    Raises
    ------
    ValueError
        If the two sequences differ in length.
    """
    if len(first) != len(second):
        raise ValueError("paired sequences must be the same length")
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    n = len(a)
    delta = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples)
    # Chunked so peak memory stays flat no matter how many
    # resamples are asked for.
    chunk = 250
    for start in range(0, resamples, chunk):
        size = min(chunk, resamples - start)
        index = rng.integers(0, n, size=(size, n))
        estimates[start : start + size] = (a[index] - b[index]).mean(axis=1)
    low, high = np.percentile(estimates, (2.5, 97.5))
    return BootstrapDelta(
        delta_pp=delta,
        ci_low_pp=float(low),
        ci_high_pp=float(high),
        resamples=resamples,
    )


@dataclass(frozen=True)
class IsotonicModel:
    """A monotone step map from margin to P(correct) (section 9).

    Attributes
    ----------
    thresholds : tuple of float
        Block-mean margins, strictly increasing.
    probabilities : tuple of float
        Observed accuracy per block, non-decreasing, one per
        threshold.
    """

    thresholds: tuple[float, ...]
    probabilities: tuple[float, ...]

    def predict(self, score: float) -> float:
        """Map a margin to P(correct), flat outside the fitted range.

        Parameters
        ----------
        score : float
            A terminal margin in reward units.

        Returns
        -------
        float
            The calibrated probability, in ``[0, 1]``.
        """
        index = bisect.bisect_right(self.thresholds, score) - 1
        index = max(0, min(index, len(self.probabilities) - 1))
        return self.probabilities[index]


def fit_isotonic(
    scores: Sequence[float], outcomes: Sequence[bool]
) -> IsotonicModel:
    """Pool-adjacent-violators fit of margin to P(correct).

    Isotonic regression is the non-parametric answer to "this score
    is ordered but not calibrated": it finds the monotone step
    function closest to the observed accuracies, merging any run of
    scores whose accuracies decrease. No shape is assumed, so a
    margin that carries no information flattens to the base rate
    instead of being forced into a line.

    Parameters
    ----------
    scores : sequence of float
        Margins from the calibration split.
    outcomes : sequence of bool
        Whether each decision was correct.

    Returns
    -------
    IsotonicModel
        The fitted step map.

    Raises
    ------
    ValueError
        If the sequences differ in length or are empty.
    """
    if len(scores) != len(outcomes):
        raise ValueError("scores and outcomes must be the same length")
    if not scores:
        raise ValueError("cannot fit isotonic regression on no data")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    # Each block holds [outcome sum, count, score sum]; the fitted
    # probability is the first over the second, the threshold the
    # third over the second.
    blocks: list[list[float]] = []
    for index in order:
        block = [1.0 if outcomes[index] else 0.0, 1.0, scores[index]]
        while blocks and blocks[-1][0] / blocks[-1][1] > block[0] / block[1]:
            previous = blocks.pop()
            block[0] += previous[0]
            block[1] += previous[1]
            block[2] += previous[2]
        blocks.append(block)
    return IsotonicModel(
        thresholds=tuple(block[2] / block[1] for block in blocks),
        probabilities=tuple(block[0] / block[1] for block in blocks),
    )


def softmax_confidence(
    values: Sequence[float], index: int, tau: float
) -> float:
    """The probability mass softmax gives one action.

    Parameters
    ----------
    values : sequence of float
        The three terminal Q-values, ordered as
        :data:`smallai.agent.TERMINALS`.
    index : int
        Which action's probability to return.
    tau : float
        The temperature; larger flattens the distribution.

    Returns
    -------
    float
        The action's probability, in ``(0, 1)``.

    Raises
    ------
    ValueError
        If ``tau`` is not positive or ``index`` is out of range.
    """
    if tau <= 0.0:
        raise ValueError(f"temperature must be positive, got {tau}")
    if not 0 <= index < len(values):
        raise ValueError(f"index {index} out of range")
    scaled = [value / tau for value in values]
    highest = max(scaled)
    weights = [math.exp(value - highest) for value in scaled]
    return weights[index] / sum(weights)


def _temperature_loss(
    log_tau: float,
    terminal_q: Sequence[tuple[float, ...]],
    correct: Sequence[bool],
    chosen: Sequence[int],
) -> float:
    """Mean binary cross-entropy of softmax P(correct) at one tau.

    Parameters
    ----------
    log_tau : float
        Log temperature, so the search runs on an unbounded scale.
    terminal_q : sequence of tuple
        Terminal Q-vectors from the calibration split.
    correct : sequence of bool
        Whether each decision was right.
    chosen : sequence of int
        Index into :data:`smallai.agent.TERMINALS` of the action
        each decision took.

    Returns
    -------
    float
        The mean negative log-likelihood.
    """
    tau = math.exp(log_tau)
    total = 0.0
    for values, outcome, index in zip(terminal_q, correct, chosen):
        probability = softmax_confidence(values, index, tau)
        probability = min(max(probability, 1e-12), 1.0 - 1e-12)
        if outcome:
            total -= math.log(probability)
        else:
            total -= math.log(1.0 - probability)
    return total / len(correct)


def fit_temperature(
    terminal_q: Sequence[tuple[float, ...]],
    correct: Sequence[bool],
    chosen: Sequence[int],
) -> float:
    """Fit the softmax temperature on the calibration split.

    Golden-section search on log tau, which is unimodal for the
    cross-entropy of a three-way softmax: the search needs no
    gradient and cannot stall on the flat regions a grid would hit.

    Parameters
    ----------
    terminal_q : sequence of tuple
        Terminal Q-vectors from the calibration split.
    correct : sequence of bool
        Whether each decision was right.
    chosen : sequence of int
        Index into :data:`smallai.agent.TERMINALS` per decision.

    Returns
    -------
    float
        The fitted temperature; 1.0 means no rescaling was needed.

    Raises
    ------
    ValueError
        If the calibration split is empty.
    """
    if not correct:
        raise ValueError("cannot fit a temperature on no data")

    def objective(log_tau: float) -> float:
        return _temperature_loss(log_tau, terminal_q, correct, chosen)

    low, high = math.log(1e-2), math.log(1e3)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = high - ratio * (high - low)
    right = low + ratio * (high - low)
    f_left, f_right = objective(left), objective(right)
    for _ in range(80):
        if f_left < f_right:
            high, right, f_right = right, left, f_left
            left = high - ratio * (high - low)
            f_left = objective(left)
        else:
            low, left, f_left = left, right, f_right
            right = low + ratio * (high - low)
            f_right = objective(right)
    return math.exp((low + high) / 2.0)


@dataclass(frozen=True)
class CalibrationReport:
    """Both section 9 maps, and whether they agree.

    Attributes
    ----------
    isotonic : IsotonicModel
        The shipped margin -> P(correct) map.
    temperature : float
        The fitted softmax temperature.
    ece_isotonic : float
        ECE of the isotonic map on the test split.
    ece_temperature : float
        ECE of the temperature map on the test split (section 9
        asks for the cross-check "reported as ECE only").
    mean_probability_difference : float
        Mean absolute gap between the two maps on the test split.
    provisional : bool
        True iff the maps disagree materially, per the section 9
        thresholds.
    """

    isotonic: IsotonicModel
    temperature: float
    ece_isotonic: float
    ece_temperature: float
    mean_probability_difference: float
    provisional: bool


def calibrate(
    margins: Sequence[float],
    terminal_q: Sequence[tuple[float, ...]],
    correct: Sequence[bool],
    actions: Sequence[str],
) -> tuple[IsotonicModel, float]:
    """Fit both confidence maps on the calibration split.

    Parameters
    ----------
    margins : sequence of float
        Terminal margins, one per calibration decision.
    terminal_q : sequence of tuple
        Terminal Q-vectors, same order.
    correct : sequence of bool
        Whether each calibration decision was right.
    actions : sequence of str
        The action each decision took, to locate it in the
        terminal Q-vector.

    Returns
    -------
    tuple
        ``(isotonic_model, temperature)``.
    """
    isotonic = fit_isotonic(margins, correct)
    chosen = [TERMINALS.index(action) for action in actions]
    temperature = fit_temperature(terminal_q, correct, chosen)
    return isotonic, temperature


def calibration_agreement(
    isotonic: IsotonicModel,
    temperature: float,
    margins: Sequence[float],
    terminal_q: Sequence[tuple[float, ...]],
    correct: Sequence[bool],
    actions: Sequence[str],
) -> CalibrationReport:
    """Compare the two maps on the test split (section 9).

    Disagreement is material when the ECE gap exceeds 0.03 or the
    mean absolute probability gap exceeds 0.05; either one flags
    the shipped isotonic map provisional rather than letting a
    silent preference decide.

    Parameters
    ----------
    isotonic : IsotonicModel
        The fitted primary map.
    temperature : float
        The fitted cross-check temperature.
    margins : sequence of float
        Test-split terminal margins.
    terminal_q : sequence of tuple
        Test-split terminal Q-vectors.
    correct : sequence of bool
        Test-split correctness.
    actions : sequence of str
        Test-split actions.

    Returns
    -------
    CalibrationReport
        Both ECEs, the probability gap and the provisional flag.
    """
    chosen = [TERMINALS.index(action) for action in actions]
    isotonic_confidence = [isotonic.predict(m) for m in margins]
    temperature_confidence = [
        softmax_confidence(values, index, temperature)
        for values, index in zip(terminal_q, chosen)
    ]
    ece_isotonic = ece(isotonic_confidence, correct)
    ece_temperature = ece(temperature_confidence, correct)
    mean_gap = sum(
        abs(a - b) for a, b in zip(isotonic_confidence, temperature_confidence)
    ) / len(actions)
    return CalibrationReport(
        isotonic=isotonic,
        temperature=temperature,
        ece_isotonic=ece_isotonic,
        ece_temperature=ece_temperature,
        mean_probability_difference=mean_gap,
        provisional=(
            abs(ece_isotonic - ece_temperature) > 0.03 or mean_gap > 0.05
        ),
    )


@dataclass(frozen=True)
class HypothesisResult:
    """One section 10 hypothesis, with the CI discipline of section 8.

    Attributes
    ----------
    name : str
        ``"H1"`` .. ``"H4"``.
    claim : str
        The claim, in words.
    delta_pp : float or None
        The measured difference in percentage points; None for H4,
        which is a measurement-count claim.
    ci_low_pp : float or None
        CI lower end, where one applies.
    ci_high_pp : float or None
        CI upper end, where one applies.
    p_value : float or None
        McNemar's exact p-value, where one applies.
    supported : bool
        Whether the point estimate favours the claim.
    note : str
        How to read the row.
    """

    name: str
    claim: str
    delta_pp: float | None
    ci_low_pp: float | None
    ci_high_pp: float | None
    p_value: float | None
    supported: bool
    note: str


def hypotheses(
    journey: PolicyScore,
    shortcut: PolicyScore,
    no_recheck: PolicyScore,
    expert_only: PolicyScore,
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[HypothesisResult]:
    """Evaluate H1-H4 with paired intervals (section 10).

    H1 is the sole confirmatory hypothesis. H2-H4 are mechanism
    checks: they are reported with the same interval discipline but
    are not gated on significance, so ``supported`` reflects the
    point estimate and the interval is printed beside it.

    Parameters
    ----------
    journey : PolicyScore
        The deliverable, Journey-Q.
    shortcut : PolicyScore
        Shortcut-CART.
    no_recheck : PolicyScore
        NoRecheck-Q.
    expert_only : PolicyScore
        ExpertOnly-Q.
    seed : int
        Master seed for the bootstrap resampling.
    resamples : int
        Bootstrap resamples per delta.

    Returns
    -------
    list of HypothesisResult
        H1, H2, H3, H4 in that order.
    """
    results: list[HypothesisResult] = []
    for name, claim, other, confirmatory in (
        (
            "H1",
            "Journey-Q accuracy > Shortcut-CART accuracy",
            shortcut,
            True,
        ),
        (
            "H2",
            "Journey-Q accuracy >= NoRecheck-Q accuracy",
            no_recheck,
            False,
        ),
        (
            "H3",
            "Journey-Q accuracy >= ExpertOnly-Q accuracy",
            expert_only,
            False,
        ),
    ):
        boot = paired_bootstrap_delta(
            journey.correct,
            other.correct,
            seed=derive_seed(seed, "bootstrap", name),
            resamples=resamples,
        )
        test = mcnemar(journey.correct, other.correct)
        if confirmatory:
            supported = boot.ci_low_pp > 0.0
            note = (
                "CI excludes zero in favour of Journey-Q"
                if supported
                else "CI includes zero: inconclusive"
            )
        else:
            supported = boot.delta_pp >= 0.0
            note = (
                "mechanism check, not gated on significance; "
                "CI printed for discipline"
            )
        results.append(
            HypothesisResult(
                name=name,
                claim=claim,
                delta_pp=boot.delta_pp,
                ci_low_pp=boot.ci_low_pp,
                ci_high_pp=boot.ci_high_pp,
                p_value=test.p_value,
                supported=supported,
                note=note,
            )
        )
    measurements = journey.mean_measurements
    results.append(
        HypothesisResult(
            name="H4",
            claim="Journey-Q mean measurements/decision < 6",
            delta_pp=None,
            ci_low_pp=None,
            ci_high_pp=None,
            p_value=None,
            supported=measurements < 6.0,
            note=(
                f"measured mean {measurements:.3f}; the tree reads "
                "all six signals by construction"
            ),
        )
    )
    return results


def visited_state_coverage(visited: int) -> float:
    """Visited states as a fraction of the state-space ceiling.

    EXPERIMENT.md section 3 asks for coverage against "states
    reachable under the sampling distribution", which has no closed
    form: reachability depends on the defect mixture and the noise,
    and estimating it would need its own Monte Carlo run. The
    reported denominator is therefore the 5^6 ceiling, which makes
    this a documented *lower* bound on the coverage the section
    means -- it can only understate how much of the reachable space
    was seen.

    Parameters
    ----------
    visited : int
        Distinct states the trained table holds.

    Returns
    -------
    float
        ``visited / 15,625``.
    """
    return visited / MAX_STATES
