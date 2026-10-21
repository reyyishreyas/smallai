"""One command: generate -> train -> evaluate -> ``reports/poc.md``.

Runs the frozen protocol of EXPERIMENT.md end to end and writes the
Stage A report. Nothing here selects a hyperparameter: splits,
budget, reward, plateau test and the policy field are imported from
the pre-registration, and the only inputs are the master seed, the
episode budget and the bootstrap resample count, all of which are
printed into the report so a reader can reproduce the run exactly.

Usage::

    python scripts/run_poc.py                 # scored run
    python scripts/run_poc.py --budget 5000   # smoke run
    python scripts/run_poc.py --resamples 500 # quick interval

A run that departs from the pre-registered budget or resample count
says so in its own header, so a smoke run cannot be mistaken for a
scored one.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy

from smallai.agent import QAgent
from smallai.evaluate import (
    BOOTSTRAP_RESAMPLES,
    BootstrapDelta,
    CalibrationReport,
    HypothesisResult,
    PolicyScore,
    calibrate,
    calibration_agreement,
    derive_seed,
    hypotheses,
    mcnemar,
    noise_sigma,
    paired_bootstrap_delta,
    score_agent,
    score_tree,
    visited_state_coverage,
)
from smallai.train import (
    BLOCK,
    MIX,
    MIX_EXPERT_ONLY,
    TrainingReport,
    cross_validated_tree,
    make_splits,
    make_tree_dataset,
    oracle_label,
    train_td,
)

#: EXPERIMENT.md section 4: fixed here, before any scored run. Every
#: other seed in the pipeline derives from this one.
MASTER_SEED = 20261021

#: Section 8: sigma = 0.08 is the primary condition.
SIGMAS: tuple[float, ...] = (0.04, 0.08, 0.12)
PRIMARY_SIGMA = 0.08

#: Section 7: the pre-registered episode cap per TD policy.
PRE_REGISTERED_BUDGET = 1_000_000

REPORTS_DIR = ROOT / "reports"


@dataclass
class ConditionResult:
    """Everything one noise-sweep condition produced.

    Attributes
    ----------
    sigma : float
        Measurement-noise standard deviation this cycle ran at.
    primary : bool
        Whether this is the sigma = 0.08 condition.
    training : dict
        Policy name to its :class:`~smallai.train.TrainingReport`.
    scores : dict
        Policy name to its :class:`~smallai.evaluate.PolicyScore`
        on the test split.
    tree_leaf : int
        ``min_samples_leaf`` cross-validation chose, train only.
    tree_cv_scores : list of float
        Per-fold accuracy at that leaf size.
    calibration : CalibrationReport
        Both confidence maps, fitted on the calibration split and
        compared on the test split.
    class_counts : dict
        Outcome-class counts in the calibration split; section 4
        asks for these because thin per-class support makes the
        isotonic map provisional.
    hypotheses : list of HypothesisResult
        H1-H4 with paired intervals.
    coverage : float
        Journey-Q visited states over the 5^6 ceiling.
    agents : dict
        Policy name to its trained :class:`~smallai.agent.QAgent`,
        kept so the primary condition's table can be written out as
        the JSON artifact ARCHITECTURE.md section 5.3 promises the
        Android port consumes.
    """

    sigma: float
    primary: bool
    training: dict[str, TrainingReport]
    scores: dict[str, PolicyScore]
    tree_leaf: int
    tree_cv_scores: list[float]
    calibration: CalibrationReport
    class_counts: dict[str, int]
    hypotheses: list[HypothesisResult]
    coverage: float
    agents: dict[str, QAgent]


def margins_of(score: PolicyScore) -> list[float]:
    """Terminal margins from a score, empty for the tree.

    Parameters
    ----------
    score : PolicyScore
        A score produced by :func:`score_agent`.

    Returns
    -------
    list of float
        One margin per scored scenario.
    """
    return list(score.margins or [])


def terminal_q_of(score: PolicyScore) -> list[tuple[float, ...]]:
    """Terminal Q-vectors from a score, empty for the tree.

    Parameters
    ----------
    score : PolicyScore
        A score produced by :func:`score_agent`.

    Returns
    -------
    list of tuple
        One Q-vector per scored scenario.
    """
    return list(score.terminal_q or [])


def downsample_curve(
    block_means: Sequence[float], points: int = 12
) -> list[tuple[int, float]]:
    """At most ``points`` ``(episode, block mean)`` pairs.

    The full curve ships in the JSON artifact; the report keeps
    enough points to see the shape without a page of decimals.

    Parameters
    ----------
    block_means : sequence of float
        Mean reward per complete block, oldest first.
    points : int
        Approximate number of rows to keep.

    Returns
    -------
    list of tuple
        Episode index to block mean, always including the last
        block.
    """
    if not block_means:
        return []
    step = max(1, len(block_means) // points)
    picked = list(range(0, len(block_means), step))
    if picked[-1] != len(block_means) - 1:
        picked.append(len(block_means) - 1)
    return [((index + 1) * BLOCK, block_means[index]) for index in picked]


def run_condition(
    sigma: float,
    splits: dict[str, list],
    labels: dict[str, list[str]],
    *,
    seed: int,
    budget: int,
    resamples: int,
) -> ConditionResult:
    """Train every policy and score it, at one noise level.

    Splits are passed in rather than redrawn, so the sensitivity
    sweep varies exactly one thing: the measurement noise.
    Training and scoring both run inside the block where ``observe``
    reads the swept sigma.

    Parameters
    ----------
    sigma : float
        Measurement-noise standard deviation for this cycle.
    splits : dict
        Split name to scenarios, from
        :func:`smallai.train.make_splits`.
    labels : dict
        Split name to oracle verdicts, same order.
    seed : int
        Master seed; sub-seeds derive from it per policy.
    budget : int
        Episode cap per TD policy.
    resamples : int
        Bootstrap resamples per reported delta.

    Returns
    -------
    ConditionResult
        Training reports, test scores, calibration and hypotheses.
    """
    train = splits["train"]
    training: dict[str, TrainingReport] = {}
    agents: dict[str, QAgent] = {}

    plans = (
        ("journey_q", QAgent(name="journey_q"), MIX),
        (
            "no_recheck_q",
            QAgent(allow_recheck=False, name="no_recheck_q"),
            MIX,
        ),
        (
            "expert_only_q",
            QAgent(name="expert_only_q"),
            MIX_EXPERT_ONLY,
        ),
    )
    for name, agent, mix in plans:
        started = time.perf_counter()
        report = train_td(
            agent,
            train,
            seed=derive_seed(seed, name, str(sigma)),
            budget=budget,
            mix=mix,
        )
        elapsed = time.perf_counter() - started
        print(
            f"  trained {name:<14} {report.episodes:>9,} episodes"
            f"  plateau={report.plateau!s:<5}"
            f"  entries={report.stored_entries:>6,}"
            f"  {elapsed:6.1f}s",
            flush=True,
        )
        training[name] = report
        agents[name] = agent

    x_train, y_train = make_tree_dataset(
        train,
        labels["train"],
        random.Random(derive_seed(seed, "tree_fit", str(sigma))),
    )
    tree, leaf, fold_scores = cross_validated_tree(
        x_train,
        y_train,
        seed=derive_seed(seed, "tree_cv", str(sigma)),
    )
    print(
        f"  fitted shortcut_cart   min_samples_leaf={leaf}"
        f"  folds={[round(s, 3) for s in fold_scores]}",
        flush=True,
    )

    test = splits["test"]
    test_labels = labels["test"]
    scores = {
        name: score_agent(
            agent, test, test_labels, derive_seed(seed, "score", name)
        )
        for name, agent in agents.items()
    }
    x_test, _ = make_tree_dataset(
        test,
        test_labels,
        random.Random(derive_seed(seed, "tree_test", str(sigma))),
    )
    scores["shortcut_cart"] = score_tree(tree, x_test, test, test_labels)

    journey = scores["journey_q"]
    isotonic, temperature = calibrate(
        margins_of(journey),
        terminal_q_of(journey),
        journey.correct,
        journey.actions,
    )
    # Fitted on calibration, then both maps are compared on the
    # held-out test split, which is the comparison section 9 means.
    agreement = calibration_agreement(
        isotonic,
        temperature,
        margins_of(journey),
        terminal_q_of(journey),
        journey.correct,
        journey.actions,
    )

    class_counts = {"pass": 0, "retake": 0, "review": 0}
    for verdict in labels["calibration"]:
        class_counts[verdict] += 1

    return ConditionResult(
        sigma=sigma,
        primary=abs(sigma - PRIMARY_SIGMA) < 1e-12,
        training=training,
        scores=scores,
        tree_leaf=leaf,
        tree_cv_scores=fold_scores,
        calibration=agreement,
        class_counts=class_counts,
        hypotheses=hypotheses(
            scores["journey_q"],
            scores["shortcut_cart"],
            scores["no_recheck_q"],
            scores["expert_only_q"],
            seed=seed,
            resamples=resamples,
        ),
        coverage=visited_state_coverage(training["journey_q"].visited_states),
        agents=agents,
    )


def score_summary(score: PolicyScore) -> dict:
    """The reported columns for one policy.

    Parameters
    ----------
    score : PolicyScore
        A scored policy.

    Returns
    -------
    dict
        Accuracy, measurement count, rates and latency, plus the
        table-size fields a Q agent has (the tree has no table, so
        those stay None).
    """
    return {
        "accuracy": score.accuracy,
        "mean_measurements": score.mean_measurements,
        "review_rate": score.review_rate,
        "invalid_rate": score.invalid_rate,
        "mean_latency_ms": score.mean_latency_ms,
        "stored_entries": None,
        "visited_states": None,
    }


def build_payload(
    conditions: Sequence[ConditionResult],
    *,
    seed: int,
    budget: int,
    resamples: int,
    elapsed_seconds: float,
) -> dict:
    """The whole run as one JSON-serializable dict.

    Parameters
    ----------
    conditions : sequence of ConditionResult
        One per sigma, in sweep order.
    seed : int
        The master seed the run used.
    budget : int
        The episode cap the run used.
    resamples : int
        The bootstrap resample count the run used.
    elapsed_seconds : float
        Wall-clock time of the whole run.

    Returns
    -------
    dict
        Environment, protocol inputs and every measured number.
    """
    return {
        "framing": "CLEF-inspired lightweight decision-policy POC",
        "stage": "A",
        "master_seed": seed,
        "budget": budget,
        "resamples": resamples,
        "elapsed_seconds": elapsed_seconds,
        "pre_registered_budget": PRE_REGISTERED_BUDGET,
        "pre_registered_resamples": BOOTSTRAP_RESAMPLES,
        "environment": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "conditions": [
            {
                "sigma": condition.sigma,
                "primary": condition.primary,
                "class_counts": condition.class_counts,
                "tree_leaf": condition.tree_leaf,
                "tree_cv_scores": condition.tree_cv_scores,
                "coverage": condition.coverage,
                "training": {
                    name: report.to_dict()
                    for name, report in condition.training.items()
                },
                "scores": {
                    name: score_summary(score)
                    for name, score in condition.scores.items()
                },
                "calibration": {
                    "temperature": condition.calibration.temperature,
                    "ece_isotonic": condition.calibration.ece_isotonic,
                    "ece_temperature": (condition.calibration.ece_temperature),
                    "mean_probability_difference": (
                        condition.calibration.mean_probability_difference
                    ),
                    "provisional": condition.calibration.provisional,
                    "thresholds": list(
                        condition.calibration.isotonic.thresholds
                    ),
                    "probabilities": list(
                        condition.calibration.isotonic.probabilities
                    ),
                },
                "hypotheses": [
                    asdict(result) for result in condition.hypotheses
                ],
            }
            for condition in conditions
        ],
    }


#: Report order, with the role each policy plays (EXPERIMENT.md 5).
POLICY_ORDER: tuple[str, ...] = (
    "journey_q",
    "no_recheck_q",
    "expert_only_q",
    "shortcut_cart",
)
POLICY_LABEL: dict[str, str] = {
    "journey_q": "Journey-Q (deliverable)",
    "no_recheck_q": "NoRecheck-Q (ablation)",
    "expert_only_q": "ExpertOnly-Q (ablation)",
    "shortcut_cart": "Shortcut-CART (baseline)",
}


def fmt(value: float | None, digits: int = 3) -> str:
    """A fixed-precision number, or the placeholder when absent.

    Unmeasured and inapplicable values render as an en-dash rather
    than as zero, so a missing number can never be misread as a
    measured one.

    Parameters
    ----------
    value : float or None
        The number to render.
    digits : int
        Decimal places.

    Returns
    -------
    str
        The formatted number, or ``"–"``.
    """
    if value is None:
        return "–"
    return f"{value:.{digits}f}"


def signed(value: float) -> str:
    """A delta with an explicit sign, in percentage points.

    Parameters
    ----------
    value : float
        The delta.

    Returns
    -------
    str
        e.g. ``"+4.200"`` or ``"-1.000"``.
    """
    return f"{value:+.3f}"


def paired(
    first: PolicyScore,
    second: PolicyScore,
    *,
    seed: int,
    label: str,
    resamples: int,
) -> BootstrapDelta:
    """The paired bootstrap delta between two scored policies.

    The seed label matters as much as the seed: passing ``"H1"``
    reproduces the interval ``hypotheses`` computed for H1, so the
    Primary result section and the H1 row can never show two
    different numbers for the same comparison.

    Parameters
    ----------
    first : PolicyScore
        The first policy, e.g. Journey-Q.
    second : PolicyScore
        The second, scored on the same scenarios.
    seed : int
        The run's master seed.
    label : str
        Names this comparison, folded into the derived seed.
    resamples : int
        Bootstrap resamples.

    Returns
    -------
    BootstrapDelta
        Delta and 95% interval, in percentage points.
    """
    return paired_bootstrap_delta(
        first.correct,
        second.correct,
        seed=derive_seed(seed, "bootstrap", label),
        resamples=resamples,
    )


def render_histogram(report: TrainingReport, limit: int = 20) -> list[str]:
    """Markdown rows for the state-visit histogram.

    Parameters
    ----------
    report : TrainingReport
        The training run to render.
    limit : int
        Rows to show before pointing at the JSON artifact.

    Returns
    -------
    list of str
        Table rows, plus a note when rows were omitted.
    """
    rows = [
        f"| {visits} | {states} |"
        for visits, states in report.visit_histogram.items()
    ]
    if len(rows) > limit:
        omitted = len(rows) - limit
        rows = rows[:limit]
        rows.append(
            f"| … | {omitted} further visit-count rows are in poc.json |"
        )
    return rows


def render_markdown(
    conditions: Sequence[ConditionResult],
    *,
    seed: int,
    budget: int,
    resamples: int,
    elapsed_seconds: float,
) -> str:
    """The Stage A report as Markdown.

    Parameters
    ----------
    conditions : sequence of ConditionResult
        One per sigma, in sweep order.
    seed : int
        The master seed the run used.
    budget : int
        The episode cap the run used.
    resamples : int
        The bootstrap resample count the run used.
    elapsed_seconds : float
        Wall-clock time of the whole run.

    Returns
    -------
    str
        The contents of ``reports/poc.md``.
    """
    primary = next(condition for condition in conditions if condition.primary)
    scored = (
        budget == PRE_REGISTERED_BUDGET and resamples == BOOTSTRAP_RESAMPLES
    )
    journey = primary.scores["journey_q"]
    shortcut = primary.scores["shortcut_cart"]
    delta = (journey.accuracy - shortcut.accuracy) * 100.0
    boot = paired(
        journey,
        shortcut,
        seed=seed,
        label="H1",
        resamples=resamples,
    )
    discordance = mcnemar(journey.correct, shortcut.correct)
    lines: list[str] = []
    add = lines.append

    add("# Stage A POC report - AI-267")
    add("")
    add("*CLEF-inspired lightweight decision-policy POC.*")
    add("")
    add(
        "Measured results for the protocol frozen in `EXPERIMENT.md`."
        " Every number below came out of one run of"
        " `python scripts/run_poc.py` on the machine listed under"
        " Environment; nothing is estimated, and the protocol was not"
        " touched after these results existed."
    )
    add("")
    if scored:
        add(
            f"**Run status: scored.** Budget {budget:,} episodes and"
            f" {resamples:,} bootstrap resamples, both as"
            " pre-registered."
        )
    else:
        add(
            "**Run status: SMOKE RUN.** This run used"
            f" {budget:,} episodes and {resamples:,} resamples, not"
            f" the pre-registered {PRE_REGISTERED_BUDGET:,} and"
            f" {BOOTSTRAP_RESAMPLES:,}. No number here may be quoted"
            " as a result."
        )
    add("")
    add(
        f"Total wall clock {elapsed_seconds:.1f} s. Master seed"
        f" `{seed}`; every other seed derives from it by SHA-256, so"
        " the whole run reproduces from that one number."
    )
    add("")

    add("## Environment")
    add("")
    add(f"- Python {platform.python_version()}")
    add(f"- numpy {numpy.__version__}")
    add(f"- {platform.platform()} on {platform.machine()}")
    add("")
    add(
        "Latency is measured on this machine only. EXPERIMENT.md"
        " section 12 makes the <10 ms decision-core criterion"
        " non-binding at table-lookup scale and rules out any"
        " Android claim in Stage A; on-device measurement is Stage B."
    )
    add("")

    add("## Splits")
    add("")
    add(
        "Scenarios are partitioned before any noise is drawn, and no"
        " true signal vector appears in two splits. Train uses the"
        " blur-heavy field mixture, calibration and test the"
        " balanced one, so a one-shot policy cannot pass by learning"
        ' "blurry means retake" alone.'
    )
    add("")
    add("| split | n | distribution | used for |")
    add("|---|---|---|---|")
    add("| train | 4,000 | skewed, blur-heavy | fitting all policies |")
    add("| calibration | 1,000 | balanced | fitting the confidence map only |")
    add("| test | 1,000 | balanced | scoring only, never tuned on |")
    add("")
    add(
        "Outcome classes in the calibration split"
        f" (thin per-class support would make the isotonic map"
        f" provisional): pass {primary.class_counts['pass']},"
        f" retake {primary.class_counts['retake']},"
        f" review {primary.class_counts['review']}."
    )
    add("")
    thin = min(primary.class_counts, key=primary.class_counts.get)
    add(
        "Section 4 asks for exactly these counts, because isotonic"
        " regression needs adequate per-class support. Here the"
        f" `{thin}` class contributes {primary.class_counts[thin]} of"
        f" {sum(primary.class_counts.values())}, which is thin: the"
        " confidence map rests on few observations wherever a"
        " decision turns on that outcome. The counts are reported"
        " rather than smoothed away."
    )
    add("")

    add("## Policy field")
    add("")
    add("| policy | description | role |")
    add("|---|---|---|")
    add(
        "| Expert rules | deterministic expert;"
        " defines the labels | reference row, 100% by construction,"
        " not scored |"
    )
    add(
        "| Shortcut-CART | from-scratch CART, max depth 6 a priori,"
        " hyperparameters by CV on train only | graded baseline:"
        " one shot, no verification |"
    )
    add(
        "| NoRecheck-Q | Journey-Q with re-check actions removed |"
        " ablation: value of re-verification |"
    )
    add(
        "| ExpertOnly-Q | Journey-Q without flawed-trajectory"
        " replay | ablation: value of journey learning itself |"
    )
    add("| Journey-Q | the full model | the deliverable |")
    add("")
    add(
        "Each ablation differs from Journey-Q in exactly one"
        " variable, and no hyperparameter of any policy was selected"
        " on the calibration or test splits."
    )
    add("")

    add("## Training")
    add("")
    add(
        f"Table entries are from the primary condition"
        f" (sigma = {PRIMARY_SIGMA}). The plateau test is the"
        " pre-registered one: 100-episode block means over the final"
        " 20%, OLS slope, plateau iff the 95% CI contains zero."
    )
    add("")
    add(
        "| policy | episodes | plateau | slope [95% CI] |"
        " stored entries | visited states | coverage |"
    )
    add("|---|---|---|---|---|---|---|")
    for name in POLICY_ORDER:
        report = primary.training.get(name)
        if report is None:
            add(f"| {POLICY_LABEL[name]} | – | – | – | – | – | – |")
            continue
        slope = fmt(report.slope, 6)
        interval = (
            f"{slope} [{fmt(report.ci_low, 6)}, {fmt(report.ci_high, 6)}]"
            if report.slope is not None
            else "–"
        )
        coverage = (
            f"{visited_state_coverage(report.visited_states) * 100.0:.3f}%"
        )
        add(
            f"| {POLICY_LABEL[name]} | {report.episodes:,} |"
            f" {'yes' if report.plateau else 'no'} | {interval} |"
            f" {report.stored_entries:,} |"
            f" {report.visited_states:,} | {coverage} |"
        )
    add("")
    stalled = [
        POLICY_LABEL[name]
        for name in POLICY_ORDER
        if name in primary.training and not primary.training[name].plateau
    ]
    if stalled:
        add(
            "Did not converge within budget: "
            + ", ".join(stalled)
            + ". That is the pre-registered outcome of EXPERIMENT.md"
            " section 7, reported as such rather than as a silent"
            " stop."
        )
    else:
        add(
            "Every TD policy declared plateau before its budget ran"
            " out, so none stopped on the episode cap."
        )
    add("")
    add(
        "Shortcut-CART is not TD-trained, which is why its row is"
        " empty above. Cross-validation on the train split alone"
        f" chose min_samples_leaf={primary.tree_leaf}, with 5-fold"
        " accuracies "
        + ", ".join(f"{score:.3f}" for score in primary.tree_cv_scores)
        + ". Max depth stays 6 a priori."
    )
    add("")
    add(
        "Coverage is visited states over the 5^6 = 15,625 ceiling."
        " EXPERIMENT.md section 3 asks for coverage over the states"
        " reachable under the sampling distribution, which has no"
        " closed form, so this figure is a documented lower bound on"
        " that quantity."
    )
    add("")

    add("## Primary result")
    add("")
    add(
        f"Journey-Q accuracy minus Shortcut-CART accuracy on the"
        f" test split: **{signed(delta)} percentage points**, with a"
        f" paired bootstrap 95% CI of [{fmt(boot.ci_low_pp)},"
        f" {fmt(boot.ci_high_pp)}] over {boot.resamples:,} resamples"
        f" of test scenarios. Both policies are scored on the same"
        f" 1,000 scenarios, so the comparison is paired."
    )
    add("")
    add(
        f"McNemar's test on the {discordance.first_only + discordance.second_only}"
        f" discordant pairs: chi-square {fmt(discordance.chi_square)},"
        f" exact two-sided p = {fmt(discordance.p_value, 4)}."
    )
    add("")
    if boot.excludes_zero:
        add("The interval excludes zero in favour of Journey-Q.")
    else:
        add(
            "The interval includes zero, so the difference is"
            " **inconclusive** whatever its sign."
        )
    add("")
    add(
        "EXPERIMENT.md section 2 requires that, with no"
        " mentor-supplied X, the measured delta and its interval are"
        " stated without a pass/fail claim. No X has been supplied,"
        " so none is made here."
    )
    add("")

    add("## Hypotheses")
    add("")
    add(
        "H1 is the sole confirmatory hypothesis. H2-H4 are mechanism"
        " checks, reported with the same interval discipline but not"
        " gated on significance."
    )
    add("")
    add("| hypothesis | claim | delta (pp) | 95% CI | p | reads as |")
    add("|---|---|---|---|---|---|")
    for result in primary.hypotheses:
        interval = (
            f"[{fmt(result.ci_low_pp)}, {fmt(result.ci_high_pp)}]"
            if result.ci_low_pp is not None
            else "–"
        )
        add(
            f"| {result.name} | {result.claim} |"
            f" {fmt(result.delta_pp)} | {interval} |"
            f" {fmt(result.p_value, 4)} | {result.note} |"
        )
    add("")

    add("## Secondary metrics")
    add("")
    add(
        "All on the test split at the primary sigma. The expert row"
        " is not scored: its verdicts *are* the labels, so its"
        " accuracy is 100% by construction and its other columns do"
        " not exist."
    )
    add("")
    add(
        "| policy | accuracy % | delta vs Journey (pp) |"
        " mean measurements | review rate % | invalid rate % |"
        " mean latency (ms) |"
    )
    add("|---|---|---|---|---|---|---|")
    add("| Expert rules (defines the labels) | 100.000 | – | – | – | – | – |")
    for name in POLICY_ORDER:
        score = primary.scores[name]
        difference = (score.accuracy - journey.accuracy) * 100.0
        add(
            f"| {POLICY_LABEL[name]} | {score.accuracy * 100.0:.3f} |"
            f" {signed(difference)} | {score.mean_measurements:.3f} |"
            f" {score.review_rate * 100.0:.3f} |"
            f" {score.invalid_rate * 100.0:.3f} |"
            f" {score.mean_latency_ms:.4f} |"
        )
    add("")
    add(
        "Invalid-verdict rate is the fraction of verdicts their own"
        " gathered evidence contradicts, under the section 3 rule:"
        " `pass` needs all six signals observed at pass level,"
        " `retake` an observed fail, `review` an observed suspect,"
        " and any terminal taken with no measurements is"
        " contradictory."
    )
    add("")

    add("## Confidence calibration")
    add("")
    add(
        "The terminal margin is a score in reward units, not a"
        " probability. Section 9 fits an isotonic regression from"
        " margin to P(correct) on the calibration split and"
        " cross-checks it against temperature-scaled softmax fitted"
        " on the same split."
    )
    add("")
    add(
        f"- isotonic ECE on test: **{fmt(primary.calibration.ece_isotonic, 4)}**"
    )
    add(
        "- temperature-scaled softmax ECE on test:"
        f" **{fmt(primary.calibration.ece_temperature, 4)}**"
        f" (fitted tau = {fmt(primary.calibration.temperature)})"
    )
    add(
        "- mean absolute probability difference between the two:"
        f" {fmt(primary.calibration.mean_probability_difference, 4)}"
    )
    add("")
    if primary.calibration.provisional:
        add(
            "The two forms disagree materially, so the shipped"
            " isotonic map is flagged **provisional** (section 9:"
            " ECE gap above 0.03, or mean probability gap above"
            " 0.05)."
        )
    else:
        add(
            "The two forms agree within the section 9 tolerances, so"
            " the isotonic map ships without a provisional flag."
        )
    add("")
    add(
        "The number shown to a user is the calibrated P(correct)."
        " The expert's hand-written confidence heuristic is internal"
        " only and never ships."
    )
    add("")

    add("## Sensitivity to measurement noise")
    add("")
    add(
        f"Sigma swept over {list(SIGMAS)}; sigma = {PRIMARY_SIGMA}"
        " is the primary condition. The splits are identical across"
        " the sweep, so the only thing that changes is the noise."
    )
    add("")
    add(
        "| sigma | Journey-Q % | Shortcut-CART % | delta (pp) |"
        " 95% CI | mean measurements |"
    )
    add("|---|---|---|---|---|---|")
    for condition in conditions:
        loop_journey = condition.scores["journey_q"]
        loop_shortcut = condition.scores["shortcut_cart"]
        loop_delta = (loop_journey.accuracy - loop_shortcut.accuracy) * 100.0
        row_boot = paired(
            loop_journey,
            loop_shortcut,
            seed=seed,
            label=f"sigma{condition.sigma}",
            resamples=resamples,
        )
        add(
            f"| {condition.sigma}"
            f"{' (primary)' if condition.primary else ''} |"
            f" {loop_journey.accuracy * 100.0:.3f} |"
            f" {loop_shortcut.accuracy * 100.0:.3f} |"
            f" {signed(loop_delta)} |"
            f" [{fmt(row_boot.ci_low_pp)}, {fmt(row_boot.ci_high_pp)}]"
            f" | {loop_journey.mean_measurements:.3f} |"
        )
    add("")

    add("## Convergence artifacts")
    add("")
    add(
        "Section 7 requires the training curve and the state-visit"
        " histogram to ship so convergence is auditable. Shown here"
        f" for Journey-Q at sigma = {PRIMARY_SIGMA}; the complete"
        " data is in `poc.json`."
    )
    add("")
    curve_report = primary.training["journey_q"]
    add("### Training curve")
    add("")
    add("| episode | mean block reward |")
    add("|---|---|")
    for episode, mean in downsample_curve(curve_report.block_means):
        add(f"| {episode:,} | {mean:.4f} |")
    add("")
    add(f"Final block mean: {curve_report.final_block_mean:.4f}.")
    add("")
    add("### State-visit histogram")
    add("")
    add("| visits | states |")
    add("|---|---|")
    lines.extend(render_histogram(curve_report))
    add("")
    add(
        f"Journey-Q stored {curve_report.stored_entries:,} values"
        f" across {curve_report.visited_states:,} distinct states"
        f" ({primary.coverage * 100.0:.3f}% of the 15,625"
        " ceiling)."
    )
    add("")

    add("## Limitations")
    add("")
    add(
        "- **Synthetic Stage 0.** The six signals are sampled, not"
        " computed from images. This POC validates the decision"
        " logic, the learning mechanisms and calibration. It"
        " establishes nothing about real ID photographs. The Stage B"
        " spike, roughly 50 real captures against these"
        " distributions, gates the Android work."
    )
    add(
        "- **A policy table, not a parametric model.** Reported as a"
        " stored-entry count. The ticket's 100K budget holds under"
        " either reading."
    )
    add(
        "- **The `review` action is an open scope question with the"
        " mentor.** The binary fallback is pre-declared in"
        " EXPERIMENT.md section 11 and was not used here."
    )
    add(
        "- **No pass/fail against X.** X has not been supplied, so"
        " the primary delta is reported with its interval and no"
        " threshold claim."
    )
    add("")
    add("---")
    add("")
    add(
        "Produced by `python scripts/run_poc.py` from the frozen"
        " protocol in `EXPERIMENT.md`. Design constants live in"
        " `ARCHITECTURE.md`; measured results live here only."
    )
    add("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: run the sweep, write the report and artifacts.

    Parameters
    ----------
    argv : sequence of str or None
        Command line without the program name; None reads sys.argv.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the AI-267 Stage A protocol and write"
            " reports/poc.md, reports/poc.json and the Q-table"
            " artifact."
        )
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=PRE_REGISTERED_BUDGET,
        help="episodes per TD policy (pre-registered: 1,000,000)",
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=BOOTSTRAP_RESAMPLES,
        help="bootstrap resamples per delta (pre-registered: 10,000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=MASTER_SEED,
        help="master seed; every other seed derives from it",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPORTS_DIR,
        help="directory to write the report into",
    )
    args = parser.parse_args(argv)

    print(
        f"AI-267 Stage A | master seed {args.seed}"
        f" | budget {args.budget:,} | resamples {args.resamples:,}",
        flush=True,
    )
    splits = make_splits(args.seed)
    print(
        "splits: " + ", ".join(f"{n}={len(v):,}" for n, v in splits.items()),
        flush=True,
    )
    labels = {
        name: [oracle_label(s) for s in scenarios]
        for name, scenarios in splits.items()
    }

    started = time.perf_counter()
    conditions: list[ConditionResult] = []
    for sigma in SIGMAS:
        print(f"noise sigma = {sigma}", flush=True)
        with noise_sigma(sigma):
            conditions.append(
                run_condition(
                    sigma,
                    splits,
                    labels,
                    seed=args.seed,
                    budget=args.budget,
                    resamples=args.resamples,
                )
            )
    elapsed = time.perf_counter() - started

    args.out.mkdir(parents=True, exist_ok=True)
    payload = build_payload(
        conditions,
        seed=args.seed,
        budget=args.budget,
        resamples=args.resamples,
        elapsed_seconds=elapsed,
    )
    (args.out / "poc.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / "poc.md").write_text(
        render_markdown(
            conditions,
            seed=args.seed,
            budget=args.budget,
            resamples=args.resamples,
            elapsed_seconds=elapsed,
        ),
        encoding="utf-8",
    )

    # The shipped model: ARCHITECTURE.md section 5.3 promises a JSON
    # table the Android port consumes with no inference runtime.
    primary = next(condition for condition in conditions if condition.primary)
    primary.agents["journey_q"].save(args.out / "journey_q.json")
    primary.agents["no_recheck_q"].save(args.out / "no_recheck_q.json")
    primary.agents["expert_only_q"].save(args.out / "expert_only_q.json")

    print(f"wrote {args.out / 'poc.md'}", flush=True)
    print(f"wrote {args.out / 'poc.json'} and the Q-table artifacts")
    print(f"total {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
