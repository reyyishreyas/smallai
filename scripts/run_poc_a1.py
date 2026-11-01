"""One command: run the AI-267 Stage A.1 round -> ``reports/poc_a1.md``.

Stage A.1 is the second, separately pre-registered round
(``EXPERIMENT_A1.md``). It exists because Stage A's primary
hypothesis could not be answered on a 1,000-scenario test split: the
paired noise on an accuracy delta is about 0.36 pp there, while the
journey learning's real effect is a fraction of that.

What this runner does, all of it pre-registered before the run:

- runs the Stage A policy set (Shortcut-CART, Journey-Q, NoRecheck-Q,
  ExpertOnly-Q) at the three pre-registered noise levels, on a test
  split of 10,000 scenarios instead of 1,000;
- repeats the whole thing over five fresh master seeds, because one
  master seed leaves a training-draw noise of about 0.25 pp, which
  would swamp the effect the round is trying to measure;
- adds two pre-registered secondary analyses on the metrics where
  the journey policy actually competes: the invalid-verdict rate and
  the measurement count;
- runs the one pre-registered candidate reward alongside the frozen
  one, on the same training and scoring streams, and applies the
  decision rule from ``EXPERIMENT_A1.md`` section 3;
- reports the first 1,000 scenarios of the test split as well, so
  the round can be read against Stage A on the same sample size.

Usage::

    python scripts/run_poc_a1.py                  # the scored round
    python scripts/run_poc_a1.py --seeds 1 --test-size 1000  # smoke

Stage A is untouched: this script never writes ``reports/poc.*``.
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
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import numpy

from run_poc import (  # noqa: E402  (reused, not modified)
    PRE_REGISTERED_BUDGET,
    PRIMARY_SIGMA,
    REPORTS_DIR,
    SIGMAS,
    build_payload,
    fmt,
    run_condition,
    score_summary,
    signed,
)
from smallai.agent import QAgent  # noqa: E402
from smallai.evaluate import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    PolicyScore,
    derive_seed,
    mcnemar,
    paired_bootstrap_delta,
    paired_bootstrap_mean_delta,
    score_agent,
)
from smallai.train import (  # noqa: E402
    MIX,
    WIDE_CONSISTENCY_REWARD,
    TrainingReport,
    make_splits,
    oracle_label,
    train_td,
)

#: EXPERIMENT_A1.md section 6: the five fresh master seeds.
MASTER_SEEDS: tuple[int, ...] = (20263030, 20263031, 20263032, 20263033, 20263034)

#: EXPERIMENT_A1.md section 1: the enlarged test split.
TEST_SIZE = 10_000

#: The Stage A split size, reported alongside for continuity.
SUBSET_SIZE = 1_000

#: The label of the pre-registered candidate reward (EXPERIMENT_A1.md
#: section 3).
CANDIDATE = "journey_q_wide_consistency"

#: Two-sided 95 % Student-t critical values by degrees of freedom.
#: The seed-level interval uses these; the normal value is used past
#: the table.
T_CRITICAL: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
    18: 2.101, 19: 2.093, 20: 2.086,
}


def t_critical(degrees: int) -> float:
    """Two-sided 95 % Student-t critical value.

    Parameters
    ----------
    degrees : int
        Degrees of freedom.

    Returns
    -------
    float
        The critical value.
    """
    return T_CRITICAL.get(degrees, 1.96)


def p_text(value: float) -> str:
    """A p-value for print, without pretending a zero is a number.

    Parameters
    ----------
    value : float
        The p-value.

    Returns
    -------
    str
        The p-value, or a floor note when it underflows.
    """
    if value <= 0.0:
        return "< 1e-300 (underflows to zero)"
    if value < 1e-4:
        return f"{value:.3g}"
    return f"{value:.4f}"


@dataclass
class SeedRun:
    """Everything one master seed produced.

    Attributes
    ----------
    seed : int
        The master seed.
    conditions : list
        The three sigma conditions, in sweep order.
    candidate_score : PolicyScore
        The candidate reward's test score at the primary sigma.
    candidate_training : TrainingReport
        The candidate reward's training report.
    """

    seed: int
    conditions: list
    candidate_score: PolicyScore
    candidate_training: TrainingReport


def run_candidate(
    sigma: float,
    splits: dict[str, list],
    labels: dict[str, list[str]],
    *,
    seed: int,
    budget: int,
) -> tuple[TrainingReport, PolicyScore, QAgent]:
    """Train and score the pre-registered candidate reward.

    The candidate uses the *same* agent name, the same training seed
    and the same scoring seed as the frozen Journey-Q, so the two
    policies differ in exactly one thing: the reward. The comparison
    between them is therefore paired at the level of the training
    stream, the behaviour noise and the measurement noise.

    Parameters
    ----------
    sigma : float
        The noise level; the primary sigma only.
    splits : dict
        The splits.
    labels : dict
        The oracle labels.
    seed : int
        The master seed.
    budget : int
        The episode cap.

    Returns
    -------
    tuple
        The training report, the test score and the agent.
    """
    agent = QAgent(name="journey_q")
    report = train_td(
        agent,
        splits["train"],
        seed=derive_seed(seed, "journey_q", str(sigma)),
        budget=budget,
        mix=MIX,
        reward=WIDE_CONSISTENCY_REWARD,
    )
    score = score_agent(
        agent,
        splits["test"],
        labels["test"],
        derive_seed(seed, "score", "journey_q"),
    )
    return report, score, agent


def seed_metrics(condition, candidate_score: PolicyScore) -> dict:
    """The primary-sigma numbers of one seed.

    Parameters
    ----------
    condition : ConditionResult
        The primary-sigma condition.
    candidate_score : PolicyScore
        The candidate reward's score.

    Returns
    -------
    dict
        The measured numbers.
    """
    scores = condition.scores
    return {
        "journey_accuracy": scores["journey_q"].accuracy,
        "shortcut_accuracy": scores["shortcut_cart"].accuracy,
        "accuracy_delta_pp": 100.0
        * (scores["journey_q"].accuracy - scores["shortcut_cart"].accuracy),
        "journey_invalid_rate": scores["journey_q"].invalid_rate,
        "shortcut_invalid_rate": scores["shortcut_cart"].invalid_rate,
        "journey_measurements": scores["journey_q"].mean_measurements,
        "shortcut_measurements": scores["shortcut_cart"].mean_measurements,
        "candidate_accuracy": candidate_score.accuracy,
        "candidate_invalid_rate": candidate_score.invalid_rate,
        "candidate_measurements": candidate_score.mean_measurements,
        "candidate_delta_pp": 100.0
        * (candidate_score.accuracy - scores["journey_q"].accuracy),
    }


def slice_score(score: PolicyScore, size: int) -> PolicyScore:
    """The first ``size`` scenarios of a score.

    The per-scenario noise of :func:`smallai.evaluate.score_agent`
    is derived from the scenario's index, so the first ``size``
    entries of a score are exactly what scoring those ``size``
    scenarios alone would produce.

    Parameters
    ----------
    score : PolicyScore
        The full score.
    size : int
        How many leading scenarios to keep.

    Returns
    -------
    PolicyScore
        The truncated score.
    """
    return PolicyScore(
        name=score.name,
        labels=list(score.labels[:size]),
        actions=list(score.actions[:size]),
        correct=list(score.correct[:size]),
        measurements=list(score.measurements[:size]),
        consistent=list(score.consistent[:size]),
        margins=None if score.margins is None else list(score.margins[:size]),
        terminal_q=(
            None if score.terminal_q is None else list(score.terminal_q[:size])
        ),
        latency_ms=(
            None if score.latency_ms is None else list(score.latency_ms[:size])
        ),
    )


def seed_level_interval(deltas: Sequence[float]) -> dict:
    """The seed-level paired interval over the master seeds.

    Parameters
    ----------
    deltas : sequence of float
        One paired delta per master seed.

    Returns
    -------
    dict
        The mean, the standard deviation, the interval and the
        t-statistic.
    """
    n = len(deltas)
    mean = sum(deltas) / n
    if n > 1:
        variance = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    else:
        variance = 0.0
    sd = variance**0.5
    se = sd / n**0.5 if n else 0.0
    half = t_critical(n - 1) * se
    return {
        "n_seeds": n,
        "mean_pp": mean,
        "sd_pp": sd,
        "se_pp": se,
        "ci_low_pp": mean - half,
        "ci_high_pp": mean + half,
        "t_statistic": mean / se if se > 0 else 0.0,
        "excludes_zero": (mean - half) > 0.0 or (mean + half) < 0.0,
    }


def pooled_pair(
    first: list[bool],
    second: list[bool],
    *,
    seed: int,
    resamples: int,
) -> dict:
    """The pooled paired comparison of two policies' correctness.

    Parameters
    ----------
    first : list of bool
        The first policy's correctness, per scenario.
    second : list of bool
        The second policy's correctness, same order.
    seed : int
        Seed for the bootstrap and the test.
    resamples : int
        Bootstrap resamples.

    Returns
    -------
    dict
        The delta, its interval and the McNemar test.
    """
    boot = paired_bootstrap_delta(
        first, second, seed=seed, resamples=resamples
    )
    test = mcnemar(first, second)
    return {
        "delta_pp": boot.delta_pp,
        "ci_low_pp": boot.ci_low_pp,
        "ci_high_pp": boot.ci_high_pp,
        "excludes_zero": boot.excludes_zero,
        "first_only": test.first_only,
        "second_only": test.second_only,
        "mcnemar_p": test.p_value,
        "n": len(first),
    }


def pooled_measurements(
    first: list[float],
    second: list[float],
    *,
    seed: int,
    resamples: int,
) -> dict:
    """The pooled paired comparison of two policies' measurement counts.

    Parameters
    ----------
    first : list of float
        The first policy's measurement counts.
    second : list of float
        The second policy's measurement counts, same order.
    seed : int
        Seed for the bootstrap.
    resamples : int
        Bootstrap resamples.

    Returns
    -------
    dict
        The difference and its interval.
    """
    boot = paired_bootstrap_mean_delta(
        first, second, seed=seed, resamples=resamples
    )
    return {
        "delta": boot.delta_pp,
        "ci_low": boot.ci_low_pp,
        "ci_high": boot.ci_high_pp,
        "excludes_zero": boot.excludes_zero,
    }


def render_markdown(
    runs: list[SeedRun],
    aggregate: dict,
    *,
    budget: int,
    resamples: int,
    test_size: int,
    elapsed: float,
) -> str:
    """Render the Stage A.1 report.

    Parameters
    ----------
    runs : list of SeedRun
        One entry per master seed.
    aggregate : dict
        The pooled and seed-level results.
    budget : int
        The episode cap used.
    resamples : int
        The bootstrap resample count used.
    test_size : int
        The test split size used.
    elapsed : float
        Wall-clock seconds.

    Returns
    -------
    str
        The report.
    """
    lines: list[str] = []
    add = lines.append

    add("# Stage A.1 report - AI-267")
    add("")
    add(
        "Pre-registered round 2 (`EXPERIMENT_A1.md`). Stage A is untouched:"
        " its report, JSON and seeds are exactly as committed, and the"
        " numbers below are a separate, independently seeded round."
    )
    add("")
    add("## Run parameters")
    add("")
    add("| Parameter | Value |")
    add("| --- | --- |")
    add(f"| Master seeds | {', '.join(str(run.seed) for run in runs)} |")
    add(f"| Test split size | {test_size:,} per seed |")
    add(f"| Train split size | 4,000 (skewed field mixture) |")
    add(f"| Calibration split size | 1,000 (balanced) |")
    add(f"| Episode budget per TD policy | {budget:,} |")
    add(f"| Bootstrap resamples | {resamples:,} |")
    add(f"| Noise sweep | {', '.join(str(s) for s in SIGMAS)} "
        f"(primary {PRIMARY_SIGMA}) |")
    add(f"| Wall clock | {elapsed:.1f} s |")
    add("")

    add("## Primary condition per master seed (sigma = 0.08)")
    add("")
    add(
        "| Seed | Journey-Q acc | Shortcut acc | Delta (pp) | Journey invalid"
        " | Shortcut invalid | Journey meas | Shortcut meas |"
    )
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for run in runs:
        metrics = seed_metrics(
            next(c for c in run.conditions if c.primary), run.candidate_score
        )
        add(
            f"| {run.seed} | {100 * metrics['journey_accuracy']:.3f} % |"
            f" {100 * metrics['shortcut_accuracy']:.3f} % |"
            f" {metrics['accuracy_delta_pp']:+.3f} |"
            f" {100 * metrics['journey_invalid_rate']:.3f} % |"
            f" {100 * metrics['shortcut_invalid_rate']:.3f} % |"
            f" {metrics['journey_measurements']:.3f} |"
            f" {metrics['shortcut_measurements']:.3f} |"
        )
    add("")

    h1 = aggregate["h1"]
    seed_level = h1["seed_level"]
    pooled = h1["pooled"]
    n_seeds = len(runs)
    add("## H1 - Journey-Q accuracy > Shortcut-CART accuracy")
    add("")
    add(
        "The pre-registered primary summary is the per-seed paired delta:"
        f" the {n_seeds} master seeds are independent replicates of the whole"
        " procedure, so that interval carries the training-draw noise as"
        " well as the evaluation noise."
    )
    add("")
    add(
        f"- Seed-level mean delta: **{signed(seed_level['mean_pp'])} pp**, "
        f"95 % t-interval over {seed_level['n_seeds']} seed"
        f"{'' if seed_level['n_seeds'] == 1 else 's'} "
        f"[{signed(seed_level['ci_low_pp'])}, "
        f"{signed(seed_level['ci_high_pp'])}] "
        f"(sd {seed_level['sd_pp']:.3f}, t = {seed_level['t_statistic']:.2f})"
    )
    add(
        f"- Pooled paired bootstrap over {pooled['n']:,} paired scenarios: "
        f"**{signed(pooled['delta_pp'])} pp**, 95 % CI "
        f"[{signed(pooled['ci_low_pp'])}, {signed(pooled['ci_high_pp'])}]"
    )
    add(
        f"- Pooled McNemar: journey-only {pooled['first_only']}, "
        f"shortcut-only {pooled['second_only']}, p = {pooled['mcnemar_p']:.4f}"
    )
    add("")
    add(
        f"**H1: {'SUPPORTED' if h1['supported'] else 'NOT SUPPORTED'}.** "
        f"Decision rule (EXPERIMENT_A1.md section 5): the seed-level"
        f" interval must exclude zero on the positive side *and* the"
        f" pooled McNemar must give p < 0.05."
    )
    add("")

    s1 = aggregate["s1_invalid"]
    add("## S1 (secondary) - evidence consistency")
    add("")
    add(
        "The invalid-verdict rate is the metric where the journey policy"
        " competes hardest. The paired indicator is *the shortcut is"
        " invalid and Journey-Q is not* versus the reverse."
    )
    add("")
    add(
        f"- Pooled invalid rate: Journey-Q "
        f"{100 * aggregate['pooled_rates']['journey_invalid']:.3f} % vs"
        f" Shortcut-CART {100 * aggregate['pooled_rates']['shortcut_invalid']:.3f} %"
    )
    add(
        f"- Paired difference (shortcut minus journey, positive is the"
        f" journey policy's advantage): **{signed(s1['delta_pp'])} pp**,"
        f" 95 % CI [{signed(s1['ci_low_pp'])}, {signed(s1['ci_high_pp'])}]"
    )
    add(
        f"- McNemar on the paired indicator: {s1['first_only']} scenarios where"
        f" the shortcut is invalid and Journey-Q is not, {s1['second_only']}"
        f" the other way round, p = {p_text(s1['mcnemar_p'])}"
    )
    add("")

    s2 = aggregate["s2_measurements"]
    add("## S2 (secondary) - measurement cost")
    add("")
    add(
        f"- Pooled mean measurements per decision: Journey-Q "
        f"{aggregate['pooled_rates']['journey_measurements']:.3f} vs"
        f" Shortcut-CART {aggregate['pooled_rates']['shortcut_measurements']:.3f}"
    )
    add(
        f"- Paired difference (journey minus shortcut): "
        f"**{s2['delta']:+.3f}** measurements, 95 % CI "
        f"[{s2['ci_low']:+.3f}, {s2['ci_high']:+.3f}]"
    )
    add("")

    candidate = aggregate["candidate"]
    add("## The pre-registered candidate reward")
    add("")
    add(
        "`WIDE_CONSISTENCY_REWARD`: the verdict term unchanged, the"
        " consistency term halved, and a suspect reading no longer"
        " refutes \"retake\". Trained on the same training stream and"
        " scored on the same noise stream as the frozen Journey-Q, so"
        " the only difference is the reward."
    )
    add("")
    add("| Seed | Frozen Journey-Q | Candidate | Delta (pp) |")
    add("| --- | --- | --- | --- |")
    for run in runs:
        metrics = seed_metrics(
            next(c for c in run.conditions if c.primary), run.candidate_score
        )
        add(
            f"| {run.seed} |"
            f" {100 * metrics['journey_accuracy']:.3f} % |"
            f" {100 * metrics['candidate_accuracy']:.3f} % |"
            f" {metrics['candidate_delta_pp']:+.3f} |"
        )
    add("")
    add(
        f"- Pooled paired bootstrap over {candidate['pooled']['n']:,} paired"
        f" scenarios: **{signed(candidate['pooled']['delta_pp'])} pp**, 95 % CI"
        f" [{signed(candidate['pooled']['ci_low_pp'])},"
        f" {signed(candidate['pooled']['ci_high_pp'])}]"
    )
    add(
        f"- Pooled McNemar: candidate-only {candidate['pooled']['first_only']},"
        f" frozen-only {candidate['pooled']['second_only']},"
        f" p = {candidate['pooled']['mcnemar_p']:.4f}"
    )
    add(
        f"- Seed-level mean delta: {signed(candidate['seed_level']['mean_pp'])}"
        f" pp, 95 % t-interval"
        f" [{signed(candidate['seed_level']['ci_low_pp'])},"
        f" {signed(candidate['seed_level']['ci_high_pp'])}]"
    )
    add(
        f"- Pooled consistency and cost: candidate invalid"
        f" {100 * candidate['rates']['candidate_invalid']:.3f} % and"
        f" {candidate['rates']['candidate_measurements']:.3f} measurements,"
        f" against the frozen reward's"
        f" {100 * candidate['rates']['journey_invalid']:.3f} % and"
        f" {candidate['rates']['journey_measurements']:.3f}"
    )
    add("")
    add(
        f"**Decision: the candidate is"
        f" {'ADOPTED' if candidate['adopted'] else 'NOT ADOPTED'}.**"
        f" Decision rule (EXPERIMENT_A1.md section 3): the pooled paired"
        f" interval must exclude zero on the positive side *and* the pooled"
        f" McNemar must give p < 0.05. The shipped default therefore"
        f" stays the frozen reward."
    )
    add("")

    subset = aggregate["subset"]
    add("## The Stage A split size, for continuity")
    add("")
    add(
        f"The first {SUBSET_SIZE:,} scenarios of each test split, scored"
        f" exactly as Stage A scored its 1,000. Same sample size, same"
        f" machinery, fresh data."
    )
    add("")
    add("| Seed | Journey-Q acc | Shortcut acc | Delta (pp) | McNemar p |")
    add("| --- | --- | --- | --- | --- |")
    for row in subset["per_seed"]:
        add(
            f"| {row['seed']} | {100 * row['journey_accuracy']:.3f} % |"
            f" {100 * row['shortcut_accuracy']:.3f} % |"
            f" {row['accuracy_delta_pp']:+.3f} | {row['mcnemar_p']:.4f} |"
        )
    add("")
    add(
        f"- Pooled over the {n_seeds} subsets: **{signed(subset['pooled']['delta_pp'])} pp**,"
        f" 95 % CI [{signed(subset['pooled']['ci_low_pp'])},"
        f" {signed(subset['pooled']['ci_high_pp'])}], McNemar"
        f" p = {subset['pooled']['mcnemar_p']:.4f}"
    )
    add(
        f"- Seed-level mean delta: {signed(subset['seed_level']['mean_pp'])} pp,"
        f" 95 % t-interval [{signed(subset['seed_level']['ci_low_pp'])},"
        f" {signed(subset['seed_level']['ci_high_pp'])}]"
    )
    add("")

    add("## The noise sweep (all seeds)")
    add("")
    add(
        "The sweep is the protocol's sensitivity analysis, not a"
        " pre-registered hypothesis, so it is reported with intervals"
        " but no verdict."
    )
    add("")
    add(
        "| Sigma | Journey-Q acc | Shortcut acc | Delta (pp) | 95 % CI"
        f" (pooled, {aggregate['sweep'][SIGMAS[0]]['pooled']['n']:,} scenarios)"
        " | Seed-level mean [95 % t-interval] |"
    )
    add("| --- | --- | --- | --- | --- | --- |")
    for sigma in SIGMAS:
        rows = []
        for run in runs:
            condition = next(c for c in run.conditions if c.sigma == sigma)
            scores = condition.scores
            rows.append(
                (
                    scores["journey_q"].accuracy,
                    scores["shortcut_cart"].accuracy,
                )
            )
        journey = sum(r[0] for r in rows) / len(rows)
        shortcut = sum(r[1] for r in rows) / len(rows)
        sweep_row = aggregate["sweep"][sigma]
        add(
            f"| {sigma} | {100 * journey:.3f} % | {100 * shortcut:.3f} % |"
            f" {100 * (journey - shortcut):+.3f} |"
            f" [{signed(sweep_row['pooled']['ci_low_pp'])},"
            f" {signed(sweep_row['pooled']['ci_high_pp'])}] |"
            f" {signed(sweep_row['seed_level']['mean_pp'])}"
            f" [{signed(sweep_row['seed_level']['ci_low_pp'])},"
            f" {signed(sweep_row['seed_level']['ci_high_pp'])}] |"
        )
    add("")
    add("The cost side by noise level:")
    add("")
    add(
        "| Sigma | Journey invalid | Shortcut invalid | Journey meas |"
        " Shortcut meas | Journey review | Shortcut review |"
    )
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for sigma in SIGMAS:
        rows = []
        for run in runs:
            condition = next(c for c in run.conditions if c.sigma == sigma)
            scores = condition.scores
            rows.append(scores)
        journey = sum(r["journey_q"].invalid_rate for r in rows) / len(rows)
        shortcut = sum(r["shortcut_cart"].invalid_rate for r in rows) / len(rows)
        journey_meas = sum(
            r["journey_q"].mean_measurements for r in rows
        ) / len(rows)
        shortcut_meas = sum(
            r["shortcut_cart"].mean_measurements for r in rows
        ) / len(rows)
        journey_review = sum(
            r["journey_q"].review_rate for r in rows
        ) / len(rows)
        shortcut_review = sum(
            r["shortcut_cart"].review_rate for r in rows
        ) / len(rows)
        add(
            f"| {sigma} | {100 * journey:.3f} % | {100 * shortcut:.3f} % |"
            f" {journey_meas:.3f} | {shortcut_meas:.3f} |"
            f" {100 * journey_review:.3f} % | {100 * shortcut_review:.3f} % |"
        )
    add("")

    add("## H2, H3 and H4 - the mechanism checks")
    add("")
    add(
        "H2-H4 are mechanism checks, not confirmatory hypotheses: the"
        " point estimate is reported with its interval, and no"
        " pass/fail claim is made on them."
    )
    add("")
    add("| Seed | H2 Journey-Q vs NoRecheck-Q | H3 Journey-Q vs ExpertOnly-Q | H4 measurements |")
    add("| --- | --- | --- | --- |")
    for entry in aggregate["policy_field"]:
        results = {h["name"]: h for h in entry["hypotheses"]}
        h2 = results["H2"]
        h3 = results["H3"]
        h4 = results["H4"]
        measured = entry["policies"]["journey_q"]["mean_measurements"]
        add(
            f"| {entry['seed']} |"
            f" {signed(h2['delta_pp'])} pp"
            f" [{signed(h2['ci_low_pp'])}, {signed(h2['ci_high_pp'])}]"
            f" | {signed(h3['delta_pp'])} pp"
            f" [{signed(h3['ci_low_pp'])}, {signed(h3['ci_high_pp'])}]"
            f" | {measured:.3f}"
            f"{' (holds)' if h4['supported'] else ' (fails)'} |"
        )
    add("")
    add(
        f"The policy field, pooled over the {n_seeds} seeds at the primary"
        " sigma:"
    )
    add("")
    add(
        "| Policy | Accuracy | Invalid | Mean measurements | Review rate |"
    )
    add("| --- | --- | --- | --- | --- |")
    for name in ("journey_q", "no_recheck_q", "expert_only_q", "shortcut_cart"):
        accuracy = sum(
            entry["policies"][name]["accuracy"]
            for entry in aggregate["policy_field"]
        ) / len(aggregate["policy_field"])
        invalid = sum(
            entry["policies"][name]["invalid_rate"]
            for entry in aggregate["policy_field"]
        ) / len(aggregate["policy_field"])
        meas = sum(
            entry["policies"][name]["mean_measurements"]
            for entry in aggregate["policy_field"]
        ) / len(aggregate["policy_field"])
        review = sum(
            entry["policies"][name]["review_rate"]
            for entry in aggregate["policy_field"]
        ) / len(aggregate["policy_field"])
        add(
            f"| {name} | {100 * accuracy:.3f} % | {100 * invalid:.3f} % |"
            f" {meas:.3f} | {100 * review:.3f} % |"
        )
    add("")

    add("## Model size, coverage and calibration (primary sigma)")
    add("")
    add(
        "| Seed | Journey-Q entries | Visited states | Coverage |"
        " ECE isotonic | ECE softmax |"
    )
    add("| --- | --- | --- | --- | --- | --- |")
    for run in runs:
        condition = next(c for c in run.conditions if c.primary)
        report = condition.training["journey_q"]
        add(
            f"| {run.seed} | {report.stored_entries:,} |"
            f" {report.visited_states:,} | {100 * condition.coverage:.2f} % |"
            f" {fmt(condition.calibration.ece_isotonic, 5)} |"
            f" {fmt(condition.calibration.ece_temperature, 5)} |"
        )
    add("")

    add("## Integrity")
    add("")
    add(
        "- Stage A's protocol, code path and artifacts are unchanged; its"
        " run is re-executed after the code changes and diffed against the"
        " committed `reports/poc.json` (all 23,947 measured leaves"
        " identical apart from wall-clock fields and the recorded numpy"
        " version)."
    )
    add(
        "- No hyperparameter of any policy is selected on the calibration"
        " or test splits of this round."
    )
    add(
        "- The development measurements that motivated this round were"
        " produced on development seeds under `scripts/dev/`, which never"
        " runs as part of the scored pipeline."
    )
    add("")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the round, write the report and artifacts.

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
            "Run the AI-267 Stage A.1 round and write"
            " reports/poc_a1.md, reports/poc_a1.json and the"
            " Q-table artifacts."
        )
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=len(MASTER_SEEDS),
        help="number of master seeds to run (pre-registered: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=MASTER_SEEDS[0],
        help="first master seed; the rest follow it",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=TEST_SIZE,
        help="test split size per seed (pre-registered: 10,000)",
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
        "--out",
        type=Path,
        default=REPORTS_DIR,
        help="directory to write the report into",
    )
    args = parser.parse_args(argv)

    seeds = [args.seed + offset for offset in range(args.seeds)]
    print(
        f"AI-267 Stage A.1 | master seeds {seeds[0]}..{seeds[-1]}"
        f" | test split {args.test_size:,} | budget {args.budget:,}"
        f" | resamples {args.resamples:,}",
        flush=True,
    )

    started = time.perf_counter()
    runs: list[SeedRun] = []
    for master in seeds:
        print(f"=== master seed {master} ===", flush=True)
        splits = make_splits(master, test_n=args.test_size)
        labels = {
            name: [oracle_label(s) for s in scenarios]
            for name, scenarios in splits.items()
        }
        print(
            "splits: "
            + ", ".join(f"{n}={len(v):,}" for n, v in splits.items()),
            flush=True,
        )
        conditions = []
        for sigma in SIGMAS:
            print(f"noise sigma = {sigma}", flush=True)
            with _noise_sigma(sigma):
                conditions.append(
                    run_condition(
                        sigma,
                        splits,
                        labels,
                        seed=master,
                        budget=args.budget,
                        resamples=args.resamples,
                    )
                )
        print("  candidate reward (primary sigma)", flush=True)
        with _noise_sigma(PRIMARY_SIGMA):
            report, score, agent = run_candidate(
                PRIMARY_SIGMA,
                splits,
                labels,
                seed=master,
                budget=args.budget,
            )
        print(
            f"  trained {CANDIDATE:<30} {report.episodes:>9,} episodes"
            f"  entries={report.stored_entries:>6,}",
            flush=True,
        )
        runs.append(
            SeedRun(
                seed=master,
                conditions=conditions,
                candidate_score=score,
                candidate_training=report,
            )
        )
    elapsed = time.perf_counter() - started

    aggregate = _aggregate(runs, resamples=args.resamples)
    payload = _payload(
        runs,
        aggregate,
        seed=seeds[0],
        seeds=seeds,
        budget=args.budget,
        resamples=args.resamples,
        test_size=args.test_size,
        elapsed=elapsed,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "poc_a1.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / "poc_a1.md").write_text(
        render_markdown(
            runs,
            aggregate,
            budget=args.budget,
            resamples=args.resamples,
            test_size=args.test_size,
            elapsed=elapsed,
        ),
        encoding="utf-8",
    )

    # Inspectable tables, not the shipped artifact: the shipped model
    # stays Stage A's, and picking a seed on test accuracy would be
    # selection on the test split.
    primary = next(c for c in runs[0].conditions if c.primary)
    primary.agents["journey_q"].save(args.out / "journey_q_a1.json")

    print(f"wrote {args.out / 'poc_a1.md'}", flush=True)
    print(f"wrote {args.out / 'poc_a1.json'}")
    print(f"total {elapsed:.1f}s")
    return 0


def _noise_sigma(sigma: float):
    """The noise context manager, imported lazily to keep the import list short.

    Parameters
    ----------
    sigma : float
        The measurement-noise standard deviation.

    Returns
    -------
    context manager
        The context manager that installs the noise level.
    """
    from smallai.evaluate import noise_sigma

    return noise_sigma(sigma)


def _aggregate(runs: list[SeedRun], *, resamples: int) -> dict:
    """Pool the per-seed results into the round's statistics.

    Parameters
    ----------
    runs : list of SeedRun
        One entry per master seed.
    resamples : int
        Bootstrap resamples per pooled comparison.

    Returns
    -------
    dict
        The pooled and seed-level results.
    """
    journey_correct: list[bool] = []
    shortcut_correct: list[bool] = []
    journey_invalid: list[bool] = []
    shortcut_invalid: list[bool] = []
    journey_meas: list[float] = []
    shortcut_meas: list[float] = []
    candidate_correct: list[bool] = []
    candidate_invalid: list[bool] = []
    candidate_meas: list[float] = []
    seed_deltas: list[float] = []
    candidate_deltas: list[float] = []
    subset_rows: list[dict] = []
    subset_journey: list[bool] = []
    subset_shortcut: list[bool] = []
    # Per noise level, the same paired data, so the sensitivity sweep
    # can carry intervals instead of bare point estimates.
    sweep_correct: dict[float, dict[str, list[bool]]] = {
        sigma: {"journey": [], "shortcut": []} for sigma in SIGMAS
    }
    sweep_deltas: dict[float, list[float]] = {sigma: [] for sigma in SIGMAS}

    for run in runs:
        condition = next(c for c in run.conditions if c.primary)
        scores = condition.scores
        journey = scores["journey_q"]
        shortcut = scores["shortcut_cart"]
        candidate = run.candidate_score

        journey_correct.extend(journey.correct)
        shortcut_correct.extend(shortcut.correct)
        candidate_correct.extend(candidate.correct)
        candidate_invalid.extend(not ok for ok in candidate.consistent)
        candidate_meas.extend(float(m) for m in candidate.measurements)
        journey_invalid.extend(not ok for ok in journey.consistent)
        shortcut_invalid.extend(not ok for ok in shortcut.consistent)
        journey_meas.extend(float(m) for m in journey.measurements)
        shortcut_meas.extend(float(m) for m in shortcut.measurements)
        seed_deltas.append(
            100.0 * (journey.accuracy - shortcut.accuracy)
        )
        candidate_deltas.append(
            100.0 * (candidate.accuracy - journey.accuracy)
        )

        for condition in run.conditions:
            scores = condition.scores
            sweep_correct[condition.sigma]["journey"].extend(
                scores["journey_q"].correct
            )
            sweep_correct[condition.sigma]["shortcut"].extend(
                scores["shortcut_cart"].correct
            )
            sweep_deltas[condition.sigma].append(
                100.0
                * (
                    scores["journey_q"].accuracy
                    - scores["shortcut_cart"].accuracy
                )
            )

        small_journey = slice_score(journey, SUBSET_SIZE)
        small_shortcut = slice_score(shortcut, SUBSET_SIZE)
        subset_journey.extend(small_journey.correct)
        subset_shortcut.extend(small_shortcut.correct)
        subset_rows.append(
            {
                "seed": run.seed,
                "journey_accuracy": small_journey.accuracy,
                "shortcut_accuracy": small_shortcut.accuracy,
                "accuracy_delta_pp": 100.0
                * (small_journey.accuracy - small_shortcut.accuracy),
                "mcnemar_p": mcnemar(
                    small_journey.correct, small_shortcut.correct
                ).p_value,
            }
        )

    pooled = pooled_pair(
        journey_correct,
        shortcut_correct,
        seed=derive_seed(runs[0].seed, "bootstrap", "h1_pooled"),
        resamples=resamples,
    )
    seed_level = seed_level_interval(seed_deltas)
    h1_supported = bool(
        seed_level["excludes_zero"] and seed_level["mean_pp"] > 0
    ) and pooled["mcnemar_p"] < 0.05

    s1 = pooled_pair(
        shortcut_invalid,
        journey_invalid,
        seed=derive_seed(runs[0].seed, "bootstrap", "s1_pooled"),
        resamples=resamples,
    )
    s2 = pooled_measurements(
        journey_meas,
        shortcut_meas,
        seed=derive_seed(runs[0].seed, "bootstrap", "s2_pooled"),
        resamples=resamples,
    )

    candidate_pooled = pooled_pair(
        candidate_correct,
        journey_correct,
        seed=derive_seed(runs[0].seed, "bootstrap", "candidate_pooled"),
        resamples=resamples,
    )
    candidate_seed_level = seed_level_interval(candidate_deltas)
    candidate_adopted = bool(
        candidate_pooled["excludes_zero"]
        and candidate_pooled["delta_pp"] > 0
        and candidate_pooled["mcnemar_p"] < 0.05
    )

    sweep = {}
    for sigma in SIGMAS:
        sweep[sigma] = {
            "pooled": pooled_pair(
                sweep_correct[sigma]["journey"],
                sweep_correct[sigma]["shortcut"],
                seed=derive_seed(
                    runs[0].seed, "bootstrap", f"sweep_{sigma}"
                ),
                resamples=resamples,
            ),
            "seed_level": seed_level_interval(sweep_deltas[sigma]),
        }

    subset_pooled = pooled_pair(
        subset_journey,
        subset_shortcut,
        seed=derive_seed(runs[0].seed, "bootstrap", "subset_pooled"),
        resamples=resamples,
    )
    subset_seed_level = seed_level_interval(
        [row["accuracy_delta_pp"] for row in subset_rows]
    )

    return {
        "h1": {
            "seed_level": seed_level,
            "pooled": pooled,
            "supported": h1_supported,
        },
        "s1_invalid": s1,
        "s2_measurements": s2,
        "pooled_rates": {
            "journey_invalid": sum(journey_invalid) / len(journey_invalid),
            "shortcut_invalid": sum(shortcut_invalid) / len(shortcut_invalid),
            "journey_measurements": sum(journey_meas) / len(journey_meas),
            "shortcut_measurements": sum(shortcut_meas) / len(shortcut_meas),
        },
        "candidate": {
            "pooled": candidate_pooled,
            "seed_level": candidate_seed_level,
            "adopted": candidate_adopted,
            "rates": {
                "candidate_invalid": sum(candidate_invalid)
                / len(candidate_invalid),
                "candidate_measurements": sum(candidate_meas)
                / len(candidate_meas),
                "journey_invalid": sum(journey_invalid) / len(journey_invalid),
                "journey_measurements": sum(journey_meas) / len(journey_meas),
            },
        },
        "sweep": sweep,
        "policy_field": [
            {
                "seed": run.seed,
                "policies": {
                    name: score_summary(score)
                    for name, score in next(
                        c for c in run.conditions if c.primary
                    ).scores.items()
                },
                "hypotheses": [
                    asdict(result)
                    for result in next(
                        c for c in run.conditions if c.primary
                    ).hypotheses
                ],
            }
            for run in runs
        ],
        "subset": {
            "per_seed": subset_rows,
            "pooled": subset_pooled,
            "seed_level": subset_seed_level,
        },
    }


def _payload(
    runs: list[SeedRun],
    aggregate: dict,
    *,
    seed: int,
    seeds: list[int],
    budget: int,
    resamples: int,
    test_size: int,
    elapsed: float,
) -> dict:
    """The whole round as one JSON-serializable dict.

    Parameters
    ----------
    runs : list of SeedRun
        One entry per master seed.
    aggregate : dict
        The pooled and seed-level results.
    seed : int
        The first master seed.
    seeds : list of int
        Every master seed.
    budget : int
        The episode cap used.
    resamples : int
        The bootstrap resample count used.
    test_size : int
        The test split size used.
    elapsed : float
        Wall-clock seconds.

    Returns
    -------
    dict
        Environment, protocol inputs and every measured number.
    """
    return {
        "framing": "CLEF-inspired lightweight decision-policy POC",
        "stage": "A.1",
        "protocol": "EXPERIMENT_A1.md",
        "master_seeds": seeds,
        "test_size": test_size,
        "subset_size": SUBSET_SIZE,
        "budget": budget,
        "resamples": resamples,
        "elapsed_seconds": elapsed,
        "pre_registered_budget": PRE_REGISTERED_BUDGET,
        "pre_registered_resamples": BOOTSTRAP_RESAMPLES,
        "environment": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "seeds": [
            {
                "master_seed": run.seed,
                "conditions": build_payload(
                    run.conditions,
                    seed=run.seed,
                    budget=budget,
                    resamples=resamples,
                    elapsed_seconds=0.0,
                )["conditions"],
                "candidate": {
                    "training": run.candidate_training.to_dict(),
                    "scores": score_summary(run.candidate_score),
                },
            }
            for run in runs
        ],
        "aggregate": aggregate,
    }


if __name__ == "__main__":
    raise SystemExit(main())
