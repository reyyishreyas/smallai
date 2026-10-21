"""Gates for evaluation: consistency, calibration, uncertainty.

The two things worth pinning hardest here: that ``walk`` cannot
diverge from the ``decide`` the package actually ships, and that the
statistical primitives return the textbook answer on inputs whose
answer can be worked out by hand.
"""

from __future__ import annotations

import random

import pytest

from smallai import environment
from smallai.agent import TERMINALS, QAgent
from smallai.environment import SIGNALS, Scenario, observe
from smallai.evaluate import (
    BootstrapDelta,
    McNemarResult,
    accuracy,
    brier,
    calibrate,
    calibration_agreement,
    derive_seed,
    ece,
    fit_isotonic,
    fit_temperature,
    hypotheses,
    is_consistent,
    levels_of,
    mcnemar,
    noise_sigma,
    paired_bootstrap_delta,
    score_agent,
    score_tree,
    softmax_confidence,
    visited_state_coverage,
    walk,
)
from smallai.train import (
    ShortcutTree,
    cross_validated_tree,
    make_splits,
    make_tree_dataset,
    oracle_label,
    train_td,
)

ALL_PASS = ["pass"] * len(SIGNALS)


def scenario(**overrides: float) -> Scenario:
    """Every signal at 0.95 except the named overrides."""
    signals = {name: 0.95 for name in SIGNALS}
    signals.update(overrides)
    return Scenario(signals=signals, cause="custom")


def trained_agent(seed: int = 12, budget: int = 800) -> QAgent:
    """A briefly trained agent, so margins are not all zero."""
    train = make_splits(seed=11)["train"][:60]
    agent = QAgent(name="journey_q")
    train_td(
        agent,
        train,
        seed=seed,
        budget=budget,
        min_episodes=budget,
        check_interval=budget,
    )
    return agent


# --- walk must not diverge from decide ---


def test_walk_agrees_with_decide_under_the_same_seed() -> None:
    """The scoring replay must describe the shipped policy.

    ``walk`` repeats the loop in ``QAgent.decide`` so it can keep the
    terminal Q-vector (needed to softmax) and the final state
    (needed to test evidence consistency), both of which ``decide``
    discards. If the two ever disagreed, every reported metric would
    describe a policy nobody ships, so action and margin are pinned
    together.
    """
    agent = trained_agent()
    test = make_splits(seed=13)["test"][:25]
    for index, capture in enumerate(test):
        step = walk(agent, capture, random.Random(1000 + index))
        decision = agent.decide(capture, random.Random(1000 + index))
        assert step.action == decision.action
        assert step.margin == pytest.approx(decision.confidence, abs=1e-4)


def test_walk_counts_every_measurement() -> None:
    """The measurement count is the checks plus the re-checks."""
    agent = trained_agent()
    step = walk(agent, scenario(), random.Random(0))
    assert step.measurements == len(step.state) - step.state.count(0)
    assert step.measurements <= 12


def test_walk_keeps_all_three_terminal_values() -> None:
    """The Q-vector is ordered as TERMINALS, so it can be softmaxed."""
    agent = trained_agent()
    step = walk(agent, scenario(blur=0.10), random.Random(1))
    assert len(step.terminal_q) == len(TERMINALS)
    assert step.action in TERMINALS


# --- evidence consistency (EXPERIMENT.md section 3) ---


def test_pass_needs_all_six_observed_pass_level() -> None:
    """A capture cannot be certified on partial evidence."""
    assert is_consistent("pass", ALL_PASS) is True
    assert is_consistent("pass", ALL_PASS[:-1]) is False
    assert is_consistent("pass", ALL_PASS[:-1] + ["suspect"]) is False
    assert is_consistent("pass", ["fail"] + ALL_PASS[1:]) is False


def test_retake_needs_one_observed_fail() -> None:
    assert is_consistent("retake", ["fail"]) is True
    assert is_consistent("retake", ["pass", "suspect"]) is False


def test_review_needs_one_observed_suspect() -> None:
    assert is_consistent("review", ["suspect"]) is True
    assert is_consistent("review", ["pass", "fail"]) is False


def test_zero_evidence_is_contradictory_for_every_action() -> None:
    """The section 3 corner case, for all three terminals."""
    for action in TERMINALS:
        assert is_consistent(action, []) is False


def test_unknown_action_raises() -> None:
    with pytest.raises(ValueError):
        is_consistent("maybe", ALL_PASS)


def test_levels_ignore_unchecked_signals() -> None:
    """An unchecked signal is not evidence and is not counted."""
    assert levels_of((0, 0, 0, 0, 0, 0)) == []
    assert levels_of((1, 2, 3, 4, 0, 0)) == [
        "fail",
        "suspect",
        "suspect",
        "pass",
    ]


# --- scalar metrics, on hand-checkable inputs ---


def test_accuracy() -> None:
    assert accuracy([True, True, False, False]) == pytest.approx(0.5)
    assert accuracy([True, True]) == pytest.approx(1.0)
    assert accuracy([False, False]) == pytest.approx(0.0)


def test_perfect_confidence_has_zero_ece_and_brier() -> None:
    confidences = [1.0, 1.0, 0.0, 0.0]
    correct = [True, True, False, False]
    assert ece(confidences, correct) == pytest.approx(0.0)
    assert brier(confidences, correct) == pytest.approx(0.0)


def test_uniform_confidence_scores_the_base_rate() -> None:
    """Half right at stated 0.5 is perfectly calibrated."""
    confidences = [0.5] * 10
    correct = [True, False] * 5
    assert ece(confidences, correct) == pytest.approx(0.0)
    assert brier(confidences, correct) == pytest.approx(0.25)


def test_ece_penalises_a_confidently_wrong_policy() -> None:
    confidences = [1.0] * 10
    correct = [False] * 10
    assert ece(confidences, correct) == pytest.approx(1.0)
    assert brier(confidences, correct) == pytest.approx(1.0)


# --- McNemar, exact ---


def test_mcnemar_on_a_worked_example() -> None:
    """Ten gains, no losses: chi-square 10, p = 2/2^10."""
    first = [True] * 10 + [False] * 10
    second = [False] * 10 + [False] * 10
    result = mcnemar(first, second)
    assert result.first_only == 10
    assert result.second_only == 0
    assert result.chi_square == pytest.approx(10.0)
    assert result.p_value == pytest.approx(2.0 / 1024.0)


def test_mcnemar_with_no_discordance_is_uninformative() -> None:
    verdicts = [True, False, True, True]
    result = mcnemar(verdicts, verdicts)
    assert result == McNemarResult(0, 0, 0.0, 1.0)


def test_mcnemar_balanced_discordance_has_no_direction() -> None:
    """Two and two: nothing separates the policies."""
    first = [True, True, False, False]
    second = [False, False, True, True]
    result = mcnemar(first, second)
    assert result.chi_square == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)


def test_mcnemar_rejects_unpaired_input() -> None:
    with pytest.raises(ValueError):
        mcnemar([True], [True, False])


# --- paired bootstrap ---


def test_identical_policies_give_a_zero_delta() -> None:
    verdicts = [True, False, True, True, False]
    delta = paired_bootstrap_delta(verdicts, verdicts, seed=1, resamples=200)
    assert delta.delta_pp == pytest.approx(0.0)
    assert delta.ci_low_pp == pytest.approx(0.0)
    assert delta.ci_high_pp == pytest.approx(0.0)
    assert delta.excludes_zero is False


def test_separated_policies_give_a_ci_away_from_zero() -> None:
    first = [True] * 100
    second = [False] * 100
    delta = paired_bootstrap_delta(first, second, seed=2, resamples=200)
    assert delta.delta_pp == pytest.approx(100.0)
    assert delta.ci_low_pp == pytest.approx(100.0)
    assert delta.excludes_zero is True


def test_bootstrap_is_reproducible_under_its_seed() -> None:
    first = [True, False, True, False, True]
    second = [False, False, True, True, False]
    a = paired_bootstrap_delta(first, second, seed=3, resamples=200)
    b = paired_bootstrap_delta(first, second, seed=3, resamples=200)
    assert a == b
    c = paired_bootstrap_delta(first, second, seed=4, resamples=200)
    assert a.delta_pp == c.delta_pp  # point estimate is seed-free


def test_bootstrap_rejects_unpaired_input() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_delta([True], [True, False], seed=0)


def test_bootstrap_delta_carries_its_resample_count() -> None:
    delta = paired_bootstrap_delta(
        [True, False], [True, False], seed=0, resamples=50
    )
    assert delta == BootstrapDelta(0.0, 0.0, 0.0, 50)


# --- isotonic regression ---


def test_isotonic_recovers_a_clean_step() -> None:
    """Low margins wrong, high margins right: a clean two-block fit."""
    model = fit_isotonic([0.1, 0.2, 0.8, 0.9], [False, False, True, True])
    assert model.predict(0.1) == pytest.approx(0.0)
    assert model.predict(0.9) == pytest.approx(1.0)


def test_isotonic_is_monotone_even_when_the_data_is_not() -> None:
    """A margin carrying no information flattens to the base rate."""
    model = fit_isotonic([0.1, 0.2, 0.3, 0.4], [True, False, True, False])
    assert list(model.probabilities) == sorted(model.probabilities)
    assert model.predict(0.25) == pytest.approx(0.5)


def test_isotonic_clamps_outside_the_fitted_range() -> None:
    """A margin wider than any seen at calibration stays flat."""
    model = fit_isotonic([0.1, 0.2, 0.3], [False, False, True])
    assert model.predict(-5.0) == pytest.approx(0.0)
    assert model.predict(99.0) == pytest.approx(1.0)


def test_isotonic_rejects_empty_or_unpaired_input() -> None:
    with pytest.raises(ValueError):
        fit_isotonic([], [])
    with pytest.raises(ValueError):
        fit_isotonic([0.1], [True, False])


# --- temperature scaling ---


def test_softmax_is_uniform_when_the_terminals_tie() -> None:
    for index in range(len(TERMINALS)):
        assert softmax_confidence((0.0, 0.0, 0.0), index, 1.0) == (
            pytest.approx(1.0 / 3.0)
        )


def test_low_temperature_concentrates_on_the_argmax() -> None:
    values = (2.0, 0.0, 0.0)
    assert softmax_confidence(values, 0, 0.001) > 0.999


def test_softmax_rejects_a_non_positive_temperature() -> None:
    with pytest.raises(ValueError):
        softmax_confidence((1.0, 0.0, 0.0), 0, 0.0)


def test_fitted_temperature_is_positive_and_finite() -> None:
    terminal_q = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)] * 20
    correct = [True, True] * 20
    chosen = [0, 1] * 20
    tau = fit_temperature(terminal_q, correct, chosen)
    assert 0.0 < tau < 1e3


def test_temperature_fits_the_dominant_action() -> None:
    """Every correct decision picked the largest Q: tau shrinks."""
    terminal_q = [(3.0, 0.0, 0.0)] * 40
    correct = [True] * 40
    chosen = [0] * 40
    tau = fit_temperature(terminal_q, correct, chosen)
    assert softmax_confidence((3.0, 0.0, 0.0), 0, tau) > 0.99


# --- the section 9 agreement rule ---


def test_agreement_flags_provisional_when_the_maps_disagree() -> None:
    """Margins that invert accuracy force the maps apart."""
    margins = [0.1, 0.2, 0.3, 0.4] * 10
    correct = [bool(i % 2) for i in range(len(margins))]
    terminal_q = [(1.0 if ok else 0.0, 0.0, 0.0) for ok in correct]
    actions = ["review"] * len(margins)
    isotonic, temperature = calibrate(margins, terminal_q, correct, actions)
    report = calibration_agreement(
        isotonic, temperature, margins, terminal_q, correct, actions
    )
    assert report.temperature == temperature
    assert 0.0 <= report.ece_isotonic <= 1.0
    assert 0.0 <= report.ece_temperature <= 1.0


def test_agreement_reports_the_probability_gap() -> None:
    margins = [0.5] * 20
    correct = [True] * 20
    terminal_q = [(1.0, 0.0, 0.0)] * 20
    actions = ["review"] * 20
    isotonic, temperature = calibrate(margins, terminal_q, correct, actions)
    report = calibration_agreement(
        isotonic, temperature, margins, terminal_q, correct, actions
    )
    assert report.mean_probability_difference >= 0.0
    assert report.isotonic is isotonic


# --- scoring ---


def test_cold_start_agent_measures_everything_then_defers() -> None:
    """Ties resolve toward evidence, then toward review."""
    score = score_agent(QAgent(), [scenario()], ["pass"], seed=5)
    assert score.measurements == [len(SIGNALS)]
    assert score.actions == ["review"]
    assert score.margins == [0.0]


def test_score_agent_reports_one_row_per_scenario() -> None:
    captures = [scenario(), scenario(blur=0.10), scenario(brightness=0.5)]
    labels = [oracle_label(c) for c in captures]
    score = score_agent(trained_agent(), captures, labels, seed=6)
    assert len(score.actions) == len(captures)
    assert len(score.correct) == len(captures)
    assert len(score.measurements) == len(captures)
    assert len(score.consistent) == len(captures)
    assert len(score.latency_ms) == len(captures)
    assert all(ms >= 0.0 for ms in score.latency_ms)


def test_score_agent_requires_matching_lengths() -> None:
    with pytest.raises(ValueError):
        score_agent(QAgent(), [scenario()], [], seed=0)


def test_tree_score_always_reads_all_six_signals() -> None:
    """The one-shot baseline has no early exit by construction."""
    train = make_splits(seed=7)["train"][:80]
    labels = [oracle_label(s) for s in train]
    x, y = make_tree_dataset(train, labels, random.Random(0))
    tree = ShortcutTree().fit(x, y)
    captures = [scenario(), scenario(blur=0.10)]
    test_labels = [oracle_label(c) for c in captures]
    test_x, _ = make_tree_dataset(captures, test_labels, random.Random(1))
    score = score_tree(tree, test_x, captures, test_labels)
    assert score.measurements == [len(SIGNALS)] * len(captures)
    assert score.margins is None
    assert score.terminal_q is None


def test_tree_cv_runs_end_to_end() -> None:
    """The graded baseline's whole path: CV on train, score on test."""
    splits = make_splits(seed=8)
    train = splits["train"][:120]
    labels = [oracle_label(s) for s in train]
    x, y = make_tree_dataset(train, labels, random.Random(0))
    tree, chosen, scores = cross_validated_tree(x, y, seed=1, folds=3)
    assert chosen in (1, 5, 20)
    assert len(scores) == 3
    test = splits["test"][:40]
    test_labels = [oracle_label(s) for s in test]
    test_x, _ = make_tree_dataset(test, test_labels, random.Random(1))
    score = score_tree(tree, test_x, test, test_labels)
    assert 0.0 <= score.accuracy <= 1.0


# --- hypotheses (EXPERIMENT.md section 10) ---


def test_hypotheses_cover_the_four_pre_registered_claims() -> None:
    captures = [scenario(), scenario(blur=0.10)] * 5
    labels = [oracle_label(c) for c in captures]
    agent = trained_agent()
    journey = score_agent(agent, captures, labels, seed=9)
    shortcut = score_agent(agent, captures, labels, seed=9)
    results = hypotheses(
        journey,
        shortcut,
        journey,
        journey,
        seed=9,
        resamples=100,
    )
    assert [r.name for r in results] == ["H1", "H2", "H3", "H4"]
    # Identical verdicts: every accuracy delta is exactly zero.
    for result in results[:3]:
        assert result.delta_pp == pytest.approx(0.0)
        assert result.ci_low_pp == pytest.approx(0.0)
    assert results[0].supported is False  # CI includes zero
    assert results[3].delta_pp is None


def test_h4_reads_the_measured_measurement_mean() -> None:
    captures = [scenario()] * 6
    labels = [oracle_label(c) for c in captures]
    journey = score_agent(trained_agent(), captures, labels, seed=10)
    results = hypotheses(
        journey, journey, journey, journey, seed=10, resamples=50
    )
    h4 = results[3]
    assert h4.supported is (journey.mean_measurements < 6.0)
    assert f"{journey.mean_measurements:.3f}" in h4.note


# --- plumbing ---


def test_noise_sigma_is_restored_after_the_block() -> None:
    """The sensitivity sweep must not leak its noise level."""
    original = environment.NOISE_SIGMA
    with noise_sigma(0.12):
        assert environment.NOISE_SIGMA == 0.12
    assert environment.NOISE_SIGMA == original


def test_noise_sigma_restores_even_when_the_block_raises() -> None:
    original = environment.NOISE_SIGMA
    with pytest.raises(RuntimeError), noise_sigma(0.04):
        raise RuntimeError("boom")
    assert environment.NOISE_SIGMA == original


def test_noise_sigma_actually_changes_the_reading() -> None:
    """Same true value and same seed, different spread."""
    capture = scenario(brightness=0.50)
    with noise_sigma(0.04):
        quiet = observe(capture, "brightness", random.Random(0))
    with noise_sigma(0.12):
        loud = observe(capture, "brightness", random.Random(0))
    assert quiet != loud


def test_derive_seed_is_stable_and_separable() -> None:
    assert derive_seed(1, "a") == derive_seed(1, "a")
    assert derive_seed(1, "a") != derive_seed(1, "b")
    assert derive_seed(1, "a") != derive_seed(2, "a")
    assert 0 <= derive_seed(1, "a") < 2**64


def test_coverage_uses_the_state_space_ceiling() -> None:
    assert visited_state_coverage(15625) == pytest.approx(1.0)
    assert visited_state_coverage(0) == pytest.approx(0.0)
    assert visited_state_coverage(1) == pytest.approx(1 / 15625)
