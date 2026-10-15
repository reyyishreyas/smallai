"""Gates for the environment: thresholds, weights, sampling."""

from __future__ import annotations

import random

from smallai.environment import (
    FAIL,
    PASS,
    SIGNALS,
    TEST_WEIGHTS,
    TRAIN_WEIGHTS,
    Scenario,
    observe,
    sample_scenario,
)


def test_weight_dicts_sum_to_one() -> None:
    """Both mixtures are proper distributions (audit fix)."""
    assert abs(sum(TRAIN_WEIGHTS.values()) - 1.0) < 1e-9
    assert abs(sum(TEST_WEIGHTS.values()) - 1.0) < 1e-9


def test_weight_dicts_cover_every_cause() -> None:
    """Every cause has a weight in both splits."""
    from smallai.environment import CAUSES

    assert set(TRAIN_WEIGHTS) == set(CAUSES)
    assert set(TEST_WEIGHTS) == set(CAUSES)


def test_train_is_blur_heavy_test_is_balanced() -> None:
    """The skew that makes the shortcut measurable exists."""
    # Blur is the dominant single defect in train (field reality)...
    assert TRAIN_WEIGHTS["blur"] > TRAIN_WEIGHTS["glare"]
    assert TRAIN_WEIGHTS["blur"] == max(
        TRAIN_WEIGHTS[c] for c in TRAIN_WEIGHTS if c != "clean"
    )
    # ...and every named defect carries equal weight in test.
    defects = [
        TEST_WEIGHTS[c] for c in TEST_WEIGHTS if c not in ("clean", "multi")
    ]
    assert len(set(defects)) == 1


def test_level_zones() -> None:
    """Below FAIL is fail, between is suspect, >= PASS is pass."""
    signals = {name: 0.5 for name in SIGNALS}
    scenario = Scenario(signals=signals, cause="custom")
    assert scenario.level("blur", FAIL["blur"] - 0.01) == "fail"
    assert scenario.level("blur", FAIL["blur"]) == "suspect"
    assert scenario.level("blur", PASS["blur"] - 0.01) == "suspect"
    assert scenario.level("blur", PASS["blur"]) == "pass"


def test_observe_clips_to_unit_interval() -> None:
    """Noise never escapes [0, 1]."""
    rng = random.Random(7)
    scenario = Scenario(
        signals={name: 0.02 for name in SIGNALS}, cause="custom"
    )
    draws = [observe(scenario, "blur", rng) for _ in range(200)]
    assert min(draws) >= 0.0
    assert max(draws) <= 1.0


def test_sampling_is_deterministic_under_seed() -> None:
    """Same seed, same scenario (protocol section 4)."""
    a = sample_scenario(random.Random(42), TRAIN_WEIGHTS)
    b = sample_scenario(random.Random(42), TRAIN_WEIGHTS)
    assert a.signals == b.signals
    assert a.cause == b.cause


def test_single_cause_weights_pin_the_cause() -> None:
    """A point-mass mixture always samples its cause, in fail zone."""
    scenario = sample_scenario(random.Random(3), {"blur": 1.0, "clean": 0.0})
    assert scenario.cause == "blur"
    assert scenario.level("blur") == "fail"
