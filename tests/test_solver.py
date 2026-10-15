"""Gates for the expert, including the suspect-only re-check rule.

The re-check tests use a scripted rng (fixed gauss draws) so the
exact number of measurements is deterministic: that count *is* the
behaviour under test.
"""

from __future__ import annotations

from smallai.environment import SIGNALS, Scenario
from smallai.solver import ORDER, ExpertPolicy


class ScriptedRng:
    """A stand-in for random.Random returning fixed gauss draws.

    Parameters
    ----------
    values : list of float
        Successive values to return from gauss().
    """

    def __init__(self, values: list[float]) -> None:
        self.values = list(values)
        self.calls = 0

    def gauss(self, mu: float, sigma: float) -> float:
        """Return the next scripted draw."""
        value = self.values[self.calls]
        self.calls += 1
        return value


def uniform(value: float) -> Scenario:
    """A scenario with every signal at the same true quality."""
    return Scenario(signals={name: value for name in SIGNALS}, cause="custom")


def uniform_except(**overrides: float) -> Scenario:
    """Every signal at 0.95 except the named overrides."""
    signals = {name: 0.95 for name in SIGNALS}
    signals.update(overrides)
    return Scenario(signals=signals, cause="custom")


def test_deep_fail_exits_after_one_measurement() -> None:
    """Cheapest-to-fail first: hopeless capture, one measurement."""
    rng = ScriptedRng([0.0])
    decision = ExpertPolicy(noisy=True).decide(
        Scenario(
            signals={
                "blur": 0.10,
                "glare": 0.95,
                "brightness": 0.95,
                "coverage": 0.95,
                "skew": 0.95,
                "resolution": 0.95,
            },
            cause="custom",
        ),
        rng,
    )
    assert decision.action == "retake"
    assert decision.path == ("blur",)
    assert rng.calls == 1


def test_suspect_reading_gets_exactly_one_recheck() -> None:
    """Six checks + one re-check: the intermediate verification.

    Brightness true 0.50 sits mid-band (FAIL 0.35, PASS 0.60); the
    other five are comfortably clean. No uniform value is suspect
    for all six signals (coverage's FAIL 0.55 meets resolution's
    PASS 0.55), so the ARCHITECTURE section 8 shape is used.
    """
    rng = ScriptedRng([0.0] * 7)
    decision = ExpertPolicy(noisy=True).decide(
        uniform_except(brightness=0.50), rng
    )
    assert decision.action == "review"
    assert rng.calls == 7


def test_fail_reading_near_threshold_is_not_rechecked() -> None:
    """Regression: 0.34 is a fail (audit fix suspect-only rule).

    The old near-threshold rule burned a second measurement on a
    reading already decisively below FAIL.
    """
    scenario = Scenario(
        signals={
            "blur": 0.34,
            "glare": 0.95,
            "brightness": 0.95,
            "coverage": 0.95,
            "skew": 0.95,
            "resolution": 0.95,
        },
        cause="custom",
    )
    rng = ScriptedRng([0.0])
    decision = ExpertPolicy(noisy=True).decide(scenario, rng)
    assert decision.action == "retake"
    assert rng.calls == 1


def test_pass_reading_near_threshold_is_not_rechecked() -> None:
    """Regression: 0.66 (just above PASS 0.65) is final, one draw."""
    scenario = Scenario(
        signals={
            "blur": 0.95,
            "glare": 0.95,
            "brightness": 0.66,
            "coverage": 0.95,
            "skew": 0.95,
            "resolution": 0.95,
        },
        cause="custom",
    )
    # ORDER is blur, brightness, ...: 5 remaining clean checks.
    rng = ScriptedRng([0.0] * 6)
    decision = ExpertPolicy(noisy=True).decide(scenario, rng)
    assert decision.action == "pass"
    assert rng.calls == 6


def test_review_defers_with_zero_point_five_confidence() -> None:
    """Deferral is uncertainty, not a confident claim."""
    decision = ExpertPolicy(noisy=False).decide(
        uniform_except(brightness=0.50), None
    )
    assert decision.action == "review"
    assert decision.confidence == 0.5


def test_oracle_mode_reads_true_values() -> None:
    """noisy=False ignores rng entirely (environment sanity)."""
    decision = ExpertPolicy(noisy=False).decide(uniform(0.95), None)
    assert decision.action == "pass"
    assert len(decision.path) == len(ORDER)
