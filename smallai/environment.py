"""The decision environment: offline document-capture validation.

A field officer photographs an ID or form while offline. Six
deterministic image-quality signals, each in ``[0, 1]`` with higher
better, describe the capture:

- ``blur`` -- sharpness (1 minus normalized blur strength)
- ``glare`` -- absence of blown-out reflections
- ``brightness`` -- closeness to a usable exposure
- ``coverage`` -- how completely the document fills the frame
- ``skew`` -- alignment to the frame axes
- ``resolution`` -- usable pixel size

Each signal has two documented thresholds: below ``FAIL`` the capture
is unusable (retake), at or above ``PASS`` it is fine, in between it
is suspect. Ground truth is the deterministic expert in
:mod:`smallai.solver`, not an annotator: the ticket (AI-267)
explicitly allows synthetic trajectories from a deterministic solver
when sequential decision data is scarce.

Readings are noisy: ``observe`` adds Gaussian measurement noise to a
scenario's true quality, so a policy that re-checks a borderline
signal before committing can outperform one that trusts its first
reading. That distinction is the whole point of the ticket's
"journey learning" requirement.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

SIGNALS: tuple[str, ...] = (
    "blur",
    "glare",
    "brightness",
    "coverage",
    "skew",
    "resolution",
)

# Engineering defaults, documented rather than fitted: below FAIL a
# capture cannot be salvaged by any reviewer; at or above PASS no
# reviewer would object. Chosen from typical mobile-capture quality
# ranges; re-tuning them is a one-line change per signal.
FAIL: dict[str, float] = {
    "blur": 0.35,
    "glare": 0.30,
    "brightness": 0.35,
    "coverage": 0.55,
    "skew": 0.40,
    "resolution": 0.30,
}
PASS: dict[str, float] = {
    "blur": 0.65,
    "glare": 0.60,
    "brightness": 0.60,
    "coverage": 0.80,
    "skew": 0.70,
    "resolution": 0.55,
}

# Measurement noise on each reading. A reading inside the suspect
# band (at or above FAIL, below PASS) is re-checked once before it
# is believed; noise is what makes that re-check worth anything.
NOISE_SIGMA = 0.08

# Defect-cause classes the sampler draws from. Weights are set per
# split and each dict sums to 1.0: training data mirrors field
# reality (blur dominates bad captures), test data is balanced so a
# policy cannot pass by learning "blurry -> retake" alone -- the
# shortcut the ticket's journey-learning comparison is designed to
# expose.
CAUSES: tuple[str, ...] = (
    "clean",
    "blur",
    "glare",
    "brightness",
    "coverage",
    "skew",
    "resolution",
    "multi",
)
TRAIN_WEIGHTS: dict[str, float] = {
    "clean": 0.35,
    "blur": 0.30,
    "glare": 0.05,
    "brightness": 0.05,
    "coverage": 0.05,
    "skew": 0.05,
    "resolution": 0.05,
    "multi": 0.10,
}
# Sum exactly 1.0: clean matches the train share, every named
# defect sits at 0.10 (blur no longer dominates), multi at 0.05.
TEST_WEIGHTS: dict[str, float] = {
    "clean": 0.35,
    "blur": 0.10,
    "glare": 0.10,
    "brightness": 0.10,
    "coverage": 0.10,
    "skew": 0.10,
    "resolution": 0.10,
    "multi": 0.05,
}


@dataclass(frozen=True)
class Scenario:
    """One capture's true (noise-free) signal quality.

    Attributes
    ----------
    signals : dict[str, float]
        True quality per signal, in ``[0, 1]``.
    cause : str
        The defect class this scenario was sampled from; used to
        report per-cause behaviour, never as a model input.
    """

    signals: dict[str, float]
    cause: str

    def level(self, name: str, reading: float | None = None) -> str:
        """Classify a signal as fail, suspect or pass.

        Parameters
        ----------
        name : str
            The signal to classify.
        reading : float or None
            A measured reading; defaults to the true value.

        Returns
        -------
        str
            ``"fail"``, ``"suspect"`` or ``"pass"``.
        """
        value = self.signals[name] if reading is None else reading
        if value < FAIL[name]:
            return "fail"
        if value < PASS[name]:
            return "suspect"
        return "pass"


def observe(scenario: Scenario, name: str, rng: random.Random) -> float:
    """Measure one signal, adding sensor noise.

    Parameters
    ----------
    scenario : Scenario
        The capture being measured.
    name : str
        The signal to measure.
    rng : random.Random
        Seeded source of randomness.

    Returns
    -------
    float
        The reading, clipped to ``[0, 1]``.
    """
    noisy = scenario.signals[name] + rng.gauss(0.0, NOISE_SIGMA)
    return min(1.0, max(0.0, noisy))


def sample_scenario(rng: random.Random, weights: dict[str, float]) -> Scenario:
    """Draw a synthetic capture from the defect-class mixture.

    Parameters
    ----------
    rng : random.Random
        Seeded source of randomness; sampling is deterministic for a
        given seed.
    weights : dict[str, float]
        Mixture weights over :data:`CAUSES`, e.g.
        :data:`TRAIN_WEIGHTS` or :data:`TEST_WEIGHTS`.

    Returns
    -------
    Scenario
        A capture with true signal qualities for its cause class.
    """
    cause = rng.choices(list(weights), weights=list(weights.values()))[0]
    signals = {name: rng.uniform(0.85, 1.0) for name in SIGNALS}
    if cause == "multi":
        defective = rng.sample(list(SIGNALS[1:]), k=2)
        for name in defective:
            signals[name] = rng.uniform(0.05, 0.5)
    elif cause != "clean":
        # The named defect dips below its FAIL threshold; everything
        # else stays comfortably usable.
        signals[cause] = rng.uniform(0.02, 0.32)
    return Scenario(signals=signals, cause=cause)
