"""The deterministic expert: ground truth and the rules baseline.

The expert checks signals in a fixed order -- cheapest-to-fail first,
so a hopeless capture is rejected after one measurement -- and exits
early on the first hard failure. A suspect reading -- one inside
the band between the fail and pass thresholds -- is measured twice
before being believed: that re-check is the "intermediate
verification step" the ticket (AI-267) asks the learned policy to
discover on its own.

The final decision:

- any observed fail -> ``retake`` (a re-prompt is cheaper than a
  reviewer's time),
- else any suspect after re-checking -> ``review``,
- else ``pass``.

Confidence is deterministic in the margins: deep below a threshold or
comfortably above the pass line raises it; ``review`` is the explicit
defer bucket and is always reported at 0.5, because deferral is an
admission of uncertainty, not a confident claim.
"""

from __future__ import annotations

from dataclasses import dataclass

from smallai.environment import (
    FAIL,
    PASS,
    SIGNALS,
    Scenario,
    observe,
)

# Cheapest-to-fail first: blur and exposure usually kill a capture,
# alignment and pixel size rarely do.
ORDER: tuple[str, ...] = (
    "blur",
    "brightness",
    "glare",
    "coverage",
    "skew",
    "resolution",
)


@dataclass(frozen=True)
class Decision:
    """A structured decision, Clef-style: typed action plus evidence.

    Attributes
    ----------
    action : str
        ``"pass"``, ``"retake"`` or ``"review"``.
    confidence : float
        The decider's confidence in its own action, in ``(0, 1]``.
    reasons : tuple[str, ...]
        Human-readable evidence, e.g. ``"blur=0.21 failed (<0.35)"``.
    path : tuple[str, ...]
        The checks evaluated, in order; the intermediate states a
        reviewer can audit.
    """

    action: str
    confidence: float
    reasons: tuple[str, ...]
    path: tuple[str, ...]


class ExpertPolicy:
    """The deterministic expert whose labels define ground truth.

    Parameters
    ----------
    noisy : bool
        If True, measurements go through
        :func:`smallai.environment.observe` (field conditions); if
        False, the expert reads true values (oracle mode, used to
        sanity-check the environment itself).
    """

    name = "expert_rules"

    def __init__(self, noisy: bool = True) -> None:
        self.noisy = noisy

    def decide(self, scenario: Scenario, rng=None) -> Decision:
        """Evaluate a capture and return a structured decision.

        Parameters
        ----------
        scenario : Scenario
            The capture to judge.
        rng : random.Random or None
            Seed source for noisy readings; required when
            ``noisy`` is True.

        Returns
        -------
        Decision
            The action, confidence, evidence and check path.
        """
        checked: dict[str, float] = {}
        path: list[str] = []

        for name in ORDER:
            reading = self._measure(scenario, name, rng)
            # A suspect reading gets one second measurement before
            # it is believed (suspect-only; matches the agent's
            # re-check(suspect_pre) semantics in EXPERIMENT.md 3).
            if scenario.level(name, reading) == "suspect":
                reading = self._measure(scenario, name, rng)
            checked[name] = reading
            path.append(name)
            if scenario.level(name, reading) == "fail":
                return Decision(
                    action="retake",
                    confidence=self._retake_confidence(name, reading),
                    reasons=(
                        f"{name}={reading:.2f} failed (<{FAIL[name]:.2f})",
                    ),
                    path=tuple(path),
                )

        suspects = [
            name
            for name in ORDER
            if scenario.level(name, checked[name]) == "suspect"
        ]
        if suspects:
            name = suspects[0]
            return Decision(
                action="review",
                confidence=0.5,
                reasons=tuple(
                    f"{s}={checked[s]:.2f} suspect" for s in suspects
                ),
                path=tuple(path),
            )
        return Decision(
            action="pass",
            confidence=self._pass_confidence(checked),
            reasons=tuple(f"{n}={checked[n]:.2f} ok" for n in ORDER),
            path=tuple(path),
        )

    def _measure(self, scenario: Scenario, name: str, rng) -> float:
        """Return the reading for one signal, noisy or true."""
        if self.noisy and rng is not None:
            return observe(scenario, name, rng)
        return scenario.signals[name]

    @staticmethod
    def _retake_confidence(name: str, reading: float) -> float:
        """Confidence grows the deeper the reading sits below FAIL."""
        severity = 1.0 - min(1.0, max(0.0, reading) / FAIL[name])
        return round(0.55 + 0.45 * severity, 4)

    @staticmethod
    def _pass_confidence(checked: dict[str, float]) -> float:
        """Confidence grows with the tightest margin over PASS."""
        margin = min((checked[n] - PASS[n]) / (1.0 - PASS[n]) for n in SIGNALS)
        margin = min(1.0, max(0.0, margin))
        return round(0.55 + 0.45 * margin, 4)
