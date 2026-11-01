"""Bayes-ceiling analysis for AI-267 observation representations.

What this measures
------------------
For a given *observation encoding* (how a raw reading is turned into
the code the policy stores) and a given *measurement path* (which
signals get measured, and which get re-measured), this computes the
accuracy of the **best possible terminal decision rule** allowed to
see only that encoding.

That number is a ceiling. It answers two questions at once:

- *decision-rule headroom* -- if the policy kept its exact behaviour
  but chose the Bayes-optimal verdict, how much accuracy would it
  gain? (i.e. is the learned table or the state representation at
  fault?)
- *representation headroom* -- how much does a finer encoding buy at
  the same path?

Method
------
The generative model is known exactly, so the posterior over the
oracle label given the observations is available in closed form:

1. Enumerate the 17 hypotheses over the defect structure (clean, one
   of six named defects, one of ten multi-defect pairs) with their
   exact priors from :data:`smallai.environment.TEST_WEIGHTS`.
2. Under each hypothesis every signal's true value is uniform on a
   known range. A reading lands in a bin with a known likelihood
   ``L(v)`` (Gaussian sensor model, including the clipping to
   ``[0, 1]``).
3. Integrate ``L`` over the hypothesis' value range to get the
   signal's evidence and its zone posterior (fail / suspect / pass).
   All of these integrals have closed forms, so the whole analysis is
   exact and fast.
4. Marginalise the hypothesis, read off ``P(label | observations)``.

The ceiling is then ``E[max_l P(label = l | observations)]``.

This is development-time analysis only. It never touches the scored
protocol's test split.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from smallai.agent import TERMINALS, UNCHECKED, QAgent, transition
from smallai.environment import (
    FAIL,
    NOISE_SIGMA,
    PASS,
    SIGNALS,
    Scenario,
    observe,
)
from smallai.evaluate import derive_seed
from smallai.solver import ORDER
from smallai.train import MIX, make_splits, oracle_label, train_td

DEV_SEED = 20261030

LABELS = ("pass", "retake", "review")


# --------------------------------------------------------------------
# Numerics
# --------------------------------------------------------------------


def _cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _antiderivative(edge: float, v: float, sigma: float) -> float:
    """Antiderivative of ``Phi((edge - v) / sigma)`` in ``v``.

    Parameters
    ----------
    edge : float
        The bin edge.
    v : float
        Evaluation point.
    sigma : float
        Sensor noise scale.

    Returns
    -------
    float
        A value whose derivative is ``Phi((edge - v) / sigma)``.
    """
    u = (edge - v) / sigma
    return -sigma * (u * _cdf(u) + _pdf(u))


def _bin_mass(
    bins: tuple, index: int, x: float, y: float, sigma: float
) -> float:
    """Integral of a bin's likelihood over ``[x, y]``.

    Parameters
    ----------
    bins : tuple
        The signal's bins.
    index : int
        The bin index.
    x : float
        Lower limit.
    y : float
        Upper limit.
    sigma : float
        Sensor noise scale.

    Returns
    -------
    float
        The integral.
    """
    lo, hi = bins[index]
    if hi >= 1.0:
        up_y, up_x = y, x
    else:
        up_y, up_x = _antiderivative(hi, y, sigma), _antiderivative(
            hi, x, sigma
        )
    if lo <= 0.0:
        low_y, low_x = 0.0, 0.0
    else:
        low_y, low_x = _antiderivative(lo, y, sigma), _antiderivative(
            lo, x, sigma
        )
    return (up_y - up_x) - (low_y - low_x)


def _point_mass(reading: float, x: float, y: float, sigma: float) -> float:
    """Integral of a point observation's likelihood over ``[x, y]``.

    Parameters
    ----------
    reading : float
        The exact observed reading.
    x : float
        Lower limit.
    y : float
        Upper limit.
    sigma : float
        Sensor noise scale.

    Returns
    -------
    float
        The integral.
    """
    return _cdf((y - reading) / sigma) - _cdf((x - reading) / sigma)


def _simpson(integrand, x: float, y: float, breakpoints, panels: int = 6) -> float:
    """Composite Simpson integration over ``[x, y]``.

    Parameters
    ----------
    integrand : callable
        The function to integrate.
    x : float
        Lower limit.
    y : float
        Upper limit.
    breakpoints : list
        Interior points forced onto the grid (kinks).
    panels : int
        Simpson panels per breakpoint interval.

    Returns
    -------
    float
        The integral.
    """
    if y <= x:
        return 0.0
    edges = sorted({x, y} | {b for b in breakpoints if x < b < y})
    total = 0.0
    for start, end in zip(edges, edges[1:]):
        width = end - start
        step = width / panels
        acc = integrand(start) + integrand(end)
        for j in range(1, panels):
            acc += integrand(start + j * step) * (4 if j % 2 else 2)
        total += acc * step / 3.0
    return total


# --------------------------------------------------------------------
# Representations
# --------------------------------------------------------------------


@dataclass(frozen=True)
class SignalRep:
    """One signal's encoding.

    Attributes
    ----------
    bins : tuple
        Atomic bins ``(lo, hi)``, each lying entirely inside one zone.
    zone : tuple
        The zone of each atomic bin.
    recheck : frozenset
        Atomic bins whose first reading triggers a second measurement.
    point : bool
        True when the encoding keeps the exact reading.
    """

    bins: tuple
    zone: tuple
    recheck: frozenset
    point: bool = False

    def bin_of(self, reading: float) -> int:
        """Atomic bin a reading falls in.

        Parameters
        ----------
        reading : float
            The measured value.

        Returns
        -------
        int
            The bin index.
        """
        for index, (lo, hi) in enumerate(self.bins):
            if lo <= reading < hi:
                return index
        return len(self.bins) - 1


@dataclass(frozen=True)
class Representation:
    """A complete per-signal encoding.

    Attributes
    ----------
    name : str
        Label for reports.
    signals : dict
        Signal name to :class:`SignalRep`.
    """

    name: str
    signals: dict[str, SignalRep]

    def bin_of(self, signal: str, reading: float) -> int:
        """Atomic bin a reading falls in."""
        return self.signals[signal].bin_of(reading)


def _split_at_zones(edges: list[float], signal: str) -> tuple:
    """Split bin edges at the signal's zone boundaries.

    Splitting keeps every atomic bin inside exactly one zone, so the
    zone posterior is a sum over whole bins.

    Parameters
    ----------
    edges : list
        Bin edges from 0 to 1.
    signal : str
        The signal name.

    Returns
    -------
    tuple
        The atomic bins.
    """
    points = sorted(set(edges) | {FAIL[signal], PASS[signal], 0.0, 1.0})
    out = []
    for lo, hi in zip(points, points[1:]):
        if hi - lo > 1e-12:
            out.append((lo, hi))
    return tuple(out)


def _zone_of(signal: str, value: float) -> str:
    """Zone a value falls in."""
    if value < FAIL[signal]:
        return "fail"
    if value < PASS[signal]:
        return "suspect"
    return "pass"


def _make_signal_rep(
    signal: str, edges: list[float], wide: bool, point: bool = False
) -> SignalRep:
    """Build one signal's encoding.

    Parameters
    ----------
    signal : str
        The signal name.
    edges : list
        Bin edges (ignored when ``point`` is True).
    wide : bool
        Whether a borderline *fail* also triggers a re-check.
    point : bool
        Keep the exact reading instead of binning.

    Returns
    -------
    SignalRep
        The encoding.
    """
    if point:
        return SignalRep(
            bins=((0.0, 1.0),),
            zone=("any",),
            recheck=frozenset(),
            point=True,
        )
    bins = _split_at_zones(edges, signal)
    zone = tuple(_zone_of(signal, (lo + hi) / 2.0) for lo, hi in bins)
    if wide:
        # The atomic bin immediately below the fail threshold.
        trigger = frozenset(
            index
            for index, (lo, hi) in enumerate(bins)
            if zone[index] == "fail" and hi >= FAIL[signal] - 1e-12
        )
    else:
        trigger = frozenset(
            index for index, z in enumerate(zone) if z == "suspect"
        )
    return SignalRep(bins=bins, zone=zone, recheck=trigger)


def _rel_edges(signal: str, fail_bins: int, suspect_bins: int) -> list[float]:
    """Threshold-relative bin edges for one signal.

    Parameters
    ----------
    signal : str
        The signal name.
    fail_bins : int
        How many bins to split the fail region into.
    suspect_bins : int
        How many bins to split the suspect band into.

    Returns
    -------
    list
        The edges.
    """
    fail = FAIL[signal]
    pas = PASS[signal]
    edges = [fail * j / fail_bins for j in range(1, fail_bins + 1)]
    edges += [
        fail + (pas - fail) * j / suspect_bins
        for j in range(1, suspect_bins + 1)
    ]
    edges.append(1.0)
    return edges


def _uniform_edges(count: int) -> list[float]:
    """Equal-width bin edges over ``[0, 1]``."""
    return [j / count for j in range(1, count + 1)]


def _make_representations() -> list[Representation]:
    """Build every candidate encoding to compare."""
    reps: list[Representation] = []

    zone_edges = {
        name: [FAIL[name], PASS[name], 1.0] for name in SIGNALS
    }
    reps.append(
        Representation(
            "zone3 (current)",
            {
                name: _make_signal_rep(name, zone_edges[name], wide=False)
                for name in SIGNALS
            },
        )
    )

    rel4 = {name: _rel_edges(name, 2, 1) for name in SIGNALS}
    reps.append(
        Representation(
            "rel4 (2+1+1) narrow",
            {
                name: _make_signal_rep(name, rel4[name], wide=False)
                for name in SIGNALS
            },
        )
    )
    reps.append(
        Representation(
            "rel4 (2+1+1) wide",
            {
                name: _make_signal_rep(name, rel4[name], wide=True)
                for name in SIGNALS
            },
        )
    )

    for fail_bins, suspect_bins, label in (
        (3, 2, "rel6 (3+2+1) wide"),
        (4, 3, "rel8 (4+3+1) wide"),
        (6, 4, "rel11 (6+4+1) wide"),
    ):
        edges = {
            name: _rel_edges(name, fail_bins, suspect_bins)
            for name in SIGNALS
        }
        reps.append(
            Representation(
                label,
                {
                    name: _make_signal_rep(name, edges[name], wide=True)
                    for name in SIGNALS
                },
            )
        )

    for count, label in ((8, "uni8"), (32, "uni32")):
        edges = _uniform_edges(count)
        reps.append(
            Representation(
                label,
                {
                    name: _make_signal_rep(name, edges, wide=False)
                    for name in SIGNALS
                },
            )
        )

    reps.append(
        Representation(
            "exact reading",
            {
                name: _make_signal_rep(name, [], wide=False, point=True)
                for name in SIGNALS
            },
        )
    )
    return reps


# --------------------------------------------------------------------
# Hypotheses
# --------------------------------------------------------------------


@dataclass(frozen=True)
class Hypothesis:
    """One candidate defect structure.

    Attributes
    ----------
    name : str
        Label.
    prior : float
        Probability under the test mixture.
    defective : dict
        Signal name to the ``(lo, hi)`` range of its true value.
    """

    name: str
    prior: float
    defective: dict[str, tuple[float, float]]


def _hypotheses() -> list[Hypothesis]:
    """The 17 defect structures the sampler can produce."""
    from smallai.environment import TEST_WEIGHTS

    out = [Hypothesis("clean", TEST_WEIGHTS["clean"], {})]
    for name in SIGNALS:
        out.append(
            Hypothesis(
                f"single:{name}", TEST_WEIGHTS[name], {name: (0.02, 0.32)}
            )
        )
    pool = list(SIGNALS[1:])
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            a, b = pool[i], pool[j]
            out.append(
                Hypothesis(
                    f"multi:{a}+{b}",
                    TEST_WEIGHTS["multi"] / 10.0,
                    {a: (0.05, 0.5), b: (0.05, 0.5)},
                )
            )
    total = sum(h.prior for h in out)
    if abs(total - 1.0) > 1e-9:
        raise AssertionError(f"Hypothesis priors sum to {total!r}")
    return out


# --------------------------------------------------------------------
# Pre-integrated tables
# --------------------------------------------------------------------


@dataclass
class SignalTable:
    """Evidence and zone posteriors for one signal under one hypothesis.

    Attributes
    ----------
    evidence : dict
        Observation key to the probability of that observation.
    zones : dict
        Observation key to the ``(fail, suspect, pass)`` posterior.
    """

    evidence: dict
    zones: dict


def _build_tables(
    rep: Representation,
    hypotheses: list[Hypothesis],
    sigma: float,
    triggers: dict | None = None,
) -> list[dict[str, SignalTable]]:
    """Pre-compute evidence and zone posteriors for every hypothesis.

    Parameters
    ----------
    rep : Representation
        The encoding.
    hypotheses : list of Hypothesis
        The defect structures.
    sigma : float
        Sensor noise scale.
    triggers : dict or None
        When given, signal name to the set of atomic bins whose first
        reading triggers a re-check. Used to replay a fixed path, where
        the re-check rule belongs to the path and not to the encoding.

    Returns
    -------
    list
        One dict per hypothesis, mapping signal name to
        :class:`SignalTable`.
    """
    tables = []
    for hyp in hypotheses:
        row: dict[str, SignalTable] = {}
        for signal in SIGNALS:
            sig = rep.signals[signal]
            lo_range, hi_range = hyp.defective.get(signal, (0.85, 1.0))
            width = hi_range - lo_range
            fail_edge = FAIL[signal]
            pass_edge = PASS[signal]
            zone_bounds = {
                "fail": (lo_range, min(hi_range, fail_edge)),
                "suspect": (max(lo_range, fail_edge), min(hi_range, pass_edge)),
                "pass": (max(lo_range, pass_edge), hi_range),
            }
            breakpoints = sorted(
                {b for pair in sig.bins for b in pair}
                | {lo_range, hi_range, fail_edge, pass_edge}
            )

            if sig.point:
                def mass(reading: float, x: float, y: float) -> float:
                    return _point_mass(reading, x, y, sigma)

                evidence: dict = {}
                zones: dict = {}
                # A point observation keeps the reading, so the table
                # is built per observed reading instead of per bin.
                row[signal] = SignalTable(evidence, zones)
                continue

            evidence = {}
            zones = {}
            for index in range(len(sig.bins)):
                total = _bin_mass(sig.bins, index, lo_range, hi_range, sigma)
                evidence[index] = total / width
                parts = []
                for zone in ("fail", "suspect", "pass"):
                    x, y = zone_bounds[zone]
                    parts.append(
                        _bin_mass(sig.bins, index, x, y, sigma)
                        if y > x
                        else 0.0
                    )
                norm = sum(parts)
                zones[index] = (
                    tuple(p / norm for p in parts) if norm > 0 else (0.0, 0.0, 1.0)
                )

            trigger = (
                triggers[signal]
                if triggers is not None
                else sig.recheck
            )
            if trigger:
                for index in range(len(sig.bins)):
                    # The product of two bin likelihoods needs
                    # numerical integration; build it from densities.
                    def likelihood(bin_index: int, v: float) -> float:
                        lo, hi = sig.bins[bin_index]
                        upper = 1.0 if hi >= 1.0 else _cdf((hi - v) / sigma)
                        lower = 0.0 if lo <= 0.0 else _cdf((lo - v) / sigma)
                        return upper - lower

                    def integrand(v: float, index=index) -> float:
                        first = sum(likelihood(k, v) for k in trigger)
                        return first * likelihood(index, v)

                    total = _simpson(
                        integrand, lo_range, hi_range, breakpoints
                    )
                    evidence[-1 - index] = total / width
                    parts = []
                    for zone in ("fail", "suspect", "pass"):
                        x, y = zone_bounds[zone]
                        parts.append(
                            _simpson(integrand, x, y, breakpoints)
                            if y > x
                            else 0.0
                        )
                    norm = sum(parts)
                    zones[-1 - index] = (
                        tuple(p / norm for p in parts)
                        if norm > 0
                        else (0.0, 0.0, 1.0)
                    )

            evidence[None] = 1.0
            fail_prior = max(
                0.0, min(hi_range, fail_edge) - lo_range
            ) / width
            pass_prior = max(0.0, hi_range - max(lo_range, pass_edge)) / width
            zones[None] = (
                fail_prior,
                max(0.0, 1.0 - fail_prior - pass_prior),
                pass_prior,
            )
            row[signal] = SignalTable(evidence, zones)
        tables.append(row)
    return tables


# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------


@dataclass
class Observation:
    """What a path observed.

    Attributes
    ----------
    readings : dict
        Signal name to the list of readings taken for it.
    measurements : int
        Total signal acquisitions.
    """

    readings: dict[str, list[float]]
    measurements: int


def _path_expert(
    scenario: Scenario, rng: random.Random, rep: Representation, sigma: float
) -> Observation:
    """The expert's path: canonical order, early exit on a fail.

    Parameters
    ----------
    scenario : Scenario
        The capture.
    rng : random.Random
        Seeded noise source.
    rep : Representation
        The encoding, which decides when a re-check happens.
    sigma : float
        Sensor noise scale.

    Returns
    -------
    Observation
        What was observed.
    """
    return _path(
        scenario, rng, rep, sigma, order=ORDER, early_exit=True, full=False
    )


def _path_full(
    scenario: Scenario, rng: random.Random, rep: Representation, sigma: float
) -> Observation:
    """Measure every signal, never exiting early.

    Parameters
    ----------
    scenario : Scenario
        The capture.
    rng : random.Random
        Seeded noise source.
    rep : Representation
        The encoding.
    sigma : float
        Sensor noise scale.

    Returns
    -------
    Observation
        What was observed.
    """
    return _path(
        scenario, rng, rep, sigma, order=ORDER, early_exit=False, full=True
    )


def _path(
    scenario: Scenario,
    rng: random.Random,
    rep: Representation,
    sigma: float,
    order: tuple[str, ...],
    early_exit: bool,
    full: bool,
) -> Observation:
    """Simulate a measurement path.

    Parameters
    ----------
    scenario : Scenario
        The capture.
    rng : random.Random
        Seeded noise source.
    rep : Representation
        The encoding.
    sigma : float
        Sensor noise scale.
    order : tuple
        The order to check signals in.
    early_exit : bool
        Stop after a decisive fail.
    full : bool
        Never stop.

    Returns
    -------
    Observation
        What was observed.
    """
    import smallai.environment as env

    previous = env.NOISE_SIGMA
    env.NOISE_SIGMA = sigma
    readings: dict[str, list[float]] = {}
    measurements = 0
    try:
        for name in order:
            reading = observe(scenario, name, rng)
            measurements += 1
            readings[name] = [reading]
            index = rep.bin_of(name, reading)
            if index in rep.signals[name].recheck:
                second = observe(scenario, name, rng)
                measurements += 1
                readings[name].append(second)
                reading = second
            if early_exit and reading < FAIL[name]:
                break
            if full:
                continue
    finally:
        env.NOISE_SIGMA = previous
    return Observation(readings, measurements)


def _policy_path(
    agent: QAgent,
    scenario: Scenario,
    rng: random.Random,
    sigma: float,
) -> tuple[Observation, str]:
    """Replay the trained policy's own measurement path.

    Parameters
    ----------
    agent : QAgent
        The trained policy.
    scenario : Scenario
        The capture.
    rng : random.Random
        Seeded noise source.
    sigma : float
        Sensor noise scale.

    Returns
    -------
    tuple
        ``(observation, policy_action)``.
    """
    import smallai.environment as env

    previous = env.NOISE_SIGMA
    env.NOISE_SIGMA = sigma
    readings: dict[str, list[float]] = {}
    measurements = 0
    state = (UNCHECKED,) * len(SIGNALS)
    try:
        while True:
            action = agent.act(state, rng, epsilon=0.0)
            if action in TERMINALS:
                break
            name = action.partition(":")[2]
            reading = observe(scenario, name, rng)
            measurements += 1
            readings.setdefault(name, []).append(reading)
            state = transition(state, action, reading)
    finally:
        env.NOISE_SIGMA = previous
    return Observation(readings, measurements), action, state


# --------------------------------------------------------------------
# Posterior
# --------------------------------------------------------------------


def _posterior(
    observations: Observation,
    rep: Representation,
    hypotheses: list[Hypothesis],
    tables: list[dict[str, SignalTable]],
    sigma: float,
    triggers: dict | None = None,
) -> dict[str, float]:
    """Posterior over the oracle label given the observations.

    Parameters
    ----------
    observations : Observation
        What the path saw.
    rep : Representation
        The encoding.
    hypotheses : list of Hypothesis
        The defect structures.
    tables : list
        Pre-integrated tables.
    sigma : float
        Sensor noise scale.

    Returns
    -------
    dict
        Label to probability.
    """
    weights = []
    keys = []
    for index, hyp in enumerate(hypotheses):
        weight = hyp.prior
        key = {}
        for signal in SIGNALS:
            if signal not in observations.readings:
                continue
            reads = observations.readings[signal]
            if rep.signals[signal].point:
                key[signal] = (reads[-1], len(reads))
                if len(reads) == 1:
                    lo_range, hi_range = hyp.defective.get(
                        signal, (0.85, 1.0)
                    )
                    width = hi_range - lo_range
                    mass = _point_mass(reads[-1], lo_range, hi_range, sigma)
                    weight *= mass / width
                    zone_bounds = {
                        "fail": (lo_range, min(hi_range, FAIL[signal])),
                        "suspect": (
                            max(lo_range, FAIL[signal]),
                            min(hi_range, PASS[signal]),
                        ),
                        "pass": (max(lo_range, PASS[signal]), hi_range),
                    }
                    parts = [
                        _point_mass(reads[-1], x, y, sigma) if y > x else 0.0
                        for x, y in zone_bounds.values()
                    ]
                    norm = sum(parts)
                    key[signal] = (
                        tuple(p / norm for p in parts) if norm > 0 else (0.0, 0.0, 1.0)
                    )
                else:
                    lo_range, hi_range = hyp.defective.get(
                        signal, (0.85, 1.0)
                    )
                    width = hi_range - lo_range
                    r1, r2 = reads[0], reads[1]
                    def integrand(v: float) -> float:
                        return (
                            _pdf((r1 - v) / sigma)
                            * _pdf((r2 - v) / sigma)
                            / sigma
                        )
                    breakpoints = sorted(
                        [lo_range, hi_range, FAIL[signal], PASS[signal]]
                    )
                    total = _simpson(integrand, lo_range, hi_range, breakpoints)
                    weight *= total / width
                    zone_bounds = {
                        "fail": (lo_range, min(hi_range, FAIL[signal])),
                        "suspect": (
                            max(lo_range, FAIL[signal]),
                            min(hi_range, PASS[signal]),
                        ),
                        "pass": (max(lo_range, PASS[signal]), hi_range),
                    }
                    parts = [
                        _simpson(integrand, x, y, breakpoints)
                        if y > x
                        else 0.0
                        for x, y in zone_bounds.values()
                    ]
                    norm = sum(parts)
                    key[signal] = (
                        tuple(p / norm for p in parts)
                        if norm > 0
                        else (0.0, 0.0, 1.0)
                    )
            else:
                bin_index = rep.bin_of(signal, reads[-1])
                n_reads = len(reads)
                table = tables[index][signal]
                key[signal] = (
                    bin_index if n_reads == 1 else -1 - bin_index
                )
                weight *= table.evidence.get(key[signal], 0.0)
                if n_reads == 2:
                    trigger = (
                        triggers[signal]
                        if triggers is not None
                        else rep.signals[signal].recheck
                    )
                    lo_range, hi_range = hyp.defective.get(
                        signal, (0.85, 1.0)
                    )
                    width = hi_range - lo_range
                    def likelihood(bin_index: int, v: float) -> float:
                        lo, hi = rep.signals[signal].bins[bin_index]
                        upper = (
                            1.0 if hi >= 1.0 else _cdf((hi - v) / sigma)
                        )
                        lower = (
                            0.0 if lo <= 0.0 else _cdf((lo - v) / sigma)
                        )
                        return upper - lower

                    def integrand(v: float, bin_index=bin_index) -> float:
                        first = sum(likelihood(k, v) for k in trigger)
                        return first * likelihood(bin_index, v)

                    breakpoints = sorted(
                        {b for pair in rep.signals[signal].bins for b in pair}
                        | {lo_range, hi_range, FAIL[signal], PASS[signal]}
                    )
                    total = _simpson(
                        integrand, lo_range, hi_range, breakpoints
                    )
                    weight *= total / width
                    zone_bounds = {
                        "fail": (lo_range, min(hi_range, FAIL[signal])),
                        "suspect": (
                            max(lo_range, FAIL[signal]),
                            min(hi_range, PASS[signal]),
                        ),
                        "pass": (max(lo_range, PASS[signal]), hi_range),
                    }
                    parts = [
                        _simpson(integrand, x, y, breakpoints)
                        if y > x
                        else 0.0
                        for x, y in zone_bounds.values()
                    ]
                    norm = sum(parts)
                    key[signal] = (
                        tuple(p / norm for p in parts)
                        if norm > 0
                        else (0.0, 0.0, 1.0)
                    )
                else:
                    key[signal] = table.zones.get(
                        key[signal], table.zones[None]
                    )
            if weight == 0.0:
                break
        weights.append(weight)
        keys.append(key)

    total = sum(weights)
    if total <= 0:
        return {label: 1.0 / 3 for label in LABELS}
    out = {label: 0.0 for label in LABELS}
    for weight, hyp, key in zip(weights, hypotheses, keys):
        if weight == 0.0:
            continue
        weight = weight / total
        defective = hyp.defective
        if not defective:
            out["pass"] += weight
            continue
        zones = []
        for signal in defective:
            value = key.get(signal)
            if value is None:
                lo_range, hi_range = hyp.defective[signal]
                width = hi_range - lo_range
                fail_prior = max(
                    0.0, min(hi_range, FAIL[signal]) - lo_range
                ) / width
                pass_prior = max(
                    0.0, hi_range - max(lo_range, PASS[signal])
                ) / width
                value = (
                    fail_prior,
                    max(0.0, 1.0 - fail_prior - pass_prior),
                    pass_prior,
                )
            zones.append(value)
        if len(zones) == 1:
            zf, zs, zp = zones[0]
            out["retake"] += weight * zf
            out["review"] += weight * zs
            out["pass"] += weight * zp
        else:
            (zf1, zs1, zp1), (zf2, zs2, zp2) = zones
            out["review"] += weight * zs1 * zs2
            out["pass"] += weight * zp1 * zp2
            out["retake"] += weight * (1.0 - (1.0 - zf1) * (1.0 - zf2))
    return out


# --------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------


def _score(
    rep: Representation,
    scenarios: list[Scenario],
    labels: list[str],
    seed: int,
    sigma: float,
    hypotheses: list[Hypothesis],
    tables: list[dict[str, SignalTable]],
    path_kind: str,
    agent: QAgent | None = None,
    triggers: dict | None = None,
) -> dict:
    """Measure the ceiling of one encoding on one path.

    Parameters
    ----------
    rep : Representation
        The encoding.
    scenarios : list of Scenario
        The evaluation captures.
    labels : list of str
        Their oracle labels.
    seed : int
        Seeded noise source.
    sigma : float
        Sensor noise scale.
    hypotheses : list of Hypothesis
        The defect structures.
    tables : list
        Pre-integrated tables.
    path_kind : str
        ``"policy"``, ``"expert"`` or ``"full"``.
    agent : QAgent or None
        The trained policy, required for ``"policy"``.
    triggers : dict or None
        Signal name to the set of atomic bins whose first reading
        triggers a re-check, for a fixed path.

    Returns
    -------
    dict
        Ceiling accuracy, mean measurements, and the policy's own
        accuracy on the same path when available.
    """
    rng = random.Random(seed)
    correct = 0
    expected = 0.0
    measurements = 0
    policy_correct = 0
    confusion: dict[str, int] = {}
    for index, (scenario, label) in enumerate(zip(scenarios, labels)):
        policy_action = None
        if path_kind == "policy":
            policy_rng = random.Random(
                derive_seed(seed, "journey_q", str(index))
            )
            observation, policy_action, _state = _policy_path(
                agent, scenario, policy_rng, sigma
            )
        elif path_kind == "expert":
            observation = _path_expert(scenario, rng, rep, sigma)
        else:
            observation = _path_full(scenario, rng, rep, sigma)
        measurements += observation.measurements
        posterior = _posterior(
            observation, rep, hypotheses, tables, sigma, triggers
        )
        # The ceiling is the expected accuracy of the Bayes rule, not
        # the realised accuracy of one argmax on one finite sample: a
        # sharper posterior can flip one scenario the wrong way while
        # being better on average, and only the expectation is
        # monotone in the encoding's resolution.
        expected += max(posterior.values())
        guess = max(posterior.items(), key=lambda kv: (kv[1], kv[0]))[0]
        confusion[f"{guess}->{label}"] = confusion.get(f"{guess}->{label}", 0) + 1
        if guess == label:
            correct += 1
        if policy_action == label:
            policy_correct += 1
    n = len(scenarios)
    return {
        "ceiling": expected / n,
        "accuracy": correct / n,
        "mean_measurements": measurements / n,
        "policy_accuracy": policy_correct / n,
        "confusion": confusion,
    }


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEV_SEED)
    parser.add_argument("--sigma", type=float, default=NOISE_SIGMA)
    args = parser.parse_args(argv)

    splits = make_splits(args.seed)
    scenarios = splits["test"]
    labels = [oracle_label(s) for s in scenarios]
    hypotheses = _hypotheses()

    agent = QAgent(name="journey_q")
    train_td(
        agent,
        splits["train"],
        seed=derive_seed(args.seed, "train", "journey_q"),
        budget=200_000,
        mix=MIX,
    )

    print("=" * 92)
    print("BAYES CEILING BY ENCODING AND PATH")
    print(
        f"dev test split, {len(scenarios):,} scenarios, sigma={args.sigma}"
    )
    print("=" * 92)
    print(
        f"{'encoding':<22}"
        f"{'policy path':>13}{'argmax':>9}{'meas':>8}"
        f"{'  vs zone3':>11}"
        f"{'expert path':>13}{'full scan':>12}"
    )
    baseline = {}
    for rep in _make_representations():
        # On the policy's own path the re-check rule belongs to the
        # policy: it re-checks a suspect reading and nothing else.
        policy_triggers = {
            name: frozenset(
                index
                for index, zone in enumerate(rep.signals[name].zone)
                if zone == "suspect"
            )
            for name in SIGNALS
        }
        policy_tables = _build_tables(
            rep, hypotheses, args.sigma, triggers=policy_triggers
        )
        tables = _build_tables(rep, hypotheses, args.sigma)
        row = {}
        for kind in ("policy", "expert", "full"):
            row[kind] = _score(
                rep,
                scenarios,
                labels,
                args.seed,
                args.sigma,
                hypotheses,
                policy_tables if kind == "policy" else tables,
                kind,
                agent,
                policy_triggers if kind == "policy" else None,
            )
        baseline[rep.name] = row
        print(
            f"{rep.name:<22}"
            f"{100 * row['policy']['ceiling']:>13.3f}"
            f"{100 * row['policy']['accuracy']:>9.3f}"
            f"{row['policy']['mean_measurements']:>8.3f}"
            f"{'':>11}"
            f"{100 * row['expert']['ceiling']:>13.3f}"
            f"{100 * row['full']['ceiling']:>12.3f}"
        )

    print()
    print("policy's own accuracy on its own path (for reference):")
    reference = baseline["zone3 (current)"]["policy"]
    print(
        f"  {100 * reference['policy_accuracy']:.3f} %  "
        f"(mean {reference['mean_measurements']:.3f} measurements)"
    )
    print()
    print("ceiling delta vs zone3 on the policy's path (pp):")
    for name, row in baseline.items():
        delta = 100.0 * (row["policy"]["ceiling"] - reference["ceiling"])
        print(f"  {name:<22}{delta:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
