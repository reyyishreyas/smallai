"""The learned policy: a sparse tabular Q-table (AI-267 Stage A).

The model is a lookup table ``Q[state][action] -> expected return``,
at most 5^6 = 15,625 states (EXPERIMENT.md section 3), stored
sparsely: only visited states occupy memory, and the stored-entry
count is reported as the model size. No neural network, no
scikit-learn, pure Python.

States are 6-tuples, one code per signal (see :data:`UNCHECKED`
through :data:`PASS_FINAL`). Actions are strings: ``"check:blur"``
measures an unchecked signal, ``"recheck:blur"`` re-measures a
suspect one (legal *only* from ``suspect_pre``, so re-check loops
are structurally impossible), and the terminals ``"pass"``,
``"retake"``, ``"review"`` are legal at any time and end the
episode.

Deciding (:meth:`QAgent.decide`) walks the table greedily from the
all-unchecked state, drawing a fresh noisy reading from the
environment for each measurement action, and stops at the first
terminal the table prefers. The reported confidence is the raw
terminal margin -- best minus second-best terminal Q-value, in
reward units -- which evaluate.py maps to P(correct) with the
isotonic regression fitted on the calibration split
(EXPERIMENT.md section 9). It is a score, not a probability.

Ties (all-zero at cold start, or exact equality after training) are
resolved toward gathering more evidence, signals in canonical
order, and -- when only terminals remain tied -- toward ``review``,
the safest deferral. A cold-start agent therefore measures
everything and defers; training replaces that prior.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from smallai.environment import FAIL, PASS, SIGNALS, Scenario, observe
from smallai.solver import Decision

# Per-signal state codes, frozen in EXPERIMENT.md section 3.
UNCHECKED: int = 0
FAIL_FINAL: int = 1
SUSPECT_PRE: int = 2
SUSPECT_POST: int = 3
PASS_FINAL: int = 4

#: Terminal actions, ordered safest-first for deterministic
#: tie-breaking: at equal Q, deferring to a human beats guessing.
TERMINALS: tuple[str, ...] = ("review", "retake", "pass")

#: 5 codes ^ 6 signals: the pre-registered state-space ceiling.
MAX_STATES: int = 5 ** len(SIGNALS)

State = tuple[int, int, int, int, int, int]


def check_action(name: str) -> str:
    """The measure-first action for one signal.

    Parameters
    ----------
    name : str
        A signal name from :data:`smallai.environment.SIGNALS`.

    Returns
    -------
    str
        ``"check:<signal>"``.
    """
    return f"check:{name}"


def recheck_action(name: str) -> str:
    """The measure-again action for one suspect signal.

    Parameters
    ----------
    name : str
        A signal name from :data:`smallai.environment.SIGNALS`.

    Returns
    -------
    str
        ``"recheck:<signal>"``.
    """
    return f"recheck:{name}"


def legal_actions(state: State, allow_recheck: bool = True) -> list[str]:
    """The actions available in a state (EXPERIMENT.md section 3).

    ``check`` for each unchecked signal, ``recheck`` for each
    suspect_pre signal (suppressed when ``allow_recheck`` is False,
    the NoRecheck-Q ablation), and the three terminals always.

    Parameters
    ----------
    state : tuple of int
        One code per signal.
    allow_recheck : bool
        Whether re-check actions exist at all.

    Returns
    -------
    list of str
        Legal action strings, measurements first (signal order),
        then terminals.
    """
    actions = [
        check_action(SIGNALS[i])
        for i, code in enumerate(state)
        if code == UNCHECKED
    ]
    if allow_recheck:
        actions += [
            recheck_action(SIGNALS[i])
            for i, code in enumerate(state)
            if code == SUSPECT_PRE
        ]
    return actions + list(TERMINALS)


def _reading_code(reading: float, name: str, rechecked: bool) -> int:
    """Map one noisy reading to its state code.

    Parameters
    ----------
    reading : float
        The measured value.
    name : str
        The signal measured.
    rechecked : bool
        Whether this was a re-check (a still-suspect reading then
        becomes suspect_post instead of suspect_pre).

    Returns
    -------
    int
        ``FAIL_FINAL``, ``SUSPECT_PRE``, ``SUSPECT_POST`` or
        ``PASS_FINAL``.
    """
    if reading < FAIL[name]:
        return FAIL_FINAL
    if reading < PASS[name]:
        return SUSPECT_POST if rechecked else SUSPECT_PRE
    return PASS_FINAL


def transition(state: State, action: str, reading: float) -> State:
    """Apply one measurement action, returning the next state.

    Parameters
    ----------
    state : tuple of int
        The current state.
    action : str
        A ``check`` or ``recheck`` action string.
    reading : float
        The fresh noisy reading it produced.

    Returns
    -------
    tuple of int
        The state with the measured signal advanced.

    Raises
    ------
    ValueError
        If the action is a terminal, names an unknown signal, or is
        illegal in this state (re-check from anything but
        suspect_pre, check of an already-measured signal).
    """
    kind, sep, name = action.partition(":")
    if not sep or name not in SIGNALS:
        raise ValueError(f"Unknown measurement action {action!r}")
    index = SIGNALS.index(name)
    current = state[index]
    if kind == "check":
        if current != UNCHECKED:
            raise ValueError(f"check is illegal for {name!r} in state {state}")
        code = _reading_code(reading, name, rechecked=False)
    elif kind == "recheck":
        if current != SUSPECT_PRE:
            raise ValueError(
                f"recheck is illegal for {name!r} in state {state}"
            )
        code = _reading_code(reading, name, rechecked=True)
    else:
        raise ValueError(f"Unknown action kind {kind!r}")
    return state[:index] + (code,) + state[index + 1 :]


class QAgent:
    """A sparse tabular Q-learning policy over capture-check journeys.

    Parameters
    ----------
    lr : float
        Learning rate for TD updates (0.3 per EXPERIMENT.md
        section 7): the fraction of each new sample the table
        absorbs.
    gamma : float
        Discount (0.99 per section 7): how much later rewards count
        relative to immediate ones.
    allow_recheck : bool
        Whether re-check actions exist. False builds the
        NoRecheck-Q ablation: identical in every other respect.
    name : str
        Label recorded in the JSON artifact (``"journey_q"``,
        ``"no_recheck_q"``, ...).

    Attributes
    ----------
    q : dict
        ``state -> {action -> float}``, sparse: an unvisited state
        has no entry and behaves as all-zero.
    """

    def __init__(
        self,
        lr: float = 0.3,
        gamma: float = 0.99,
        allow_recheck: bool = True,
        name: str = "journey_q",
    ) -> None:
        self.lr = lr
        self.gamma = gamma
        self.allow_recheck = allow_recheck
        self.name = name
        self.q: dict[State, dict[str, float]] = {}

    @property
    def stored_entries(self) -> int:
        """Total stored (state, action) values: the model size."""
        return sum(len(actions) for actions in self.q.values())

    @property
    def visited_states(self) -> int:
        """Number of distinct states with at least one stored value."""
        return len(self.q)

    def act(
        self, state: State, rng: random.Random, epsilon: float = 0.0
    ) -> str:
        """Pick an action for a state, epsilon-greedy.

        Parameters
        ----------
        state : tuple of int
            The current state.
        rng : random.Random
            Seeded randomness for the exploratory branch.
        epsilon : float
            Probability of a uniformly random legal action; 0.0
            (default) is the greedy decision-time behaviour.

        Returns
        -------
        str
            The chosen legal action.
        """
        legal = legal_actions(state, self.allow_recheck)
        if epsilon > 0.0 and rng.random() < epsilon:
            return rng.choice(legal)
        values = self.q.get(state, {})
        best = max(values.get(a, 0.0) for a in legal)
        tied = [a for a in legal if values.get(a, 0.0) >= best - 1e-12]
        measurements = [a for a in tied if a not in TERMINALS]
        if not measurements:
            return tied[0]
        return min(
            measurements,
            key=lambda a: SIGNALS.index(a.partition(":")[2]),
        )

    def update(
        self,
        state: State,
        action: str,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        """One TD(0) update (EXPERIMENT.md section 7).

        ``Q[s][a] += lr * (target - Q[s][a])`` where the target is
        the observed reward plus gamma times the best legal value
        of the next state, or just the reward at episode end.

        Parameters
        ----------
        state : tuple of int
            The state the action was taken in.
        action : str
            The action taken.
        reward : float
            The immediate reward (frozen in section 6).
        next_state : tuple of int
            The resulting state.
        done : bool
            Whether the action ended the episode.
        """
        current = self.q.setdefault(state, {}).get(action, 0.0)
        target = reward
        if not done:
            legal = legal_actions(next_state, self.allow_recheck)
            best_next = max(
                self.q.get(next_state, {}).get(a, 0.0) for a in legal
            )
            target += self.gamma * best_next
        self.q[state][action] = current + self.lr * (target - current)

    def decide(self, scenario: Scenario, rng: random.Random) -> Decision:
        """Run one greedy episode and return the structured verdict.

        Walks from the all-unchecked state, executing each chosen
        measurement against the environment (fresh noisy reading)
        until a terminal action is chosen. Structurally bounded at
        12 measurements (6 checks + at most 6 re-checks).

        Parameters
        ----------
        scenario : Scenario
            The capture to judge.
        rng : random.Random
            Seeded randomness for the noisy readings; required,
            because field readings are noisy by design.

        Returns
        -------
        Decision
            Terminal action, raw terminal margin as confidence
            (reward units, 0.0 at cold start), evidence strings in
            check order (re-checked signals show both readings),
            and the path of checks performed.

        Raises
        ------
        ValueError
            If rng is None.
        RuntimeError
            If the walk somehow exceeds the structural measurement
            bound (unreachable with the transition rules; a guard).
        """
        if rng is None:
            raise ValueError(
                "decide() needs an rng: readings are noisy by design"
            )
        state: State = (UNCHECKED,) * len(SIGNALS)
        readings: dict[str, list[float]] = {}
        path: list[str] = []
        for _ in range(2 * len(SIGNALS) + 1):
            action = self.act(state, rng, epsilon=0.0)
            if action in TERMINALS:
                margin = self.terminal_margin(state)
                reasons = self._reasons(scenario, readings)
                return Decision(
                    action=action,
                    confidence=round(margin, 4),
                    reasons=reasons,
                    path=tuple(path),
                )
            name = action.partition(":")[2]
            readings.setdefault(name, []).append(observe(scenario, name, rng))
            if action.startswith("check:"):
                path.append(name)
            state = transition(state, action, readings[name][-1])
        raise RuntimeError(
            "decision walk exceeded 12 measurements; the transition"
            " rules make this unreachable"
        )

    def terminal_margin(self, state: State) -> float:
        """Best minus second-best terminal Q-value at a state.

        The raw confidence signal of EXPERIMENT.md section 9:
        reward units, not a probability. Unvisited states score
        0.0 (all terminals tie).

        Parameters
        ----------
        state : tuple of int
            The state to score the terminals of.

        Returns
        -------
        float
            The margin, >= 0.0.
        """
        values = self.q.get(state, {})
        ranked = sorted((values.get(a, 0.0) for a in TERMINALS), reverse=True)
        return ranked[0] - ranked[1]

    @staticmethod
    def _reasons(
        scenario: Scenario, readings: dict[str, list[float]]
    ) -> tuple[str, ...]:
        """Evidence strings per checked signal, in check order.

        A signal measured once shows one reading; a re-checked
        signal shows both, with the levels they landed in.

        Parameters
        ----------
        scenario : Scenario
            The capture (for threshold classification only).
        readings : dict
            Signal name to the actual readings drawn during the
            walk, in draw order.

        Returns
        -------
        tuple of str
            One entry per checked signal.
        """
        reasons = []
        for name, drawn in readings.items():
            if len(drawn) == 2:
                first, second = drawn
                reasons.append(
                    f"{name}={first:.2f}"
                    f" {scenario.level(name, first)}, re-checked"
                    f" {second:.2f} {scenario.level(name, second)}"
                )
            else:
                reasons.append(
                    f"{name}={drawn[0]:.2f} {scenario.level(name, drawn[0])}"
                )
        return tuple(reasons)

    def to_dict(self) -> dict:
        """The full model as a JSON-friendly dict.

        Returns
        -------
        dict
            ``format``/``version`` guard, ``meta`` (name, lr,
            gamma, allow_recheck), and ``q`` keyed by
            comma-joined state codes.
        """
        return {
            "format": "smallai-qtable",
            "version": 1,
            "meta": {
                "name": self.name,
                "lr": self.lr,
                "gamma": self.gamma,
                "allow_recheck": self.allow_recheck,
            },
            "q": {
                ",".join(str(code) for code in state): dict(actions)
                for state, actions in self.q.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> QAgent:
        """Rebuild an agent from :meth:`to_dict` output.

        Parameters
        ----------
        data : dict
            A previously serialized model.

        Returns
        -------
        QAgent
            The restored agent.

        Raises
        ------
        ValueError
            If the format tag, version, a state key, or an action
            name is not recognized.
        """
        if data.get("format") != "smallai-qtable":
            raise ValueError("Not a smallai Q-table artifact")
        if data.get("version") != 1:
            raise ValueError(
                f"Unsupported artifact version {data.get('version')!r}"
            )
        meta = data.get("meta", {})
        agent = cls(
            lr=float(meta.get("lr", 0.3)),
            gamma=float(meta.get("gamma", 0.99)),
            allow_recheck=bool(meta.get("allow_recheck", True)),
            name=str(meta.get("name", "journey_q")),
        )
        for key, actions in data.get("q", {}).items():
            state = cls._parse_state(key)
            agent.q[state] = {
                cls._parse_action(a): float(v) for a, v in actions.items()
            }
        return agent

    @staticmethod
    def _parse_state(key: str) -> State:
        """Parse a comma-joined state key, validating every code."""
        try:
            state = tuple(int(c) for c in key.split(","))
        except ValueError as err:
            raise ValueError(f"Malformed state key {key!r}") from err
        if len(state) != len(SIGNALS) or any(
            code < UNCHECKED or code > PASS_FINAL for code in state
        ):
            raise ValueError(f"Malformed state key {key!r}")
        return state  # type: ignore[return-value]

    @staticmethod
    def _parse_action(action: str) -> str:
        """Validate an action string, returning it unchanged."""
        if action in TERMINALS:
            return action
        kind, sep, name = action.partition(":")
        if sep and kind in ("check", "recheck") and name in SIGNALS:
            return action
        raise ValueError(f"Malformed action {action!r}")

    def save(self, path: str | Path) -> None:
        """Write the model as formatted JSON.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file.
        """
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> QAgent:
        """Read a model written by :meth:`save`.

        Parameters
        ----------
        path : str or pathlib.Path
            The artifact file.

        Returns
        -------
        QAgent
            The restored agent.
        """
        return cls.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )
