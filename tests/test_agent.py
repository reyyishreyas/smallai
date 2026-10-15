"""Gates for the agent: state machine, learning rule, JSON I/O."""

from __future__ import annotations

import json
import random

import pytest

from smallai.agent import (
    FAIL_FINAL,
    MAX_STATES,
    PASS_FINAL,
    SUSPECT_POST,
    SUSPECT_PRE,
    UNCHECKED,
    QAgent,
    check_action,
    legal_actions,
    recheck_action,
    transition,
)
from smallai.environment import SIGNALS, Scenario

CLEAN = 0.95
ALL_CLEAN: tuple[int, ...] = (PASS_FINAL,) * 6
START: tuple[int, ...] = (UNCHECKED,) * 6


def state(**overrides: int) -> tuple[int, ...]:
    """A 6-code state; named signals overridden, rest unchecked."""
    codes = [UNCHECKED] * 6
    for name, code in overrides.items():
        codes[SIGNALS.index(name)] = code
    return tuple(codes)  # type: ignore[return-value]


def scenario(**overrides: float) -> Scenario:
    """Every signal clean unless overridden."""
    signals = {name: CLEAN for name in SIGNALS}
    signals.update(overrides)
    return Scenario(signals=signals, cause="custom")


# --- state space and action legality (EXPERIMENT.md section 3) ---


def test_state_space_ceiling() -> None:
    """5 codes ^ 6 signals = 15,625, the pre-registered bound."""
    assert MAX_STATES == 5**6 == 15625


def test_start_state_offers_checks_and_terminals_only() -> None:
    """Nothing suspect yet: no re-check exists at the start."""
    actions = legal_actions(START)
    assert len(actions) == 6 + 3
    assert all(not a.startswith("recheck") for a in actions)


def test_recheck_is_legal_only_from_suspect_pre() -> None:
    """The structural no-loop guarantee."""
    pre = state(blur=SUSPECT_PRE)
    assert recheck_action("blur") in legal_actions(pre)
    for code in (UNCHECKED, FAIL_FINAL, SUSPECT_POST, PASS_FINAL):
        assert recheck_action("blur") not in legal_actions(state(blur=code))


def test_checked_signal_offers_no_second_check() -> None:
    """A final signal cannot be checked again."""
    for code in (FAIL_FINAL, SUSPECT_POST, PASS_FINAL):
        assert check_action("blur") not in legal_actions(state(blur=code))


def test_no_recheck_ablation_removes_recheck_actions() -> None:
    """NoRecheck-Q: identical minus re-check (section 5)."""
    pre = state(blur=SUSPECT_PRE)
    assert recheck_action("blur") not in legal_actions(
        pre, allow_recheck=False
    )
    assert check_action("glare") in legal_actions(pre, allow_recheck=False)


# --- transitions ---


def test_check_transitions_to_the_right_code() -> None:
    """Fail / suspect / pass readings map to their codes."""
    assert transition(START, check_action("blur"), 0.20)[0] == (FAIL_FINAL)
    assert transition(START, check_action("blur"), 0.50)[0] == (SUSPECT_PRE)
    assert transition(START, check_action("blur"), 0.90)[0] == (PASS_FINAL)


def test_recheck_lands_on_post_or_final_never_pre() -> None:
    """A re-checked suspect becomes suspect_post, fail or pass."""
    pre = state(blur=SUSPECT_PRE)
    assert transition(pre, recheck_action("blur"), 0.50)[0] == (SUSPECT_POST)
    assert transition(pre, recheck_action("blur"), 0.20)[0] == (FAIL_FINAL)
    assert transition(pre, recheck_action("blur"), 0.90)[0] == (PASS_FINAL)


def test_illegal_transitions_raise() -> None:
    """Structural illegality, not training-time discouragement."""
    with pytest.raises(ValueError):
        transition(state(blur=SUSPECT_POST), recheck_action("blur"), 0.5)
    with pytest.raises(ValueError):
        transition(START, recheck_action("blur"), 0.5)
    with pytest.raises(ValueError):
        transition(state(blur=PASS_FINAL), check_action("blur"), 0.9)
    with pytest.raises(ValueError):
        transition(START, "check:notasignal", 0.5)
    with pytest.raises(ValueError):
        transition(START, "review", 0.5)


# --- the learning rule (EXPERIMENT.md section 7) ---


def test_td_update_matches_frozen_formula() -> None:
    """Q += lr * (reward + gamma * maxQ' - Q), lr .3 gamma .99."""
    agent = QAgent(lr=0.3, gamma=0.99)
    next_state = state(blur=FAIL_FINAL)
    agent.update(START, check_action("blur"), -0.05, next_state, True)
    assert agent.q[START][check_action("blur")] == pytest.approx(-0.015)


def test_td_update_bootstraps_the_next_state() -> None:
    """Non-terminal targets include gamma * max Q(s')."""
    agent = QAgent(lr=0.3, gamma=0.99)
    next_state = state(blur=FAIL_FINAL)
    agent.q[next_state] = {"retake": 1.0}
    agent.update(START, check_action("blur"), -0.05, next_state, False)
    assert agent.q[START][check_action("blur")] == pytest.approx(
        0.3 * (-0.05 + 0.99 * 1.0)
    )


def test_unvisited_states_are_stored_sparsely() -> None:
    """One update touches one state; the table stays sparse."""
    agent = QAgent()
    agent.update(START, check_action("blur"), 1.0, ALL_CLEAN, True)
    assert agent.visited_states == 1
    assert agent.stored_entries == 1


# --- deciding ---


def test_decide_returns_a_structured_decision() -> None:
    """Same output shape as the expert (Clef-style typed action)."""
    decision = QAgent().decide(scenario(), random.Random(0))
    assert decision.action in ("pass", "retake", "review")
    assert 0.0 <= decision.confidence <= 1.0
    assert set(decision.path) <= set(SIGNALS)


def test_cold_start_measures_all_and_defers() -> None:
    """Untrained table: gather everything, then review (safest)."""
    decision = QAgent().decide(scenario(), random.Random(1))
    assert decision.action == "review"
    assert len(decision.path) == 6
    assert decision.confidence == 0.0  # terminals tie at zero


def test_taught_table_greedy_walk_exits_early() -> None:
    """A one-measurement retake, the ARCHITECTURE 5.4 walk-through."""
    agent = QAgent()
    fail_state = state(blur=FAIL_FINAL)
    agent.q[START] = {check_action("blur"): 1.0}
    agent.q[fail_state] = {"retake": 1.0, "review": 0.2, "pass": 0.0}
    decision = agent.decide(scenario(blur=CLEAN - 0.85), random.Random(0))
    assert decision.action == "retake"
    assert decision.path == ("blur",)
    assert decision.confidence == pytest.approx(0.8, abs=1e-3)


def test_decide_walk_stays_within_twelve_measurements() -> None:
    """The structural budget of EXPERIMENT.md section 3."""
    decision = QAgent().decide(scenario(brightness=0.50), random.Random(2))
    assert len(decision.path) <= 6
    # reasons count each measurement, re-checks included
    assert len(decision.reasons) <= 12


def test_decide_requires_an_rng() -> None:
    """Readings are noisy by design; no silent oracle mode."""
    with pytest.raises(ValueError):
        QAgent().decide(scenario(), None)  # type: ignore[arg-type]


def test_recheck_ablation_never_rechecks_in_decision() -> None:
    """NoRecheck-Q walks the same world without re-check actions."""
    agent = QAgent(allow_recheck=False, name="no_recheck_q")
    decision = agent.decide(scenario(brightness=0.50), random.Random(3))
    assert len(decision.path) == 6
    assert all("re-checked" not in reason for reason in decision.reasons)


# --- JSON artifact (the Android-consumable deliverable) ---


def test_json_roundtrip_preserves_the_table(tmp_path) -> None:
    """save/load is lossless; meta rides along."""
    agent = QAgent(lr=0.5, gamma=0.9, name="no_recheck_q", allow_recheck=False)
    agent.update(START, check_action("blur"), -0.05, ALL_CLEAN, True)
    path = tmp_path / "table.json"
    agent.save(path)
    loaded = QAgent.load(path)
    assert loaded.to_dict() == agent.to_dict()
    assert loaded.lr == 0.5
    assert loaded.gamma == 0.9
    assert loaded.allow_recheck is False
    assert loaded.name == "no_recheck_q"


def test_artifact_is_readable_json(tmp_path) -> None:
    """Sorted keys, one space indent: diffable in review."""
    agent = QAgent()
    agent.update(START, check_action("blur"), -0.05, ALL_CLEAN, True)
    path = tmp_path / "table.json"
    agent.save(path)
    text = path.read_text(encoding="utf-8")
    assert '"format": "smallai-qtable"' in text
    data = json.loads(text)
    assert data["q"] == {"0,0,0,0,0,0": {"check:blur": pytest.approx(-0.015)}}


def test_from_dict_rejects_foreign_artifacts() -> None:
    """Guard rails on the load path."""
    with pytest.raises(ValueError):
        QAgent.from_dict({"format": "other"})
    with pytest.raises(ValueError):
        QAgent.from_dict({"format": "smallai-qtable", "version": 2})
    with pytest.raises(ValueError):
        QAgent.from_dict(
            {"format": "smallai-qtable", "version": 1, "q": {"9": {}}}
        )
    with pytest.raises(ValueError):
        QAgent.from_dict(
            {
                "format": "smallai-qtable",
                "version": 1,
                "q": {"0,0,0,0,0,0": {"fly:blur": 1.0}},
            }
        )
