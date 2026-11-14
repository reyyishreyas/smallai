"""Tests for the Stage B-Q tabular Q-learning capture gate (experimental)."""

import numpy as np

from smallai import stage_b_q as sbq
from smallai.env import image_quality as iq


def test_quantize_signals_bins():
    sig = {
        "sharpness": np.array([0.02, 0.07, 0.11, 0.16, 0.30]),
        "ink_ratio": np.array([0.0, 0.0025, 0.25, 0.005, 0.25]),
        "brightness": np.array([0.05, 0.8, 1.0, 0.8, 1.0]),
        "contrast": np.array([0.05, 0.25, 0.9, 0.25, 0.9]),
    }
    states = sbq.quantize_signals(sig)
    assert states[0] == (0, 0, 0, 0)  # all bottom buckets
    assert states[1] == (1, 1, 3, 1)  # just above the edge cutoffs
    assert states[3] == (3, 2, 3, 1)  # mid buckets
    assert states[4] == (4, 2, 3, 2)  # top sharpness bucket
    # dense print (ink 0.25) is NOT an outlier bucket — no upper ink cut
    assert len(set(states)) == 5
    assert sbq.STATE_EDGES["ink_ratio"] == (0.002, 0.003)


def test_reward_matrix_is_asymmetric():
    # forwarding an unusable capture must cost more than a wasted re-take
    assert sbq.REWARD[iq.PASS, iq.RETAKE] < sbq.REWARD[iq.RETAKE, iq.PASS]
    # correct decisions are the only zero-cost ones
    assert (np.diag(sbq.COST) == 0).all()


def test_q_agent_learns_sweep_truth():
    states, experts = sbq.samples_from_sweep(n_docs=6, seed=1)
    agent = sbq.QCaptureAgent(seed=2)
    agent.learn(states, experts, epochs=40)
    actions = agent.predict(states)
    assert np.mean([a == e for a, e in zip(actions, experts, strict=True)]) >= 0.95
    # the whole table stays tiny
    st = agent.stats()
    assert st["n_params"] <= st["max_possible_states"] * 3
    assert st["n_params"] < 1000


def test_q_agent_deterministic():
    states, experts = sbq.samples_from_sweep(n_docs=4, seed=3)
    a1 = sbq.QCaptureAgent(seed=7)
    a1.learn(states, experts, epochs=10)
    a2 = sbq.QCaptureAgent(seed=7)
    a2.learn(states, experts, epochs=10)
    assert a1.q.keys() == a2.q.keys()
    assert all(np.array_equal(a1.q[s], a2.q[s]) for s in a1.q)


def test_agent_predicts_hard_levels():
    # the sweep's extreme states must map to their labels after training
    states, experts = sbq.samples_from_sweep(n_docs=12, seed=0)
    agent = sbq.QCaptureAgent(seed=0)
    agent.learn(states, experts, epochs=60)
    dark = [s for s, e in zip(states, experts, strict=True) if s[2] == 0]  # bright bucket 0
    sharp = [s for s, e in zip(states, experts, strict=True) if s[0] >= 4]  # very sharp
    got_dark = {agent.act(s, greedy=True) for s in dark}
    got_sharp = {agent.act(s, greedy=True) for s in sharp}
    assert got_dark == {iq.RETAKE}
    assert got_sharp == {iq.PASS}


def test_evaluate_metrics():
    actions = [iq.PASS, iq.RETAKE, iq.REVIEW]
    experts = [iq.PASS, iq.PASS, iq.REVIEW]
    m = sbq.evaluate(actions, experts)
    assert m["agreement"] == 2 / 3
    # one PASS-on-RETAKE is the expensive mistake
    assert m["mean_cost"] > 0


def test_compare_policies_counts_disagreements():
    fixed = [iq.PASS, iq.PASS, iq.RETAKE]
    q = [iq.PASS, iq.REVIEW, iq.RETAKE]
    c = sbq.compare_policies(fixed, q)
    assert c["n"] == 3
    assert c["n_disagreements"] == 1
    assert c["agreement"] == 2 / 3
    assert c["pairs"][iq.PASS][iq.REVIEW] == 1


def test_per_state_report_breaks_down_labels_and_safety():
    states = [(0, 0, 0, 0), (0, 0, 0, 0), (1, 2, 3, 2), (1, 2, 3, 2)]
    experts = [iq.PASS, iq.REVIEW, iq.RETAKE, iq.PASS]
    fixed = [iq.PASS, iq.PASS, iq.RETAKE, iq.PASS]
    q = [iq.PASS, iq.REVIEW, iq.RETAKE, iq.RETAKE]
    rep = sbq.per_state_report(states, experts, fixed, q)

    assert rep["n_samples"] == 4
    assert rep["n_states_observed"] == 2
    assert rep["q_vs_fixed"]["n_disagreements"] == 2
    # Q flagged the mixed-label image; the fixed gate forwarded it
    assert rep["safety"]["q"]["pass_on_review"] == 0
    assert rep["safety"]["fixed"]["pass_on_review"] == 1
    # Q rejected a PASS-labelled image: a wasted recapture, not a contract break
    assert rep["safety"]["q"]["retake_on_pass"] == 1

    row = {tuple(r["state"]): r for r in rep["per_state"]}[(0, 0, 0, 0)]
    assert row["n"] == 2
    assert row["experts"] == {"RETAKE": 0, "REVIEW": 1, "PASS": 1}
    assert row["disagreements"] == 1
    # the Q policy matched both labels in this state; the fixed gate did not
    assert row["label_disagreements"] == 0


def test_disagreement_report_falls_back_to_sweep(tmp_path):
    rep = sbq.disagreement_report(
        npz_path=tmp_path / "absent.npz",
        real_ladder_dir=tmp_path / "absent",
    )
    assert rep["source"].startswith("sweep")
    assert rep["n_samples"] == 96
    assert rep["n_states_observed"] == len(rep["per_state"])
    assert set(rep["q_vs_fixed"]) == {"n", "agreement", "n_disagreements", "pairs"}
    assert 0.0 <= rep["q_vs_fixed"]["agreement"] <= 1.0


def test_run_experiment_reports_policy_comparison(tmp_path):
    r = sbq.run_experiment(
        npz_path=tmp_path / "absent.npz",
        real_ladder_dir=tmp_path / "absent",
    )
    assert r["q_vs_fixed"]["n"] == r["n_samples"]
    assert 0.0 <= r["q_vs_fixed"]["agreement"] <= 1.0
    assert r["n_training_samples"] == r["n_samples"]


def test_check_images_with_q(tmp_path):
    # the learned policy should route a pristine doc to PASS, and expose the
    # quantized state + per-action Q-values alongside the fixed-gate verdict
    from PIL import Image

    from smallai import stage_b as sb

    doc = (sb.make_document(np.random.default_rng(1)) * 255).astype(np.uint8)
    p = tmp_path / "doc.png"
    Image.fromarray(doc, "RGB").save(p)
    res = sbq.check_images_with_q([p], npz_path=tmp_path / "absent.npz")  # sweep fallback
    r = res[0]
    assert r["verdict_q"] == "PASS"
    assert r["verdict_fixed"] == "PASS"  # policies agree on an obvious doc
    assert len(r["state"]) == 4
    assert max(r["q_values"]) > min(r["q_values"])  # the agent actually learned


def test_resolve_state_generalizes_or_fails_closed():
    states, experts = sbq.samples_from_sweep(n_docs=8, seed=0)
    agent = sbq.QCaptureAgent(seed=0)
    agent.learn(states, experts, epochs=40)
    trained = list(agent.q)
    s = trained[0]
    # perturb exactly one bucket -> still resolvable
    for i in range(4):
        for delta in (-1, 1):
            nov = list(s)
            nov[i] = max(nov[i] + delta, 0)
            nov = tuple(nov)
            if nov == s or nov in agent.q:
                continue
            resolved, d = agent.resolve_state(nov)
            assert resolved == s and d == 1  # snaps to the trained neighbour
            break
        else:
            continue
        break
    # find a truly novel state (>= 2 L1 bucket units from everything trained)
    far = None
    for cand in [(4, 2, 3, 2), (0, 0, 0, 0), (2, 2, 2, 2)]:
        d_best = min(sum(abs(a - b) for a, b in zip(cand, t, strict=False)) for t in agent.q)
        if d_best >= 2:
            far = cand
            break
    if far is not None:
        assert agent.resolve_state(far) == (None, None)
