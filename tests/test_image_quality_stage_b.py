"""Stage B tests: real image-quality signals + offline decision policy."""
from __future__ import annotations

import numpy as np

from smallai import stage_b as sb
from smallai.env import image_quality as iq


def _img(shape, value=0.5):
    return np.full((shape[0], shape[1], shape[2]), float(value), dtype=np.float64)


def test_extract_signals_shapes():
    imgs = np.random.default_rng(0).random((4, 8, 8, 3))
    sig = iq.extract(imgs)
    assert set(sig) == {"sharpness", "brightness", "exposure"}
    assert sig["sharpness"].shape == (4,)
    assert sig["brightness"].shape == (4,)
    assert sig["exposure"].shape == (4,)


def test_decide_retakes_blurry():
    # perfectly uniform (zero gradient) image -> blurry -> retake
    sig = iq.extract(np.full((1, 8, 8, 3), 0.5))
    out = iq.decide(sig)
    assert out.shape == (1,)
    assert out[0] == iq.RETAKE


def test_decide_passes_sharp_image():
    rng = np.random.default_rng(1)
    imgs = rng.random((2, 16, 16, 3))
    sig = iq.extract(imgs)
    out = iq.decide(sig)
    # random noise images are sharp -> likely pass
    assert out.shape == (2,)


def test_decision_labels_valid():
    assert iq.RETAKE in (iq.RETAKE, iq.REVIEW, iq.PASS)
    assert iq.N_DECISIONS == 3
    assert set(iq.DECISION_NAMES) == {"RETAKE", "REVIEW", "PASS"}
    assert len({iq.RETAKE, iq.REVIEW, iq.PASS}) == 3


def test_verdict_name_mapping():
    assert iq.verdict_name(iq.RETAKE) == "RETAKE"
    assert iq.verdict_name(iq.REVIEW) == "REVIEW"
    assert iq.verdict_name(iq.PASS) == "PASS"


def test_decide_offline_batch():
    imgs = np.stack([sb.make_capture(np.random.default_rng(0), h=32, w=24) for _ in range(3)])
    verdicts = iq.decide_offline(imgs)
    assert verdicts.shape == (3,)
    assert all(int(v) in (iq.RETAKE, iq.REVIEW, iq.PASS) for v in verdicts)


def test_corrupt_exposure_dark_retakes():
    img = sb.make_capture(np.random.default_rng(0), h=32, w=24)
    dark = sb.apply_exposure(img, 0.1)
    assert int(iq.decide_offline(dark[None])[0]) == iq.RETAKE


def test_sweep_levels():
    res = sb.run_sweep(n_images=8, h=32, w=24, seed=0)
    # pristine -> PASS, heavy blur -> REVIEW, dark exposure -> RETAKE
    assert res["none:0"]["verdict_rates"]["PASS"] > 0.9
    assert res["blur:8"]["verdict_rates"]["REVIEW"] > 0.9
    assert res["exposure:0.1"]["verdict_rates"]["RETAKE"] > 0.9


def test_sweep_deterministic():
    a = sb.run_sweep(n_images=8, h=32, w=24, seed=0)
    b = sb.run_sweep(n_images=8, h=32, w=24, seed=0)
    assert set(a) == set(b)
    for k in a:
        assert a[k]["verdict_rates"] == b[k]["verdict_rates"]
        assert a[k]["mean_sharpness"] == b[k]["mean_sharpness"]
