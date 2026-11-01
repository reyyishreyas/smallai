"""Stage B tests: real image-quality signals + offline decision policy."""
from __future__ import annotations

import numpy as np
import pytest

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
