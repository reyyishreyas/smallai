"""Stage B (AI-267): real image-quality signals + offline decision policy.

Stage A of the CLEF-inspired decision-policy PoC scored a tabular Q-table over
*synthetic* image-quality signals. Stage B moves to *real* image-quality signals
measured directly from the captured document image, and keeps the discrete
decision (pass / retake / review) as a policy that is separate from OCR /
content verification (Victor's PoC scope).

Everything here is implemented from scratch with NumPy only — no CV/ML
libraries and no LLM in the inference path. The policy stays tiny and fully
offline, which is the property the PoC is validating.

Public surface:
  extract(images)        -> {sharpness, brightness, exposure} per image (real signals)
  decide(signals, ...)   -> RETAKE | REVIEW | PASS per image
  decide_offline(images) -> decide(extract(images))  (one-shot offline gate)
  verdict_name(v)        -> "RETAKE" | "REVIEW" | "PASS"
"""
from __future__ import annotations

import numpy as np

RETAKE, REVIEW, PASS = 0, 1, 2
N_DECISIONS = 3
DECISION_NAMES = ("RETAKE", "REVIEW", "PASS")

# Stage B quality gates (real-signal thresholds).
RETAKE_SHARP_MIN = 0.02   # mean gradient magnitude below this -> too blurry
EXPOSURE_MAX = 0.5        # |luma - 0.5| above this -> over/under exposed


def _luma(img: np.ndarray) -> np.ndarray:
    """Per-image mean luma from an (N, H, W, 3) float image batch."""
    img = np.asarray(img, dtype=np.float64)
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def _grad_mag(img: np.ndarray) -> np.ndarray:
    """Per-image mean gradient magnitude (proxy for sharpness; high = sharp)."""
    l = _luma(img)  # (N, H, W)
    gx = np.abs(np.diff(l, axis=1))  # (N, H, W-1)
    gy = np.abs(np.diff(l, axis=2))  # (N, H-1, W)
    gx = gx.mean(axis=(1, 2))        # (N,)
    gy = gy.mean(axis=(1, 2))        # (N,)
    return 0.5 * (gx + gy)


def _brightness(img: np.ndarray) -> np.ndarray:
    """Per-image mean luma."""
    l = _luma(img)
    return l.mean(axis=tuple(range(1, l.ndim)))


def extract(images: np.ndarray) -> dict[str, np.ndarray]:
    """Real image-quality signals for a batch of images (N, H, W, 3).

    Returns a dict of per-image signal arrays:
      sharpness : mean gradient magnitude (blur proxy; low -> blurry)
      brightness: mean luma
      exposure  : |mean luma - 0.5| (over/under exposure)
    """
    img = np.asarray(images, dtype=np.float64)
    sharpness = _grad_mag(img)
    brightness = _brightness(img)
    exposure = np.abs(brightness - 0.5)
    return {"sharpness": sharpness, "brightness": brightness, "exposure": exposure}


def decide(
    signals: dict[str, np.ndarray],
    *,
    retake_sharp_min: float = RETAKE_SHARP_MIN,
    exposure_max: float = EXPOSURE_MAX,
    bright_low: float = 0.12,
    bright_high: float = 0.88,
    review_sharp_max: float = 0.05,
    review_exposure_max: float = 0.3,
    review_bright_low: float = 0.2,
    review_bright_high: float = 0.8,
) -> np.ndarray:
    """Stage B offline decision: pass / retake / review from real signals.

    - retake: image is unusable (blurry, over/under exposed, near-black/bright).
    - review : usable but borderline.
    - pass   : usable.

    All thresholds are real-signal-derived and overridable per call.
    """
    sharp = np.asarray(signals["sharpness"], dtype=np.float64)
    bright = np.asarray(signals["brightness"], dtype=np.float64)
    expo = np.asarray(signals["exposure"], dtype=np.float64)
    retake = (sharp < retake_sharp_min) | (expo > exposure_max) | (bright < bright_low) | (bright > bright_high)
    review = (~retake) & ((sharp < review_sharp_max) | (expo > review_exposure_max) | (bright < review_bright_low) | (bright > review_bright_high))
    return np.where(retake, RETAKE, np.where(review, REVIEW, PASS)).astype(np.int64)


def decide_offline(images: np.ndarray, **kwargs) -> np.ndarray:
    """Full offline decision gate: images -> verdict (RETAKE | REVIEW | PASS).

    This is the single entry point the field-officer mobile runtime (and the
    PoC CLI) uses: measure real quality signals from the captured image, then
    apply the discrete offline policy.
    """
    return decide(extract(images), **kwargs)


def verdict_name(verdict) -> str:
    """Human-readable name for a verdict value (RETAKE / REVIEW / PASS)."""
    v = int(np.asarray(verdict).reshape(-1)[0]) if np.asarray(verdict).size else RETAKE
    return DECISION_NAMES[v]
