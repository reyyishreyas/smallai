"""Stage B (AI-267) — real image-quality signals through the offline gate.

Stage A scored a tabular Q-table over *synthetic* quality signals. Stage B
measures *real* quality signals from captured document images and applies the
discrete offline policy (RETAKE / REVIEW / PASS).

This module runs the full Stage B experiment:

  1. Build a deterministic corpus of document-capture-like images
     (`make_capture`) — sharp, mid-exposure images in [0, 1].
  2. Apply controlled degradations (`blur`, `add_noise`, `apply_exposure`).
  3. Run the real-signal offline gate over each level (`decide_offline`).
  4. Tabulate verdict rates (RETAKE / REVIEW / PASS) per level.

The experiment is deterministic (fixed seeds) and fully NumPy — no CV/ML
libraries, no GPU, no LLM. The verdict contract (RETAKE | REVIEW | PASS) is the
stable interface the Android / Kotlin runtime will consume; OCR and content
verification stay out of scope for this stage.
"""
from __future__ import annotations

import numpy as np

from .env import image_quality as iq

RETAKE, REVIEW, PASS = iq.RETAKE, iq.REVIEW, iq.PASS
DECISION_NAMES = iq.DECISION_NAMES

# Degradation sweep: (kind, level). kind in {"none", "blur", "noise", "exposure"}.
SWEEP: tuple[tuple[str, float], ...] = (
    ("none", 0.0),
    ("blur", 2.0),
    ("blur", 8.0),
    ("exposure", 0.1),
)


def make_capture(rng: np.random.Generator, h: int = 32, w: int = 24, luma: float = 0.0, amp: float = 2.2) -> np.ndarray:
    """Deterministic document-capture-like image (H, W, 3), values in [0, 1].

    Smooth 2-D luma field (product of two cosines) -> high gradient magnitude
    -> a *sharp* image, so the pristine level of the sweep is expected to pass
    the offline gate.
    """
    xg = np.linspace(0.0, 1.0, w, dtype=np.float64)[None, :]
    yg = np.linspace(0.0, 1.0, h, dtype=np.float64)[:, None]
    field = (0.5 + 0.5 * np.cos(2 * np.pi * xg)) * (0.5 + 0.5 * np.cos(2 * np.pi * yg))
    luma_img = np.clip(luma + amp * field, 0.0, 1.0)
    return np.stack([luma_img] * 3, axis=-1)


def blur(img: np.ndarray, r: float) -> np.ndarray:
    """Box blur (pure NumPy, separable via padded accumulation). Handles
    (H, W, 3) and batch (N, H, W, 3) images."""
    r = max(0, round(float(r)))
    if r == 0:
        return img.copy()
    lead = img.ndim - 2  # spatial axes are the last two
    pad_width = [(0, 0)] * lead + [(r, r), (r, r)]
    p = np.pad(img, pad_width, mode="reflect")
    out = np.zeros_like(img)
    H, W = img.shape[-2], img.shape[-1]
    for i in range(2 * r + 1):
        out += p[(slice(None),) * lead + (slice(i, i + H), slice(i, i + W))]
    return out / (2 * r + 1)


def add_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise, clipped to [0, 1]."""
    return np.clip(img + float(sigma) * rng.standard_normal(img.shape), 0.0, 1.0)


def apply_exposure(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale brightness (factor < 1 -> under-exposed, > 1 -> over-exposed)."""
    return np.clip(img * float(factor), 0.0, 1.0)


def corrupt(img: np.ndarray, kind: str, level: float, rng: np.random.Generator) -> np.ndarray:
    """Apply a degradation from the Stage B sweep."""
    kind = str(kind)
    level = float(level)
    if kind == "none":
        return img.copy()
    if kind == "blur":
        return blur(img, level)
    if kind == "noise":
        return add_noise(img, level, rng)
    if kind == "exposure":
        return apply_exposure(img, level)
    raise ValueError(f"unknown Stage B degradation kind: {kind!r}")


def run_sweep(
    *,
    n_images: int = 64,
    h: int = 32,
    w: int = 24,
    seed: int = 0,
) -> dict:
    """Run the full Stage B degradation sweep; return per-level verdict stats."""
    rng = np.random.default_rng(seed)
    results: dict[str, dict] = {}
    for kind, level in SWEEP:
        imgs = np.stack(
            [corrupt(make_capture(rng, h=h, w=w), kind, level, rng) for _ in range(n_images)]
        )
        verdicts = iq.decide_offline(imgs)
        sig = iq.extract(imgs)
        results[f"{kind}:{level:g}"] = {
            "kind": kind,
            "level": level,
            "n_images": int(n_images),
            "verdict_rates": verdict_distribution(verdicts),
            "mean_sharpness": float(np.mean(sig["sharpness"])),
            "mean_brightness": float(np.mean(sig["brightness"])),
            "mean_exposure": float(np.mean(sig["exposure"])),
        }
    return results


def verdict_distribution(verdicts: np.ndarray) -> dict[str, float]:
    """Return the fraction of RETAKE / REVIEW / PASS verdicts."""
    v = np.asarray(verdicts).reshape(-1)
    return {name: float(np.mean(v == idx)) for idx, name in enumerate(DECISION_NAMES)}
