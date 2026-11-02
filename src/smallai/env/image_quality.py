"""Stage B (AI-267): real image-quality signals + offline decision policy (v2).

Stage A of the CLEF-inspired decision-policy PoC scored a tabular Q-table over
*synthetic* image-quality signals. Stage B moves to *real* image-quality signals
measured directly from the captured document image, and keeps the discrete
decision (pass / retake / review) as a policy that is separate from OCR /
content verification (Victor's PoC scope).

Everything here is implemented from scratch with NumPy only — no CV/ML
libraries and no LLM in the inference path. The policy stays tiny and fully
offline, which is the property the PoC is validating.

v2 — document-aware feature engineering
---------------------------------------
The v1 gate (mean gradient + bright>0.88) mis-flagged genuinely readable
prescription scans (dark ink on white paper) as RETAKE:

1. mean gradient is diluted by the dominant flat background — a document is
   >95% paper, so the *text edges* (the only sharpness evidence) vanish into
   the average;
2. `brightness > 0.88 -> RETAKE` reads white paper as over-exposure — but a
   white page with dark ink is exactly what a readable document looks like.

v2 therefore measures quality *where the content is*:

  sharpness : mean gradient magnitude inside a 1-px dilation of the ink mask
              (fallback: whole-frame mean gradient when the mask is empty)
  contrast  : p99.5 - p0.5 of luma (robust luminance range; collapses when
              the capture is flat or crushed by exposure)
  ink_ratio : fraction of pixels darker than (p90 luma - 0.25) — the "is there
              actually content on the page" detector; a blank page or a photo
              of a wall has ink_ratio ~ 0
  brightness: mean luma (kept for reporting + the too-dark gate)
  exposure  : |mean luma - 0.5| (kept for reporting / downstream use)

Decisions use contrast, ink_ratio and per-region sharpness as the primary
evidence; there is deliberately *no* high-brightness gate — over-bright paper
is normal, and a truly blank/blown capture trips ink_ratio instead.

Public surface:
  extract(images)        -> {sharpness, contrast, ink_ratio, brightness, exposure}
  decide(signals, ...)   -> RETAKE | REVIEW | PASS per image
  decide_offline(images) -> decide(extract(images))  (one-shot offline gate)
  verdict_name(v)        -> "RETAKE" | "REVIEW" | "PASS"
"""
from __future__ import annotations

import numpy as np

RETAKE, REVIEW, PASS = 0, 1, 2
N_DECISIONS = 3
DECISION_NAMES = ("RETAKE", "REVIEW", "PASS")

# ---------------------------------------------------------------------------
# Signal-measurement constants (fixed, not tuned per call)
# ---------------------------------------------------------------------------
INK_MARGIN = 0.25   # ink pixel: luma < (p90 luma) - INK_MARGIN
INK_BG_PCT = 90     # background estimate = 90th percentile of luma
CONTRAST_PCTS = (0.5, 99.5)  # robust luminance range (robust to <1% ink)

# ---------------------------------------------------------------------------
# Gate thresholds (calibrated on the real prescription corpus +
# controlled blur/exposure degradations of it; all overridable per call)
# ---------------------------------------------------------------------------
RETAKE_BRIGHT_MIN = 0.10    # mean luma below this -> too dark to read
RETAKE_CONTRAST_MIN = 0.10  # luminance range below this -> flat, nothing readable
RETAKE_INK_MIN = 0.002      # less ink than this -> blank page / no content
RETAKE_SHARP_MIN = 0.055    # text-edge strength below this -> unreadably blurry
REVIEW_BRIGHT_MIN = 0.25    # dim but possibly readable
REVIEW_CONTRAST_MIN = 0.18  # weak but possibly readable
REVIEW_INK_MIN = 0.005      # sparse content
REVIEW_SHARP_MIN = 0.095    # soft text edges (mild blur / faint ink)


def _luma(img: np.ndarray) -> np.ndarray:
    """Per-pixel BT.709 luma from an (N, H, W, 3) float image batch."""
    img = np.asarray(img, dtype=np.float64)
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def _grad_mag_map(l: np.ndarray) -> np.ndarray:
    """Per-pixel gradient magnitude of an (N, H, W) luma stack."""
    gx = np.abs(np.diff(l, axis=2))   # (N, H, W-1)
    gy = np.abs(np.diff(l, axis=1))   # (N, H-1, W)
    g = np.zeros_like(l)
    g[..., :, 1:] += 0.5 * gx
    g[..., 1:, :] += 0.5 * gy
    return g


def _dilate(mask: np.ndarray) -> np.ndarray:
    """1-px 4-connected dilation of an (N, H, W) boolean mask."""
    m = mask.copy()
    for axis, shift in ((1, 1), (1, -1), (2, 1), (2, -1)):
        m |= np.roll(mask, shift, axis=axis)
    return m


def extract(images: np.ndarray) -> dict[str, np.ndarray]:
    """Document-aware quality signals for a batch of images (N, H, W, 3).

    Returns a dict of per-image signal arrays:
      sharpness : mean gradient magnitude within the ink neighbourhood
                  (document text-edge strength; low -> blurry/unreadable)
      contrast  : p99.5 - p0.5 of luma (robust luminance range; robust to
                  documents with under 1% ink)
      ink_ratio : fraction of pixels darker than (p90 luma - 0.25)
      brightness: mean luma
      exposure  : |mean luma - 0.5| (reporting aid)
    """
    img = np.asarray(images, dtype=np.float64)
    l = _luma(img)                      # (N, H, W)
    g = _grad_mag_map(l)                # (N, H, W)

    bg = np.percentile(l, INK_BG_PCT, axis=(1, 2))          # (N,)
    ink_mask = l < (bg[:, None, None] - INK_MARGIN)
    ink_ratio = ink_mask.mean(axis=(1, 2))                  # (N,)

    near = _dilate(ink_mask)
    near_count = near.sum(axis=(1, 2))
    ink_grad_sum = (g * near).sum(axis=(1, 2))
    per_count = np.maximum(near_count, 1)
    ink_sharpness = ink_grad_sum / per_count
    # Fallback for images with no ink mask at all (uniform / crushed captures):
    global_sharpness = g.mean(axis=(1, 2))
    sharpness = np.where(near_count > 0, ink_sharpness, global_sharpness)

    p_lo = np.percentile(l, CONTRAST_PCTS[0], axis=(1, 2))
    p_hi = np.percentile(l, CONTRAST_PCTS[1], axis=(1, 2))
    contrast = p_hi - p_lo
    brightness = l.mean(axis=(1, 2))
    exposure = np.abs(brightness - 0.5)

    return {
        "sharpness": sharpness,
        "contrast": contrast,
        "ink_ratio": ink_ratio,
        "brightness": brightness,
        "exposure": exposure,
    }


def decide(
    signals: dict[str, np.ndarray],
    *,
    retake_sharp_min: float = RETAKE_SHARP_MIN,
    retake_contrast_min: float = RETAKE_CONTRAST_MIN,
    retake_ink_min: float = RETAKE_INK_MIN,
    retake_bright_min: float = RETAKE_BRIGHT_MIN,
    review_sharp_min: float = REVIEW_SHARP_MIN,
    review_contrast_min: float = REVIEW_CONTRAST_MIN,
    review_ink_min: float = REVIEW_INK_MIN,
    review_bright_min: float = REVIEW_BRIGHT_MIN,
) -> np.ndarray:
    """Stage B offline decision: pass / retake / review from real signals.

    - retake: image is unusable — too dark, flat (no luminance range),
      no detectable content (blank page), or text edges too smeared to read.
    - review : usable but borderline (mild blur / faint ink / weak contrast).
    - pass   : usable.

    There is deliberately no high-brightness gate: white paper is the normal
    background of a document capture; blank captures are detected by
    ink_ratio instead. All thresholds are overridable per call.
    """
    sharp = np.asarray(signals["sharpness"], dtype=np.float64)
    contrast = np.asarray(signals["contrast"], dtype=np.float64)
    ink = np.asarray(signals["ink_ratio"], dtype=np.float64)
    bright = np.asarray(signals["brightness"], dtype=np.float64)

    retake = (
        (bright < retake_bright_min)
        | (contrast < retake_contrast_min)
        | (ink < retake_ink_min)
        | (sharp < retake_sharp_min)
    )
    review = (~retake) & (
        (bright < review_bright_min)
        | (contrast < review_contrast_min)
        | (ink < review_ink_min)
        | (sharp < review_sharp_min)
    )
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
