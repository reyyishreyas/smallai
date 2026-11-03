"""Stage B (AI-267): real image-quality signals + offline decision policy (v3).

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

v2 therefore measures quality *where the content is*.

v3 — resolution invariance + supervised calibration
----------------------------------------------------
The v2 signals were calibrated on n=3 images and turned out resolution-
dependent (sharpness scales with capture size). v3:

3. resamples every capture to a fixed analysis grid (max side ANALYSIS_MAX_SIDE,
   pure-NumPy bilinear) before measuring, so thresholds transfer across devices;
4. replaces whole-frame luminance-percentile contrast with **ink-region
   contrast** (bg − p25 of the ink pixels) — percentile-rank contrast breaks
   whenever ink covers fewer pixels than the percentile (both the 1% p1-cliff
   and the 0.5% p0.5-cliff verified on real data);
5. retunes all thresholds **supervised on the full 2000-image Kaggle corpus**
   (`dataset/images` + its per-image blur labels): normalized sharpness
   separates blur-labelled (max 0.092) from sharp-labelled (min 0.208) with no
   overlap. Cuts: RETAKE < 0.08, REVIEW < 0.13 — zero false RETAKEs on the 952
   sharp-labelled images and *no blurred capture ever passes* (everything
   blurred-labelled lands in RETAKE or REVIEW).

v3.1 — the readable-blur band (field evidence)
----------------------------------------------
Real field captures vindicated widening the REVIEW band downward: a blurry-but
-human-readable printed-form photo measured sharpness 0.088 — inside the
blur corpus's own class range, yet plainly legible. Camera photos of printed
pages degrade differently from the synthetic blur kernel, so the hard-RETAKE
cut was lowered 0.10 → 0.08: below it text is genuinely gone; the 0.08–0.13
zone is REVIEW (flag, let a human judge). The supervised guarantees hold
unchanged: zero sharp-labelled false rejects, zero blurred-labelled passes.

Signals (all measured on the fixed analysis grid):

  sharpness : mean gradient magnitude inside a 1-px dilation of the ink mask
              (fallback: whole-frame mean gradient when the mask is empty)
  contrast  : bg − p25(ink pixels) — ink-region luminance contrast, coverage-
              independent (whole-frame p99.5 − p0.5 only as no-ink fallback)
  ink_ratio : fraction of pixels darker than (p90 luma − 0.25) — the "is there
              actually content on the page" detector; a blank page or a photo
              of a wall has ink_ratio ~ 0
  brightness: mean luma (feeds the too-dark gate)
  exposure  : |mean luma − 0.5| (reporting aid)

Decisions use sharpness + ink_ratio as the primary RETAKE evidence and
brightness/contrast/ink for the REVIEW band; there is deliberately *no*
high-brightness gate — over-bright paper is normal (white page + dark ink is
what a readable document looks like), and truly blank captures trip ink_ratio.

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
ANALYSIS_MAX_SIDE = 512  # images are resampled to this max side before
                         # measuring, so thresholds are resolution-invariant
                         # (calibrated against the 2000-image corpus)
INK_MARGIN = 0.25   # ink pixel: luma < (p90 luma) - INK_MARGIN
INK_BG_PCT = 90     # background estimate = 90th percentile of luma
MIN_INK_FOR_CONTRAST = 16  # min ink pixels before contrast uses the ink region
CONTRAST_PCTS = (0.5, 99.5)  # whole-frame fallback when no ink is detectable

# ---------------------------------------------------------------------------
# Gate thresholds v3.1 — supervised on the full 2000-image Kaggle corpus
# (metadata blur labels), real mirror trio, and field evidence:
#   normalized sharpness separates the classes cleanly:
#     blur-labelled: 0.02-0.092 | readable-blur band: 0.08-0.13 | readable: > 0.13
#     (blur=0 corpus floor 0.208; 0 false retakes on all 952 sharp-labelled
#      images at cut 0.08; nothing blurred-labelled ever reaches PASS)
#   v3.1: hard-RETAKE lowered 0.10 -> 0.08 after a real human-readable
#   printed-form photo measured 0.088 — camera blur ≠ the synthetic blur
#   kernel, so 0.08-0.10 is REVIEW ("flag, human judges"), not RETAKE.
#   ink_ratio catches blank/dark-crushed/wall captures (no-content),
#   contrast separates washed-out-but-readable faint print into REVIEW.
# ---------------------------------------------------------------------------
RETAKE_BRIGHT_MIN = 0.10    # mean luma below this -> too dark to read
RETAKE_INK_MIN = 0.002      # less ink than this -> blank page / wall / crushed
RETAKE_SHARP_MIN = 0.08     # text-edge strength below this -> unreadably blurry
REVIEW_BRIGHT_MIN = 0.25    # dim but possibly readable
REVIEW_CONTRAST_MIN = 0.18  # washed-out print / crushed capture
REVIEW_INK_MIN = 0.003      # near-blank content
REVIEW_SHARP_MIN = 0.13     # soft text edges (mild blur / faint ink)


def _luma(img: np.ndarray) -> np.ndarray:
    """Per-pixel BT.709 luma from an (N, H, W, 3) float image batch."""
    img = np.asarray(img, dtype=np.float64)
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def _resize_bilinear(l: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Pure-NumPy bilinear resample of an (N, H, W) luma stack (area-preserving
    enough for quality signals; keeps thresholds resolution-invariant).

    Corner gathers use adjacent advanced indices (y, x in one expression) —
    mixing advanced indices with a basic slice in between would move the
    broadcast dims to the front and mis-index (NumPy indexing rule).
    """
    _, h, w = l.shape
    if h == out_h and w == out_w:
        return l
    yy = np.linspace(0.0, max(h - 1, 1), out_h)
    xx = np.linspace(0.0, max(w - 1, 1), out_w)
    y0 = np.floor(yy).astype(np.int64)
    x0 = np.floor(xx).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    y0 = np.minimum(y0, h - 1)
    x0 = np.minimum(x0, w - 1)
    wy = (yy - y0)[None, :, None]     # (1, out_h, 1)
    wx = (xx - x0)[None, None, :]     # (1, 1, out_w)
    tl = l[:, y0[:, None], x0[None, :]]  # (N, out_h, out_w) top-left
    tr = l[:, y0[:, None], x1[None, :]]  # top-right
    bl = l[:, y1[:, None], x0[None, :]]  # bottom-left
    br = l[:, y1[:, None], x1[None, :]]  # bottom-right
    out = (tl * (1 - wy) * (1 - wx) + tr * (1 - wy) * wx
           + bl * wy * (1 - wx) + br * wy * wx)
    return out


def _normalize_luma(img: np.ndarray) -> np.ndarray:
    """Batch (N, H, W, 3) -> luma stack (N, H', W') on a fixed analysis grid
    (max side ANALYSIS_MAX_SIDE, aspect preserved, uniform per batch)."""
    l = _luma(img)  # (N, H, W)
    h, w = l.shape[-2], l.shape[-1]
    scale = ANALYSIS_MAX_SIDE / max(h, w)
    out_h = max(round(h * scale), 16)
    out_w = max(round(w * scale), 16)
    if h == out_h and w == out_w:
        return l
    return _resize_bilinear(l, out_h, out_w)


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

    Images are first resampled to a fixed analysis grid (max side
    ANALYSIS_MAX_SIDE, bilinear), so the signals — and therefore the gate
    thresholds — are resolution-invariant across capture scales.

    Returns a dict of per-image signal arrays:
      sharpness : mean gradient magnitude within the ink neighbourhood
                  (document text-edge strength; low -> blurry/unreadable)
      contrast  : bg - p25(ink pixels) — ink-region luminance contrast,
                  measured where the content is (ink-coverage independent;
                  whole-frame p99.5 - p0.5 only as a no-ink fallback)
      ink_ratio : fraction of pixels darker than (p90 luma - 0.25)
      brightness: mean luma
      exposure  : |mean luma - 0.5| (reporting aid)
    """
    img = np.asarray(images, dtype=np.float64)
    if img.ndim == 3:  # single grayscale image -> batch of 1 with a luma channel
        img = img[..., None]
    l = _normalize_luma(img)            # (N, H', W') on the analysis grid
    g = _grad_mag_map(l)                # (N, H', W')

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

    # Ink-region contrast: background minus the 25th-percentile ink luminance.
    # Coverage-independent (works with arbitrarily sparse strokes), unlike
    # whole-frame luminance percentiles whose rank depends on ink fraction.
    contrast = np.empty(img.shape[0], dtype=np.float64)
    for i in range(img.shape[0]):
        vals = l[i][ink_mask[i]]
        if vals.size >= MIN_INK_FOR_CONTRAST:
            contrast[i] = max(float(bg[i] - np.percentile(vals, 25)), 0.0)
        else:
            p_lo = np.percentile(l[i], CONTRAST_PCTS[0])
            p_hi = np.percentile(l[i], CONTRAST_PCTS[1])
            contrast[i] = max(float(p_hi - p_lo), 0.0)
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
    retake_ink_min: float = RETAKE_INK_MIN,
    retake_bright_min: float = RETAKE_BRIGHT_MIN,
    review_sharp_min: float = REVIEW_SHARP_MIN,
    review_contrast_min: float = REVIEW_CONTRAST_MIN,
    review_ink_min: float = REVIEW_INK_MIN,
    review_bright_min: float = REVIEW_BRIGHT_MIN,
) -> np.ndarray:
    """Stage B offline decision: pass / retake / review from real signals.

    - retake: image is unusable — too dark, no detectable content (blank page,
      crushed or washed-out copy of nothing), or text edges too smeared to read.
    - review : usable but borderline (mild blur / faint ink / low contrast).
    - pass   : usable.

    There is deliberately no high-brightness gate: white paper is the normal
    background of a document capture; blank captures are detected by
    ink_ratio instead, and washed-out-but-legible print lands in REVIEW via
    the contrast band. All thresholds are overridable per call.
    """
    sharp = np.asarray(signals["sharpness"], dtype=np.float64)
    contrast = np.asarray(signals["contrast"], dtype=np.float64)
    ink = np.asarray(signals["ink_ratio"], dtype=np.float64)
    bright = np.asarray(signals["brightness"], dtype=np.float64)

    retake = (
        (bright < retake_bright_min)
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
