"""Stage B (AI-267) — real image-quality signals through the offline gate.

Stage A scored a tabular Q-table over *synthetic* quality signals. Stage B
measures *real* quality signals from captured document images and applies the
discrete offline policy (RETAKE / REVIEW / PASS).

This module runs the full Stage B experiment:

  1. Build a deterministic corpus of *document-like* captures
     (`make_document`) — white paper with wavy handwriting strokes, so the
     synthetic corpus exercises the same feature space as real scans
     (paper background ~0.9 luma, dark ink strokes with crisp edges).
  2. Apply controlled degradations (`blur`, `add_noise`, `apply_exposure`).
  3. Run the document-aware offline gate over each level (`decide_offline`).
  4. Tabulate verdict rates (RETAKE / REVIEW / PASS) per level.

The experiment is deterministic (fixed seeds) and fully NumPy — no CV/ML
libraries, no GPU, no LLM. The verdict contract (RETAKE | REVIEW | PASS) is the
stable interface the Android / Kotlin runtime will consume; OCR and content
verification stay out of scope for this stage.

Real corpora (e.g. the synthetic medical prescription OCR images mirrored at
`chinmays18/medical-prescription-dataset`) are supported through
`load_images_from_dir` / `run_real_corpus`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .env import image_quality as iq

RETAKE, REVIEW, PASS = iq.RETAKE, iq.REVIEW, iq.PASS
DECISION_NAMES = iq.DECISION_NAMES

# Degradation sweep: (kind, level). kind in {"none", "blur", "noise", "exposure"}.
# Levels are calibrated against the real 2000-image prescription corpus: blur
# r=2 leaves text readable (PASS), blur r=8 sits in the designed borderline
# band (REVIEW), and darkness at 10% exposure is unusable (RETAKE).
SWEEP: tuple[tuple[str, float], ...] = (
    ("none", 0.0),
    ("blur", 2.0),
    ("blur", 8.0),
    ("exposure", 0.1),
)

SWEEP_H = 512
SWEEP_W = 384


def make_document(
    rng: np.random.Generator,
    h: int = SWEEP_H,
    w: int = SWEEP_W,
    *,
    lines_range: tuple[int, int] = (6, 11),
    word_len: tuple[float, float] = (0.05, 0.14),
) -> np.ndarray:
    """Deterministic document-like capture (H, W, 3), values in [0, 1].

    White paper (~0.9 luma, gentle illumination field) with dark handwriting-
    like strokes: lines of wavy scribbles with ~1 px stroke width, tuned so the
    synthetic corpus lives in the same feature space as real document captures
    (paper-dominated frame, ink_ratio ~1-2%, crisp stroke edges). Pure NumPy;
    deterministic under a fixed seeded `rng`.
    """
    paper = 0.90 + 0.02 * rng.random()
    yy, xx = np.mgrid[0:h, 0:w]
    xf = xx / max(w - 1, 1)
    yf = yy / max(h - 1, 1)
    # gentle illumination field: slight corner falloff
    illum = 1.0 - 0.10 * ((xf - 0.5) ** 2 + (yf - 0.5) ** 2) * 2.0
    paper_l = np.clip(paper * illum, 0.0, 1.0)

    img = np.repeat(paper_l[:, :, None], 3, axis=2)

    n_lines = int(rng.integers(lines_range[0], lines_range[1]))
    line_ys = np.linspace(0.10, 0.86, n_lines)
    ink_lo, ink_hi = 0.02, 0.12
    y_dev = 0.012 * h          # stroke-scale vertical wobble (px)
    for y0 in line_ys:
        # slight line tilt (captures are rarely perfectly axis-aligned)
        y_end = y0 + rng.uniform(-0.02, 0.02)
        n_words = int(rng.integers(2, 6))
        cursor = 0.06 + rng.random() * 0.05
        for _ in range(n_words):
            wlen = word_len[0] + rng.random() * (word_len[1] - word_len[0])
            x0, x1 = cursor * w, min((cursor + wlen) * w, 0.94 * w)
            cursor = x1 / w + 0.03 + rng.random() * 0.05
            if x1 - x0 < 3:
                continue
            t = np.linspace(x0, x1, max(int(x1 - x0) * 2, 4))
            phase = rng.uniform(0.0, 2.0 * np.pi)
            freq = rng.uniform(0.8, 2.2)
            base_y = y0 * h + (y_end - y0) * h * (t / w)
            y_path = base_y + y_dev * np.sin(2 * np.pi * freq * t / w + phase)
            ink_val = rng.uniform(ink_lo, ink_hi)
            # dashed near-solid pen: sample the path densely (real print
            # strokes are ~1 px at capture resolution), with slight pen jitter
            step = 0.75
            ts = np.arange(x0, x1, step)
            if ts.size == 0:
                continue
            y_s = np.interp(ts, t, y_path) + rng.uniform(-0.6, 0.6, size=ts.shape)
            ys = np.clip(np.round(y_s).astype(np.int64), 0, h - 1)
            xs = np.clip(np.round(ts).astype(np.int64), 0, w - 1)
            img[ys, xs] = np.minimum(img[ys, xs], ink_val)
    # mild per-pixel paper noise (grain)
    img = np.clip(img + rng.normal(0.0, 0.008, size=img.shape), 0.0, 1.0)
    return img


def make_capture(rng: np.random.Generator, h: int = SWEEP_H, w: int = SWEEP_W) -> np.ndarray:
    """Backwards-compatible alias: `make_capture` -> `make_document`."""
    return make_document(rng, h=h, w=w)


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
    h: int = SWEEP_H,
    w: int = SWEEP_W,
    seed: int = 0,
) -> dict:
    """Run the full Stage B degradation sweep; return per-level verdict stats."""
    rng = np.random.default_rng(seed)
    results: dict[str, dict] = {}
    for kind, level in SWEEP:
        imgs = np.stack(
            [corrupt(make_document(rng, h=h, w=w), kind, level, rng) for _ in range(n_images)]
        )
        verdicts = iq.decide_offline(imgs)
        sig = iq.extract(imgs)
        results[f"{kind}:{level:g}"] = {
            "kind": kind,
            "level": level,
            "n_images": int(n_images),
            "verdict_rates": verdict_distribution(verdicts),
            "mean_sharpness": float(np.mean(sig["sharpness"])),
            "mean_contrast": float(np.mean(sig["contrast"])),
            "mean_ink_ratio": float(np.mean(sig["ink_ratio"])),
            "mean_brightness": float(np.mean(sig["brightness"])),
            "mean_exposure": float(np.mean(sig["exposure"])),
        }
    return results


def run_real_corpus(directory: str | Path, n: int | None = None,
                    pattern: str = "*.png") -> dict:
    """Run the offline quality gate over real images in a directory.

    Used with real-world corpora such as the synthetic medical prescription
    OCR images (Kaggle `priyanshuyav13/synthetic-medical-prescription-ocr-dataset`,
    mirrored at `chinmays18/medical-prescription-dataset`). Images are
    processed **one at a time** — sizes may vary within the corpus and memory
    stays flat even for thousands of captures. Returns verdict rates and mean
    quality signals over the (first `n`) images.
    """
    from PIL import Image

    d = Path(directory)
    files = sorted(d.glob(pattern))
    if n is not None:
        files = files[:n]

    verdicts: list[int] = []
    sums = {"sharpness": 0.0, "contrast": 0.0, "ink_ratio": 0.0, "brightness": 0.0}
    count = 0
    for f in files:
        with Image.open(f) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
        sig = iq.extract(arr[None])
        verdicts.append(int(iq.decide(sig)[0]))
        for k in sums:
            sums[k] += float(sig[k][0])
        count += 1

    if count == 0:
        return {"n_images": 0, "verdict_rates": verdict_distribution(np.zeros(0, dtype=np.int64)),
                "mean_sharpness": float("nan"), "mean_contrast": float("nan"),
                "mean_ink_ratio": float("nan"), "mean_brightness": float("nan")}
    return {
        "n_images": count,
        "verdict_rates": verdict_distribution(np.array(verdicts, dtype=np.int64)),
        "mean_sharpness": sums["sharpness"] / count,
        "mean_contrast": sums["contrast"] / count,
        "mean_ink_ratio": sums["ink_ratio"] / count,
        "mean_brightness": sums["brightness"] / count,
    }


def verdict_distribution(verdicts: np.ndarray) -> dict[str, float]:
    """Return the fraction of RETAKE / REVIEW / PASS verdicts."""
    v = np.asarray(verdicts).reshape(-1)
    return {name: float(np.mean(v == idx)) for idx, name in enumerate(DECISION_NAMES)}


def load_images_from_dir(directory: str | Path, pattern: str = "*.png", n: int | None = None) -> np.ndarray:
    """Load images from a directory (sorted by name) -> (N, H, W, 3) float in [0, 1].

    Uses Pillow only for IO; all quality math stays pure NumPy. `n` truncates
    the batch to the first N images (after sorting).
    """
    from PIL import Image

    d = Path(directory)
    files = sorted(d.glob(pattern))
    if n is not None:
        files = files[:n]
    imgs: list[np.ndarray] = []
    for f in files:
        with Image.open(f) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        imgs.append(arr)
    if not imgs:
        return np.zeros((0, 32, 32, 3), dtype=np.float64)
    return np.stack(imgs, axis=0)
