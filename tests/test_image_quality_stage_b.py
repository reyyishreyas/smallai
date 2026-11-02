"""Stage B tests: real image-quality signals + offline decision policy (v2)."""
from __future__ import annotations

import numpy as np

from smallai import stage_b as sb
from smallai.env import image_quality as iq


def _img(shape, value=0.5):
    return np.full((shape[0], shape[1], shape[2]), float(value), dtype=np.float64)


def test_extract_signals_shapes():
    imgs = np.random.default_rng(0).random((4, 16, 16, 3))
    sig = iq.extract(imgs)
    assert set(sig) == {"sharpness", "contrast", "ink_ratio", "brightness", "exposure"}
    for key in ("sharpness", "contrast", "ink_ratio", "brightness", "exposure"):
        assert sig[key].shape == (4,)


def test_decide_retakes_uniform_image():
    # perfectly uniform image: flat + no content -> retake
    sig = iq.extract(np.full((1, 16, 16, 3), 0.5))
    out = iq.decide(sig)
    assert out.shape == (1,)
    assert out[0] == iq.RETAKE


def test_decide_retakes_blank_paper():
    # white page with no ink -> blank capture -> retake via ink_ratio
    sig = iq.extract(np.full((1, 64, 64, 3), 0.93))
    out = iq.decide(sig)
    assert out[0] == iq.RETAKE


def test_decide_passes_readable_document():
    # synthetic document capture with crisp handwriting -> PASS
    img = sb.make_document(np.random.default_rng(0))[None]
    out = iq.decide_offline(img)
    assert out[0] == iq.PASS


def test_decide_retakes_dark_capture():
    img = sb.make_document(np.random.default_rng(0))[None]
    dark = sb.apply_exposure(img, 0.1)
    assert iq.decide_offline(dark)[0] == iq.RETAKE


def test_contrast_feature_robust_to_low_ink():
    # regression: a document with <1% ink of pixels must not have its
    # luminance range collapse (p0.5/p99.5 percentiles, not p1/p99)
    img = np.full((1, 64, 64, 3), 0.9)
    # sprinkle ~30 dark pixels (0.7% of 4096) of text-like ink
    rng = np.random.default_rng(3)
    ys = rng.integers(8, 56, size=30)
    xs = rng.integers(8, 56, size=30)
    img[0, ys, xs] = 0.1
    sig = iq.extract(img)
    assert float(sig["ink_ratio"][0]) < 0.01
    assert float(sig["contrast"][0]) > 0.5


def test_decision_labels_valid():
    assert iq.RETAKE in {iq.RETAKE, iq.REVIEW, iq.PASS}
    assert iq.N_DECISIONS == 3
    assert set(iq.DECISION_NAMES) == {"RETAKE", "REVIEW", "PASS"}
    assert len({iq.RETAKE, iq.REVIEW, iq.PASS}) == 3


def test_verdict_name_mapping():
    assert iq.verdict_name(iq.RETAKE) == "RETAKE"
    assert iq.verdict_name(iq.REVIEW) == "REVIEW"
    assert iq.verdict_name(iq.PASS) == "PASS"


def test_decide_offline_batch():
    imgs = np.stack([sb.make_document(np.random.default_rng(i)) for i in range(3)])
    verdicts = iq.decide_offline(imgs)
    assert verdicts.shape == (3,)
    assert all(int(v) in (iq.RETAKE, iq.REVIEW, iq.PASS) for v in verdicts)


def test_sweep_levels():
    res = sb.run_sweep(n_images=16, seed=0)
    # pristine & mild blur -> PASS; heavy blur -> REVIEW; dark -> RETAKE
    assert res["none:0"]["verdict_rates"]["PASS"] > 0.9
    assert res["blur:2"]["verdict_rates"]["PASS"] > 0.9
    assert res["blur:8"]["verdict_rates"]["REVIEW"] > 0.9
    assert res["exposure:0.1"]["verdict_rates"]["RETAKE"] > 0.9


def test_extract_resolution_invariance():
    # thresholds are calibrated on a fixed analysis grid, so the same content
    # captured at a different scale must produce (near-)identical signals
    img = sb.make_document(np.random.default_rng(3))[None]
    big = np.repeat(np.repeat(img, 2, axis=1), 2, axis=2)  # 2x capture scale
    s1 = iq.extract(img)
    s2 = iq.extract(big)
    assert iq.verdict_name(iq.decide(s1)[0]) == iq.verdict_name(iq.decide(s2)[0])
    assert abs(float(s1["sharpness"][0]) - float(s2["sharpness"][0])) < 0.2 * abs(float(s1["sharpness"][0]))


def test_sweep_deterministic():
    a = sb.run_sweep(n_images=4, seed=0)
    b = sb.run_sweep(n_images=4, seed=0)
    assert set(a) == set(b)
    for k in a:
        assert a[k]["verdict_rates"] == b[k]["verdict_rates"]
        assert a[k]["mean_sharpness"] == b[k]["mean_sharpness"]


def test_make_document_deterministic():
    a = sb.make_document(np.random.default_rng(5))
    b = sb.make_document(np.random.default_rng(5))
    assert a.shape == b.shape == (sb.SWEEP_H, sb.SWEEP_W, 3)
    assert np.array_equal(a, b)
    # paper-dominated frame with visible ink, in [0, 1]
    assert a.min() >= 0.0 and a.max() <= 1.0
    assert a.mean() > 0.6  # paper background dominates


def test_make_document_passes_gate():
    # pristine synthetic documents should land in the usable band
    verdicts = [int(iq.decide_offline(sb.make_document(np.random.default_rng(i))[None])[0])
                for i in range(8)]
    assert all(v == iq.PASS for v in verdicts)


def test_load_images_from_dir(tmp_path):
    from PIL import Image

    for name, val in (("a.png", 0.5), ("b.png", 0.2)):
        Image.fromarray(np.full((8, 8, 3), round(val * 255), dtype=np.uint8), "RGB").save(tmp_path / name)
    imgs = sb.load_images_from_dir(tmp_path, n=2)
    assert imgs.shape[0] == 2
    assert imgs.shape[1:] == (8, 8, 3)
    assert imgs.min() >= 0.0 and imgs.max() <= 1.0


def test_load_images_from_dir_empty(tmp_path):
    imgs = sb.load_images_from_dir(tmp_path, n=4)
    assert imgs.shape[0] == 0


def test_run_real_corpus(tmp_path):
    from PIL import Image

    Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8), "RGB").save(tmp_path / "a.png")
    stats = sb.run_real_corpus(tmp_path, n=1)
    assert stats["n_images"] == 1
    assert set(stats["verdict_rates"]) == {"RETAKE", "REVIEW", "PASS"}
    assert set(stats) >= {"n_images", "verdict_rates", "mean_sharpness", "mean_contrast",
                          "mean_ink_ratio", "mean_brightness"}


def test_run_real_corpus_mixed_sizes(tmp_path):
    # real corpora have varying capture sizes: the runner must stream images
    # one at a time instead of stacking a uniform batch
    from PIL import Image

    Image.fromarray(np.full((20, 30, 3), 235, dtype=np.uint8), "RGB").save(tmp_path / "a.png")
    Image.fromarray(np.full((12, 16, 3), 235, dtype=np.uint8), "RGB").save(tmp_path / "b.png")
    stats = sb.run_real_corpus(tmp_path)
    assert stats["n_images"] == 2
    assert set(stats["verdict_rates"]) == {"RETAKE", "REVIEW", "PASS"}


def test_check_images(tmp_path):
    # check_images is the arbitrary-file input path: any format Pillow reads,
    # per-image verdict + missing-gate reasons + error resilience
    from PIL import Image

    doc = (sb.make_document(np.random.default_rng(1)) * 255).astype(np.uint8)
    p_doc = tmp_path / "doc.png"
    Image.fromarray(doc, "RGB").save(p_doc)
    p_blank = tmp_path / "blank.jpg"  # blank page -> RETAKE via ink_ratio
    Image.fromarray(np.full((200, 300, 3), 245, dtype=np.uint8), "RGB").save(p_blank)
    p_gray = tmp_path / "gray.png"    # single-channel (L-mode) file
    Image.fromarray(np.full((100, 140), 30, dtype=np.uint8), "L").save(p_gray)

    res = sb.check_images([p_doc, p_blank, p_gray, tmp_path / "missing.png"])
    assert len(res) == 4

    r_doc = res[0]
    assert r_doc["verdict"] == "PASS" and r_doc["reasons"] == []
    assert r_doc["size"] == [384, 512]  # (w, h) of the synthetic doc
    assert set(r_doc["signals"]) == {"sharpness", "contrast", "ink_ratio",
                                     "brightness", "exposure"}

    r_blank = res[1]
    assert r_blank["verdict"] == "RETAKE"
    assert any("no readable content" in r for r in r_blank["reasons"])
    assert r_blank["size"] == [300, 200]

    assert res[2]["verdict"] == "RETAKE"  # grayscale input works like RGB
    assert "file not found" in res[3]["error"]  # missing file -> error entry, no crash
