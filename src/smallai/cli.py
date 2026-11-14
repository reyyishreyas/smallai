"""CLI: python -m smallai {stage-b,quality,qgate} [--image ...] [--json]"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import stage_b as stage_b_mod
from .env import image_quality as iq


def _log(msg: str) -> None:
    print(msg, flush=True)


IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
    ".heic",
    ".heif",
}


def _quality_file_list(image_args: list | None) -> list[Path]:
    """Expand --image groups (files or folders, repeatable) into a flat list.

    Raises ValueError when a folder has no supported image files (the CLI
    handlers turn that into exit code 2 with a readable message).
    """
    files: list[Path] = []
    for group in image_args or []:
        entries = group if isinstance(group, (list, tuple)) else [group]
        for item in entries:
            p = Path(item).expanduser()
            if p.is_dir():
                found = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
                if not found:
                    raise ValueError(
                        f"no image files in {p} (supported: png/jpg/jpeg/webp/bmp/tif/gif[/heic])"
                    )
                files.extend(found)
            else:
                files.append(p)
    return files


def _cmd_quality(args) -> int:
    """Quality-check arbitrary image files (single, several, or directories).

    Flexible input is the point: any file path, several paths, or folders
    (which are expanded to all supported image files inside) work. Each image
    is scored through the offline gate; output is human-readable by
    default, stable JSON with --json.
    """
    try:
        files = _quality_file_list(args.image)
    except ValueError as e:
        _log(str(e))
        return 2
    if not files:
        _log("no input images: use --image <file-or-dir> [more...]")
        return 2

    results = stage_b_mod.check_images(files)

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    for r in results:
        path = Path(r["path"])
        print(f"\n{path.name}  ({path})", end="")
        if "error" in r:
            print(f"\n  ERROR: {r['error']}")
            continue
        w, h = r["size"]
        print(f"  {w}×{h}px")
        s = r["signals"]
        print(f"  VERDICT: {r['verdict']}")
        print(
            f"    sharpness  {s['sharpness']:.3f}"
            + (
                f"   (RETAKE < {iq.RETAKE_SHARP_MIN:.2f}, REVIEW < {iq.REVIEW_SHARP_MIN:.2f})"
                if r["verdict"] != "PASS"
                else ""
            )
        )
        print(
            f"    contrast   {s['contrast']:.3f}"
            + (f"   (REVIEW < {iq.REVIEW_CONTRAST_MIN:.2f})" if r["verdict"] != "PASS" else "")
        )
        print(
            f"    ink_ratio  {s['ink_ratio']:.4f}"
            + (
                f"   (RETAKE < {iq.RETAKE_INK_MIN:.3f}, REVIEW < {iq.REVIEW_INK_MIN:.3f})"
                if r["verdict"] != "PASS"
                else ""
            )
        )
        print(
            f"    brightness {s['brightness']:.3f}"
            + (
                f"   (RETAKE < {iq.RETAKE_BRIGHT_MIN:.2f}, REVIEW < {iq.REVIEW_BRIGHT_MIN:.2f})"
                if r["verdict"] != "PASS"
                else ""
            )
        )
        for reason in r["reasons"]:
            print(f"    ⚠ {reason}")
    print()
    counts = {"PASS": 0, "REVIEW": 0, "RETAKE": 0}
    for r in results:
        if "verdict" in r:
            counts[r["verdict"]] += 1
    n = max(len(results), 1)
    print(
        f"summary: {len(results)} image(s) — "
        f"PASS {counts['PASS']}, REVIEW {counts['REVIEW']}, "
        f"RETAKE {counts['RETAKE']}  ({counts['PASS'] / n:.0%} pass)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smallai", description=__doc__)
    parser.add_argument("command", choices=["stage-b", "quality", "qgate"])
    parser.add_argument(
        "--image",
        action="append",
        nargs="+",
        default=None,
        help="image file(s) or folder(s) for the quality gate "
        "(any format Pillow reads; folders are expanded "
        "to their image files). Repeatable.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="quality command: emit machine-readable JSON",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="directory of real images for the offline gate (e.g. dataset/images)",
    )
    parser.add_argument(
        "--n", type=int, default=None, help="max real images to evaluate in the gate"
    )
    parser.add_argument(
        "--real-images",
        type=str,
        default="data/prescriptions",
        help="directory of real document images behind the qgate "
        "real-ladder supervision (default: data/prescriptions)",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        default=False,
        help="qgate command: per-state comparison of the Q policy against the "
        "fixed gate (agreement, label disagreement, reward, safety counters)",
    )
    args = parser.parse_args(argv)

    if args.command == "stage-b":
        if args.images_dir is None:
            _log("--images-dir is required for the stage-b command (directory of real images)")
            return 2
        stats = stage_b_mod.run_real_corpus(args.images_dir, n=args.n)
        print(json.dumps(stats, indent=2, default=float))
        return 0
    if args.command == "quality":
        return _cmd_quality(args)
    if args.command == "qgate":
        from . import stage_b_q as sbq

        if args.diagnose:
            print(
                json.dumps(
                    sbq.disagreement_report(real_ladder_dir=args.real_images),
                    indent=2,
                    default=list,
                )
            )
            return 0
        if args.image:
            results = sbq.check_images_with_q(
                _quality_file_list(args.image), real_ladder_dir=args.real_images
            )
            print(json.dumps(results, indent=2))
            return 0
        print(
            json.dumps(
                sbq.run_experiment(real_ladder_dir=args.real_images),
                indent=2,
                default=list,
            )
        )
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
