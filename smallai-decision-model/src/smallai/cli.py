"""CLI: python -m smallai {generate,train,evaluate,benchmark,predict} [--config ...]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import benchmark as bench_mod
from . import config as cfg_mod
from . import evaluate as eval_mod
from . import runs as runs_mod
from .env import oracle
from .model import SmallAIModel


def _log(msg: str) -> None:
    print(msg, flush=True)


def _resolve(cfg_path: str, cfg: cfg_mod.Config, root: Path, args) -> cfg_mod.Config:
    """Apply CLI overrides (--seed/--epochs/--n-train/--quick)."""
    if args.seed is not None:
        cfg.seed = args.seed
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.n_train is not None:
        cfg.data.n_train = args.n_train
    if args.quick:
        cfg.data.n_train = 2_000
        cfg.data.n_val = 500
        cfg.data.n_test_clean = 500
        cfg.data.n_verifier_eval = 200
        cfg.train.epochs = 3
        cfg.train.early_stop_patience = 3
    # fresh seeds require regenerating the dataset and retraining
    if args.seed is not None or args.n_train is not None or args.quick:
        _forget_cache(root)
        runs_dir = root / "runs"
        if runs_dir.exists():
            import shutil
            shutil.rmtree(runs_dir)
    return cfg


def _forget_cache(root: Path) -> None:
    data = root / "data" / "dataset.npz"
    if data.exists():
        data.unlink()


def _load_model(cfg: cfg_mod.Config, root: Path) -> SmallAIModel:
    ds = runs_mod.load_or_generate_dataset(cfg, root, logger=None)
    model = runs_mod.run_kind(cfg, root, "M", dataset=ds, logger=None)
    model.decode_weight = cfg.eval.verifier_decode_weight
    return model


def _cmd_predict(args) -> int:
    """Feed arbitrary raw feature vectors and get decisions + traces."""
    path = Path(args.input).expanduser()
    payload = json.loads(path.read_text())
    rows_in = None
    if isinstance(payload, dict):
        rows_in = payload.get("features")
        if rows_in is None:
            raise ValueError("predict input dict must contain a 'features' array")
    else:
        rows_in = payload
    if not isinstance(rows_in, list) or (len(rows_in) > 0 and not isinstance(rows_in[0], (list, dict))):
        # a flat list of numbers = a single feature row
        rows_in = [rows_in]
    rows = []
    for r in rows_in:
        arr = np.asarray(r, dtype=float).reshape(-1)
        if arr.size != oracle.N_FEATURES:
            raise ValueError(
                f"each feature row must have {oracle.N_FEATURES} values, got {arr.size}")
        rows.append(arr)
    cfg, root = cfg_mod.load_config(args.config)
    cfg = _resolve(args.config, cfg, root, args)
    model = _load_model(cfg, root)
    features = np.stack(rows) if rows else np.zeros((0, oracle.N_FEATURES))
    rows_out = model.prediction(features, verifier_threshold=cfg.eval.verifier_threshold,
                                decode_weight=cfg.eval.verifier_decode_weight)
    out = []
    for r in rows_out:
        out.append({
            "features": r["features"],
            "states": r["states"],
            "state_names": r["state_names"],
            "action": r["action"],
            "action_name": r["action_name"],
            "validities": r["validities"],
            "low_confidence": r["low_confidence"],
            "trace": r["trace"],
            "final_probs": r["final_probs"],
        })
    print(json.dumps(out, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smallai", description=__doc__)
    parser.add_argument("command", choices=["generate", "train", "evaluate", "benchmark", "predict"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None, help="override the config seed")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="tiny sizes for a smoke run")
    parser.add_argument("--force", action="store_true",
                        help="ignore cached dataset/runs and retrain")
    parser.add_argument("--input", type=str, default=None,
                        help="JSON file with one feature array per row: "
                             "[[batt,wind,temp,gps,link,payload,dist,sensor,hour,terrain], ...] "
                             "(missing keys default to 0)")
    args = parser.parse_args(argv)

    cfg, root = cfg_mod.load_config(args.config)
    cfg = _resolve(args.config, cfg, root, args)
    if args.force:
        _forget_cache(root)
        runs_dir = root / "runs"
        if runs_dir.exists():
            import shutil
            shutil.rmtree(runs_dir)

    if args.command == "generate":
        runs_mod.load_or_generate_dataset(cfg, root, logger=_log)
        return 0
    if args.command == "train":
        ds = runs_mod.load_or_generate_dataset(cfg, root, logger=_log)
        for kind in ("M", "B2", "B1"):
            runs_mod.run_kind(cfg, root, kind, dataset=ds, logger=_log)
        return 0
    if args.command == "evaluate":
        results = eval_mod.full_report(cfg, root, logger=_log)
        print(f"\nheadline: M testC acc {results['M']['testC']['acc']:.4f}")
        return 0
    if args.command == "benchmark":
        bench_mod.run(cfg, root, logger=_log)
        return 0
    if args.command == "predict":
        if args.input is None:
            _log("--input is required for the predict command (JSON feature rows)")
            return 2
        return _cmd_predict(args)
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
