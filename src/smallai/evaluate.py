"""Evaluation metrics and the results harness (PRD Task 4, section 6.2).

All metrics are implemented from scratch (accuracy, macro-F1, confusion
matrix, rank-based AUC for the verifier). The harness evaluates every model
across Test A (clean), Test B (noisy) and Test C (shifted), plus verifier
quality and the step-consistency rate, and writes tables to reports/.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import runs as runs_mod
from .baselines import Heuristic, llm_proxy_estimate
from .env import oracle

MODEL_ORDER = ["M", "B2", "B1", "B0"]


# --------------------------------------------------------------------- metrics
def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(np.int64), y_pred.astype(np.int64)):
        cm[t, p] += 1
    return cm


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = float(((y_true == c) & (y_pred == c)).sum())
        fp = float(((y_true != c) & (y_pred == c)).sum())
        fn = float(((y_true == c) & (y_pred != c)).sum())
        if tp == 0.0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0.0:
            f1s.append(0.0)
        else:
            f1s.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(f1s))


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U, ties get average rank)."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0  # average rank (1-based)
        i = j + 1
    sum_pos_ranks = ranks[labels == 1].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ----------------------------------------------------------------- evaluation
def evaluate_final(model, x: np.ndarray, gt_states: np.ndarray,
                   gt_action: np.ndarray) -> dict:
    """Final-decision metrics plus per-step accuracy for step models."""
    pred_states, pred_action = model.predict_chain(x)
    out = {
        "acc": accuracy(gt_action, pred_action),
        "macro_f1": macro_f1(gt_action, pred_action, oracle.N_ACTIONS),
        "confusion": confusion(gt_action, pred_action, oracle.N_ACTIONS).tolist(),
    }
    if hasattr(model, "step_heads"):  # step-wise models only: M and B2
        per_step = [float((pred_states[:, k] == gt_states[:, k]).mean())
                    for k in range(oracle.N_STEPS)]
        out["step_acc"] = per_step
        # step-consistency: does the final head obey the rule table for its own steps?
        legal = oracle.legal_action_for_steps(pred_states)
        out["step_consistency"] = float((legal == pred_action).mean())
    return out


def verifier_report(model, ds) -> dict:
    """Verifier precision/recall/AUC for detecting corrupted steps.

    Scores are evaluated per (row, step) pair; labels are the per-step
    validity labels of the corrupted-evaluation rows."""
    if not model.journey:
        return {"verifier": False}
    scores_per_k = model.verifier_scores(ds.verifier_x, ds.verifier_states)
    # align per-k score blocks with per-k label blocks (same (row, step) pairs)
    labels = np.concatenate([ds.verifier_matches[:, k] for k in range(oracle.N_STEPS)])
    scores = np.concatenate(scores_per_k)
    pred = scores >= 0.5
    tp = float(((labels == 1) & pred).sum())
    fp = float(((labels == 0) & pred).sum())
    fn = float(((labels == 1) & ~pred).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)

    strats = np.concatenate([ds.verifier_strategy for _ in range(oracle.N_STEPS)])
    per_strategy = {}
    from .env.generate import STRATEGY_NAMES
    for code in (1, 2, 3):
        sel = strats == code
        strat_auc = auc(labels[sel], scores[sel])
        # `shortcut` corrupts only the final action, so its steps are always
        # valid: there are no negative labels and AUC is undefined
        per_strategy[STRATEGY_NAMES[code]] = (
            strat_auc if not np.isnan(strat_auc) else "n/a (steps always valid)")
    return {
        "verifier": True,
        "auc": auc(labels, scores),
        "precision": precision,
        "recall": recall,
        "auc_per_strategy": per_strategy,
    }


def low_confidence_rate(model, x: np.ndarray, threshold: float) -> float | None:
    if not model.journey:
        return None
    rows = model.prediction(x[:200], verifier_threshold=threshold)
    return float(np.mean([r["low_confidence"] for r in rows]))


def full_report(cfg, root: Path, logger=None) -> dict:
    """Evaluate every model across the three test sets and write reports/."""
    log = (lambda *a: None) if logger is None else logger
    ds = runs_mod.load_or_generate_dataset(cfg, root, logger)

    models = {}
    for kind in ("M", "B2", "B1"):
        models[kind] = runs_mod.run_kind(cfg, root, kind, dataset=ds, logger=logger)
    models["B0"] = Heuristic()

    results: dict[str, dict] = {}
    for name, model in models.items():
        results[name] = {}
        for tag, (x, s, a) in {
            "testA": (ds.test_a_x, ds.test_a_states, ds.test_a_action),
            "testB": (ds.test_b_x, ds.test_b_states, ds.test_b_action),
            "testC": (ds.test_c_x, ds.test_c_states, ds.test_c_action),
        }.items():
            res = evaluate_final(model, x, s, a)
            low = low_confidence_rate(model, x, cfg.eval.verifier_threshold)
            if low is not None:
                res["low_confidence_rate"] = low
            results[name][tag] = res
        if name == "M":
            results[name]["verifier"] = verifier_report(models["M"], ds)
        log(f"evaluated {name}: " + ", ".join(
            f"{t}={results[name][t]['acc']:.3f}" for t in ("testA", "testB", "testC")))

    results["llm_proxy"] = llm_proxy_estimate(cfg.llm_proxy)
    results["meta"] = {
        "param_count_M": models["M"].param_count(),
        "param_count_B1": models["B1"].param_count(),
        "seed": cfg.seed,
    }

    write_tables(cfg, root, results, models)
    write_plots(cfg, root, results, models, ds)
    results.pop("confusion", None)
    for name in list(results):
        if isinstance(results[name], dict):
            results[name].pop("confusion", None)
    return results


# --------------------------------------------------------------------- output
def _fmt(v) -> str:
    return f"{v * 100:.2f}"


def write_tables(cfg, root: Path, results: dict, models: dict) -> None:
    rep = Path(root) / "reports"
    rep.mkdir(parents=True, exist_ok=True)

    lines = ["# Results", "",
             "Accuracy (%). Test A = clean, Test B = sensor noise, Test C = distribution shift.", ""]
    lines += [
        "| Model | Test A acc | Test A F1 | Test B acc | Test B F1 | Test C acc | Test C F1 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in MODEL_ORDER:
        r = results.get(name, {})
        row = [f"**{name}**" if name == "M" else name]
        for tag in ("testA", "testB", "testC"):
            if tag in r:
                row += [_fmt(r[tag]["acc"]), _fmt(r[tag]["macro_f1"])]
            else:
                row += ["-", "-"]
        lines.append("| " + " | ".join(row) + " |")

    if "M" in results and "B2" in results:
        delta = (results["M"]["testC"]["acc"] - results["B2"]["testC"]["acc"]) * 100
        lines += ["", f"**Journey-learning delta on Test C (M − B2): {delta:+.2f} points**"]

    # step-level metrics (M and B2 make actual intermediate predictions)
    lines += ["", "## Step-level metrics", "",
              "| Model | Test | Step 1 acc | Step 2 acc | Step 3 acc | Step-consistency |",
              "| --- | --- | --- | --- | --- | --- |"]
    for name in ("M", "B2"):
        for tag in ("testA", "testB", "testC"):
            r = results.get(name, {}).get(tag, {})
            if "step_acc" not in r:
                continue
            lines.append(f"| {name} | {tag} | "
                         + " | ".join(f"{s * 100:.2f}" for s in r["step_acc"])
                         + f" | {r.get('step_consistency', float('nan')) * 100:.2f} |")
    vrep = results.get("M", {}).get("verifier", {})
    if vrep.get("verifier"):
        lines += ["",
                  "Low-confidence detection for M "
                  f"(threshold {cfg.eval.verifier_threshold} of the verifier): "
                  + ", ".join(f"{t}={results['M'][t].get('low_confidence_rate', 0) * 100:.2f}%"
                              for t in ("testA", "testB", "testC"))]

    vrep = results.get("M", {}).get("verifier", {})
    if vrep.get("verifier"):
        lines += ["", "## Verifier quality (corrupted-step detection)", "",
                  f"- AUC: {vrep['auc']:.4f}",
                  f"- Precision @0.5: {vrep['precision']:.4f}",
                  f"- Recall @0.5: {vrep['recall']:.4f}",
                  "- AUC per corruption strategy: " + ", ".join(
                      f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in vrep["auc_per_strategy"].items())]

    pc = results["meta"]
    lines += ["", "## Model size", "",
              f"- Full model (M): {pc['param_count_M']:,} parameters",
              f"- Shortcut MLP (B1): {pc['param_count_B1']:,} parameters"]

    llm = results["llm_proxy"]
    lines += ["", "## LLM cost proxy (B3, analytical)", "",
              f"- Model: {llm['model']}",
              f"- Tokens per decision: ~{llm['tokens_per_decision']}",
              f"- Cost per 1,000 decisions: ${llm['usd_per_1000_decisions']:.4f}",
              f"- Cost per 1M decisions: ${llm['usd_per_1m_decisions']:.2f}",
              f"- Latency p50/p95: {llm['latency_ms_p50']:.0f}/{llm['latency_ms_p95']:.0f} ms"]

    for name in MODEL_ORDER:
        for tag in ("testA", "testB", "testC"):
            cm = results.get(name, {}).get(tag, {}).get("confusion")
            if cm is None:
                continue
            lines += ["", f"### Confusion matrix — {name} / {tag}",
                      "", "| true\\pred | " + " | ".join(oracle.ACTION_NAMES) + " |",
                      "| --- | " + " | ".join(["---"] * oracle.N_ACTIONS) + " |"]
            for i, row in enumerate(cm):
                lines.append("| " + oracle.ACTION_NAMES[i] + " | "
                             + " | ".join(str(c) for c in row) + " |")

    (rep / "results.md").write_text("\n".join(lines) + "\n")

    # CSV of headline numbers
    with open(rep / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "test", "accuracy", "macro_f1", "p95_latency_ms"])
        for name in MODEL_ORDER:
            for tag in ("testA", "testB", "testC"):
                r = results.get(name, {}).get(tag, {})
                if r:
                    w.writerow([name, tag, f"{r['acc']:.4f}", f"{r['macro_f1']:.4f}", ""])


def write_plots(cfg, root: Path, results: dict, models: dict, ds) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # matplotlib is an optional dependency: skip plots
        print("matplotlib unavailable — skipping plots")
        return
    rep = Path(root) / "reports"
    rep.mkdir(parents=True, exist_ok=True)

    # loss curves from each run's history
    fig, ax = plt.subplots(figsize=(6, 4))
    for kind in ("M", "B2", "B1"):
        hist_file = Path(root) / "runs" / kind / "history.json"
        if not hist_file.exists():
            continue
        hist = json.loads(hist_file.read_text())
        ax.plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist],
                label=f"{kind} train loss")
        ax.plot([h["epoch"] for h in hist], [h["val_acc"] for h in hist],
                linestyle="--", label=f"{kind} val acc")
    ax.set_xlabel("epoch")
    ax.set_title("Training curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(rep / "loss_curves.png", dpi=150)
    plt.close(fig)

    # confusion matrices for M on the shifted set
    r = results.get("M", {}).get("testC", {}).get("confusion")
    if r is not None:
        cm = np.array(r, dtype=float)
        cm = cm / max(cm.sum(), 1.0)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(oracle.N_ACTIONS), oracle.ACTION_NAMES, rotation=30)
        ax.set_yticks(range(oracle.N_ACTIONS), oracle.ACTION_NAMES)
        for i in range(oracle.N_ACTIONS):
            for j in range(oracle.N_ACTIONS):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xlabel("predicted (M, Test C)")
        ax.set_ylabel("true")
        fig.tight_layout()
        fig.savefig(rep / "confusion_M_testC.png", dpi=150)
        plt.close(fig)
