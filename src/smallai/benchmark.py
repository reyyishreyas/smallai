"""Benchmarks (PRD Task 4, section 6.2 / definition of done): latency p50/p95
over 10K single-file decisions, analytical FLOPs per decision, peak memory via
tracemalloc, and the LLM cost proxy. Writes reports/benchmark.md.
"""
from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from . import runs as runs_mod
from .baselines import llm_proxy_estimate
from .env import oracle
from .nn.layers import Dense


def analytic_flops(model) -> int:
    """Multiply-add FLOPs for one inference pass of the chain decode:
    encoder + verifier encoder once, each step head once, the final head, and
    3 verifier head calls per step (the fused decode scores all 3 candidate
    states). Counted analytically from layer shapes."""
    flops = 0
    stacks = [model.encoder, model.verifier_encoder, model.final_head] \
        + list(model.step_heads) + [model.verifier_head] * (2 * oracle.N_STEPS)
    for stack in stacks:
        for layer in stack:
            if isinstance(layer, Dense):
                flops += 2 * layer.W.shape[0] * layer.W.shape[1]
    return flops


def latency_bench(model, x: np.ndarray, cfg_bench) -> dict:
    """p50/p95 per-decision latency over single-row calls with perf_counter,
    after warmup, single thread."""
    n_rows = x.shape[0]
    for i in range(cfg_bench.warmup):
        model.prediction(x[i:i + 1])
    times = []
    for i in range(cfg_bench.runs):
        row = x[i % n_rows:i % n_rows + 1]
        t0 = time.perf_counter()
        model.prediction(row)
        times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000.0
    return {
        "runs": len(times),
        "latency_ms_mean": float(times.mean()),
        "latency_ms_p50": float(np.percentile(times, 50)),
        "latency_ms_p95": float(np.percentile(times, 95)),
    }


def throughput_bench(model, x: np.ndarray, cfg_bench) -> dict:
    """Vectorised batches: decisions per second."""
    n = min(2 * cfg_bench.batch_size, x.shape[0])
    t0 = time.perf_counter()
    for start in range(0, n, cfg_bench.batch_size):
        model.predict_chain(x[start:start + cfg_bench.batch_size])
    dt = time.perf_counter() - t0
    return {"batch_size": cfg_bench.batch_size, "decisions_per_second": n / dt}


def memory_bench(model, x: np.ndarray) -> dict:
    tracemalloc.start()
    model.prediction(x[:1])
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return {"peak_memory_bytes_single_decision": int(peak)}


def run(cfg, root: Path, logger=None) -> dict:
    log = (lambda *a: None) if logger is None else logger
    ds = runs_mod.load_or_generate_dataset(cfg, root, logger)
    model = runs_mod.run_kind(cfg, root, "M", dataset=ds, logger=logger)

    out = {
        "parameters": model.param_count(),
        "flops_per_decision": analytic_flops(model),
        **latency_bench(model, ds.test_a_x, cfg.benchmark),
        **throughput_bench(model, ds.test_a_x, cfg.benchmark),
        **memory_bench(model, ds.test_a_x),
        "llm_proxy": llm_proxy_estimate(cfg.llm_proxy),
    }

    rep = Path(root) / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "benchmark.json").write_text(json.dumps(out, indent=2))

    lines = [
        "# Benchmark", "",
        f"- Parameters: {out['parameters']:,}",
        f"- FLOPs per decision (analytic, multiply-adds): {out['flops_per_decision']:,}",
        (f"- Latency over {out['runs']:,} single decisions: "
        f"mean {out['latency_ms_mean']:.3f} ms | p50 {out['latency_ms_p50']:.3f} ms | "
        f"p95 {out['latency_ms_p95']:.3f} ms"),
        (f"- Vectorised throughput: {out['decisions_per_second']:,.0f} decisions/s "
        f"(batch {out['batch_size']})"),
        f"- Peak memory, single decision: {out['peak_memory_bytes_single_decision'] / 1024:.1f} KiB",
        "",
        "## LLM cost proxy (analytical, no API in the loop)",
        "",
        f"- {out['llm_proxy']['model']}: ~{out['llm_proxy']['tokens_per_decision']} tokens/decision",
        (f"- ~${out['llm_proxy']['usd_per_1m_decisions']:.2f} per 1M decisions, "
        f"p50 latency {out['llm_proxy']['latency_ms_p50']:.0f} ms "
        f"vs ≈$0 API cost at {out['latency_ms_p50']:.3f} ms p50 for this model."),
    ]
    (rep / "benchmark.md").write_text("\n".join(lines) + "\n")
    log(f"benchmark written to {rep / 'benchmark.md'}")
    return out
