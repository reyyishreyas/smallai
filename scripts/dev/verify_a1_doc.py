"""Check every figure in AI-267_STAGE_A1_RESULTS.md against the JSON.

Development tool. The results document quotes a lot of numbers;
this script recomputes each of them from ``reports/poc_a1.json`` and
fails loudly if the document and the run ever disagree. It is the
same discipline the Stage A document was written under.

Usage::

    python scripts/dev/verify_a1_doc.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = ROOT / "reports" / "poc_a1.json"


def main() -> int:
    """Recompute the document's figures and compare them."""
    data = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    agg = data["aggregate"]
    h1 = agg["h1"]
    s1 = agg["s1_invalid"]
    s2 = agg["s2_measurements"]
    cand = agg["candidate"]
    subset = agg["subset"]
    sweep = agg["sweep"]
    rates = agg["pooled_rates"]
    field = agg["policy_field"]

    checks: list[tuple[str, float, float, float]] = []

    def check(name: str, actual: float, expected: float, tol: float = 5e-4) -> None:
        checks.append((name, actual, expected, tol))

    # run facts
    check("stage", 1.0 if data["stage"] == "A.1" else 0.0, 1.0)
    check("master seeds", float(len(data["master_seeds"])), 5.0)
    check("test size", float(data["test_size"]), 10_000.0)
    check("budget", float(data["budget"]), 1_000_000.0)
    check("resamples", float(data["resamples"]), 10_000.0)
    check("elapsed", data["elapsed_seconds"], 402.0, tol=5.0)
    check("subset size", float(data["subset_size"]), 1_000.0)

    # H1
    check("H1 seed-level mean", h1["seed_level"]["mean_pp"], 0.170)
    check("H1 seed-level sd", h1["seed_level"]["sd_pp"], 0.240)
    check("H1 seed-level se", h1["seed_level"]["se_pp"], 0.107)
    check("H1 seed-level low", h1["seed_level"]["ci_low_pp"], -0.127)
    check("H1 seed-level high", h1["seed_level"]["ci_high_pp"], 0.467)
    check("H1 t", h1["seed_level"]["t_statistic"], 1.59, tol=5e-3)
    check("H1 pooled delta", h1["pooled"]["delta_pp"], 0.170)
    check("H1 pooled low", h1["pooled"]["ci_low_pp"], 0.070)
    check("H1 pooled high", h1["pooled"]["ci_high_pp"], 0.268)
    check("H1 pooled n", float(h1["pooled"]["n"]), 50_000.0)
    check("H1 journey-only", float(h1["pooled"]["first_only"]), 367.0)
    check("H1 shortcut-only", float(h1["pooled"]["second_only"]), 282.0)
    check("H1 mcnemar p", h1["pooled"]["mcnemar_p"], 0.0010)
    check(
        "H1 not supported",
        0.0 if h1["supported"] else 1.0,
        1.0,
    )

    # S1
    check("S1 journey invalid", rates["journey_invalid"] * 100, 1.230)
    check("S1 shortcut invalid", rates["shortcut_invalid"] * 100, 7.874)
    check("S1 delta", s1["delta_pp"], 6.644)
    check("S1 low", s1["ci_low_pp"], 6.396)
    check("S1 high", s1["ci_high_pp"], 6.890)
    check("S1 shortcut-invalid-only", float(s1["first_only"]), 3_769.0)
    check("S1 journey-invalid-only", float(s1["second_only"]), 447.0)
    check("S1 p underflows", 1.0 if s1["mcnemar_p"] == 0.0 else 0.0, 1.0)

    # S2
    check("S2 journey measurements", rates["journey_measurements"], 4.592)
    check("S2 shortcut measurements", rates["shortcut_measurements"], 6.000)
    check("S2 delta", s2["delta"], -1.408)
    check("S2 low", s2["ci_low"], -1.423)
    check("S2 high", s2["ci_high"], -1.392)

    # candidate
    check("candidate pooled delta", cand["pooled"]["delta_pp"], 0.034)
    check("candidate pooled low", cand["pooled"]["ci_low_pp"], -0.062)
    check("candidate pooled high", cand["pooled"]["ci_high_pp"], 0.132)
    check("candidate mcnemar p", cand["pooled"]["mcnemar_p"], 0.5161)
    check("candidate candidate-only", float(cand["pooled"]["first_only"]), 312.0)
    check("candidate frozen-only", float(cand["pooled"]["second_only"]), 295.0)
    check("candidate seed-level mean", cand["seed_level"]["mean_pp"], 0.034)
    check("candidate seed-level low", cand["seed_level"]["ci_low_pp"], -0.179)
    check("candidate seed-level high", cand["seed_level"]["ci_high_pp"], 0.247)
    check("candidate invalid rate", cand["rates"]["candidate_invalid"] * 100, 2.182)
    check("candidate measurements", cand["rates"]["candidate_measurements"], 4.519)
    check("frozen invalid rate", cand["rates"]["journey_invalid"] * 100, 1.230)
    check("frozen measurements", cand["rates"]["journey_measurements"], 4.592)
    check(
        "candidate not adopted",
        0.0 if cand["adopted"] else 1.0,
        1.0,
    )

    # the 1,000-scenario subset
    check("subset pooled delta", subset["pooled"]["delta_pp"], 0.040)
    check("subset pooled low", subset["pooled"]["ci_low_pp"], -0.260)
    check("subset pooled high", subset["pooled"]["ci_high_pp"], 0.340)
    check("subset pooled n", float(subset["pooled"]["n"]), 5_000.0)
    check("subset mcnemar p", subset["pooled"]["mcnemar_p"], 0.8939)
    check("subset seed-level mean", subset["seed_level"]["mean_pp"], 0.040)
    check("subset seed-level low", subset["seed_level"]["ci_low_pp"], -0.395)
    check("subset seed-level high", subset["seed_level"]["ci_high_pp"], 0.475)

    # the sweep
    check("sweep 0.04 delta", sweep["0.04"]["pooled"]["delta_pp"], 0.508)
    check("sweep 0.04 low", sweep["0.04"]["pooled"]["ci_low_pp"], 0.406)
    check("sweep 0.04 high", sweep["0.04"]["pooled"]["ci_high_pp"], 0.608)
    check("sweep 0.04 seed low", sweep["0.04"]["seed_level"]["ci_low_pp"], 0.373)
    check("sweep 0.04 seed high", sweep["0.04"]["seed_level"]["ci_high_pp"], 0.643)
    check("sweep 0.04 t", sweep["0.04"]["seed_level"]["t_statistic"], 10.5, tol=5e-2)
    check("sweep 0.08 delta", sweep["0.08"]["pooled"]["delta_pp"], 0.170)
    check("sweep 0.12 delta", sweep["0.12"]["pooled"]["delta_pp"], 0.040)
    check("sweep 0.12 low", sweep["0.12"]["pooled"]["ci_low_pp"], -0.090)
    check("sweep 0.12 high", sweep["0.12"]["pooled"]["ci_high_pp"], 0.168)

    # the policy field, pooled
    def pooled(name: str, key: str) -> float:
        return sum(e["policies"][name][key] for e in field) / len(field)

    check("field journey accuracy", pooled("journey_q", "accuracy") * 100, 98.192)
    check("field journey invalid", pooled("journey_q", "invalid_rate") * 100, 1.230)
    check("field journey measurements", pooled("journey_q", "mean_measurements"), 4.592)
    check("field journey review", pooled("journey_q", "review_rate") * 100, 0.960)
    check("field no_recheck accuracy", pooled("no_recheck_q", "accuracy") * 100, 97.142)
    check("field no_recheck invalid", pooled("no_recheck_q", "invalid_rate") * 100, 6.132)
    check("field no_recheck measurements", pooled("no_recheck_q", "mean_measurements"), 4.487)
    check("field no_recheck review", pooled("no_recheck_q", "review_rate") * 100, 2.128)
    check("field expert_only accuracy", pooled("expert_only_q", "accuracy") * 100, 98.050)
    check("field expert_only invalid", pooled("expert_only_q", "invalid_rate") * 100, 1.008)
    check("field expert_only measurements", pooled("expert_only_q", "mean_measurements"), 4.634)
    check("field expert_only review", pooled("expert_only_q", "review_rate") * 100, 1.202)
    check("field shortcut accuracy", pooled("shortcut_cart", "accuracy") * 100, 98.022)
    check("field shortcut invalid", pooled("shortcut_cart", "invalid_rate") * 100, 7.874)
    check("field shortcut measurements", pooled("shortcut_cart", "mean_measurements"), 6.000)
    check("field shortcut review", pooled("shortcut_cart", "review_rate") * 100, 0.232)

    # the sweep's cost table, computed from the per-seed conditions
    conditions = data["seeds"]
    for sigma, expected in (
        (
            "0.04",
            (98.608, 98.100, 0.246, 2.714, 4.498, 6.000, 1.040, 0.564),
        ),
        (
            "0.08",
            (98.192, 98.022, 1.230, 7.874, 4.592, 6.000, 0.960, 0.232),
        ),
        (
            "0.12",
            (97.594, 97.554, 2.164, 14.654, 4.687, 6.000, 1.294, 0.000),
        ),
    ):
        rows = [
            next(c for c in entry["conditions"] if c["sigma"] == float(sigma))
            for entry in conditions
        ]

        def mean(key: str, policy: str, scale: float = 1.0) -> float:
            values = [r["scores"][policy][key] for r in rows]
            return sum(values) / len(values) * scale

        jacc, sacc, jinv, sinv, jmeas, smeas, jrev, srev = expected
        check(f"sweep {sigma} journey acc", mean("accuracy", "journey_q", 100), jacc)
        check(f"sweep {sigma} shortcut acc", mean("accuracy", "shortcut_cart", 100), sacc)
        check(f"sweep {sigma} journey invalid", mean("invalid_rate", "journey_q", 100), jinv)
        check(f"sweep {sigma} shortcut invalid", mean("invalid_rate", "shortcut_cart", 100), sinv)
        check(f"sweep {sigma} journey meas", mean("mean_measurements", "journey_q"), jmeas)
        check(f"sweep {sigma} shortcut meas", mean("mean_measurements", "shortcut_cart"), smeas)
        check(f"sweep {sigma} journey review", mean("review_rate", "journey_q", 100), jrev)
        check(f"sweep {sigma} shortcut review", mean("review_rate", "shortcut_cart", 100), srev)

    # H2, H3 and H4, per seed, at the primary sigma
    for entry, expected in zip(
        field,
        [
            (1.240, 0.950, 1.530, 0.340, 0.130, 0.550, 4.648),
            (1.890, 1.500, 2.280, -0.050, -0.330, 0.220, 4.655),
            (0.500, 0.270, 0.740, 0.300, 0.070, 0.540, 4.482),
            (0.120, -0.140, 0.390, -0.070, -0.350, 0.220, 4.659),
            (1.500, 1.140, 1.870, 0.190, -0.090, 0.460, 4.517),
        ],
    ):
        results = {h["name"]: h for h in entry["hypotheses"]}
        h2, h3 = results["H2"], results["H3"]
        h4 = results["H4"]
        d2, lo2, hi2, d3, lo3, hi3, meas = expected
        check(f"H2 delta {entry['seed']}", h2["delta_pp"], d2)
        check(f"H2 low {entry['seed']}", h2["ci_low_pp"], lo2)
        check(f"H2 high {entry['seed']}", h2["ci_high_pp"], hi2)
        check(f"H3 delta {entry['seed']}", h3["delta_pp"], d3)
        check(f"H3 low {entry['seed']}", h3["ci_low_pp"], lo3)
        check(f"H3 high {entry['seed']}", h3["ci_high_pp"], hi3)
        check(f"H4 meas {entry['seed']}", entry["policies"]["journey_q"]["mean_measurements"], meas)
        check(
            f"H4 holds {entry['seed']}",
            1.0 if h4["supported"] else 0.0,
            1.0,
        )

    # per-seed primary table
    expected_rows = [
        (20263030, 98.340, 97.870, 0.470, 2.530, 7.310, 4.648),
        (20263031, 98.120, 98.240, -0.120, 0.700, 8.150, 4.655),
        (20263032, 98.130, 97.910, 0.220, 1.430, 7.990, 4.482),
        (20263033, 98.030, 97.730, 0.300, 0.740, 7.890, 4.659),
        (20263034, 98.340, 98.360, -0.020, 0.750, 8.030, 4.517),
    ]
    for entry, row in zip(field, expected_rows):
        seed, jacc, sacc, delta, jinv, sinv, jmeas = row
        policies = entry["policies"]
        check(f"seed {seed} journey acc", policies["journey_q"]["accuracy"] * 100, jacc)
        check(f"seed {seed} shortcut acc", policies["shortcut_cart"]["accuracy"] * 100, sacc)
        check(
            f"seed {seed} delta",
            100 * (policies["journey_q"]["accuracy"] - policies["shortcut_cart"]["accuracy"]),
            delta,
        )
        check(f"seed {seed} journey invalid", policies["journey_q"]["invalid_rate"] * 100, jinv)
        check(f"seed {seed} shortcut invalid", policies["shortcut_cart"]["invalid_rate"] * 100, sinv)
        check(f"seed {seed} journey meas", policies["journey_q"]["mean_measurements"], jmeas)

    failures = [
        (name, actual, expected)
        for name, actual, expected, tol in checks
        if abs(actual - expected) > tol
    ]
    for name, actual, expected in failures:
        print(f"MISMATCH {name}: document {expected} vs json {actual}")
    print(f"{len(checks) - len(failures)}/{len(checks)} figures verified")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
