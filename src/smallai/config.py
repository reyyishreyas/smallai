"""Configuration: dataclasses plus YAML loading (hyperparameters live in configs/*.yaml)."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    # Dataset sizes (section 4.3 of the PRD)
    n_train: int = 50_000
    n_val: int = 5_000
    n_test_clean: int = 5_000
    n_verifier_eval: int = 3_000
    # 1..corrupted_max corrupted trajectories per training example
    corrupted_max: int = 2
    # Corruption strategies and their sampling probabilities (see generate.py)
    corruption_probs: dict[str, float] = field(
        default_factory=lambda: {"random_flip": 0.35, "neighbour": 0.35, "shortcut": 0.30}
    )
    # A corruption strategy held out of training so the verifier has to generalise
    # to an unseen corruption style. Empty string disables the holdout.
    holdout_strategy: str = "neighbour"
    # Test B: per-feature Gaussian noise sigma added to features
    noise_sigma: dict[str, float] = field(
        default_factory=lambda: {
            "battery_pct": 2.0,
            "wind_mps": 0.8,
            "temp_c": 1.0,
            "gps_fix_quality": 0.03,
            "link_signal": 0.05,
            "payload_kg": 0.1,
            "distance_km": 0.5,
            "sensor_health": 0.03,
            "hour_of_day": 0.5,
            "terrain_difficulty": 0.03,
        }
    )
    # Test C: distribution shift applied to features
    shift: dict[str, float] = field(
        default_factory=lambda: {
            "battery_pct": -8.0,
            "wind_mps": 3.0,
            "temp_c": -4.0,
            "payload_kg": 0.5,
            "distance_km": 1.25,  # multiplicative for distance
        }
    )


@dataclass
class ModelConfig:
    hidden_enc: int = 32
    hidden_head: int = 16
    hidden_verifier: int = 16


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 1024
    lr: float = 3e-3
    # Loss weights: L = a*CE(final) + b*sum_k CE(step_k) + c*BCE(verifier) + d*L_format
    weight_a: float = 1.0
    weight_b: float = 1.0
    weight_c: float = 4.0
    weight_d: float = 0.2
    journey_learning: bool = True
    early_stop_patience: int = 5
    # upweight corrupted rows (label 0) in the verifier BCE to counter the
    # valid-vs-corrupted imbalance
    verifier_neg_weight: float = 8.0
    # The verifier module trains on every journey row for a fixed budget of its
    # own (decoupled from the early-stopped decision-head training).
    verifier_epochs: int = 300
    verifier_batch_size: int = 256


@dataclass
class EvalConfig:
    verifier_threshold: float = 0.5
    # weight of log verifier-validity in the fused step decode
    # (log p_step + weight * log validity); 0 disables verifier-guided decode
    verifier_decode_weight: float = 1.0


@dataclass
class BenchmarkConfig:
    warmup: int = 200
    runs: int = 10_000
    batch_size: int = 256


@dataclass
class LLMProxyConfig:
    """Analytical cost proxy for solving the same task with an LLM (no API in inference)."""
    model_name: str = "gpt-4o-mini (published pricing)"
    prompt_tokens: int = 420
    completion_tokens: int = 80
    usd_per_m_input: float = 0.15
    usd_per_m_output: float = 0.60
    latency_ms_p50: float = 900.0
    latency_ms_p95: float = 2200.0


@dataclass
class Config:
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    llm_proxy: LLMProxyConfig = field(default_factory=LLMProxyConfig)


_NESTED = {
    "DataConfig": DataConfig,
    "ModelConfig": ModelConfig,
    "TrainConfig": TrainConfig,
    "EvalConfig": EvalConfig,
    "BenchmarkConfig": BenchmarkConfig,
    "LLMProxyConfig": LLMProxyConfig,
}


def _build(cls: type, data: dict[str, Any]):
    """Recursively build a dataclass from a nested dict, ignoring unknown keys."""
    kwargs: dict[str, Any] = {}
    for key, value in (data or {}).items():
        fld = {f.name: f for f in fields(cls)}.get(key)
        if fld is None:
            continue
        ann = fld.type
        if isinstance(value, dict) and isinstance(ann, str) and ann in _NESTED:
            value = _build(_NESTED[ann], value)
        kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> tuple[Config, Path]:
    """Load a YAML config. Returns (config, project_root) where relative output
    paths are resolved against the project root (parent of the config dir)."""
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    cfg = _build(Config, raw)
    root = path.parent.parent
    return cfg, root
