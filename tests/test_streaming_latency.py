"""Phase 13 regression tests: streaming latency statistics and throughput.

Tests added:
    test_latency_stats_has_p50_and_p99      -- p50_ms / p99_ms keys always present
    test_latency_stats_warmup_discard       -- warmup_n correctly excludes leading samples
    test_latency_stats_throughput_positive  -- throughput_eps > 0 when latencies given
    test_latency_stats_empty_returns_zeros  -- empty list returns zeros without raising
    test_stream_latency_within_sanity_bound -- p99 < 500 ms on real data (all 6 configs)

Design note on config loading:
    The Phase 11 checkpoints were trained at D=100, hidden_size=16 (fast validation
    settings). The production configs/  *.yaml files specify D=10000, hidden_size=64.
    Loading a D=100 checkpoint into a D=10000 model raises a state_dict shape mismatch.
    The integration tests therefore load each run's own config.snapshot.yaml (which
    records the exact training settings) rather than the production config files.
    The benchmark script does the same. This is documented in PHASE13_REPORT.md.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from hdlnn.eval.metrics import compute_latency_stats
from hdlnn.common.config import load_settings
try:
    from hdlnn.pipeline import run_stream
except ImportError:
    run_stream = None


# ---------------------------------------------------------------------------
# Unit tests for compute_latency_stats()
# ---------------------------------------------------------------------------


def test_latency_stats_has_p50_and_p99() -> None:
    """compute_latency_stats() must return p50_ms and p99_ms keys."""
    result = compute_latency_stats([1.0, 2.0, 3.0, 4.0, 5.0], device="cpu")
    assert "p50_ms" in result, "p50_ms key missing from latency stats"
    assert "p99_ms" in result, "p99_ms key missing from latency stats"
    assert "throughput_eps" in result, "throughput_eps key missing from latency stats"
    assert "warmup_n" in result, "warmup_n key missing from latency stats"


def test_latency_stats_warmup_discard() -> None:
    """Warm-up discard must exclude the first N measurements from percentile calculations."""
    # First 10 values are artificially high (simulating JIT warm-up spikes).
    warmup_spikes = [1000.0] * 10
    real_measurements = [1.0, 1.5, 2.0, 1.8, 1.2, 1.6, 1.9, 1.3, 1.7, 1.4]
    latencies = warmup_spikes + real_measurements

    # With warmup_n=10, spikes must be excluded from mean/p99.
    result_warmed = compute_latency_stats(latencies, device="cpu", warmup_n=10)
    # With no warmup, spikes must dominate.
    result_raw = compute_latency_stats(latencies, device="cpu", warmup_n=0)

    assert result_warmed["mean_ms"] < 5.0, (
        f"Warm-up discarded mean should be < 5 ms, got {result_warmed['mean_ms']:.3f}"
    )
    assert result_raw["mean_ms"] > 400.0, (
        f"Raw mean (with spikes) should be >> 5 ms, got {result_raw['mean_ms']:.3f}"
    )
    assert result_warmed["warmup_n"] == 10
    assert result_warmed["throughput_eps"] > 0.0


def test_latency_stats_throughput_positive() -> None:
    """throughput_eps must be > 0 whenever latencies are provided."""
    result = compute_latency_stats([2.0, 3.0, 1.5], device="cpu")
    assert result["throughput_eps"] > 0.0, "throughput_eps should be positive"
    # Sanity: throughput_eps = n / sum_seconds
    # sum_s = (2+3+1.5)/1000 = 0.0065, n=3
    expected = 3 / ((2.0 + 3.0 + 1.5) / 1000.0)
    assert math.isclose(result["throughput_eps"], expected, rel_tol=1e-6), (
        f"throughput_eps={result['throughput_eps']:.2f} vs expected={expected:.2f}"
    )


def test_latency_stats_empty_returns_zeros() -> None:
    """Empty latency list must return all-zero stats without raising."""
    result = compute_latency_stats([], device="cpu")
    assert result["p50_ms"] == 0.0
    assert result["p99_ms"] == 0.0
    assert result["throughput_eps"] == 0.0


# ---------------------------------------------------------------------------
# Integration test: streaming p99 stays within a sanity bound
# ---------------------------------------------------------------------------

# Load each config from its run's own config.snapshot.yaml so that
# dimensionality / hidden_size match the saved checkpoint exactly.
# Phase 11 checkpoints: D=100, hidden_size=16, epochs=2, seed=42.
_RUN_CONFIGS = [
    ("runs/hybrid_cfc_v1/config.snapshot.yaml",    "hybrid_cfc_v1"),
    ("runs/hdc_only_v1/config.snapshot.yaml",      "hdc_only_v1"),
    ("runs/lnn_only_v1/config.snapshot.yaml",      "lnn_only_v1"),
    ("runs/lstm_baseline_v1/config.snapshot.yaml", "lstm_baseline_v1"),
    ("runs/cnn_baseline_v1/config.snapshot.yaml",  "cnn_baseline_v1"),
    ("runs/ltc_ablation_v1/config.snapshot.yaml",  "ltc_ablation_v1"),
]


@pytest.mark.parametrize("config_path,run_id", _RUN_CONFIGS)
def test_stream_latency_within_sanity_bound(config_path: str, run_id: str) -> None:
    """Streaming p99 latency must remain below 500 ms for all configs on real data.

    500 ms is a generous ceiling -- this test checks for pathological regressions
    (e.g. accidental synchronous training inside step(), or unbounded ODE
    integration) rather than for tight real-time performance guarantees.
    The exact latency numbers are documented in PHASE13_REPORT.md.
    """
    if run_stream is None:
        pytest.skip("run_stream not available in v0.2.0 modular architecture")
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        pytest.skip(f"Config snapshot not found: {config_path}")

    cfg = load_settings(str(cfg_file))
    result = run_stream(cfg, run_id_to_load=run_id, replay_rate=10000.0, check_consistency=False)

    p99 = result.latency_stats.get("p99_ms", 0.0)
    throughput = result.latency_stats.get("throughput_eps", 0.0)

    assert p99 < 500.0, (
        f"{run_id}: streaming p99={p99:.3f} ms exceeds 500 ms sanity ceiling. "
        f"This likely indicates a regression or unbounded computation in step()."
    )
    assert throughput > 0.0, (
        f"{run_id}: throughput_eps={throughput:.1f} must be positive."
    )
