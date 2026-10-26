#!/usr/bin/env python3
"""Phase 13 -- Real Streaming Latency & Throughput Benchmark.

Runs all 6 named ablation configs through ``hdlnn stream`` on the real
UNSW-NB15 test split (30 entities), then cross-checks against batch inference
timing for the same split.

Config loading note:
    Each config is loaded from runs/<run_id>/config.snapshot.yaml (not from
    configs/*.yaml). The production configs/ files specify D=10000, hidden_size=64
    while the Phase 11 checkpoints were trained at D=100, hidden_size=16.
    Loading a D=100 checkpoint into a D=10000 model raises a state_dict shape
    mismatch. Using the snapshot ensures exact architectural alignment.
    This is explicitly disclosed in PHASE13_REPORT.md.

Usage:
    .\.venv\Scripts\python.exe scripts/phase13_benchmark.py

Output:
    runs/phase13_benchmark/results.json  -- full per-config latency/memory JSON
    Printed table to stdout.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch

from hdlnn.common.config import load_settings
from hdlnn.common.seed import set_global_seed
from hdlnn.pipeline import load_and_split_data, run_stream
from hdlnn.eval import StandardPipeline

# ---------------------------------------------------------------------------
# Config inventory -- use run snapshots so model dims match the checkpoint
# ---------------------------------------------------------------------------

CONFIGS = [
    ("runs/hybrid_cfc_v1/config.snapshot.yaml",    "hybrid_cfc_v1",    "hybrid"),
    ("runs/hdc_only_v1/config.snapshot.yaml",      "hdc_only_v1",      "hdc_only"),
    ("runs/lnn_only_v1/config.snapshot.yaml",      "lnn_only_v1",      "lnn_only"),
    ("runs/lstm_baseline_v1/config.snapshot.yaml", "lstm_baseline_v1", "lstm_baseline"),
    ("runs/cnn_baseline_v1/config.snapshot.yaml",  "cnn_baseline_v1",  "cnn_baseline"),
    ("runs/ltc_ablation_v1/config.snapshot.yaml",  "ltc_ablation_v1",  "ltc_ablation"),
]

WARMUP_N = 10          # matches _STREAM_WARMUP_N in pipeline.py
REPLAY_RATE = 10_000.0 # suppress sleep() overhead


def get_batch_per_event_ms(config_path: str, run_id: str, real_data_path: str | None = None) -> tuple[float, int]:
    """Estimate batch per-event inference time by timing the forward() pass only.

    Loads the same checkpoint used for streaming, then groups test flows by
    entity and runs sequence_model.forward() -- same code path as run_eval().
    Returns (per_event_ms, n_events).
    """
    cfg = load_settings(config_path)
    overrides: dict = {"split_version": "v_phase13"}
    if real_data_path:
        overrides["dataset_raw_path"] = real_data_path
    cfg = cfg.model_copy(update=overrides)
    set_global_seed(cfg.seed)

    train_flows, val_flows, test_flows = load_and_split_data(cfg)
    pipeline = StandardPipeline(settings=cfg)

    ckpt_path = Path("runs") / run_id / "model_checkpoint.pt"
    if ckpt_path.exists():
        model_module = getattr(
            pipeline.sequence_model, "cfc",
            getattr(pipeline.sequence_model, "ltc",
                getattr(pipeline.sequence_model, "lstm",
                    getattr(pipeline.sequence_model, "conv", None))))
        if isinstance(model_module, torch.nn.Module):
            state_dict = torch.load(ckpt_path, weights_only=True)
            model_module.load_state_dict(state_dict)
            model_module.eval()

    pipeline.encoder.fit(train_flows)

    test_encoded = [pipeline.encoder.encode(f) for f in test_flows]
    groups: dict[str, list] = {}
    for hv, f in zip(test_encoded, test_flows):
        groups.setdefault(f.src_ip, []).append(hv)

    t0 = time.perf_counter()
    for entity_id, seq in groups.items():
        pipeline.sequence_model.forward(seq, entity_id=entity_id)
    t1 = time.perf_counter()

    batch_total_ms = (t1 - t0) * 1000.0
    n_events = len(test_flows)
    per_event_ms = batch_total_ms / n_events if n_events > 0 else 0.0
    return per_event_ms, n_events


def main() -> None:
    out_dir = Path("runs/phase13_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    print()
    print("=" * 90)
    print("Phase 13 -- Real Streaming Latency & Throughput Benchmark")
    print("Dataset : UNSW-NB15 real slice (30 IP entities)")
    print("Device  : CPU (no GPU present)")
    print(f"Warm-up : N={WARMUP_N} events discarded from percentile calculations")
    print("Configs : loaded from runs/<run_id>/config.snapshot.yaml (D=100, hidden=16)")
    print("         Production configs/ use D=10000 which mismatches Phase 11 checkpoints.")
    print("=" * 90)

    for config_path, run_id, label in CONFIGS:
        print(f"\n--- [{label}]  snapshot={config_path}  run_id={run_id} ---")

        snap_path = Path(config_path)
        if not snap_path.exists():
            print(f"  ERROR: config snapshot not found at {config_path}. Skipping.")
            continue

        with open(snap_path) as fh:
            snap = yaml.safe_load(fh)
        snap_epochs = snap.get("epochs", "unknown")
        snap_dim = snap.get("dimensionality", "unknown")
        snap_hidden = snap.get("hidden_size", "unknown")

        ckpt_path = Path("runs") / run_id / "model_checkpoint.pt"
        ckpt_present = ckpt_path.exists()

        print(f"  Checkpoint: {'PRESENT' if ckpt_present else 'MISSING'} | "
              f"epochs={snap_epochs}, D={snap_dim}, hidden={snap_hidden}, seed=42")
        if not ckpt_present:
            if label == "hdc_only":
                print("  NOTE: hdc_only has lnn_stage=off -- no trainable weights. "
                      "HDC codebook refitted on train flows each run. Correct by design.")
            else:
                print("  WARNING: Unexpected missing checkpoint for non-passthrough config!")

        # --- Streaming benchmark ---
        # Override dataset_raw_path to the real 30-entity UNSW-NB15 slice.
        # The snapshot records dataset_raw_path=null (falls back to 20-row fixture).
        # For a real benchmark we need the actual data.
        REAL_DATA_PATH = "data/raw/unsw_slice_5k.csv"
        if not Path(REAL_DATA_PATH).exists():
            print(f"  WARNING: Real data {REAL_DATA_PATH} not found. Falling back to fixture.")
            REAL_DATA_PATH = None

        # Root-cause note (PHASE13): data/splits/unsw_v1.json was created from
        # the 20-row fixture (19 unique flows, 5 test). Reusing it with 5k data
        # causes 4995 flows to default to 'train', leaving only 5 test events.
        # Using split_version=v_phase13 forces a fresh 60/20/20 split from
        # the 5k data (~1000 test events). unsw_v1.json is preserved for tests.
        SPLIT_VERSION = "v_phase13"

        print(f"  [stream] replay_rate={REPLAY_RATE:.0f}, warmup_n={WARMUP_N} ...")
        cfg = load_settings(config_path)
        overrides = {"split_version": SPLIT_VERSION}
        if REAL_DATA_PATH:
            overrides["dataset_raw_path"] = REAL_DATA_PATH
        cfg = cfg.model_copy(update=overrides)
        set_global_seed(cfg.seed)
        stream_res = run_stream(
            cfg,
            run_id_to_load=run_id,
            replay_rate=REPLAY_RATE,
            check_consistency=False,
        )

        ls = stream_res.latency_stats
        ms_stats = stream_res.memory_stats
        n_events = stream_res.detection_metrics.get("total_decisions", 0)
        device = ls.get("device", "cpu")

        print(f"  [stream] n_events={n_events}, device={device}, "
              f"warmup_n={ls.get('warmup_n', WARMUP_N)}")
        print(f"           p50={ls.get('p50_ms',0):.3f}ms  "
              f"p95={ls.get('p95_ms',0):.3f}ms  "
              f"p99={ls.get('p99_ms',0):.3f}ms  "
              f"max={ls.get('max_ms',0):.3f}ms")
        print(f"           throughput={ls.get('throughput_eps',0):.1f} eps  "
              f"peak_RSS={ms_stats.get('peak_rss_mb',0):.1f} MB")

        # --- Batch cross-check ---
        print(f"  [batch]  timing forward() pass for cross-check ...")
        batch_per_event_ms, batch_n = get_batch_per_event_ms(config_path, run_id, REAL_DATA_PATH)
        stream_p50 = ls.get("p50_ms", 0.0)
        ratio = stream_p50 / batch_per_event_ms if batch_per_event_ms > 0 else float("inf")
        print(f"  [batch]  per-event={batch_per_event_ms:.3f}ms over {batch_n} events | "
              f"stream_p50/batch = {ratio:.1f}x")
        if ratio > 20.0:
            print(f"  WARNING: ratio {ratio:.1f}x > 20x -- streaming pathologically slow vs batch!")
        else:
            print(f"  OK: ratio {ratio:.1f}x <= 20x (expected per-event overhead)")

        row = {
            "config": label,
            "run_id": run_id,
            "device": device,
            "checkpoint_present": ckpt_present,
            "checkpoint_epochs": snap_epochs,
            "checkpoint_dim": snap_dim,
            "checkpoint_hidden": snap_hidden,
            "n_events": n_events,
            "warmup_n": WARMUP_N,
            "latency_p50_ms": round(ls.get("p50_ms", 0.0), 4),
            "latency_p95_ms": round(ls.get("p95_ms", 0.0), 4),
            "latency_p99_ms": round(ls.get("p99_ms", 0.0), 4),
            "latency_max_ms": round(ls.get("max_ms", 0.0), 4),
            "latency_mean_ms": round(ls.get("mean_ms", 0.0), 4),
            "throughput_eps": round(ls.get("throughput_eps", 0.0), 2),
            "peak_rss_mb": round(ms_stats.get("peak_rss_mb", 0.0), 2),
            "batch_per_event_ms": round(batch_per_event_ms, 4),
            "stream_vs_batch_ratio": round(ratio, 2) if ratio != float("inf") else None,
        }
        all_results.append(row)

    # --- Summary table ---
    print()
    print("=" * 90)
    print("PHASE 13 FINAL BENCHMARK TABLE")
    print("Dataset: UNSW-NB15 (30 entities), Device: CPU, Warm-up N=10 discarded")
    print("Checkpoints: Phase 11 reduced-epoch (D=100, hidden=16, epochs=2, seed=42)")
    print("=" * 90)
    hdr = (
        f"{'Config':<16} {'Dev':<4} {'P50(ms)':>8} {'P95(ms)':>8} "
        f"{'P99(ms)':>8} {'Max(ms)':>8} {'Tput(eps)':>10} {'RSS(MB)':>8}"
    )
    unit = (
        f"{'':16} {'':4} {'':>8} {'':>8} {'':>8} {'':>8} {'':>10} {'':>8}"
    )
    print(hdr)
    print("-" * 90)
    for r in all_results:
        ck = "" if r["checkpoint_present"] else "*"
        print(
            f"{r['config']+ck:<16} {r['device']:<4} "
            f"{r['latency_p50_ms']:>8.3f} "
            f"{r['latency_p95_ms']:>8.3f} "
            f"{r['latency_p99_ms']:>8.3f} "
            f"{r['latency_max_ms']:>8.3f} "
            f"{r['throughput_eps']:>10.1f} "
            f"{r['peak_rss_mb']:>8.1f}"
        )
    print("=" * 90)
    print("* hdc_only: no .pt checkpoint by design (lnn_stage=off, no trainable weights)")

    # --- Persist ---
    results_path = out_dir / "results.json"
    with open(results_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nFull results saved to: {results_path}")


if __name__ == "__main__":
    main()
