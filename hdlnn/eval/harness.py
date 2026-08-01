"""Experiment harness orchestrating execution, seed initialization, artifact persistence, and ablation suites."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import subprocess
from typing import Any
import pandas as pd
from pydantic import BaseModel, Field
import torch
import yaml

from hdlnn.common.config import Settings
from hdlnn.common.logging import get_logger
from hdlnn.common.seed import set_global_seed
from hdlnn.contracts.flow import CanonicalFlow
from hdlnn.eval.interfaces import IPipeline
from hdlnn.eval.pipeline_adapter import StandardPipeline
from hdlnn.eval.run_result import RunResult


class ComparisonTable(BaseModel):
    """Container model representing aggregated multi-config ablation comparison results."""

    timestamp: str
    rows: list[dict[str, Any]] = Field(default_factory=list)
    cross_dataset_transfer: dict[str, Any] = Field(default_factory=dict)


def get_git_commit_hash() -> str:
    """Safely obtain current git commit hash via subprocess."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def run_experiment(
    config: Settings,
    pipeline: IPipeline,
    train_flows: list[CanonicalFlow],
    val_flows: list[CanonicalFlow],
    test_flows: list[CanonicalFlow],
) -> RunResult:
    """Orchestrate end-to-end experiment execution and write all run artifacts to disk.

    Args:
        config: Application Settings object.
        pipeline: IPipeline implementation (e.g. StandardPipeline).
        train_flows: Training flow list.
        val_flows: Validation flow list.
        test_flows: Testing flow list.

    Returns:
        Populated RunResult object.
    """
    logger = get_logger("hdlnn.eval.harness", run_id=config.run_id)

    # 1. Enforce global seed reproducibility (P5)
    set_global_seed(config.seed)
    logger.info("Set global seed to %d for experiment run_id='%s'", config.seed, config.run_id)

    # 2. Get git commit hash gracefully
    commit_hash = get_git_commit_hash()
    if commit_hash == "unknown":
        logger.warning("Git commit hash could not be retrieved; recording 'unknown'")

    # 3. Execute pipeline
    result = pipeline.run(train_flows, val_flows, test_flows)
    result.git_commit_hash = commit_hash

    # 4. Prepare run output directory runs/<run_id>/
    run_dir = Path("runs") / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Artifact 1: config.snapshot.yaml (config copied verbatim)
    config_path = run_dir / "config.snapshot.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)

    # Artifact 2: git_commit.txt
    commit_path = run_dir / "git_commit.txt"
    with open(commit_path, "w", encoding="utf-8") as f:
        f.write(f"{commit_hash}\n")

    # Artifact 3: metrics.json (single source of truth for reported numbers)
    metrics_payload = {
        "run_id": config.run_id,
        "detection_metrics": result.detection_metrics,
        "latency_stats": result.latency_stats,
        "memory_stats": result.memory_stats,
        "git_commit_hash": commit_hash,
    }
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # Artifact 4: decisions.parquet (Parquet format per ADR-005)
    decisions_path = run_dir / "decisions.parquet"
    if result.decisions:
        dec_dicts = [d.model_dump() for d in result.decisions]
        df = pd.DataFrame(dec_dicts)
        if "t" in df.columns:
            df["t"] = df["t"].astype(str)
        df.to_parquet(decisions_path, index=False)
    else:
        df = pd.DataFrame(
            columns=[
                "entity_id",
                "flow_id",
                "t",
                "drift_score",
                "threshold",
                "is_anomalous",
                "manifold_version",
                "ground_truth_label",
                "attack_category",
            ]
        )
        df.to_parquet(decisions_path, index=False)

    # Artifact 5: Model Checkpoint Persistence (Spec §14)
    # Readout head discarded from checkpoint as a training scaffold; only deployed sequence model weights saved.
    model_checkpoint_path = run_dir / "model_checkpoint.pt"
    if hasattr(pipeline, "sequence_model"):
        seq_model = getattr(pipeline, "sequence_model")
        model_module = getattr(
            seq_model,
            "cfc",
            getattr(
                seq_model,
                "ltc",
                getattr(seq_model, "lstm", getattr(seq_model, "conv", None)),
            ),
        )
        if isinstance(model_module, torch.nn.Module):
            torch.save(model_module.state_dict(), model_checkpoint_path)
            logger.info("Saved model checkpoint to '%s'", model_checkpoint_path)

    logger.info("Successfully persisted all run artifacts under '%s'", run_dir)
    return result


def run_cross_dataset_transfer(
    trained_pipeline: IPipeline, ciciot_flows: list[Any]
) -> dict[str, Any]:
    """Placeholder stub for zero-shot cross-dataset evaluation on CICIoT2023.

    Note: CICIoTLoader is a Phase 2 stub raising NotImplementedError. Cross-dataset numbers are not faked.
    """
    return {
        "status": "not_run: CICIoTLoader not yet implemented (Phase 2 stub)",
        "dataset": "CICIoT2023",
        "f1": None,
        "auroc": None,
        "accuracy": None,
    }


def run_ablation_suite(
    configs: list[Settings],
    train_flows: list[CanonicalFlow],
    val_flows: list[CanonicalFlow],
    test_flows: list[CanonicalFlow],
) -> ComparisonTable:
    """Execute ablation suite over all specified configuration YAMLs using the SAME split.

    Args:
        configs: List of Settings configuration objects.
        train_flows: Shared training flow list.
        val_flows: Shared validation flow list.
        test_flows: Shared testing flow list.

    Returns:
        ComparisonTable containing aggregated rows and cross-dataset transfer placeholder.
    """
    logger = get_logger("hdlnn.eval.harness.ablation_suite")
    logger.info("Starting ablation suite run with %d configurations", len(configs))

    rows: list[dict[str, Any]] = []

    suite_start = time.time()
    for idx, cfg in enumerate(configs, start=1):
        t_cfg_start = time.time()
        logger.info("[%d/%d] Executing ablation config run_id='%s'...", idx, len(configs), cfg.run_id)
        pipeline = StandardPipeline(settings=cfg)
        result = run_experiment(cfg, pipeline, train_flows, val_flows, test_flows)
        t_cfg_elapsed = time.time() - t_cfg_start
        logger.info(
            "[%d/%d] Finished ablation config run_id='%s' in %.2fs (F1=%.4f, AUROC=%.4f)",
            idx,
            len(configs),
            cfg.run_id,
            t_cfg_elapsed,
            result.detection_metrics.get("f1", 0.0),
            result.detection_metrics.get("auroc", 0.0),
        )

        row_dict = {
            "run_id": result.run_id,
            "hdc_encoder": cfg.hdc_encoder,
            "lnn_stage": cfg.lnn_stage,
            "sequence_model_type": cfg.sequence_model_type,
            "f1": result.detection_metrics.get("f1", 0.0),
            "precision": result.detection_metrics.get("precision", 0.0),
            "recall": result.detection_metrics.get("recall", 0.0),
            "auroc": result.detection_metrics.get("auroc", 0.0),
            "fpr_at_95_tpr": result.detection_metrics.get("fpr_at_95_tpr", 0.0),
            "oov_rate": result.detection_metrics.get("oov_rate", 0.0),
            "mean_latency_ms": result.latency_stats.get("mean_ms", 0.0),
            "p95_latency_ms": result.latency_stats.get("p95_ms", 0.0),
            "peak_rss_mb": result.memory_stats.get("peak_rss_mb", 0.0),
            "model_size_mb": result.memory_stats.get("model_size_mb", 0.0),
        }
        rows.append(row_dict)

    t_suite_elapsed = time.time() - suite_start
    logger.info("Completed ablation suite run across %d configs in %.2fs", len(configs), t_suite_elapsed)

    cross_dataset_placeholder = run_cross_dataset_transfer(None, [])

    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    table = ComparisonTable(
        timestamp=ts_str,
        rows=rows,
        cross_dataset_transfer=cross_dataset_placeholder,
    )

    # Persist ComparisonTable json and csv under runs/ablation_suite_<timestamp>/
    suite_dir = Path("runs") / f"ablation_suite_{ts_str}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    table_payload = table.model_dump()
    with open(suite_dir / "comparison_table.json", "w", encoding="utf-8") as f:
        json.dump(table_payload, f, indent=2)

    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(suite_dir / "comparison_table.csv", index=False)

    logger.info("Saved ablation comparison suite table to '%s'", suite_dir)
    return table
