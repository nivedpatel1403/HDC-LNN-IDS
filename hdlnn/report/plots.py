"""Plotting utilities for trajectory divergence curves and ablation comparison bar charts."""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

from hdlnn.eval.harness import ComparisonTable


def plot_trajectory_divergence(
    run_id: str, entity_id: str, runs_dir: str = "runs"
) -> Figure:
    """Plot trajectory drift score over time against decision threshold line from decisions.parquet artifact.

    Note: Strictly reads pre-computed drift scores and thresholds from decisions.parquet artifact;
    never recomputes scores or thresholds inline.

    Args:
        run_id: Identifier of the experiment run.
        entity_id: Identifier string of the entity to plot.
        runs_dir: Base runs directory path.

    Returns:
        Matplotlib Figure object.

    Raises:
        FileNotFoundError: If decisions.parquet does not exist.
        ValueError: If entity_id is not found in decisions.parquet.
    """
    parquet_path = Path(runs_dir) / run_id / "decisions.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Decisions artifact not found at '{parquet_path}'")

    df = pd.read_parquet(parquet_path)
    df_entity = df[df["entity_id"] == entity_id]

    if df_entity.empty:
        raise ValueError(f"Entity '{entity_id}' not found in decisions artifact '{parquet_path}'")

    fig, ax = plt.subplots(figsize=(10, 5))

    steps = list(range(len(df_entity)))
    drift_scores = df_entity["drift_score"].values
    is_anomalous = df_entity["is_anomalous"].values
    threshold = float(df_entity["threshold"].iloc[0])

    # Plot drift score line trajectory
    ax.plot(steps, drift_scores, color="gray", linestyle="-", linewidth=1.5, alpha=0.7, label="Drift Score Trajectory")

    # Mark normal points in blue and anomaly alerts in red
    normal_indices = [i for i, anomaly in enumerate(is_anomalous) if not anomaly]
    anomaly_indices = [i for i, anomaly in enumerate(is_anomalous) if anomaly]

    if normal_indices:
        ax.scatter(
            normal_indices,
            drift_scores[normal_indices],
            color="blue",
            label="Normal",
            zorder=3,
            s=30,
        )

    if anomaly_indices:
        ax.scatter(
            anomaly_indices,
            drift_scores[anomaly_indices],
            color="red",
            marker="^",
            s=60,
            label="Anomaly Alert",
            zorder=4,
        )

    # Plot horizontal reference threshold line
    ax.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Calibrated Threshold ({threshold:.3f})",
    )

    ax.set_title(f"Trajectory Drift Score — Entity: '{entity_id}' (Run: '{run_id}')", fontsize=12)
    ax.set_xlabel("Flow Step Index", fontsize=10)
    ax.set_ylabel("Mahalanobis Drift Score", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

    fig.tight_layout()
    return fig


def plot_comparison_bars(
    table: ComparisonTable, metric: str = "f1"
) -> Figure:
    """Plot bar chart comparing a chosen metric across configuration rows in ComparisonTable.

    Note: Strictly reads pre-computed metric values from ComparisonTable; never recomputes metrics.

    Args:
        table: ComparisonTable object.
        metric: Metric field key to compare (e.g. 'f1', 'auroc', 'mean_latency_ms', 'p95_latency_ms').

    Returns:
        Matplotlib Figure object.
    """
    if not table.rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Empty ComparisonTable", ha="center", va="center")
        return fig

    run_ids = [r.get("run_id", f"run_{i}") for i, r in enumerate(table.rows)]
    values = [float(r.get(metric, 0.0)) for r in table.rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(run_ids, values, color="skyblue", edgecolor="navy", width=0.5)

    # Annotate value text above each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}" if height < 10 else f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(f"Ablation Suite Comparison — Metric: {metric.upper()}", fontsize=12)
    ax.set_xlabel("Configuration Run ID", fontsize=10)
    ax.set_ylabel(metric.replace("_", " ").upper(), fontsize=10)
    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels(run_ids, rotation=30, ha="right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()
    return fig
