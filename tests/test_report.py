"""Ablation tier tests for L5 Reporting Layer (table formatting, plot generation, dashboard)."""

from pathlib import Path
import matplotlib
import pytest

from hdlnn.eval.harness import ComparisonTable
from hdlnn.report.plots import plot_comparison_bars, plot_trajectory_divergence
from hdlnn.report.tables import render_csv, render_markdown_table

matplotlib.use("Agg")  # Non-interactive backend for headless unit tests


@pytest.mark.ablation
def test_table_rendering_markdown_and_csv(tmp_path):
    row_dict = {
        "run_id": "hybrid_cfc_v1",
        "hdc_encoder": "on",
        "lnn_stage": "on",
        "sequence_model_type": "cfc",
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.96,
        "f1": 0.95,
        "auroc": 0.98,
        "p95_latency_ms": 1.25,
        "throughput_fps": 800.0,
        "peak_memory_mb": 12.5,
        "oov_rate": 0.01,
    }
    table = ComparisonTable(
        timestamp="20260801_100000",
        rows=[row_dict],
    )

    md = render_markdown_table(table)
    assert "| Run ID | HDC | LNN | Model | F1 | Precision | Recall | AUROC |" in md
    assert "hybrid_cfc_v1" in md

    csv_str = render_csv(table)
    csv_path = tmp_path / "table.csv"
    csv_path.write_text(csv_str, encoding="utf-8")
    assert csv_path.exists()
    content = csv_path.read_text(encoding="utf-8")
    assert "hybrid_cfc_v1" in content


@pytest.mark.ablation
def test_plot_generation(tmp_path):
    row_dict = {
        "run_id": "hybrid_cfc_v1",
        "hdc_encoder": "on",
        "lnn_stage": "on",
        "sequence_model_type": "cfc",
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.96,
        "f1": 0.95,
        "auroc": 0.98,
        "p95_latency_ms": 1.25,
        "throughput_fps": 800.0,
        "peak_memory_mb": 12.5,
        "oov_rate": 0.01,
    }
    table = ComparisonTable(
        timestamp="20260801_100000",
        rows=[row_dict],
    )

    fig_bars = plot_comparison_bars(table, metric="f1")
    assert fig_bars is not None

    chart_path = tmp_path / "bar_chart.png"
    fig_bars.savefig(chart_path)
    assert chart_path.exists()


@pytest.mark.ablation
def test_reporting_layer_does_not_compute_metrics():
    """Static audit asserting hdlnn/report module files contain zero metric calculation functions."""
    report_dir = Path(__file__).parent.parent / "hdlnn" / "report"

    for py_file in report_dir.glob("*.py"):
        code = py_file.read_text(encoding="utf-8")
        assert "sklearn.metrics" not in code, f"Forbidden import found in {py_file}"
        assert "confusion_matrix" not in code, f"Forbidden calculation found in {py_file}"
        assert "f1_score" not in code, f"Forbidden calculation found in {py_file}"
