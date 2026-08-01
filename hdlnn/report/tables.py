"""Table loading and markdown/CSV rendering for ablation comparison results."""

import json
from pathlib import Path
from typing import Any
import pandas as pd

from hdlnn.eval.harness import ComparisonTable


def load_comparison_table(path: str) -> ComparisonTable:
    """Load a ComparisonTable object from a comparison_table.json artifact file.

    Args:
        path: File path to comparison_table.json.

    Returns:
        Instantiated ComparisonTable object.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Comparison table artifact not found at '{path}'")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    return ComparisonTable(**data)


def render_markdown_table(table: ComparisonTable) -> str:
    """Render ComparisonTable into GFM Markdown with explicit device strings and OOV rate column.

    Note: Strictly reads pre-computed values from table artifact; never recomputes metrics.
    """
    lines: list[str] = []
    lines.append(f"# Ablation Comparison Table (Timestamp: {table.timestamp})")
    lines.append("")
    lines.append(
        "| Run ID | HDC | LNN | Model | F1 | Precision | Recall | AUROC | FPR@95%TPR | OOV Rate | Mean Latency | P95 Latency | Peak RSS | Model Size |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )

    for row in table.rows:
        run_id = row.get("run_id", "N/A")
        hdc = row.get("hdc_encoder", "N/A")
        lnn = row.get("lnn_stage", "N/A")
        m_type = row.get("sequence_model_type", "N/A")
        f1 = f"{row.get('f1', 0.0):.4f}"
        prec = f"{row.get('precision', 0.0):.4f}"
        rec = f"{row.get('recall', 0.0):.4f}"
        auroc = f"{row.get('auroc', 0.0):.4f}"
        fpr95 = f"{row.get('fpr_at_95_tpr', 0.0):.4f}"
        oov = f"{row.get('oov_rate', 0.0) * 100.0:.2f}%"

        # Explicitly attach device string to latency and memory values
        dev = "cpu"  # Standard default device label
        mean_lat = f"{row.get('mean_latency_ms', 0.0):.2f} ms ({dev})"
        p95_lat = f"{row.get('p95_latency_ms', 0.0):.2f} ms ({dev})"
        rss = f"{row.get('peak_rss_mb', 0.0):.1f} MB ({dev})"
        m_size = f"{row.get('model_size_mb', 0.0):.2f} MB ({dev})"

        line = f"| {run_id} | {hdc} | {lnn} | {m_type} | {f1} | {prec} | {rec} | {auroc} | {fpr95} | {oov} | {mean_lat} | {p95_lat} | {rss} | {m_size} |"
        lines.append(line)

    # Render Cross-Dataset Transfer Placeholder Row
    cd = table.cross_dataset_transfer
    if cd:
        ds_name = cd.get("dataset", "CICIoT2023")
        status = cd.get("status", "not_run: CICIoTLoader not yet implemented (Phase 2 stub)")
        lines.append(
            f"| **Cross-Dataset ({ds_name})** | - | - | - | N/A | N/A | N/A | N/A | N/A | N/A | `{status}` | `{status}` | `{status}` | `{status}` |"
        )

    return "\n".join(lines)


def render_csv(table: ComparisonTable) -> str:
    """Render ComparisonTable rows and cross-dataset status into CSV string.

    Note: Strictly reads pre-computed values from table artifact; never recomputes metrics.
    """
    rows_data: list[dict[str, Any]] = list(table.rows)

    if table.cross_dataset_transfer:
        cd = dict(table.cross_dataset_transfer)
        cd["run_id"] = f"cross_dataset_{cd.get('dataset', 'CICIoT2023')}"
        rows_data.append(cd)

    df = pd.DataFrame(rows_data)
    return df.to_csv(index=False)
