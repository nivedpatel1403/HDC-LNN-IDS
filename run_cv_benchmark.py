"""Dedicated Cross-Validation Benchmark Runner for HDC-LNN and Baselines.

Executes K-Fold (default: 5-Fold) Entity-Disjoint Cross-Validation across benchmark datasets,
evaluating all 7 baseline architectures:
  - HDC-LNN (Continuous-time Liquid Neural Network + HDC)
  - Mamba-2 (Pure PyTorch SSM)
  - 1D-CNN (Temporal Conv)
  - LSTM-Autoencoder (Reconstruction AE)
  - LSTM (Recurrent baseline)
  - Transformer-Encoder (Bidirectional Multi-Head Attention)
  - Mini-GPT (Causal Autoregressive Decoder)

Usage:
    python run_cv_benchmark.py --dataset UNSW_NB15_testing-set.csv --cv-folds 5
    python run_cv_benchmark.py --quick --baselines transformer-enc mini-gpt
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from hdlnn.common.config import load_config
from hdlnn.data.loaders import load_dataset_flows
from hdlnn.data.cv_splitter import kfold_entity_splits
from hdlnn.eval.harness import run_experiment
from hdlnn.eval.injector import inject_threat_scenarios
from hdlnn.report.tables import _format_metric

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("run_cv_benchmark")

ALL_BASELINES = [
    "mamba2",
    "mamba3",
    "autoencoder",
    "cnn",
    "hdc-lnn",
    "lstm",
    "transformer-enc",
    "mini-gpt",
]

DISPLAY_NAMES = {
    "mamba2": "Mamba-2 (Pure PyTorch SSM)",
    "mamba3": "Mamba-3 (Complex SSM & Heavy-Tail)",
    "autoencoder": "Autoencoder (LSTM-AE)",
    "cnn": "1D-CNN (Temporal Conv)",
    "hdc-lnn": "HDC-LNN (Ours)",
    "lstm": "LSTM (Recurrent Baseline)",
    "transformer-enc": "Transformer-Encoder (Self-Attention)",
    "mini-gpt": "Mini-GPT (Causal Decoder)",
}


def generate_cv_summary_table(cv_results: Dict[str, Dict[str, Any]]) -> str:
    """Generates a comprehensive Markdown table summarizing K-Fold CV metrics (mean +/- std)."""
    headers = [
        "Model Architecture", "# Params", "Model Size", "F1 Score", "Precision", "Recall", 
        "AUROC", "PR-AUC", "FPR @ 95% TPR", "Latency (ms)", "Throughput (f/s)", "Peak RSS (MB)"
    ]

    rows = []
    for b_key, data in cv_results.items():
        name = DISPLAY_NAMES.get(b_key, b_key)
        msize = data.get("model_size", {})
        p_count = msize.get("total_params", 0)
        p_str = f"{p_count:,}" if p_count > 0 else "-"
        mb_size = msize.get("size_mb", 0.0)
        mb_str = f"{mb_size:.2f} MB" if mb_size > 0 else "-"

        f1_str = f"{data.get('f1_score', 0.0):.4f} ± {data.get('f1_score_std', 0.0):.4f}"
        prec_str = f"{data.get('precision', 0.0):.4f} ± {data.get('precision_std', 0.0):.4f}"
        rec_str = f"{data.get('recall', 0.0):.4f} ± {data.get('recall_std', 0.0):.4f}"
        auroc_str = f"{data.get('auroc', 0.0):.4f} ± {data.get('auroc_std', 0.0):.4f}"
        prauc_str = f"{data.get('pr_auc', 0.0):.4f} ± {data.get('pr_auc_std', 0.0):.4f}"
        fpr_str = f"{data.get('fpr_at_95_tpr', 0.0):.4f} ± {data.get('fpr_at_95_tpr_std', 0.0):.4f}"
        lat_str = f"{data.get('latency_ms_per_flow', 0.0):.2f} ms"
        thru_str = f"{data.get('throughput_flows_sec', 0.0):.1f}"
        mem_str = f"{data.get('peak_rss_mb', 0.0):.1f}"

        rows.append([
            name, p_str, mb_str, f1_str, prec_str, rec_str, auroc_str, prauc_str, fpr_str, lat_str, thru_str, mem_str
        ])

    col_widths = [max(len(str(r[i])) for r in rows + [headers]) for i in range(len(headers))]
    header_str = " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers)))
    sep_str = "-|-".join("-" * col_widths[i] for i in range(len(headers)))

    lines = [f"| {header_str} |", f"|:{sep_str}:|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[i]).ljust(col_widths[i]) for i in range(len(headers))) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run K-Fold Cross-Validation on HDC-LNN and Baselines.")
    parser.add_argument("--dataset", type=str, default="UNSW_NB15_testing-set.csv", help="Target dataset CSV file in datasets/.")
    parser.add_argument("--datasets-dir", type=str, default="datasets", help="Directory containing dataset files.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds (default: 5).")
    parser.add_argument("--baselines", type=str, nargs="+", default=None, help="Subset of baselines to evaluate.")
    parser.add_argument("--limit", type=int, default=30000, help="Flow limit per evaluation (default: 30000).")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test mode (5000 flows, 1 epoch).")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Output directory for runs.")
    parser.add_argument("--output-report", type=str, default="cv_benchmark_summary.md", help="Output markdown summary path.")
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    dataset_file = datasets_dir / args.dataset
    if not dataset_file.exists():
        logger.error(f"Dataset file not found: {dataset_file}")
        return

    limit = 5000 if args.quick else args.limit
    k = args.cv_folds
    baselines = args.baselines if args.baselines else ALL_BASELINES
    baselines = [b for b in baselines if b in ALL_BASELINES]

    logger.info(f"Starting {k}-Fold Cross-Validation on {dataset_file.name}")
    logger.info(f"Target baselines: {baselines} (limit={limit}, quick={args.quick})")

    # Match configuration
    base_config_path = Path("configs/base.yaml")
    dataset_config_path = Path("configs/dataset_unsw_nb15.yaml")
    if "kdd" in dataset_file.name.lower():
        dataset_config_path = Path("configs/dataset_kdd.yaml")
    elif "ton" in dataset_file.name.lower():
        dataset_config_path = Path("configs/dataset_ton_iot.yaml")

    config = load_config(base_config_path, dataset_config_path)

    # 1. Load data once
    logger.info(f"Loading dataset flows from {dataset_file.name} (limit={limit})...")
    flows = load_dataset_flows(config, Path("."), source_file=str(dataset_file), limit=limit)

    # 2. Inject threat scenarios once
    logger.info("Injecting threat scenarios...")
    flows = inject_threat_scenarios(
        flows,
        seed=config.seed,
        categorical_cols=config.categorical_columns,
        numerical_cols=config.numerical_columns
    )

    # 3. Create K entity-disjoint folds
    logger.info(f"Generating {k} entity-disjoint folds...")
    cv_folds_data = list(kfold_entity_splits(flows, k=k, seed=config.seed))

    runs_dir = Path(args.runs_dir)
    cv_summary: Dict[str, Dict[str, Any]] = {}

    for b_idx, baseline in enumerate(baselines, 1):
        logger.info(f"\n=======================================================")
        logger.info(f"[{b_idx}/{len(baselines)}] Evaluating Baseline: {baseline.upper()} across {k} Folds")
        logger.info(f"=======================================================")

        fold_metrics = []
        for fold_idx, tr_flows, v_flows, te_flows in cv_folds_data:
            run_id = f"cv_{dataset_file.stem}_{baseline.replace('-', '')}_f{fold_idx + 1}"
            cfg = load_config(base_config_path, dataset_config_path)
            if args.quick and cfg._data.get("model", {}).get("epochs", 2) > 1:
                cfg._data["model"]["epochs"] = 1

            start_t = time.time()
            try:
                m = run_experiment(
                    config=cfg,
                    baseline_name=baseline,
                    run_id=run_id,
                    output_dir=runs_dir,
                    pre_split_flows=(tr_flows, v_flows, te_flows)
                )
                fold_metrics.append(m)
                logger.info(f"  Fold {fold_idx + 1}/{k} [{baseline}] took {time.time() - start_t:.2f}s | F1: {m.get('f1_score', 0.0):.4f} | AUROC: {m.get('auroc', 0.0):.4f}")
            except Exception as e:
                logger.error(f"Error in fold {fold_idx + 1} for {baseline}: {e}", exc_info=True)

        if fold_metrics:
            keys = ["f1_score", "precision", "recall", "auroc", "pr_auc", "fpr_at_95_tpr", "latency_ms_per_flow", "throughput_flows_sec", "peak_rss_mb"]
            b_summary: Dict[str, Any] = {}
            for k_metric in keys:
                vals = [m.get(k_metric, 0.0) for m in fold_metrics if m.get(k_metric) is not None]
                b_summary[k_metric] = float(np.mean(vals)) if vals else 0.0
                b_summary[f"{k_metric}_std"] = float(np.std(vals)) if vals else 0.0

            if "model_size" in fold_metrics[0]:
                b_summary["model_size"] = fold_metrics[0]["model_size"]

            b_summary["folds_count"] = len(fold_metrics)
            cv_summary[baseline] = b_summary

            logger.info(
                f"\n>>> {baseline.upper()} {k}-Fold Result: "
                f"F1 = {b_summary['f1_score']:.4f} ± {b_summary['f1_score_std']:.4f}, "
                f"AUROC = {b_summary['auroc']:.4f} ± {b_summary['auroc_std']:.4f}, "
                f"Throughput = {b_summary['throughput_flows_sec']:.1f} flows/s"
            )

    # Generate Markdown Summary
    table_md = generate_cv_summary_table(cv_summary)
    full_report = (
        f"# {k}-Fold Cross-Validation Benchmark Summary\n\n"
        f"- **Dataset**: `{dataset_file.name}`\n"
        f"- **Folds**: {k} Entity-Disjoint Folds\n"
        f"- **Flow Limit**: {limit:,} flows\n"
        f"- **Evaluated Models**: {len(cv_summary)} architectures\n\n"
        f"## Cross-Validation Performance (Mean ± Standard Deviation)\n\n"
        f"{table_md}\n"
    )

    out_path = Path(args.output_report)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_report)

    print("\n" + "=" * 60)
    print(full_report)
    print("=" * 60 + "\n")
    logger.info(f"Cross-Validation Summary successfully written to {out_path}")


if __name__ == "__main__":
    main()
