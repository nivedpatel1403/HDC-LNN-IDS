import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, Any, Optional

from hdlnn.common.dataset_utils import compute_encoder_separability as _lib_compute_encoder_separability

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_encoder_separability(
    dataset_path: str = "datasets/UNSW_NB15_testing-set.csv",
    limit: int = 5000,
    D: int = 10000,
    output_json: Optional[Path] = Path("runs/encoder_separability.json")
) -> Dict[str, Dict[str, float]]:
    """Evaluates geometric separability between normal and attack traffic representations across encoders."""
    return _lib_compute_encoder_separability(
        dataset_path=dataset_path,
        limit=limit,
        D=D,
        output_json=output_json
    )


def print_separability_table(results: Dict[str, Dict[str, float]]) -> str:
    """Formats a markdown table displaying geometric separability metrics across encoders."""
    if not results:
        return "No empirical encoder separability metrics available."
    headers = [
        "Encoder Architecture", "Centroid Cos Dist", "Norm Euclidean Dist", 
        "Fisher Ratio", "Intra Sim", "Inter Sim", "Separability Margin"
    ]
    rows = []
    for enc_name, m in results.items():
        rows.append([
            enc_name,
            f"{m.get('cos_dist', 0.0):.4f}",
            f"{m.get('eucl_dist', 0.0):.4f}",
            f"{m.get('fisher_ratio', 0.0):.6f}",
            f"{m.get('intra_sim', 0.0):.4f}",
            f"{m.get('inter_sim', 0.0):.4f}",
            f"{m.get('margin', 0.0):.4f}"
        ])
        
    col_widths = [max(len(str(row[i])) for row in rows + [headers]) for i in range(len(headers))]
    header_str = " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers)))
    sep_str = "-|-".join("-" * col_widths[i] for i in range(len(headers)))
    
    lines = [
        f"| {header_str} |",
        f"|:{sep_str}:|"
    ]
    for row in rows:
        lines.append(f"| " + " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers))) + " |")
        
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate empirical geometric separability across HDC encoders.")
    parser.add_argument("--dataset", type=str, default="datasets/UNSW_NB15_testing-set.csv", help="Dataset CSV path.")
    parser.add_argument("--limit", type=int, default=5000, help="Number of flows to evaluate.")
    parser.add_argument("--d", type=int, default=10000, help="HDC vector dimension.")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Directory for output runs (default: runs).")
    parser.add_argument("--output-json", type=str, default=None, help="Destination JSON path for separability metrics.")
    args = parser.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        logger.error(f"Dataset path does not exist: {ds_path}")
        sys.exit(1)

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.output_json) if args.output_json else runs_dir / "encoder_separability.json"

    res = compute_encoder_separability(
        dataset_path=str(ds_path),
        limit=args.limit,
        D=args.d,
        output_json=out_json
    )
    tbl = print_separability_table(res)
    print("\n--- ENCODER-ONLY SEPARABILITY COMPARISON TABLE ---")
    print(tbl)
    print("---------------------------------------------------\n")


if __name__ == "__main__":
    main()
