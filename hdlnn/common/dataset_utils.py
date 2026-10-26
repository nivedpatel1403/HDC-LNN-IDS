"""Dataset and config resolution utilities for HDC-LNN benchmark harness."""

import math
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch

from hdlnn.hdc.encoder import RecordEncoder
from hdlnn.hdc.alternative_encoders import SAXEncoder, RFFEncoder
from hdlnn.data.loaders import _load_csv_as_flows

logger = logging.getLogger(__name__)

BASELINES: List[str] = [
    "mamba2",
    "cnn",
    "autoencoder",
    "hdc-lnn",
    "lstm",
    "ft-transformer",
    "saint"
]


def find_dataset_files(
    data_dir: Path = Path("datasets"),
    target_file: Optional[str] = None,
    all_datasets: bool = False
) -> List[Path]:
    """Discovers benchmark dataset files in the datasets folder."""
    datasets_dir = Path(data_dir)
    if not datasets_dir.exists():
        logger.error(f"Datasets directory not found: {datasets_dir}")
        return []

    if target_file:
        t_path = Path(target_file) if Path(target_file).exists() else datasets_dir / target_file
        if t_path.exists():
            return [t_path]
        else:
            logger.error(f"Requested dataset file not found: {target_file}")
            return []

    if all_datasets:
        # Discover all CSV files in directory (ignoring metadata schema definition files)
        return [
            f for f in sorted(datasets_dir.glob("*.csv"))
            if not f.name.endswith("_Names.csv") and not f.name.endswith("_Feature_Names.csv")
        ]

    # Priority search for primary evaluation datasets across different domains
    primary_candidates = [
        "UNSW_NB15_testing-set.csv",
        "kdd_test.csv",
        "ton-iot.csv"
    ]
    found = [datasets_dir / name for name in primary_candidates if (datasets_dir / name).exists()]

    if not found:
        # Fallback to all CSV files in directory (ignoring metadata schema definition files)
        found = [
            f for f in sorted(datasets_dir.glob("*.csv"))
            if not f.name.endswith("_Names.csv") and not f.name.endswith("_Feature_Names.csv")
        ]

    return found


def match_config_for_dataset(
    dataset_path: Path,
    configs_dir: Optional[Path] = None
) -> Path:
    """Matches appropriate config YAML for the given dataset file."""
    cfg_dir = Path(configs_dir) if configs_dir is not None else Path("configs")
    dataset_path = Path(dataset_path)
    name_lower = dataset_path.name.lower()

    if "kdd" in name_lower:
        cfg = cfg_dir / "dataset_kdd.yaml"
        if cfg.exists():
            return cfg
    if "ton" in name_lower or ("iot" in name_lower and "botnet" not in name_lower and "unsw" not in name_lower):
        cfg = cfg_dir / "dataset_ton_iot.yaml"
        if cfg.exists():
            return cfg
    if "botnet" in name_lower:
        cfg = cfg_dir / "dataset_unsw_botnet.yaml"
        if cfg.exists():
            return cfg
    if "dataset" in name_lower and any(c.isdigit() for c in name_lower) and "unsw" not in name_lower:
        cfg = cfg_dir / "dataset_iot23.yaml"
        if cfg.exists():
            return cfg
    if "unsw" in name_lower or "nb15" in name_lower:
        cfg = cfg_dir / "dataset_unsw_nb15.yaml"
        if cfg.exists():
            return cfg

    # Inspect CSV header if file exists to detect dataset schema dynamically
    if dataset_path.exists():
        try:
            with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline().strip()
                header = first_line.lower()
                cols = [c.strip() for c in first_line.split(",")]
                if "src_bytes" in header and "dst_bytes" in header and "count" in header:
                    cfg = cfg_dir / "dataset_kdd.yaml"
                    if cfg.exists():
                        return cfg
                elif len(cols) in (41, 42, 43) and cols[0].isdigit() and cols[1] in ("tcp", "udp", "icmp"):
                    # Standard headerless KDD/NSL-KDD CSV format
                    cfg = cfg_dir / "dataset_kdd.yaml"
                    if cfg.exists():
                        return cfg
                elif "ts" in header and "conn_state" in header:
                    cfg = cfg_dir / "dataset_ton_iot.yaml"
                    if cfg.exists():
                        return cfg
                elif "sbytes" in header and "dbytes" in header:
                    cfg = cfg_dir / "dataset_unsw_nb15.yaml"
                    if cfg.exists():
                        return cfg
        except Exception:
            pass

    return cfg_dir / "dataset_unsw_nb15.yaml"


def compute_encoder_separability(
    dataset_path: str = "datasets/UNSW_NB15_testing-set.csv",
    limit: int = 5000,
    D: int = 10000,
    output_json: Optional[Path] = Path("runs/encoder_separability.json")
) -> Dict[str, Dict[str, float]]:
    """Evaluates geometric separability between normal and attack traffic representations across encoders."""
    categorical_cols = ["proto", "service", "state"]
    numerical_cols = [
        "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
        "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
        "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
        "dmean", "trans_depth", "response_body_len"
    ]

    csv_path = Path(dataset_path)
    if not csv_path.exists():
        logger.warning(f"Dataset path {dataset_path} does not exist for separability evaluation.")
        return {}

    flows = _load_csv_as_flows(
        csv_path=csv_path,
        categorical_columns=categorical_cols,
        numerical_columns=numerical_cols,
        label_column="label",
        entity_id_column="srcip",
        limit=limit
    )
    logger.info(f"Loaded {len(flows)} flows for encoder separability evaluation.")

    normal_flows = [f for f in flows if f.label == 0]
    attack_flows = [f for f in flows if f.label == 1]

    if not normal_flows or not attack_flows:
        logger.warning(f"Dataset {dataset_path} does not contain both normal and attack flows for separability evaluation.")
        return {}

    split_idx = int(len(normal_flows) * 0.6)
    train_normals = normal_flows[:split_idx]
    eval_normals = normal_flows[split_idx:]
    eval_attacks = attack_flows[:len(eval_normals)]

    if len(train_normals) == 0 or len(eval_normals) == 0 or len(eval_attacks) == 0:
        logger.warning(f"Insufficient normal/attack flows after split in {dataset_path}.")
        return {}

    logger.info(
        f"Train Normals (fitting): {len(train_normals)}, "
        f"Eval Normals: {len(eval_normals)}, Eval Attacks: {len(eval_attacks)}"
    )

    encoders = {
        "HDC (RecordEncoder)": RecordEncoder(
            D=D,
            categorical_columns=categorical_cols,
            numerical_columns=numerical_cols
        ),
        "SAX (Symbolic Aggregate)": SAXEncoder(
            D=D,
            alphabet_size=8,
            categorical_columns=categorical_cols,
            numerical_columns=numerical_cols
        ),
        "RFF (Random Fourier)": RFFEncoder(
            D=D,
            gamma=0.5,
            categorical_columns=categorical_cols,
            numerical_columns=numerical_cols
        )
    }

    results = {}

    for name, encoder in encoders.items():
        logger.info(f"Fitting {name} on {len(train_normals)} normal flows...")
        encoder.fit(train_normals)

        logger.info(f"Encoding eval normal and attack flows with {name}...")
        norm_hvs = torch.stack([hv.vector for hv in encoder.encode_batch(eval_normals)])
        att_hvs = torch.stack([hv.vector for hv in encoder.encode_batch(eval_attacks)])

        # 1. Centroids
        mu_norm = norm_hvs.mean(dim=0)
        mu_att = att_hvs.mean(dim=0)

        # 2. Centroid Cosine Distance (1 - cos_sim)
        cos_sim = torch.cosine_similarity(mu_norm.unsqueeze(0), mu_att.unsqueeze(0)).item()
        centroid_cos_dist = float(1.0 - cos_sim)

        # 3. Normalized Euclidean Centroid Distance (||mu_0 - mu_1|| / sqrt(D))
        diff_norm = torch.norm(mu_norm - mu_att, p=2).item()
        eucl_dist = float(diff_norm / math.sqrt(D))

        # 4. Fisher-like Class Separability Ratio: ||mu_0 - mu_1||^2 / (var_0 + var_1)
        var_norm = norm_hvs.var(dim=0).sum().item()
        var_att = att_hvs.var(dim=0).sum().item()
        fisher_ratio = float((diff_norm ** 2) / (var_norm + var_att + 1e-8))

        # 5. Mean Intra-Normal Cosine Similarity vs Inter-Class Similarity
        norm_unit = norm_hvs / norm_hvs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        att_unit = att_hvs / att_hvs.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        sub_n = min(len(norm_unit), 200)
        sub_a = min(len(att_unit), 200)
        intra_norm_sim = float(torch.mm(norm_unit[:sub_n], norm_unit[:sub_n].T).mean().item())
        inter_class_sim = float(torch.mm(norm_unit[:sub_n], att_unit[:sub_a].T).mean().item())
        margin = float(intra_norm_sim - inter_class_sim)

        results[name] = {
            "cos_dist": centroid_cos_dist,
            "eucl_dist": eucl_dist,
            "fisher_ratio": fisher_ratio,
            "intra_sim": intra_norm_sim,
            "inter_sim": inter_class_sim,
            "margin": margin
        }

    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(exist_ok=True, parents=True)
        with open(out_p, "w", encoding="utf-8") as jf:
            json.dump(results, jf, indent=2)
        logger.info(f"Saved empirical encoder separability metrics to {out_p}")

    return results
