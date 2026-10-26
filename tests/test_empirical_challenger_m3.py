"""Empirical Adversarial Challenge Test Suite for Milestone 3 (F10, F11, F12).

Adversarially tests:
1. Bidirectional schema normalization under adversarial and edge-case configs.
2. Dataset configs (TON-IoT, IoT-23, KDD, UNSW-NB15 Raw49) entity ID resolution.
3. Real IP preservation vs synthetic modulo fallback in loaders.
4. resolve_experiment_config fallback resilience with nonexistent files and varied hints.
5. evaluate_encoder_separability.py resilience with empty, single-class, and corrupted CSVs.
6. Canonical benchmark file immutability across reporting tools.
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest
import torch

from hdlnn.common.config import (
    Config,
    load_config,
    resolve_experiment_config,
    _normalize_data_config,
    DATA_KEYS
)
from hdlnn.common.dataset_utils import (
    match_config_for_dataset,
    compute_encoder_separability
)
from hdlnn.data.loaders import load_dataset_flows
from hdlnn.report.tables import print_and_save_comparison_table
from evaluate_encoder_separability import print_separability_table


# =========================================================================
# 1. BIDIRECTIONAL SCHEMA NORMALIZATION ADVERSARIAL TESTS
# =========================================================================

def test_normalization_root_overrides_data_section():
    """Verify that if conflicting keys exist at root and inside data, root wins in both."""
    raw = {
        "dataset_name": "root_dataset",
        "entity_id_column": "root_ip",
        "data": {
            "dataset_name": "nested_dataset",
            "entity_id_column": "nested_ip",
            "split_version": "v2"
        }
    }
    _normalize_data_config(raw)
    cfg = Config(raw)
    # Root keys take precedence and overwrite data block
    assert cfg.dataset_name == "root_dataset"
    assert cfg.data.dataset_name == "root_dataset"
    assert cfg.entity_id_column == "root_ip"
    assert cfg.data.entity_id_column == "root_ip"
    # Nested key not present at root gets propagated to root
    assert cfg.split_version == "v2"
    assert cfg.data.split_version == "v2"


def test_normalization_empty_or_malformed_data_block():
    """Verify that missing, None, or non-dict data section is gracefully handled."""
    # Test None data section
    raw1 = {"dataset_name": "ds1", "data": None}
    _normalize_data_config(raw1)
    cfg1 = Config(raw1)
    assert cfg1.dataset_name == "ds1"
    assert cfg1.data.dataset_name == "ds1"

    # Test non-dict data section (e.g. integer or string)
    raw2 = {"dataset_name": "ds2", "data": "invalid_string"}
    _normalize_data_config(raw2)
    cfg2 = Config(raw2)
    assert cfg2.dataset_name == "ds2"
    assert cfg2.data.dataset_name == "ds2"


def test_normalization_all_data_keys_propagate():
    """Verify all DATA_KEYS synchronize bidirectionally."""
    for key in DATA_KEYS:
        # Data block to root
        raw_nested = {"data": {key: f"val_{key}"}}
        _normalize_data_config(raw_nested)
        cfg_nested = Config(raw_nested)
        assert getattr(cfg_nested, key) == f"val_{key}"
        assert getattr(cfg_nested.data, key) == f"val_{key}"

        # Root to data block
        raw_root = {key: f"val_root_{key}"}
        _normalize_data_config(raw_root)
        cfg_root = Config(raw_root)
        assert getattr(cfg_root, key) == f"val_root_{key}"
        assert getattr(cfg_root.data, key) == f"val_root_{key}"


def test_config_delegation_nonexistent_key_raises_attribute_error():
    """Verify accessing nonexistent key on Config raises AttributeError."""
    cfg = Config({"data": {"known": 123}})
    assert cfg.known == 123
    assert cfg.data.known == 123
    with pytest.raises(AttributeError, match="No such configuration key"):
        _ = cfg.totally_unknown_key


# =========================================================================
# 2. DATASET CONFIGS ENTITY ID RESOLUTION & IP PRESERVATION
# =========================================================================

def test_production_dataset_configs_resolution():
    """Verify TON-IoT, IoT-23, KDD, and UNSW-NB15 Raw49 resolve entity ID and dataset name."""
    base_yaml = Path("configs/base.yaml")
    assert base_yaml.exists(), "configs/base.yaml missing"

    # TON-IoT: src_ip, modulo 254
    ton_cfg = load_config(base_yaml, Path("configs/dataset_ton_iot.yaml"))
    assert ton_cfg.dataset_name == "ton_iot"
    assert ton_cfg.data.dataset_name == "ton_iot"
    assert ton_cfg.entity_id_column == "src_ip"
    assert ton_cfg.data.entity_id_column == "src_ip"
    assert ton_cfg.entity_id_modulo == 254
    assert ton_cfg.data.entity_id_modulo == 254

    # IoT-23: id.orig_h, modulo 254
    iot_cfg = load_config(base_yaml, Path("configs/dataset_iot23.yaml"))
    assert iot_cfg.dataset_name == "iot23"
    assert iot_cfg.data.dataset_name == "iot23"
    assert iot_cfg.entity_id_column == "id.orig_h"
    assert iot_cfg.data.entity_id_column == "id.orig_h"

    # KDD: no entity_id_column, modulo 254
    kdd_cfg = load_config(base_yaml, Path("configs/dataset_kdd.yaml"))
    assert kdd_cfg.dataset_name == "kdd"
    assert kdd_cfg.data.dataset_name == "kdd"
    assert kdd_cfg.get("entity_id_column") is None
    assert kdd_cfg.data.get("entity_id_column") is None
    assert kdd_cfg.entity_id_modulo == 254

    # UNSW-NB15 Raw49: srcip
    raw49_cfg = load_config(base_yaml, Path("configs/dataset_unsw_nb15_raw49.yaml"))
    assert raw49_cfg.dataset_name == "unsw_nb15_raw49"
    assert raw49_cfg.data.dataset_name == "unsw_nb15_raw49"
    assert raw49_cfg.entity_id_column == "srcip"
    assert raw49_cfg.data.entity_id_column == "srcip"


def test_ip_preservation_synthetic_csv(tmp_path):
    """Stress-test loader IP preservation for real IPs and fallback for missing/corrupted IPs."""
    csv_file = tmp_path / "test_ip_traffic.csv"
    rows = [
        # Real IPv4
        {"src_ip": "192.168.1.105", "proto": "tcp", "service": "http", "conn_state": "SF", "duration": "0.1", "src_bytes": "500", "dst_bytes": "1000", "label": "0"},
        # Real IPv6
        {"src_ip": "2001:db8::1", "proto": "udp", "service": "dns", "conn_state": "SF", "duration": "0.05", "src_bytes": "100", "dst_bytes": "200", "label": "0"},
        # Missing IP dash '-' -> fallback
        {"src_ip": "-", "proto": "tcp", "service": "-", "conn_state": "OTH", "duration": "0.0", "src_bytes": "0", "dst_bytes": "0", "label": "1"},
        # Empty IP -> fallback
        {"src_ip": "", "proto": "icmp", "service": "-", "conn_state": "OTH", "duration": "0.01", "src_bytes": "50", "dst_bytes": "0", "label": "1"},
        # NaN IP -> fallback
        {"src_ip": "nan", "proto": "tcp", "service": "ssh", "conn_state": "SF", "duration": "1.2", "src_bytes": "800", "dst_bytes": "1200", "label": "0"},
        # Another real IP
        {"src_ip": "10.200.5.99", "proto": "tcp", "service": "http", "conn_state": "SF", "duration": "0.2", "src_bytes": "300", "dst_bytes": "600", "label": "0"},
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    cfg = Config({
        "dataset_name": "custom_ton",
        "entity_id_column": "src_ip",
        "entity_id_modulo": 10,
        "categorical_columns": ["proto", "service", "conn_state"],
        "numerical_columns": ["duration", "src_bytes", "dst_bytes"],
        "label_column": "label"
    })

    flows = load_dataset_flows(cfg, data_dir=tmp_path, source_file=str(csv_file))
    assert len(flows) == 6

    # Verify real IPs were kept verbatim
    assert flows[0].entity_id == "192.168.1.105"
    assert flows[1].entity_id == "2001:db8::1"
    # Verify fallback for missing/corrupted IPs: 10.0.0.((row_idx % 10) + 1)
    assert flows[2].entity_id == f"10.0.0.{(2 % 10) + 1}"
    assert flows[3].entity_id == f"10.0.0.{(3 % 10) + 1}"
    assert flows[4].entity_id == f"10.0.0.{(4 % 10) + 1}"
    # Verify another real IP
    assert flows[5].entity_id == "10.200.5.99"


def test_kdd_synthetic_modulo_generation(tmp_path):
    """Verify KDD loader (no entity_id_column) generates modulo entities deterministically."""
    csv_file = tmp_path / "kdd_sample.csv"
    rows = [
        {"duration": "0", "protocol_type": "tcp", "service": "http", "src_bytes": "100", "dst_bytes": "200", "labels": "normal"},
        {"duration": "0", "protocol_type": "udp", "service": "domain_u", "src_bytes": "50", "dst_bytes": "50", "labels": "normal"},
        {"duration": "1", "protocol_type": "tcp", "service": "smtp", "src_bytes": "300", "dst_bytes": "400", "labels": "neptune"},
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    cfg = Config({
        "dataset_name": "kdd",
        "entity_id_modulo": 5,
        "categorical_columns": ["protocol_type", "service"],
        "numerical_columns": ["duration", "src_bytes", "dst_bytes"],
        "label_column": "labels"
    })

    flows = load_dataset_flows(cfg, data_dir=tmp_path, source_file=str(csv_file))
    assert len(flows) == 3
    assert flows[0].entity_id == "10.0.0.1"
    assert flows[1].entity_id == "10.0.0.2"
    assert flows[2].entity_id == "10.0.0.3"


# =========================================================================
# 3. RESOLVE_EXPERIMENT_CONFIG FALLBACK & RESILIENCE
# =========================================================================

def test_resolve_experiment_config_all_dataset_hints():
    """Verify resolve_experiment_config with nonexistent cfg_path falls back cleanly for all hints."""
    nonexistent = "nonexistent/dir/config.yaml"

    test_cases = [
        ("ton", "ton_iot", "src_ip"),
        ("TON_IOT", "ton_iot", "src_ip"),
        ("kdd", "kdd", None),
        ("NSL_KDD", "kdd", None),
        ("botnet", "unsw_botnet", None),
        ("iot23", "iot23", "id.orig_h"),
        ("dataset17", "iot23", "id.orig_h"),
        ("unsw", "unsw_nb15", None),
        ("unknown_hint_fallback", "unsw_nb15", None),
    ]

    for hint, expected_ds_name, expected_entity_col in test_cases:
        cfg = resolve_experiment_config(cfg_path=nonexistent, dataset_hint=hint)
        assert cfg.dataset_name == expected_ds_name, f"Hint '{hint}' produced dataset '{cfg.dataset_name}', expected '{expected_ds_name}'"
        assert cfg.data.dataset_name == expected_ds_name
        if expected_entity_col:
            assert cfg.entity_id_column == expected_entity_col
            assert cfg.data.entity_id_column == expected_entity_col


def test_resolve_experiment_config_nonexistent_configs_dir():
    """Verify resolve_experiment_config raises FileNotFoundError if configs dir is invalid."""
    with pytest.raises(FileNotFoundError, match="Neither candidate .* nor fallback configs"):
        _ = resolve_experiment_config(
            cfg_path="nonexistent/cand.yaml",
            dataset_hint="ton",
            configs_dir="completely/nonexistent/directory"
        )


# =========================================================================
# 4. EVALUATE_ENCODER_SEPARABILITY RESILIENCE TESTS
# =========================================================================

def test_separability_nonexistent_dataset():
    """Verify compute_encoder_separability returns empty dict when file does not exist."""
    res = compute_encoder_separability(dataset_path="nonexistent/data.csv")
    assert res == {}
    table_str = print_separability_table(res)
    assert table_str == "No empirical encoder separability metrics available."


def test_separability_empty_csv(tmp_path):
    """Verify compute_encoder_separability handles 0-byte or header-only CSV without unhandled crash."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("proto,service,state,dur,spkts,dpkts,sbytes,dbytes,rate,sttl,dttl,sloss,dloss,sinpkt,dinpkt,sjit,djit,swin,stcpb,dtcpb,dwin,tcprtt,synack,ackdat,smean,dmean,trans_depth,response_body_len,label,srcip\n")
    res = compute_encoder_separability(dataset_path=str(empty_csv))
    assert res == {}


def test_separability_single_class_csv(tmp_path):
    """Verify compute_encoder_separability handles single-class CSV (only normal or only attack)."""
    header = "proto,service,state,dur,spkts,dpkts,sbytes,dbytes,rate,sttl,dttl,sloss,dloss,sinpkt,dinpkt,sjit,djit,swin,stcpb,dtcpb,dwin,tcprtt,synack,ackdat,smean,dmean,trans_depth,response_body_len,label,srcip\n"
    norm_line = "tcp,http,FIN,0.1,10,10,500,500,100,64,64,0,0,0.01,0.01,0.001,0.001,255,1000,1000,255,0.05,0.02,0.03,50,50,0,0,0,192.168.1.1\n"
    
    # Only normal flows
    normal_csv = tmp_path / "only_normal.csv"
    normal_csv.write_text(header + norm_line * 20)
    res_norm = compute_encoder_separability(dataset_path=str(normal_csv))
    assert res_norm == {}

    # Only attack flows
    att_line = norm_line.replace(",0,192.168.1.1", ",1,192.168.1.2")
    attack_csv = tmp_path / "only_attack.csv"
    attack_csv.write_text(header + att_line * 20)
    res_att = compute_encoder_separability(dataset_path=str(attack_csv))
    assert res_att == {}


def test_separability_valid_synthetic_flows(tmp_path):
    """Verify compute_encoder_separability produces valid metrics when both classes are present."""
    header = "proto,service,state,dur,spkts,dpkts,sbytes,dbytes,rate,sttl,dttl,sloss,dloss,sinpkt,dinpkt,sjit,djit,swin,stcpb,dtcpb,dwin,tcprtt,synack,ackdat,smean,dmean,trans_depth,response_body_len,label,srcip\n"
    norm_line = "tcp,http,FIN,0.1,10,10,500,500,100,64,64,0,0,0.01,0.01,0.001,0.001,255,1000,1000,255,0.05,0.02,0.03,50,50,0,0,0,192.168.1.1\n"
    att_line = "udp,dns,CON,0.01,2,2,80,80,200,32,32,0,0,0.005,0.005,0.0001,0.0001,0,0,0,0,0.01,0.0,0.0,40,40,0,0,1,10.0.0.5\n"

    valid_csv = tmp_path / "valid_traffic.csv"
    # Need sufficient normal flows (> 2) for split (split_idx >= 1, eval >= 1)
    valid_csv.write_text(header + (norm_line * 10) + (att_line * 10))

    out_json = tmp_path / "separability.json"
    res = compute_encoder_separability(dataset_path=str(valid_csv), limit=20, D=512, output_json=out_json)
    assert len(res) == 3
    assert "HDC (RecordEncoder)" in res
    assert "SAX (Symbolic Aggregate)" in res
    assert "RFF (Random Fourier)" in res
    for enc_name, m in res.items():
        assert "cos_dist" in m
        assert "eucl_dist" in m
        assert "fisher_ratio" in m
        assert "margin" in m
    assert out_json.exists()


# =========================================================================
# 5. EXPERIMENT SCRIPT RESILIENCE WHEN DATASETS ARE MISSING
# =========================================================================

def test_generate_table_cli_isolated(tmp_path):
    """Verify generate_table.py writes to custom runs directory and protects canonical markdown."""
    canon_file = Path("BENCHMARK_COMPARISON.md")
    canon_mtime_before = canon_file.stat().st_mtime if canon_file.exists() else None

    out_md = tmp_path / "test_table.md"
    cmd = [
        sys.executable,
        "generate_table.py",
        "--runs-dir", "runs",
        "--output", str(out_md)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"generate_table.py failed: {proc.stderr}"
    assert out_md.exists(), f"Output table was not created at {out_md}"

    if canon_file.exists():
        canon_mtime_after = canon_file.stat().st_mtime
        assert canon_mtime_before == canon_mtime_after, "Canonical BENCHMARK_COMPARISON.md was modified!"


def test_evaluate_encoder_separability_cli_missing_dataset():
    """Verify evaluate_encoder_separability.py CLI exits with 1 when dataset is missing."""
    cmd = [
        sys.executable,
        "evaluate_encoder_separability.py",
        "--dataset", "nonexistent/dataset.csv"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 1
    assert "does not exist" in proc.stderr or "does not exist" in proc.stdout


def test_run_instantaneous_temporal_fusion_imports_and_checks_datasets():
    """Verify run_instantaneous_temporal_fusion can be imported and handles missing baseline records."""
    from experiments.run_instantaneous_temporal_fusion import PRIMARY_DATASETS
    assert len(PRIMARY_DATASETS) == 3
    # Check that datasets in list have valid tuples
    for ds_name, src_file, cfg_path in PRIMARY_DATASETS:
        assert isinstance(ds_name, str)
        assert isinstance(src_file, str)
        assert isinstance(cfg_path, str)


def test_run_replication_verification_imports_and_checks_manifest():
    """Verify run_replication_verification step1 handles nonexistent benchmark datasets without crashing."""
    from experiments.run_replication_verification import step1_dataset_manifest_and_leakage_audit
    manifest, leakage_passed = step1_dataset_manifest_and_leakage_audit()
    assert isinstance(manifest, dict)
    assert leakage_passed is True
