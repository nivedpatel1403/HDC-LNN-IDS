import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import pytest
import torch

from hdlnn.report.tables import (
    generate_comparison_table,
    generate_encoder_separability_table,
    print_and_save_comparison_table,
)
from hdlnn.common.config import Config, load_config, resolve_experiment_config, _normalize_data_config
from hdlnn.common.dataset_utils import compute_encoder_separability


def test_tables_with_nonexistent_runs_dir(tmp_path):
    """Stress test: print_and_save_comparison_table when runs_dir does not exist."""
    non_existent = tmp_path / "does_not_exist"
    out_file = tmp_path / "sub" / "output.md"
    
    # Should not crash even if runs_dir does not exist
    print_and_save_comparison_table(runs_dir=non_existent, output_file=out_file)
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "No runs folder found" in content


def test_tables_with_corrupt_metrics_json(tmp_path):
    """Stress test: runs_dir contains malformed JSON or empty directories."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    
    # Empty run dir
    (runs_dir / "empty_run").mkdir()
    
    # Corrupt JSON
    bad_dir = runs_dir / "bad_run"
    bad_dir.mkdir()
    (bad_dir / "metrics.json").write_text("{ corrupt json ...", encoding="utf-8")
    
    # Incomplete JSON (missing standard fields)
    incomplete_dir = runs_dir / "incomplete_run"
    incomplete_dir.mkdir()
    (incomplete_dir / "metrics.json").write_text(json.dumps({"unknown_field": 123}), encoding="utf-8")
    
    out_file = tmp_path / "table.md"
    # Should handle gracefully without unhandled exception
    print_and_save_comparison_table(runs_dir=runs_dir, output_file=out_file)
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "Final Baseline Model Benchmark Comparison" in content


def test_generate_table_cli_with_empty_and_corrupt_runs(tmp_path):
    """Stress test: generate_table.py CLI execution with empty/missing/corrupted runs."""
    empty_runs = tmp_path / "empty_runs"
    empty_runs.mkdir()
    
    out_file = tmp_path / "deep" / "nested" / "table.md"
    cmd = [
        sys.executable, "generate_table.py",
        "--runs-dir", str(empty_runs),
        "--output-file", str(out_file)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Stderr: {res.stderr}"
    assert out_file.exists()
    
    # Verify canonical file was untouched
    canonical = Path("BENCHMARK_COMPARISON.md")
    assert canonical.exists()


def test_separability_with_zero_attack_flows(tmp_path):
    """Stress test: compute_encoder_separability when CSV contains only normal flows (no attack)."""
    csv_file = tmp_path / "normal_only.csv"
    headers = [
        "id", "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
        "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt",
        "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
        "smean", "dmean", "trans_depth", "response_body_len", "srcip", "label"
    ]
    rows = [",".join(headers)]
    for i in range(5):
        row = ["0"] * len(headers)
        row[headers.index("proto")] = "tcp"
        row[headers.index("service")] = "http"
        row[headers.index("state")] = "FIN"
        row[headers.index("srcip")] = f"192.168.1.{i+1}"
        row[headers.index("label")] = "0"
        rows.append(",".join(row))
    csv_file.write_text("\n".join(rows), encoding="utf-8")
    
    out_json = tmp_path / "nested" / "sep.json"
    results = compute_encoder_separability(
        dataset_path=str(csv_file),
        limit=10,
        D=128,
        output_json=out_json
    )
    # When no attack flows exist, should return empty dict and not crash on torch.stack
    assert results == {}


def test_separability_with_zero_normal_flows(tmp_path):
    """Stress test: compute_encoder_separability when CSV contains only attack flows (no normal)."""
    csv_file = tmp_path / "attack_only.csv"
    headers = [
        "id", "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
        "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt",
        "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
        "smean", "dmean", "trans_depth", "response_body_len", "srcip", "label"
    ]
    rows = [",".join(headers)]
    for i in range(5):
        row = ["0"] * len(headers)
        row[headers.index("proto")] = "udp"
        row[headers.index("service")] = "dns"
        row[headers.index("state")] = "CON"
        row[headers.index("srcip")] = f"10.0.0.{i+1}"
        row[headers.index("label")] = "1"
        rows.append(",".join(row))
    csv_file.write_text("\n".join(rows), encoding="utf-8")
    
    out_json = tmp_path / "nested2" / "sep.json"
    results = compute_encoder_separability(
        dataset_path=str(csv_file),
        limit=10,
        D=128,
        output_json=out_json
    )
    # When no normal flows exist, should return empty dict and not crash
    assert results == {}


def test_pipeline_mode_report_in_isolated_cwd(tmp_path):
    """Stress test: python -m hdlnn.pipeline --mode report executed in isolated dir."""
    repo_root = Path.cwd().resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    
    # Run pipeline inside isolated tmp_path so it doesn't process 89 existing runs
    cmd = [
        sys.executable, "-m", "hdlnn.pipeline",
        "--mode", "report",
        "--config-base", str(repo_root / "configs" / "base.yaml"),
        "--config-dataset", str(repo_root / "configs" / "dataset_unsw_nb15.yaml"),
        "--output-table", str(tmp_path / "pipe_runs" / "output_table.md")
    ]
    res = subprocess.run(cmd, cwd=str(tmp_path), env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"Pipeline report failed: {res.stderr}\nStdout: {res.stdout}"
    
    out_table = tmp_path / "pipe_runs" / "output_table.md"
    assert out_table.exists()
    content = out_table.read_text(encoding="utf-8")
    assert "Final Baseline Model Benchmark Comparison" in content
    # Ensure root BENCHMARK_COMPARISON.md is untouched
    assert not (tmp_path / "BENCHMARK_COMPARISON.md").exists()


def test_pipeline_mode_report_default_output(tmp_path):
    """Stress test: pipeline defaults to runs/comparison_table.md when --output-table is not specified."""
    repo_root = Path.cwd().resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    
    cmd = [
        sys.executable, "-m", "hdlnn.pipeline",
        "--mode", "report",
        "--config-base", str(repo_root / "configs" / "base.yaml"),
        "--config-dataset", str(repo_root / "configs" / "dataset_unsw_nb15.yaml"),
    ]
    res = subprocess.run(cmd, cwd=str(tmp_path), env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"Pipeline failed: {res.stderr}"
    
    default_table = tmp_path / "runs" / "comparison_table.md"
    assert default_table.exists(), "Pipeline should default to runs/comparison_table.md"
    assert not (tmp_path / "BENCHMARK_COMPARISON.md").exists()


def test_config_conflict_resolution():
    """Stress test: conflict resolution between root keys and data block keys."""
    data = {
        "dataset_name": "root_val",
        "data": {
            "dataset_name": "data_val"
        }
    }
    _normalize_data_config(data)
    assert data["data"]["dataset_name"] == "root_val"

    data2 = {
        "data": {
            "entity_id_column": "srcip_data"
        }
    }
    _normalize_data_config(data2)
    assert data2["entity_id_column"] == "srcip_data"


def test_evaluate_encoder_separability_cli_nonexistent_dataset():
    """Stress test: evaluate_encoder_separability.py CLI exits cleanly on missing dataset."""
    cmd = [
        sys.executable, "evaluate_encoder_separability.py",
        "--dataset", "nonexistent_file_path.csv"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    # Should exit with code 1 and clean log message, not Python traceback
    assert res.returncode == 1
    assert "Dataset path does not exist" in res.stderr
    assert "Traceback" not in res.stderr


def test_run_all_datasets_cli_nonexistent_datasets_dir(tmp_path):
    """Stress test: run_all_datasets.py exits gracefully when datasets dir does not exist."""
    out_file = tmp_path / "summary.md"
    cmd = [
        sys.executable, "run_all_datasets.py",
        "--datasets-dir", str(tmp_path / "missing_datasets"),
        "--output", str(out_file)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Datasets directory not found" in res.stderr
    assert not out_file.exists()
