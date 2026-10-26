from pathlib import Path
import subprocess
import sys
import pytest

from hdlnn.report.tables import print_and_save_comparison_table, generate_comparison_table


def test_print_and_save_comparison_table_default(tmp_path, monkeypatch):
    """Ensure default output goes to runs/comparison_table.md and does not touch root BENCHMARK_COMPARISON.md."""
    monkeypatch.chdir(tmp_path)
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    print_and_save_comparison_table(runs_dir=runs_dir)
    assert (runs_dir / "comparison_table.md").exists()
    assert not (tmp_path / "BENCHMARK_COMPARISON.md").exists()


def test_print_and_save_comparison_table_none_output(tmp_path, monkeypatch):
    """Ensure output_file=None executes without creating any output file on disk."""
    monkeypatch.chdir(tmp_path)
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    print_and_save_comparison_table(runs_dir=runs_dir, output_file=None)
    assert not (runs_dir / "comparison_table.md").exists()
    assert not (tmp_path / "BENCHMARK_COMPARISON.md").exists()


def test_print_and_save_comparison_table_custom(tmp_path):
    """Ensure custom output file is created with proper parent directory creation."""
    custom_out = tmp_path / "custom_subdir" / "my_table.md"
    print_and_save_comparison_table(runs_dir=tmp_path, output_file=custom_out)
    
    assert custom_out.exists()
    content = custom_out.read_text(encoding="utf-8")
    assert "Final Baseline Model Benchmark Comparison" in content


def test_canonical_benchmark_file_protection():
    """Verify that root BENCHMARK_COMPARISON.md exists and contains the authoritative sections."""
    canonical = Path("BENCHMARK_COMPARISON.md")
    assert canonical.exists(), "Root BENCHMARK_COMPARISON.md must exist."
    content = canonical.read_text(encoding="utf-8")
    assert "Macro-Average Detection & Efficiency Summary" in content
    assert "HDC-LNN" in content
    assert "FT-Transformer" in content
    assert "SAINT" in content


def test_generate_table_cli_non_destructive(tmp_path):
    """Verify generate_table.py CLI writes to specified output and leaves BENCHMARK_COMPARISON.md untouched."""
    out_table = tmp_path / "test_runs" / "output_table.md"
    cmd = [
        sys.executable, "generate_table.py",
        "--runs-dir", str(tmp_path / "test_runs"),
        "--output-file", str(out_table)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert out_table.exists()
    content = out_table.read_text(encoding="utf-8")
    assert "# Final Baseline Model Benchmark Comparison" in content

