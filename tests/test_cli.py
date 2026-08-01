"""Integration and ablation tier tests for hdlnn CLI entry point."""

from pathlib import Path
from click.testing import CliRunner
import pytest
import yaml

from hdlnn.cli import main as cli

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "unsw_sample.csv"
CONFIGS_DIR = Path(__file__).parent.parent / "configs"


@pytest.mark.integration
def test_cli_train_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    cfg_dict = {
        "run_id": "cli_train_001",
        "seed": 42,
        "dataset_raw_path": str(FIXTURE_PATH),
        "dimensionality": 100,
        "hidden_size": 16,
        "epochs": 1,
    }
    cfg_file = tmp_path / "train_cfg.yaml"
    cfg_file.write_text(yaml.dump(cfg_dict), encoding="utf-8")

    res = runner.invoke(cli, ["train", "--config", str(cfg_file)])
    assert res.exit_code == 0, f"CLI train failed: {res.output}"
    assert "Training Complete" in res.output
    assert (tmp_path / "runs" / "cli_train_001" / "metrics.json").exists()


@pytest.mark.integration
def test_cli_eval_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    cfg_dict = {
        "run_id": "cli_eval_001",
        "seed": 42,
        "dataset_raw_path": str(FIXTURE_PATH),
        "dimensionality": 100,
        "hidden_size": 16,
    }
    cfg_file = tmp_path / "eval_cfg.yaml"
    cfg_file.write_text(yaml.dump(cfg_dict), encoding="utf-8")

    res = runner.invoke(cli, ["eval", "--config", str(cfg_file)])
    assert res.exit_code == 0, f"CLI eval failed: {res.output}"
    assert "Evaluation Complete" in res.output


@pytest.mark.integration
def test_cli_stream_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    # 1. Train first
    train_cfg_dict = {
        "run_id": "cli_stream_train",
        "seed": 42,
        "dataset_raw_path": str(FIXTURE_PATH),
        "dimensionality": 100,
        "hidden_size": 16,
        "epochs": 1,
    }
    train_cfg_file = tmp_path / "stream_train_cfg.yaml"
    train_cfg_file.write_text(yaml.dump(train_cfg_dict), encoding="utf-8")
    runner.invoke(cli, ["train", "--config", str(train_cfg_file)])

    # 2. Stream execution
    stream_cfg_dict = {
        "run_id": "cli_stream_exec",
        "seed": 42,
        "dataset_raw_path": str(FIXTURE_PATH),
        "dimensionality": 100,
        "hidden_size": 16,
    }
    stream_cfg_file = tmp_path / "stream_exec_cfg.yaml"
    stream_cfg_file.write_text(yaml.dump(stream_cfg_dict), encoding="utf-8")

    res = runner.invoke(
        cli,
        [
            "stream",
            "--config",
            str(stream_cfg_file),
            "--run-id",
            "cli_stream_train",
            "--replay-rate",
            "0.0",
            "--check-consistency",
        ],
    )
    assert res.exit_code == 0, f"CLI stream failed: {res.output}"
    assert "Streaming Complete" in res.output or "Artifacts saved under" in res.output


@pytest.mark.ablation
def test_cli_ablate_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    res = runner.invoke(cli, ["ablate", "--configs-dir", str(CONFIGS_DIR)])
    assert res.exit_code == 0, f"CLI ablate failed: {res.output}"
    assert "Ablation Suite Complete" in res.output or "Comparison table" in res.output


@pytest.mark.integration
def test_cli_report_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    # Generate ablation suite first
    runner.invoke(cli, ["ablate", "--configs-dir", str(CONFIGS_DIR)])

    res = runner.invoke(cli, ["report", "--format", "markdown"])
    assert res.exit_code == 0, f"CLI report failed: {res.output}"
    assert "Ablation Comparison Table" in res.output


@pytest.mark.integration
def test_cli_import_boundary_audit():
    """Static audit asserting cli.py does not import directly from internal stage packages."""
    cli_file = Path(__file__).parent.parent / "hdlnn" / "cli.py"
    code = cli_file.read_text(encoding="utf-8")

    forbidden_imports = [
        "import hdlnn.data",
        "from hdlnn.data",
        "import hdlnn.hdc",
        "from hdlnn.hdc",
        "import hdlnn.lnn",
        "from hdlnn.lnn",
        "import hdlnn.divergence",
        "from hdlnn.divergence",
    ]

    for forbidden in forbidden_imports:
        assert forbidden not in code, f"Forbidden import '{forbidden}' found in hdlnn/cli.py"
