"""Integration and ablation tier tests for pipeline orchestrator module."""

from pathlib import Path
import pytest

from hdlnn.common.config import Settings
from hdlnn.eval.harness import ComparisonTable
from hdlnn.eval.run_result import RunResult
from hdlnn.pipeline import load_and_split_data, run_ablate, run_eval, run_stream, run_train

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "unsw_sample.csv"
CONFIGS_DIR = Path(__file__).parent.parent / "configs"


@pytest.mark.integration
def test_load_and_split_data_reuses_existing_split(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    cfg = Settings(run_id="test_split_reuse", seed=42)
    cfg.dataset_raw_path = str(FIXTURE_PATH)
    cfg.split_version = "reuse_v1"

    # Call 1: creates new split JSON
    tr1, val1, ts1 = load_and_split_data(cfg)
    split_file = tmp_path / "data" / "splits" / "unsw_reuse_v1.json"
    assert split_file.exists()

    # Call 2: loads existing split JSON
    tr2, val2, ts2 = load_and_split_data(cfg)

    # Assert exact flow_id alignment between calls
    assert [f.flow_id for f in tr1] == [f.flow_id for f in tr2]
    assert [f.flow_id for f in val1] == [f.flow_id for f in val2]
    assert [f.flow_id for f in ts1] == [f.flow_id for f in ts2]


@pytest.mark.integration
def test_run_train_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    cfg = Settings(
        run_id="test_pipe_train",
        seed=42,
        dimensionality=100,
        hidden_size=16,
        epochs=1,
    )
    cfg.dataset_raw_path = str(FIXTURE_PATH)

    result = run_train(cfg)

    assert isinstance(result, RunResult)
    assert result.run_id == "test_pipe_train"
    assert (tmp_path / "runs" / "test_pipe_train" / "model_checkpoint.pt").exists()


@pytest.mark.integration
def test_run_eval_with_checkpoint_loading(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Train first
    cfg_train = Settings(
        run_id="test_pipe_eval_train",
        seed=42,
        dimensionality=100,
        hidden_size=16,
        epochs=1,
    )
    cfg_train.dataset_raw_path = str(FIXTURE_PATH)
    run_train(cfg_train)

    # Eval loading checkpoint
    cfg_eval = Settings(
        run_id="test_pipe_eval_exec",
        seed=42,
        dimensionality=100,
        hidden_size=16,
    )
    cfg_eval.dataset_raw_path = str(FIXTURE_PATH)

    result = run_eval(cfg_eval, run_id_to_load="test_pipe_eval_train")

    assert isinstance(result, RunResult)
    assert result.run_id == "test_pipe_eval_exec"


@pytest.mark.integration
def test_run_stream_and_consistency_check_passes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Train first
    cfg_train = Settings(
        run_id="test_pipe_stream_train",
        seed=42,
        dimensionality=100,
        hidden_size=16,
        epochs=1,
    )
    cfg_train.dataset_raw_path = str(FIXTURE_PATH)
    run_train(cfg_train)

    # Stream replay with consistency check
    cfg_stream = Settings(
        run_id="test_pipe_stream_exec",
        seed=42,
        dimensionality=100,
        hidden_size=16,
    )
    cfg_stream.dataset_raw_path = str(FIXTURE_PATH)

    result = run_stream(
        cfg_stream,
        run_id_to_load="test_pipe_stream_train",
        replay_rate=0.0,
        check_consistency=True,
    )

    assert isinstance(result, RunResult)
    assert result.run_id == "test_pipe_stream_exec"
    assert len(result.decisions) > 0


@pytest.mark.ablation
def test_run_ablate_default_configs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    table = run_ablate(configs_dir=str(CONFIGS_DIR))

    assert isinstance(table, ComparisonTable)
    assert len(table.rows) >= 6
