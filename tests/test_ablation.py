"""Ablation suite tier tests for Phase 7b multi-config comparative experiments."""

from pathlib import Path
import pytest

from hdlnn.common import Settings
from hdlnn.data import Splitter, UNSWLoader
from hdlnn.eval import run_ablation_suite, run_experiment
from hdlnn.eval.harness import ComparisonTable
from hdlnn.eval.pipeline_adapter import StandardPipeline

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "unsw_sample.csv"


@pytest.mark.ablation
def test_ablation_modes_run_experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    loader = UNSWLoader(run_id="test_ablate_modes")
    flows = loader.load(str(FIXTURE_PATH))
    splitter = Splitter()
    train_flows, val_flows, test_flows = splitter.split(flows, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42)

    # Config 1: HDC Off
    cfg_hdc_off = Settings(run_id="hdc_off_run", seed=42, hdc_encoder="off", dimensionality=100, hidden_size=16, epochs=1)
    pipe_hdc_off = StandardPipeline(settings=cfg_hdc_off)
    res_hdc_off = run_experiment(cfg_hdc_off, pipe_hdc_off, train_flows, val_flows, test_flows)
    assert res_hdc_off.run_id == "hdc_off_run"

    # Config 2: LNN Off
    cfg_lnn_off = Settings(run_id="lnn_off_run", seed=42, lnn_stage="off", dimensionality=100, hidden_size=16)
    pipe_lnn_off = StandardPipeline(settings=cfg_lnn_off)
    res_lnn_off = run_experiment(cfg_lnn_off, pipe_lnn_off, train_flows, val_flows, test_flows)
    assert res_lnn_off.run_id == "lnn_off_run"


@pytest.mark.ablation
def test_run_ablation_suite_all_6_configs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    loader = UNSWLoader(run_id="test_ablation_suite")
    flows = loader.load(str(FIXTURE_PATH))
    splitter = Splitter()
    train_flows, val_flows, test_flows = splitter.split(flows, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42)

    configs = [
        Settings(run_id="hybrid_cfc_v1", seed=42, hdc_encoder="on", lnn_stage="on", sequence_model_type="cfc", dimensionality=100, hidden_size=16, epochs=1),
        Settings(run_id="hdc_only_v1", seed=42, hdc_encoder="on", lnn_stage="off", sequence_model_type="cfc", dimensionality=100, hidden_size=16),
        Settings(run_id="lnn_only_v1", seed=42, hdc_encoder="off", lnn_stage="on", sequence_model_type="cfc", dimensionality=100, hidden_size=16, epochs=1),
        Settings(run_id="lstm_baseline_v1", seed=42, hdc_encoder="on", lnn_stage="on", sequence_model_type="lstm", dimensionality=100, hidden_size=16, epochs=1),
        Settings(run_id="cnn_baseline_v1", seed=42, hdc_encoder="on", lnn_stage="on", sequence_model_type="cnn", dimensionality=100, hidden_size=16, epochs=1),
        Settings(run_id="ltc_ablation_v1", seed=42, hdc_encoder="on", lnn_stage="on", sequence_model_type="ltc", dimensionality=100, hidden_size=16, epochs=1),
    ]

    table = run_ablation_suite(configs, train_flows, val_flows, test_flows)
    assert isinstance(table, ComparisonTable)
    assert len(table.rows) == 6
