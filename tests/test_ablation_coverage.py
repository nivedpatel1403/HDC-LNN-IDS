"""Ablation tier and component tier coverage tests for Phase 7c gap-fill."""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
import torch

from hdlnn.common import Settings
from hdlnn.contracts.encoded import EncodedHypervector
from hdlnn.data import Splitter, UNSWLoader
from hdlnn.eval import run_ablation_suite, run_experiment
from hdlnn.eval.harness import (
    ComparisonTable,
    run_cross_dataset_transfer,
)
from hdlnn.eval.pipeline_adapter import StandardPipeline

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "unsw_sample.csv"


@pytest.mark.ablation
def test_run_experiment_hybrid_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loader = UNSWLoader(run_id="test_hybrid_yaml")
    flows = loader.load(str(FIXTURE_PATH))
    splitter = Splitter()
    tr, val, ts = splitter.split(flows, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42)

    cfg = Settings(run_id="hybrid_cfc_v1", seed=42, hdc_encoder="on", lnn_stage="on", sequence_model_type="cfc", dimensionality=100, hidden_size=16, epochs=1)
    pipe = StandardPipeline(settings=cfg)
    res = run_experiment(cfg, pipe, tr, val, ts)
    assert res.run_id == "hybrid_cfc_v1"
    assert res.detection_metrics["total_decisions"] == len(ts)


@pytest.mark.ablation
def test_run_experiment_hdc_only_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loader = UNSWLoader(run_id="test_hdc_only_yaml")
    flows = loader.load(str(FIXTURE_PATH))
    splitter = Splitter()
    tr, val, ts = splitter.split(flows, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42)

    cfg = Settings(run_id="hdc_only_v1", seed=42, hdc_encoder="on", lnn_stage="off", sequence_model_type="cfc", dimensionality=100, hidden_size=16)
    pipe = StandardPipeline(settings=cfg)
    res = run_experiment(cfg, pipe, tr, val, ts)
    assert res.run_id == "hdc_only_v1"
    assert res.detection_metrics["total_decisions"] == len(ts)


@pytest.mark.ablation
def test_run_experiment_lnn_only_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loader = UNSWLoader(run_id="test_lnn_only_yaml")
    flows = loader.load(str(FIXTURE_PATH))
    splitter = Splitter()
    tr, val, ts = splitter.split(flows, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42)

    cfg = Settings(run_id="lnn_only_v1", seed=42, hdc_encoder="off", lnn_stage="on", sequence_model_type="cfc", dimensionality=100, hidden_size=16, epochs=1)
    pipe = StandardPipeline(settings=cfg)
    res = run_experiment(cfg, pipe, tr, val, ts)
    assert res.run_id == "lnn_only_v1"
    assert res.detection_metrics["total_decisions"] == len(ts)


@pytest.mark.component
def test_hdc_only_bypasses_lnn_and_uses_raw_hdc_vectors(tmp_path):
    loader = UNSWLoader(run_id="test_hdc_raw_manifold")
    flows = loader.load(str(FIXTURE_PATH))
    splitter = Splitter()
    tr, val, ts = splitter.split(flows, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42)

    cfg = Settings(run_id="hdc_raw_test", seed=42, hdc_encoder="on", lnn_stage="off", dimensionality=250)
    pipe = StandardPipeline(settings=cfg)

    # Encode train flow
    pipe.encoder.fit(tr)
    enc_val = [pipe.encoder.encode(f) for f in val]

    # Process through passthrough sequence model
    val_groups = {}
    val_map = {f.flow_id: f for f in val}
    for hv, f in zip(enc_val, val):
        val_groups.setdefault(f.src_ip, []).append(hv)

    val_trajs = []
    val_labels = []
    for eid, seq in val_groups.items():
        trajs = pipe.sequence_model.forward(seq, entity_id=eid)
        for traj in trajs:
            if val_map[traj.flow_id].label == "normal":
                val_trajs.append(traj)
                val_labels.append("normal")

    pipe.divergence_scorer.fit(val_trajs, val_labels)

    # Manifold mean vector must have dimension D = 250 (raw HDC vector dimension)
    assert pipe.divergence_scorer.manifold.mean.shape == (250,)


@pytest.mark.component
def test_hdc_off_mode_bypasses_hdc_encoder():
    cfg = Settings(run_id="test_hdc_off_bypass", seed=42, hdc_encoder="off", dimensionality=100)
    pipe = StandardPipeline(settings=cfg)

    raw_flow = UNSWLoader(run_id="test_spy").load(str(FIXTURE_PATH))[0]
    res_hv = pipe.encoder.encode(raw_flow)

    assert isinstance(res_hv, EncodedHypervector)
    assert res_hv.encoder_version == "hdc_off"


@pytest.mark.ablation
def test_ablation_suite_6_configs_plus_cross_dataset_placeholder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loader = UNSWLoader(run_id="test_ablation_suite_full")
    flows = loader.load(str(FIXTURE_PATH))
    splitter = Splitter()
    tr, val, ts = splitter.split(flows, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42)

    configs = [
        Settings(run_id="hybrid_cfc_v1", seed=42, hdc_encoder="on", lnn_stage="on", sequence_model_type="cfc", dimensionality=100, hidden_size=16, epochs=1),
        Settings(run_id="hdc_only_v1", seed=42, hdc_encoder="on", lnn_stage="off", sequence_model_type="cfc", dimensionality=100, hidden_size=16),
        Settings(run_id="lnn_only_v1", seed=42, hdc_encoder="off", lnn_stage="on", sequence_model_type="cfc", dimensionality=100, hidden_size=16, epochs=1),
        Settings(run_id="lstm_baseline_v1", seed=42, hdc_encoder="on", lnn_stage="on", sequence_model_type="lstm", dimensionality=100, hidden_size=16, epochs=1),
        Settings(run_id="cnn_baseline_v1", seed=42, hdc_encoder="on", lnn_stage="on", sequence_model_type="cnn", dimensionality=100, hidden_size=16, epochs=1),
        Settings(run_id="ltc_ablation_v1", seed=42, hdc_encoder="on", lnn_stage="on", sequence_model_type="ltc", dimensionality=100, hidden_size=16, epochs=1),
    ]

    table = run_ablation_suite(configs, tr, val, ts)
    assert isinstance(table, ComparisonTable)
    assert len(table.rows) == 6

    # Verify OOV rate field is present and non-negative
    for row in table.rows:
        assert "oov_rate" in row
        assert row["oov_rate"] >= 0.0

    # Cross-dataset transfer verification
    transfer_res = run_cross_dataset_transfer(None, [])
    assert transfer_res["dataset"] == "CICIoT2023"
    assert transfer_res["status"] in ("completed", "skipped: ciciot_slice_5k.csv raw file not found")


@pytest.mark.ablation
def test_model_checkpoint_persistence_and_readout_exclusion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loader = UNSWLoader(run_id="test_ckpt")
    flows = loader.load(str(FIXTURE_PATH))
    splitter = Splitter()
    tr, val, ts = splitter.split(flows, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42)

    run_id = "ckpt_test_run"
    cfg = Settings(run_id=run_id, seed=42, hdc_encoder="on", lnn_stage="on", sequence_model_type="cfc", dimensionality=100, hidden_size=16, epochs=1)
    pipe = StandardPipeline(settings=cfg)

    res = run_experiment(cfg, pipe, tr, val, ts)
    assert res.run_id == run_id

    ckpt_path = Path("runs") / run_id / "model_checkpoint.pt"
    assert ckpt_path.exists()

    state_dict = torch.load(ckpt_path, weights_only=True)
    assert isinstance(state_dict, dict)
    # Confirm sequence model state keys are saved (e.g. cfc.rnn.weight_ih_l0 or linear weights)
    assert len(state_dict) > 0

    # Readout head excluded from inference scoring checkpoint
    assert "readout_head.weight" not in state_dict
