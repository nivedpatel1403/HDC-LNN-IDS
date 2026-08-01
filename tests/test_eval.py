"""Unit and integration tests for L4 evaluation harness core (metrics, pipeline adapter, experiment runner, artifact writing)."""

from datetime import datetime, timezone
from pathlib import Path
import json
import pytest

from hdlnn.baselines import LSTMBaseline
from hdlnn.common import Settings
from hdlnn.contracts.decision import AnomalyDecision
from hdlnn.contracts.flow import CanonicalFlow
from hdlnn.data import Splitter, UNSWLoader
from hdlnn.divergence import DivergenceScorer
from hdlnn.eval import (
    RunResult,
    StandardPipeline,
    compute_detection_metrics,
    compute_latency_stats,
    compute_memory_stats,
    run_experiment,
)
from hdlnn.hdc import HDCEncoder
from hdlnn.lnn import CfCModel

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "unsw_sample.csv"


@pytest.mark.unit
def test_compute_detection_metrics_exact_hand_calculated():
    now = datetime.now(timezone.utc)
    # Hand-crafted decisions:
    # 1. TP: ground_truth=attack, pred=True, score=0.9
    # 2. TP: ground_truth=attack, pred=True, score=0.8
    # 3. FN: ground_truth=attack, pred=False, score=0.3
    # 4. FP: ground_truth=normal, pred=True, score=0.7
    # 5. TN: ground_truth=normal, pred=False, score=0.1
    # TP=2, FP=1, FN=1, TN=1
    # Precision = 2/3, Recall = 2/3, F1 = 2/3
    decisions = [
        AnomalyDecision(entity_id="e1", flow_id="f1", t=now, drift_score=0.9, threshold=0.5, is_anomalous=True, manifold_version="v1", ground_truth_label="attack"),
        AnomalyDecision(entity_id="e1", flow_id="f2", t=now, drift_score=0.8, threshold=0.5, is_anomalous=True, manifold_version="v1", ground_truth_label="attack"),
        AnomalyDecision(entity_id="e1", flow_id="f3", t=now, drift_score=0.3, threshold=0.5, is_anomalous=False, manifold_version="v1", ground_truth_label="attack"),
        AnomalyDecision(entity_id="e1", flow_id="f4", t=now, drift_score=0.7, threshold=0.5, is_anomalous=True, manifold_version="v1", ground_truth_label="normal"),
        AnomalyDecision(entity_id="e1", flow_id="f5", t=now, drift_score=0.1, threshold=0.5, is_anomalous=False, manifold_version="v1", ground_truth_label="normal"),
    ]

    metrics = compute_detection_metrics(decisions)
    assert pytest.approx(metrics["precision"], abs=1e-4) == 2.0 / 3.0
    assert pytest.approx(metrics["recall"], abs=1e-4) == 2.0 / 3.0
    assert pytest.approx(metrics["f1"], abs=1e-4) == 2.0 / 3.0
    assert metrics["total_decisions"] == 5
    assert metrics["anomalies_detected"] == 3


@pytest.mark.unit
def test_compute_stats_device_required():
    with pytest.raises(ValueError, match="Device string is required"):
        compute_latency_stats([1.0, 2.0], device="")

    with pytest.raises(ValueError, match="Device string is required"):
        compute_memory_stats(None, device="")


@pytest.mark.integration
def test_standard_pipeline_run_cfc(tmp_path):
    loader = UNSWLoader(run_id="test_pipe_cfc")
    flows = loader.load(str(FIXTURE_PATH))

    splitter = Splitter()
    train_flows, val_flows, test_flows = splitter.split(flows, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42)

    settings = Settings(run_id="run_cfc_eval_001", seed=42)
    encoder = HDCEncoder(dim=500, representation="MAP", encoder_version="eval_hdc_v1")
    seq_model = CfCModel(input_size=500, hidden_size=16)
    scorer = DivergenceScorer(target_fpr=0.05)

    pipeline = StandardPipeline(
        encoder=encoder,
        sequence_model=seq_model,
        divergence_scorer=scorer,
        settings=settings,
        epochs=1,
        lr=1e-3,
        device="cpu",
    )

    result = pipeline.run(train_flows, val_flows, test_flows)

    assert isinstance(result, RunResult)
    assert result.run_id == "run_cfc_eval_001"
    assert result.detection_metrics["total_decisions"] == len(test_flows)
    assert result.latency_stats["device"] == "cpu"
    assert result.memory_stats["device"] == "cpu"
    assert len(result.decisions) == len(test_flows)


@pytest.mark.integration
def test_standard_pipeline_run_lstm_baseline(tmp_path):
    loader = UNSWLoader(run_id="test_pipe_lstm")
    flows = loader.load(str(FIXTURE_PATH))

    splitter = Splitter()
    train_flows, val_flows, test_flows = splitter.split(flows, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42)

    settings = Settings(run_id="run_lstm_eval_001", seed=42)
    encoder = HDCEncoder(dim=500, representation="MAP", encoder_version="eval_hdc_lstm_v1")
    seq_model = LSTMBaseline(input_size=500, hidden_size=16)
    scorer = DivergenceScorer(target_fpr=0.05)

    pipeline = StandardPipeline(
        encoder=encoder,
        sequence_model=seq_model,
        divergence_scorer=scorer,
        settings=settings,
        epochs=1,
        lr=1e-3,
        device="cpu",
    )

    result = pipeline.run(train_flows, val_flows, test_flows)

    assert isinstance(result, RunResult)
    assert result.run_id == "run_lstm_eval_001"
    assert len(result.decisions) == len(test_flows)


@pytest.mark.integration
def test_run_experiment_persists_5_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    loader = UNSWLoader(run_id="test_harness")
    flows = loader.load(str(FIXTURE_PATH))

    splitter = Splitter()
    train_flows, val_flows, test_flows = splitter.split(flows, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42)

    run_id = "exp_run_001"
    settings = Settings(run_id=run_id, seed=42)
    encoder = HDCEncoder(dim=500, representation="MAP", encoder_version="exp_hdc_v1")
    seq_model = CfCModel(input_size=500, hidden_size=16)
    scorer = DivergenceScorer(target_fpr=0.05)

    pipeline = StandardPipeline(
        encoder=encoder,
        sequence_model=seq_model,
        divergence_scorer=scorer,
        settings=settings,
        epochs=1,
        lr=1e-3,
        device="cpu",
    )

    result = run_experiment(settings, pipeline, train_flows, val_flows, test_flows)

    run_dir = Path("runs") / run_id
    assert run_dir.exists()

    # 1. config.snapshot.yaml
    assert (run_dir / "config.snapshot.yaml").exists()

    # 2. git_commit.txt
    assert (run_dir / "git_commit.txt").exists()

    # 3. metrics.json
    metrics_file = run_dir / "metrics.json"
    assert metrics_file.exists()
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics_json = json.load(f)
    assert metrics_json["detection_metrics"] == result.detection_metrics

    # 4. decisions.parquet
    assert (run_dir / "decisions.parquet").exists()

    # 5. logs/
    assert (run_dir / "logs").exists()
