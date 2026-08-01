"""Integration tier test verifying end-to-end pipeline execution against hand-computed metrics."""

from datetime import datetime, timezone
import pytest

from hdlnn.common.config import Settings
from hdlnn.common.seed import set_global_seed
from hdlnn.contracts.flow import CanonicalFlow
from hdlnn.eval.pipeline_adapter import StandardPipeline


def create_hand_computed_test_scenario() -> tuple[list[CanonicalFlow], list[CanonicalFlow], list[CanonicalFlow]]:
    """Construct a known synthetic scenario with 10 train, 10 val, and 8 test flows.

    Test flows:
      - 4 clearly normal flows (matching train/val baseline traffic pattern)
      - 4 clearly anomalous flows (extreme numeric feature values by construction)
    """
    now = datetime.now(timezone.utc)
    train_flows: list[CanonicalFlow] = []
    val_flows: list[CanonicalFlow] = []
    test_flows: list[CanonicalFlow] = []

    # 10 Normal Train Flows
    for i in range(10):
        f = CanonicalFlow(
            dataset_source="UNSW-NB15",
            flow_id=f"train_{i+1}",
            occurred_at=now,
            src_ip=f"192.168.1.{i+1}",
            src_port=1000 + i,
            dst_ip="10.0.0.1",
            dst_port=80,
            protocol="tcp",
            state="FIN",
            duration=0.1,
            src_bytes=100,
            dst_bytes=200,
            numeric_features={"dur": 0.1, "sbytes": 100.0, "dbytes": 200.0, "sttl": 64.0, "dttl": 64.0},
            categorical_features={"proto": "tcp", "state": "FIN"},
            label="normal",
        )
        f.split = "train"
        train_flows.append(f)

    # 10 Normal Val Flows
    for i in range(10):
        f = CanonicalFlow(
            dataset_source="UNSW-NB15",
            flow_id=f"val_{i+1}",
            occurred_at=now,
            src_ip=f"192.168.2.{i+1}",
            src_port=2000 + i,
            dst_ip="10.0.0.1",
            dst_port=80,
            protocol="tcp",
            state="FIN",
            duration=0.1,
            src_bytes=100,
            dst_bytes=200,
            numeric_features={"dur": 0.1, "sbytes": 100.0, "dbytes": 200.0, "sttl": 64.0, "dttl": 64.0},
            categorical_features={"proto": "tcp", "state": "FIN"},
            label="normal",
        )
        f.split = "val"
        val_flows.append(f)

    # 4 Normal Test Flows
    for i in range(4):
        f = CanonicalFlow(
            dataset_source="UNSW-NB15",
            flow_id=f"test_norm_{i+1}",
            occurred_at=now,
            src_ip=f"192.168.3.{i+1}",
            src_port=3000 + i,
            dst_ip="10.0.0.1",
            dst_port=80,
            protocol="tcp",
            state="FIN",
            duration=0.1,
            src_bytes=100,
            dst_bytes=200,
            numeric_features={"dur": 0.1, "sbytes": 100.0, "dbytes": 200.0, "sttl": 64.0, "dttl": 64.0},
            categorical_features={"proto": "tcp", "state": "FIN"},
            label="normal",
        )
        f.split = "test"
        test_flows.append(f)

    # 4 Extreme Attack Test Flows
    for i in range(4):
        f = CanonicalFlow(
            dataset_source="UNSW-NB15",
            flow_id=f"test_att_{i+1}",
            occurred_at=now,
            src_ip=f"192.168.4.{i+1}",
            src_port=4000 + i,
            dst_ip="10.0.0.1",
            dst_port=80,
            protocol="udp",
            state="INT",
            duration=9999.0 * (i + 1),
            src_bytes=int(9999999 * (i + 1)),
            dst_bytes=int(9999999 * (i + 1)),
            numeric_features={"dur": 9999.0 * (i + 1), "sbytes": 9999999.0 * (i + 1), "dbytes": 9999999.0 * (i + 1), "sttl": 255.0, "dttl": 255.0},
            categorical_features={"proto": "udp", "state": "INT"},
            label="attack",
            attack_category="DoS",
        )
        f.split = "test"
        test_flows.append(f)

    return train_flows, val_flows, test_flows


@pytest.mark.integration
def test_hand_computed_integration_metrics(tmp_path, monkeypatch):
    """Integration tier test validating full pipeline execution against hand-computed expected metrics.

    Hand-Computed Expected Breakdown for Known 8-Test-Flow Scenario:
        - Ground truth: 4 normal flows, 4 attack flows
        - Predicted: 4 normal flows (TN=4, FP=0), 4 attack flows (TP=4, FN=0)
        - Precision = TP / (TP + FP) = 4 / 4 = 1.0000
        - Recall = TP / (TP + FN) = 4 / 4 = 1.0000
        - F1 Score = 2 * (1.0 * 1.0) / (1.0 + 1.0) = 1.0000
        - Accuracy = (TP + TN) / Total = (4 + 4) / 8 = 1.0000
        - AUROC = 1.0000
        - FPR @ 95% TPR = 0.0000
    """
    monkeypatch.chdir(tmp_path)

    cfg = Settings(
        run_id="hand_computed_integration_001",
        seed=42,
        dimensionality=100,
        hidden_size=16,
        epochs=0,
        target_fpr=0.05,
    )
    set_global_seed(cfg.seed)

    train_flows, val_flows, test_flows = create_hand_computed_test_scenario()
    pipeline = StandardPipeline(settings=cfg)

    result = pipeline.run(train_flows, val_flows, test_flows)
    m = result.detection_metrics

    # Assert exact match against hand-computed metrics
    assert m["total_decisions"] == 8
    assert m["anomalies_detected"] == 4
    assert pytest.approx(m["precision"], abs=1e-4) == 1.0
    assert pytest.approx(m["recall"], abs=1e-4) == 1.0
    assert pytest.approx(m["f1"], abs=1e-4) == 1.0
    assert pytest.approx(m["auroc"], abs=1e-4) == 1.0
    assert pytest.approx(m["fpr_at_95_tpr"], abs=1e-4) == 0.0
