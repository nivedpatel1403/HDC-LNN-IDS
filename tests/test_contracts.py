"""Unit tests for L1 Contract data structures and error hierarchy."""

from datetime import datetime, timezone
import pytest

from hdlnn.contracts.decision import AnomalyDecision
from hdlnn.contracts.encoded import EncodedHypervector
from hdlnn.contracts.errors import (
    ConfigError,
    DataValidationError,
    HdlnnError,
    ItemMemoryError,
    ManifoldFitError,
)
from hdlnn.contracts.flow import CanonicalFlow
from hdlnn.contracts.trajectory import TrajectoryState


@pytest.mark.unit
def test_canonical_flow_instantiation():
    now = datetime.now(timezone.utc)
    flow = CanonicalFlow(
        dataset_source="UNSW-NB15",
        flow_id="unsw_1",
        occurred_at=now,
        src_ip="192.168.1.1",
        src_port=12345,
        dst_ip="10.0.0.1",
        dst_port=80,
        protocol="tcp",
        state="FIN",
        duration=0.5,
        src_bytes=100,
        dst_bytes=200,
        numeric_features={"sbytes": 100.0, "dbytes": 200.0},
        categorical_features={"proto": "tcp", "state": "FIN"},
        label="normal",
    )
    assert flow.flow_id == "unsw_1"
    assert flow.split is None
    assert flow.attack_category is None


@pytest.mark.unit
def test_encoded_hypervector_instantiation():
    hv = EncodedHypervector(
        flow_id="unsw_1",
        encoder_version="v1",
        dim=10000,
        representation="MAP",
        vector_ref="data/encoded/v1/unsw_1.npy",
        oov_fields=["proto"],
    )
    assert hv.dim == 10000
    assert hv.representation == "MAP"
    assert hv.oov_fields == ["proto"]


@pytest.mark.unit
def test_trajectory_state_instantiation():
    now = datetime.now(timezone.utc)
    traj = TrajectoryState(
        entity_id="192.168.1.1",
        flow_id="unsw_1",
        t=now,
        hidden_state_ref="data/encoded/v1/trajectories/192.168.1.1/unsw_1.npy",
        model_version="cfc_v1",
    )
    assert traj.entity_id == "192.168.1.1"
    assert traj.model_version == "cfc_v1"


@pytest.mark.unit
def test_anomaly_decision_instantiation():
    now = datetime.now(timezone.utc)
    dec = AnomalyDecision(
        entity_id="192.168.1.1",
        flow_id="unsw_1",
        t=now,
        drift_score=2.45,
        threshold=1.50,
        is_anomalous=True,
        manifold_version="man_v1",
        ground_truth_label="attack",
        attack_category="Exploits",
    )
    assert dec.is_anomalous is True
    assert dec.drift_score > dec.threshold
    assert dec.attack_category == "Exploits"


@pytest.mark.unit
def test_hdlnn_error_hierarchy():
    err = HdlnnError("Base error")
    assert isinstance(err, Exception)

    val_err = DataValidationError("Validation failed")
    assert isinstance(val_err, HdlnnError)

    mem_err = ItemMemoryError("Item memory failed")
    assert isinstance(mem_err, HdlnnError)

    cfg_err = ConfigError("Config invalid")
    assert isinstance(cfg_err, HdlnnError)

    man_err = ManifoldFitError("Manifold fit failed")
    assert isinstance(man_err, HdlnnError)
