"""Data Quality tier test validating NSL-KDD dataset slice (nslkdd_slice_5k.csv & KDDTrain+.txt)."""

from pathlib import Path
import pandas as pd
import pytest

from hdlnn.contracts.flow import CanonicalFlow
from hdlnn.data.loaders.nslkdd import NSLKDDLoader
from hdlnn.data.quality import check_quality

SLICE_PATH = Path(__file__).parent.parent / "data" / "raw" / "nslkdd_slice_5k.csv"
KDD_TRAIN_PATH = Path(__file__).parent.parent / "data" / "raw" / "KDDTrain+.txt"


@pytest.mark.data_quality
def test_data_quality_nslkdd_slice_structural_properties():
    """Verify structural loader properties against raw CSV on nslkdd_slice_5k.csv."""
    assert SLICE_PATH.exists(), f"Real dataset slice file not found at {SLICE_PATH}"

    raw_df = pd.read_csv(SLICE_PATH)

    loader = NSLKDDLoader(run_id="test_dq_nslkdd")
    flows = loader.load(str(SLICE_PATH))

    # 1. Total row count matches
    assert len(flows) == 5000
    assert len(raw_df) == 5000

    # 2. Label breakdown matches expectation
    loaded_attack_count = sum(1 for f in flows if f.label == "attack")
    loaded_normal_count = sum(1 for f in flows if f.label == "normal")

    raw_normal_count = int((raw_df["label"].str.strip().str.lower() == "normal").sum())
    raw_attack_count = 5000 - raw_normal_count

    assert loaded_normal_count == raw_normal_count
    assert loaded_attack_count == raw_attack_count

    # 3. Quality report validation
    report = check_quality(flows)
    assert report is not None

    assert report.total_rows == 5000
    assert report.unparseable_row_count == 0
    assert report.duplicate_flow_id_count == 0


@pytest.mark.data_quality
def test_data_quality_nslkdd_kddtrain_structural_properties():
    """Verify NSLKDDLoader execution on raw unheadered KDDTrain+.txt."""
    if not KDD_TRAIN_PATH.exists():
        pytest.skip(f"KDDTrain+.txt not found at {KDD_TRAIN_PATH}")

    loader = NSLKDDLoader(run_id="test_dq_kddtrain")
    flows = loader.load(str(KDD_TRAIN_PATH))

    assert len(flows) == 125973
    f0 = flows[0]
    assert isinstance(f0, CanonicalFlow)
    assert f0.dataset_source == "NSL-KDD"
    assert f0.flow_id == "nslkdd_0"
    assert f0.protocol in ("TCP", "UDP", "ICMP")
    assert f0.label in ("normal", "attack")

    report = check_quality(flows)
    assert report is not None


@pytest.mark.data_quality
def test_data_quality_nslkdd_slice_hand_verified_rows():
    """Verify specific CanonicalFlow field values against raw CSV rows for hand-checked samples."""
    raw_df = pd.read_csv(SLICE_PATH)

    loader = NSLKDDLoader(run_id="test_dq_nslkdd_hand")
    flows = loader.load(str(SLICE_PATH))

    # Hand-checked row 0 (Normal)
    f0 = flows[0]
    assert isinstance(f0, CanonicalFlow)
    assert f0.flow_id == "nslkdd_0"
    assert f0.dataset_source == "NSL-KDD"
    assert f0.label == "normal"
    assert f0.attack_category is None
    assert f0.protocol.upper() == "TCP"
    assert f0.src_ip == "nslkdd_entity_0"
    assert f0.dst_ip == "0.0.0.0"

    # Hand-checked Neptune row
    nep_idx = raw_df[raw_df["label"].astype(str).str.lower() == "neptune"].index[0]
    f_nep = flows[nep_idx]
    assert f_nep.label == "attack"
    assert f_nep.attack_category == "neptune"
