"""Data Quality tier test validating CICIoT2023 dataset slice (ciciot_slice_5k.csv)."""

from pathlib import Path
import pandas as pd
import pytest

from hdlnn.contracts.flow import CanonicalFlow
from hdlnn.data.loaders.ciciot import CICIoTLoader
from hdlnn.data.quality import check_quality

SLICE_PATH = Path(__file__).parent.parent / "data" / "raw" / "ciciot_slice_5k.csv"


@pytest.mark.data_quality
def test_data_quality_ciciot_slice_structural_properties():
    """Verify structural loader properties against raw CSV on ciciot_slice_5k.csv."""
    assert SLICE_PATH.exists(), f"Real dataset slice file not found at {SLICE_PATH}"

    raw_df = pd.read_csv(SLICE_PATH)

    loader = CICIoTLoader(run_id="test_dq_ciciot")
    flows = loader.load(str(SLICE_PATH))

    # 1. Total row count matches
    assert len(flows) == 5500
    assert len(raw_df) == 5500

    # 2. Label breakdown matches expectation
    loaded_attack_count = sum(1 for f in flows if f.label == "attack")
    loaded_normal_count = sum(1 for f in flows if f.label == "normal")

    assert loaded_normal_count == 2500
    assert loaded_attack_count == 3000

    # 3. Quality report validation via check_quality helper
    report = check_quality(flows)
    assert report is not None

    assert report.total_rows == 5500
    assert report.unparseable_row_count == 0
    assert report.duplicate_flow_id_count == 0


@pytest.mark.data_quality
def test_data_quality_ciciot_slice_hand_verified_rows():
    """Verify specific CanonicalFlow field values against raw CSV rows for hand-checked samples."""
    raw_df = pd.read_csv(SLICE_PATH)

    loader = CICIoTLoader(run_id="test_dq_ciciot_hand")
    flows = loader.load(str(SLICE_PATH))

    # Hand-checked Benign row
    benign_idx = raw_df[raw_df["label"].astype(str).str.lower().str.contains("benign")].index[0]
    f_benign = flows[benign_idx]
    assert isinstance(f_benign, CanonicalFlow)
    assert f_benign.flow_id == f"ciciot_{benign_idx}"
    assert f_benign.dataset_source == "CICIOT2023"
    assert f_benign.label == "normal"
    assert f_benign.attack_category is None
    assert f_benign.src_ip == f"ciciot_entity_{benign_idx}"
    assert f_benign.dst_ip == "0.0.0.0"
    assert "Rate" in f_benign.numeric_features or "rate" in f_benign.numeric_features

    # Hand-checked DDoS row
    ddos_idx = raw_df[raw_df["label"].astype(str).str.lower().str.contains("ddos")].index[0]
    f_ddos = flows[ddos_idx]
    assert f_ddos.label == "attack"
    assert f_ddos.attack_category == "DDoS-HTTP_Flood"

    # Hand-checked Recon row
    recon_idx = raw_df[raw_df["label"].astype(str).str.lower().str.contains("recon")].index[0]
    f_recon = flows[recon_idx]
    assert f_recon.label == "attack"
    assert f_recon.attack_category == "Recon-PingSweep"

    # Hand-checked Mirai row
    mirai_idx = raw_df[raw_df["label"].astype(str).str.lower().str.contains("mirai")].index[0]
    f_mirai = flows[mirai_idx]
    assert f_mirai.label == "attack"
    assert f_mirai.attack_category == "Mirai-greeth_flood"
