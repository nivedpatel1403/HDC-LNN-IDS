"""Data Quality tier test validating real UNSW-NB15 dataset slice (unsw_slice_20k.csv)."""

from pathlib import Path
import pandas as pd
import pytest

from hdlnn.contracts.flow import CanonicalFlow
from hdlnn.data.loaders.unsw import UNSWLoader

SLICE_PATH = Path(__file__).parent.parent / "data" / "raw" / "unsw_slice_20k.csv"


@pytest.mark.data_quality
def test_data_quality_real_unsw_slice_structural_properties():
    """Verify structural loader properties against manual raw CSV calculations on unsw_slice_20k.csv."""
    assert SLICE_PATH.exists(), f"Real dataset slice file not found at {SLICE_PATH}"

    raw_df = pd.read_csv(SLICE_PATH)

    loader = UNSWLoader(run_id="test_dq_real")
    flows = loader.load(str(SLICE_PATH))

    # 1. Total row count matches
    assert len(flows) == 20000
    assert len(raw_df) == 20000

    # 2. Label distribution matches manual pandas value_counts() exactly
    raw_attack_count = int((raw_df["label"] == 1).sum())
    raw_normal_count = int((raw_df["label"] == 0).sum())

    loaded_attack_count = sum(1 for f in flows if f.label == "attack")
    loaded_normal_count = sum(1 for f in flows if f.label == "normal")

    assert loaded_attack_count == 953
    assert loaded_normal_count == 19047
    assert loaded_attack_count == raw_attack_count
    assert loaded_normal_count == raw_normal_count

    # 3. Attack category set matches raw CSV set
    raw_categories = set(raw_df["attack_cat"].dropna().str.strip().unique()) - {"normal", "-", "0"}
    loaded_categories = set(f.attack_category for f in flows if f.attack_category is not None)

    assert loaded_categories.issubset(raw_categories) or loaded_categories == raw_categories
    assert "normal" not in loaded_categories  # Normal mapped to attack_category=None

    # 4. Data Quality Report metrics match expected raw file characteristics
    report = loader.last_quality_report
    assert report is not None
    assert report.total_rows == 20000
    assert report.unparseable_row_count == 0
    assert report.duplicate_flow_id_count == 0
    assert report.missing_field_counts.get("src_ip", 0) == 0
    assert report.missing_field_counts.get("dst_ip", 0) == 0
    assert report.missing_field_counts.get("src_port", 0) == 0
    assert report.missing_field_counts.get("dst_port", 0) == 0


@pytest.mark.data_quality
def test_data_quality_real_unsw_slice_hand_verified_rows():
    """Verify specific CanonicalFlow field values against raw CSV rows for hand-checked samples."""
    raw_df = pd.read_csv(SLICE_PATH)

    loader = UNSWLoader(run_id="test_dq_hand")
    flows = loader.load(str(SLICE_PATH))

    # Hand-checked row 0 (Normal)
    r0 = raw_df.iloc[0]
    f0 = flows[0]
    assert isinstance(f0, CanonicalFlow)
    assert f0.flow_id == "unsw_0"
    assert f0.src_ip == "59.166.0.8"
    assert f0.dst_ip == "149.171.126.2"
    assert f0.src_port == 44119
    assert f0.dst_port == 53
    assert f0.protocol == "udp"
    assert f0.state == "CON"
    assert pytest.approx(f0.duration, abs=1e-5) == float(r0["dur"])
    assert pytest.approx(f0.numeric_features["sbytes"], abs=1e-5) == float(r0["sbytes"])
    assert f0.label == "normal"
    assert f0.attack_category is None

    # Hand-checked DoS row
    dos_idx = raw_df[raw_df["attack_cat"].astype(str).str.lower() == "dos"].index[0]
    r_dos = raw_df.iloc[dos_idx]
    f_dos = flows[dos_idx]
    assert f_dos.label == "attack"
    assert f_dos.attack_category is not None and f_dos.attack_category.lower() == "dos"
    assert f_dos.protocol == str(r_dos["proto"]).strip()

    # Hand-checked Exploits row
    exp_idx = raw_df[raw_df["attack_cat"].astype(str).str.lower() == "exploits"].index[0]
    r_exp = raw_df.iloc[exp_idx]
    f_exp = flows[exp_idx]
    assert f_exp.label == "attack"
    assert f_exp.attack_category is not None and f_exp.attack_category.lower() == "exploits"
    assert f_exp.protocol == str(r_exp["proto"]).strip()
