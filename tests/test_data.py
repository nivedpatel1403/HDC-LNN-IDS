"""Unit tests for L2 data pipeline components (UNSWLoader, DataQualityReport, Splitter)."""

from pathlib import Path
import pytest

from hdlnn.data.interfaces import ILoader
from hdlnn.data.loaders.ciciot import CICIoTLoader
from hdlnn.data.loaders.nslkdd import NSLKDDLoader
from hdlnn.data.loaders.unsw import UNSWLoader
from hdlnn.data.quality import DataQualityReport, check_quality
from hdlnn.data.splitter import Splitter

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "unsw_sample.csv"


@pytest.mark.data_quality
def test_unsw_loader_implements_iloader():
    loader = UNSWLoader(run_id="test_unsw_iloader")
    assert isinstance(loader, ILoader)


@pytest.mark.data_quality
def test_unsw_loader_loads_synthetic_fixture():
    loader = UNSWLoader(run_id="test_unsw_fixture")
    flows = loader.load(str(FIXTURE_PATH))

    assert len(flows) > 0
    first_flow = flows[0]
    assert first_flow.flow_id.startswith("unsw_")
    assert first_flow.protocol in ["tcp", "udp", "arp", "ospf", "icmp", "unk"]
    assert "sbytes" in first_flow.numeric_features
    assert "proto" in first_flow.categorical_features
    assert first_flow.label in ["normal", "attack"]


@pytest.mark.data_quality
def test_data_quality_report():
    loader = UNSWLoader(run_id="test_quality")
    flows = loader.load(str(FIXTURE_PATH))

    report = check_quality(flows)
    assert isinstance(report, DataQualityReport)
    assert report.total_rows == len(flows)


@pytest.mark.data_quality
def test_splitter_deterministic_and_non_overwriting(tmp_path):
    loader = UNSWLoader(run_id="test_splitter")
    flows = loader.load(str(FIXTURE_PATH))

    splitter = Splitter()
    train1, val1, test1 = splitter.split(flows, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42)
    train2, val2, test2 = splitter.split(flows, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42)

    assert [f.flow_id for f in train1] == [f.flow_id for f in train2]
    assert [f.flow_id for f in val1] == [f.flow_id for f in val2]

    # Save split JSON file
    split_file = splitter.save_split(flows, dataset_name="unsw", version="test_v1", out_dir=str(tmp_path))
    assert Path(split_file).exists()

    # Saving again with same version must raise FileExistsError
    with pytest.raises(FileExistsError, match="Split version file already exists"):
        splitter.save_split(flows, dataset_name="unsw", version="test_v1", out_dir=str(tmp_path))


@pytest.mark.data_quality
def test_all_loaders_implement_iloader():
    ciciot_loader = CICIoTLoader()
    nslkdd_loader = NSLKDDLoader()

    assert isinstance(ciciot_loader, ILoader)
    assert isinstance(nslkdd_loader, ILoader)
