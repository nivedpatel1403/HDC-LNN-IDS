"""Unit and component tests for L3a HDC Encoder Layer (ItemMemory and HDCEncoder)."""

from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytest
import torch

from hdlnn.contracts.errors import ConfigError, ItemMemoryError
from hdlnn.contracts.flow import CanonicalFlow
from hdlnn.hdc.encoder import HDCEncoder
from hdlnn.hdc.interfaces import IEncoder
from hdlnn.hdc.item_memory import ItemMemory


def create_sample_flows() -> list[CanonicalFlow]:
    now = datetime.now(timezone.utc)
    f1 = CanonicalFlow(
        dataset_source="UNSW-NB15",
        flow_id="f1",
        occurred_at=now,
        src_ip="10.0.0.1",
        src_port=80,
        dst_ip="10.0.0.2",
        dst_port=443,
        protocol="tcp",
        state="FIN",
        duration=0.1,
        src_bytes=100,
        dst_bytes=200,
        numeric_features={"dur": 0.1, "sbytes": 100.0, "dbytes": 200.0},
        categorical_features={"proto": "tcp", "state": "FIN"},
        label="normal",
    )
    f1.split = "train"

    f2 = CanonicalFlow(
        dataset_source="UNSW-NB15",
        flow_id="f2",
        occurred_at=now,
        src_ip="10.0.0.1",
        src_port=80,
        dst_ip="10.0.0.2",
        dst_port=443,
        protocol="udp",
        state="INT",
        duration=0.5,
        src_bytes=50,
        dst_bytes=50,
        numeric_features={"dur": 0.5, "sbytes": 50.0, "dbytes": 50.0},
        categorical_features={"proto": "udp", "state": "INT"},
        label="normal",
    )
    f2.split = "train"

    return [f1, f2]


@pytest.mark.unit
def test_item_memory_fit_train_split_enforced():
    flows = create_sample_flows()
    flows[0].split = "val"  # Non-train split!

    item_mem = ItemMemory(dim=100, representation="MAP")
    with pytest.raises(ItemMemoryError, match="P4 violation"):
        item_mem.fit(flows)


@pytest.mark.unit
def test_item_memory_oov_reservation():
    flows = create_sample_flows()
    item_mem = ItemMemory(dim=100, representation="MAP")
    item_mem.fit(flows)

    # Known category returns assigned hypervector
    vec_tcp, is_oov_tcp = item_mem.lookup("proto", "tcp")
    assert not is_oov_tcp
    assert vec_tcp.shape == (100,)

    # Unknown category returns OOV reserved hypervector
    vec_unknown, is_oov_unkn = item_mem.lookup("proto", "HTTP_SPECIAL")
    assert is_oov_unkn
    assert vec_unknown.shape == (100,)


@pytest.mark.unit
def test_item_memory_save_load(tmp_path):
    flows = create_sample_flows()
    item_mem = ItemMemory(dim=100, representation="MAP")
    item_mem.fit(flows)

    save_path = tmp_path / "item_memory.pt"
    item_mem.save(str(save_path))
    assert save_path.exists()

    loaded_mem = ItemMemory.load(str(save_path))
    assert loaded_mem.is_fitted
    assert loaded_mem.dim == 100

    vec1, _ = item_mem.lookup("proto", "tcp")
    vec2, _ = loaded_mem.lookup("proto", "tcp")
    assert torch.equal(vec1, vec2)


@pytest.mark.component
def test_hdc_encoder_fit_and_encode(tmp_path):
    flows = create_sample_flows()

    encoder = HDCEncoder(dim=1000, representation="MAP", num_levels=10, encoder_version="test_v1")
    assert isinstance(encoder, IEncoder)

    encoder.fit(flows)

    # Encode flow
    enc_hv = encoder.encode(flows[0])
    assert enc_hv.flow_id == "f1"
    assert enc_hv.encoder_version == "test_v1"
    assert enc_hv.dim == 1000
    assert enc_hv.representation == "MAP"
    assert Path(enc_hv.vector_ref).exists()

    # Check saved numpy array shape and dtype
    vec = np.load(enc_hv.vector_ref)
    assert vec.shape == (1000,)
    assert vec.dtype == np.float32


@pytest.mark.component
def test_hdc_encoder_binary_rejected():
    with pytest.raises(ConfigError, match="ADR-001"):
        HDCEncoder(dim=1000, representation="binary")


@pytest.mark.component
def test_hdc_encoder_persistence(tmp_path):
    flows = create_sample_flows()

    encoder = HDCEncoder(dim=500, representation="MAP", num_levels=10, encoder_version="persist_v1")
    encoder.fit(flows)

    save_path = tmp_path / "item_memory.pt"
    encoder.item_memory.save(str(save_path))
    assert save_path.exists()

    loaded_mem = ItemMemory.load(str(save_path))
    assert loaded_mem.is_fitted
    assert loaded_mem.dim == 500
