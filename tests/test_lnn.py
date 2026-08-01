"""Component tests for L3b LNN Core Layer (CfCModel and LTCModel)."""

from pathlib import Path
import numpy as np
import pytest
import torch

from hdlnn.contracts.encoded import EncodedHypervector
from hdlnn.contracts.trajectory import TrajectoryState
from hdlnn.lnn.cfc_model import CfCModel
from hdlnn.lnn.interfaces import ISequenceModel
from hdlnn.lnn.ltc_model import LTCModel
from hdlnn.lnn.utils import load_hidden_state


def create_synthetic_encoded_sequence(
    tmp_path: Path, n_steps: int = 5, dim: int = 100, encoder_version: str = "hdc_v1"
) -> list[EncodedHypervector]:
    """Helper to create a synthetic sequence of EncodedHypervector contract objects."""
    enc_dir = tmp_path / "encoded" / encoder_version
    enc_dir.mkdir(parents=True, exist_ok=True)

    sequence: list[EncodedHypervector] = []
    rng = np.random.RandomState(42)

    for i in range(n_steps):
        flow_id = f"flow_{i+1}"
        vec = rng.choice([-1.0, 1.0], size=(dim,)).astype(np.float32)
        vec_path = enc_dir / f"{flow_id}.npy"
        np.save(vec_path, vec)

        hv = EncodedHypervector(
            flow_id=flow_id,
            encoder_version=encoder_version,
            dim=dim,
            representation="MAP",
            vector_ref=str(vec_path),
            oov_fields=[],
        )
        sequence.append(hv)

    return sequence


@pytest.mark.component
def test_cfc_model_batch_vs_streaming_equivalence(tmp_path):
    torch.manual_seed(42)
    sequence = create_synthetic_encoded_sequence(tmp_path, n_steps=5, dim=100)

    model = CfCModel(input_size=100, hidden_size=32, model_version="cfc_test_v1")
    assert isinstance(model, ISequenceModel)

    # Batch pass
    batch_trajs = model.forward(sequence, entity_id="entity_1")
    assert len(batch_trajs) == 5

    # Streaming pass
    stream_trajs: list[TrajectoryState] = []
    prev_state: TrajectoryState | None = None
    for hv in sequence:
        curr_state = model.step(hv, prev_state=prev_state, entity_id="entity_1")
        stream_trajs.append(curr_state)
        prev_state = curr_state

    # Assert numerical equivalence between batch and streaming
    for i in range(5):
        batch_h = load_hidden_state(batch_trajs[i].hidden_state_ref)
        stream_h = load_hidden_state(stream_trajs[i].hidden_state_ref)
        max_diff = torch.max(torch.abs(batch_h - stream_h)).item()
        assert max_diff < 1e-5, f"Step {i} batch vs streaming diff {max_diff} exceeded tolerance"


@pytest.mark.component
def test_ltc_model_batch_vs_streaming_equivalence(tmp_path):
    torch.manual_seed(42)
    sequence = create_synthetic_encoded_sequence(tmp_path, n_steps=5, dim=100)

    model = LTCModel(input_size=100, hidden_size=32, model_version="ltc_test_v1")
    assert isinstance(model, ISequenceModel)

    # Batch pass
    batch_trajs = model.forward(sequence, entity_id="entity_1")
    assert len(batch_trajs) == 5

    # Streaming pass
    stream_trajs: list[TrajectoryState] = []
    prev_state: TrajectoryState | None = None
    for hv in sequence:
        curr_state = model.step(hv, prev_state=prev_state, entity_id="entity_1")
        stream_trajs.append(curr_state)
        prev_state = curr_state

    # Assert numerical equivalence between batch and streaming
    for i in range(5):
        batch_h = load_hidden_state(batch_trajs[i].hidden_state_ref)
        stream_h = load_hidden_state(stream_trajs[i].hidden_state_ref)
        max_diff = torch.max(torch.abs(batch_h - stream_h)).item()
        assert max_diff < 1e-5, f"Step {i} batch vs streaming diff {max_diff} exceeded tolerance"


@pytest.mark.component
def test_multi_entity_state_isolation(tmp_path):
    sequence_a = create_synthetic_encoded_sequence(tmp_path, n_steps=3, dim=100)
    sequence_b = create_synthetic_encoded_sequence(tmp_path, n_steps=3, dim=100)

    model = CfCModel(input_size=100, hidden_size=32)

    state_a1 = model.step(sequence_a[0], prev_state=None, entity_id="entity_A")
    state_b1 = model.step(sequence_b[0], prev_state=None, entity_id="entity_B")
    state_a2 = model.step(sequence_a[1], prev_state=state_a1)

    assert state_a2.entity_id == "entity_A"
    assert state_b1.entity_id == "entity_B"
    assert Path(state_a2.hidden_state_ref).exists()
    assert Path(state_b1.hidden_state_ref).exists()


@pytest.mark.component
def test_trajectory_state_persistence_on_disk(tmp_path):
    sequence = create_synthetic_encoded_sequence(tmp_path, n_steps=2, dim=50)
    model = CfCModel(input_size=50, hidden_size=16)

    trajs = model.forward(sequence, entity_id="test_entity")
    for traj in trajs:
        ref_path = Path(traj.hidden_state_ref)
        assert ref_path.exists()
        arr = np.load(ref_path)
        assert arr.shape == (1, 16)
