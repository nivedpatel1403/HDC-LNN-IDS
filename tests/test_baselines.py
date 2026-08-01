"""Component tests for L4 Baseline comparative models (LSTMBaseline and CNNBaseline)."""

from pathlib import Path
import numpy as np
import pytest
import torch

from hdlnn.baselines import CNNBaseline, LSTMBaseline
from hdlnn.contracts.encoded import EncodedHypervector
from hdlnn.contracts.trajectory import TrajectoryState
from hdlnn.lnn.interfaces import ISequenceModel
from hdlnn.lnn.utils import load_hidden_state


def create_synthetic_encoded_sequence(
    tmp_path: Path, n_steps: int = 5, dim: int = 100, encoder_version: str = "hdc_base_v1"
) -> list[EncodedHypervector]:
    """Helper to create a synthetic sequence of EncodedHypervector contract objects."""
    enc_dir = tmp_path / "encoded" / encoder_version
    enc_dir.mkdir(parents=True, exist_ok=True)

    sequence: list[EncodedHypervector] = []
    rng = np.random.RandomState(42)

    for i in range(n_steps):
        flow_id = f"flow_base_{i+1}"
        vec = rng.randn(dim).astype(np.float32)
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
def test_baselines_isinstance_isequencemodel():
    lstm = LSTMBaseline(input_size=100, hidden_size=32)
    cnn = CNNBaseline(input_size=100, hidden_size=32, window_size=5)

    assert isinstance(lstm, ISequenceModel)
    assert isinstance(cnn, ISequenceModel)


@pytest.mark.component
def test_lstm_batch_vs_streaming_equivalence(tmp_path):
    torch.manual_seed(42)
    sequence = create_synthetic_encoded_sequence(tmp_path, n_steps=5, dim=100)

    model = LSTMBaseline(input_size=100, hidden_size=32, model_version="lstm_test_v1")

    # Batch pass
    batch_trajs = model.forward(sequence, entity_id="entity_lstm")
    assert len(batch_trajs) == 5

    # Streaming pass
    stream_trajs: list[TrajectoryState] = []
    prev_state: TrajectoryState | None = None
    for hv in sequence:
        curr_state = model.step(hv, prev_state=prev_state, entity_id="entity_lstm")
        stream_trajs.append(curr_state)
        prev_state = curr_state

    # Assert numerical equivalence between batch and streaming
    for i in range(5):
        batch_h = load_hidden_state(batch_trajs[i].hidden_state_ref)
        stream_h = load_hidden_state(stream_trajs[i].hidden_state_ref)
        max_diff = torch.max(torch.abs(batch_h - stream_h)).item()
        assert max_diff < 1e-5, f"LSTM Step {i} batch vs streaming diff {max_diff} exceeded tolerance"


@pytest.mark.component
def test_lstm_entity_state_isolation(tmp_path):
    sequence_a = create_synthetic_encoded_sequence(tmp_path, n_steps=3, dim=100)
    sequence_b = create_synthetic_encoded_sequence(tmp_path, n_steps=3, dim=100)

    model = LSTMBaseline(input_size=100, hidden_size=32)

    state_a1 = model.step(sequence_a[0], prev_state=None, entity_id="entity_A")
    state_b1 = model.step(sequence_b[0], prev_state=None, entity_id="entity_B")
    state_a2 = model.step(sequence_a[1], prev_state=state_a1)

    assert state_a2.entity_id == "entity_A"
    assert state_b1.entity_id == "entity_B"
    assert Path(state_a2.hidden_state_ref).exists()
    assert Path(state_b1.hidden_state_ref).exists()


@pytest.mark.component
def test_cnn_sliding_window_overflow_and_padding(tmp_path):
    sequence = create_synthetic_encoded_sequence(tmp_path, n_steps=12, dim=50)
    window_size = 5

    cnn = CNNBaseline(input_size=50, hidden_size=16, window_size=window_size)

    # 1. Under window size zero-padding handling (step 1, len 1 < 5)
    state1 = cnn.step(sequence[0], prev_state=None, entity_id="entity_cnn")
    buf_path1 = Path(state1.hidden_state_ref).parent / f"{sequence[0].flow_id}_cnn_buf.npy"
    assert buf_path1.exists()
    buf1 = np.load(buf_path1)
    assert buf1.shape == (1, 50)  # Raw unpadded stored buffer length is 1

    # 2. Sequential steps past window size capacity (steps 1..12)
    prev_state = None
    all_states: list[TrajectoryState] = []
    for hv in sequence:
        curr_state = cnn.step(hv, prev_state=prev_state, entity_id="entity_cnn")
        all_states.append(curr_state)
        prev_state = curr_state

    # Check last state buffer dropped oldest records (max capacity = window_size = 5)
    last_buf_path = Path(all_states[-1].hidden_state_ref).parent / f"{sequence[-1].flow_id}_cnn_buf.npy"
    last_buf = np.load(last_buf_path)
    assert last_buf.shape == (5, 50)


@pytest.mark.component
def test_cnn_batch_vs_streaming_equivalence(tmp_path):
    torch.manual_seed(42)
    sequence = create_synthetic_encoded_sequence(tmp_path, n_steps=8, dim=60)

    cnn = CNNBaseline(input_size=60, hidden_size=24, window_size=4)

    # Batch forward pass
    batch_trajs = cnn.forward(sequence, entity_id="entity_cnn_seq")

    # Streaming step calls
    stream_trajs: list[TrajectoryState] = []
    prev_state: TrajectoryState | None = None
    for hv in sequence:
        curr_state = cnn.step(hv, prev_state=prev_state, entity_id="entity_cnn_seq")
        stream_trajs.append(curr_state)
        prev_state = curr_state

    for i in range(8):
        batch_h = load_hidden_state(batch_trajs[i].hidden_state_ref)
        stream_h = load_hidden_state(stream_trajs[i].hidden_state_ref)
        max_diff = torch.max(torch.abs(batch_h - stream_h)).item()
        assert max_diff < 1e-5, f"CNN Step {i} batch vs streaming diff {max_diff} exceeded tolerance"
