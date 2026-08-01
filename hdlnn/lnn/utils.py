"""Internal utility functions for LNN state management and vector I/O."""

from pathlib import Path
import numpy as np
import torch

from hdlnn.contracts.encoded import EncodedHypervector


def load_encoded_vector(hv: EncodedHypervector) -> torch.Tensor:
    """Load numpy vector array from EncodedHypervector.vector_ref into a PyTorch Tensor.

    Args:
        hv: EncodedHypervector contract object.

    Returns:
        PyTorch Tensor of shape (1, 1, dim).
    """
    path = Path(hv.vector_ref)
    if not path.exists():
        raise FileNotFoundError(f"Encoded hypervector file not found at '{path}'")

    np_arr = np.load(path)
    tensor = torch.from_numpy(np_arr).float()
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, dim)
    elif tensor.dim() == 2:
        tensor = tensor.unsqueeze(1)  # Shape (1, 1, dim)
    return tensor


def save_hidden_state(
    hidden_tensor: torch.Tensor,
    encoder_version: str,
    entity_id: str,
    flow_id: str,
) -> str:
    """Save hidden state tensor to data/encoded/<encoder_version>/trajectories/<entity_id>/<flow_id>.npy.

    Args:
        hidden_tensor: Hidden state PyTorch Tensor.
        encoder_version: Version identifier string of the encoder.
        entity_id: Entity identifier string.
        flow_id: Flow identifier string.

    Returns:
        Saved file path string.
    """
    target_dir = Path("data/encoded") / encoder_version / "trajectories" / entity_id
    target_dir.mkdir(parents=True, exist_ok=True)

    out_path = target_dir / f"{flow_id}.npy"
    np_arr = hidden_tensor.detach().cpu().numpy()
    np.save(out_path, np_arr)
    return str(out_path)


def load_hidden_state(hidden_state_ref: str) -> torch.Tensor:
    """Load hidden state tensor from numpy file path.

    Args:
        hidden_state_ref: Path string to the saved .npy hidden state file.

    Returns:
        PyTorch Tensor containing the hidden state.
    """
    path = Path(hidden_state_ref)
    if not path.exists():
        raise FileNotFoundError(f"Hidden state file not found at '{path}'")

    np_arr = np.load(path)
    return torch.from_numpy(np_arr).float()
