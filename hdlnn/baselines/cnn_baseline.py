"""1D CNN baseline sequence model implementation with sliding window input buffering."""

from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from hdlnn.contracts.encoded import EncodedHypervector
from hdlnn.contracts.trajectory import TrajectoryState
from hdlnn.lnn.interfaces import ISequenceModel
from hdlnn.lnn.utils import load_encoded_vector, save_hidden_state


class CNNBaseline(ISequenceModel):
    """1D CNN baseline sequence model operating over a sliding window of EncodedHypervectors.

    Design Note: CNNs are non-recurrent. The state maintained across steps is a sliding window
    buffer of the most recent 'window_size' hypervectors. The CNN output feature embedding (1, hidden_size)
    is persisted to hidden_state_ref for seamless reference manifold scoring compatibility.
    """

    def __init__(
        self,
        input_size: int = 10000,
        hidden_size: int = 64,
        window_size: int = 10,
        model_version: str | None = None,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.model_version = model_version or f"cnn_v1_w{self.window_size}"

        # 1D Convolution stack over sliding window
        self.conv = nn.Conv1d(
            in_channels=self.input_size,
            out_channels=self.hidden_size,
            kernel_size=3,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def _process_buffer(self, buffer_tensors: list[torch.Tensor]) -> torch.Tensor:
        """Zero-pad or clip buffer tensors list to window_size and run Conv1d forward pass."""

        # Keep last window_size elements if length exceeds window_size
        recent_tensors = buffer_tensors[-self.window_size :]

        # Left zero-pad if length is less than window_size
        pad_count = self.window_size - len(recent_tensors)
        padded_tensors: list[torch.Tensor] = []
        for _ in range(pad_count):
            padded_tensors.append(torch.zeros(1, self.input_size))
        padded_tensors.extend(recent_tensors)

        # Stack to (1, window_size, input_size) -> transpose to (1, input_size, window_size)
        stacked = torch.stack([t.squeeze(0) for t in padded_tensors], dim=0).unsqueeze(0)
        x_in = stacked.transpose(1, 2)

        out = self.pool(self.relu(self.conv(x_in))).squeeze(-1)  # Shape (1, hidden_size)
        return out

    def forward(
        self, sequence: list[EncodedHypervector], entity_id: str = "default"
    ) -> list[TrajectoryState]:
        """Batch mode: process a full ordered sequence by maintaining the sliding window.

        Args:
            sequence: Ordered list of EncodedHypervector contract objects.
            entity_id: Identifier string for the entity.

        Returns:
            List of TrajectoryState objects corresponding to each record step.
        """
        if not sequence:
            return []

        results: list[TrajectoryState] = []
        buffer_tensors: list[torch.Tensor] = []

        for hv in sequence:
            hv_tensor = load_encoded_vector(hv).squeeze(0)  # Shape (1, input_size)
            buffer_tensors.append(hv_tensor)
            if len(buffer_tensors) > self.window_size:
                buffer_tensors.pop(0)

            h_output = self._process_buffer(buffer_tensors)

            ref_path = save_hidden_state(
                h_output,
                encoder_version=hv.encoder_version,
                entity_id=entity_id,
                flow_id=hv.flow_id,
            )

            # Save window buffer tensor alongside trajectory output
            buf_path = Path(ref_path).parent / f"{hv.flow_id}_cnn_buf.npy"
            buf_arr = torch.stack([t.squeeze(0) for t in buffer_tensors], dim=0).detach().cpu().numpy()
            np.save(buf_path, buf_arr)

            traj = TrajectoryState(
                entity_id=entity_id,
                flow_id=hv.flow_id,
                t=datetime.now(timezone.utc),
                hidden_state_ref=ref_path,
                model_version=self.model_version,
            )
            results.append(traj)

        return results

    def step(
        self,
        hv: EncodedHypervector,
        prev_state: TrajectoryState | None = None,
        entity_id: str | None = None,
    ) -> TrajectoryState:
        """Streaming mode: append new record to sliding window buffer and compute CNN embedding.

        Args:
            hv: Single EncodedHypervector contract object.
            prev_state: Optional previous TrajectoryState carrying window buffer reference.
            entity_id: Optional entity ID string.

        Returns:
            Updated TrajectoryState contract instance.
        """
        eid = entity_id or (prev_state.entity_id if prev_state else "default")
        buffer_tensors: list[torch.Tensor] = []

        if prev_state is not None:
            buf_path = Path(prev_state.hidden_state_ref).parent / f"{prev_state.flow_id}_cnn_buf.npy"
            if buf_path.exists():
                buf_arr = np.load(buf_path)
                buf_t = torch.from_numpy(buf_arr).float()
                for i in range(buf_t.shape[0]):
                    buffer_tensors.append(buf_t[i : i + 1, :])

        hv_tensor = load_encoded_vector(hv).squeeze(0)  # (1, input_size)
        buffer_tensors.append(hv_tensor)
        if len(buffer_tensors) > self.window_size:
            buffer_tensors.pop(0)

        h_output = self._process_buffer(buffer_tensors)

        ref_path = save_hidden_state(
            h_output,
            encoder_version=hv.encoder_version,
            entity_id=eid,
            flow_id=hv.flow_id,
        )

        buf_path = Path(ref_path).parent / f"{hv.flow_id}_cnn_buf.npy"
        buf_arr = torch.stack([t.squeeze(0) for t in buffer_tensors], dim=0).detach().cpu().numpy()
        np.save(buf_path, buf_arr)

        return TrajectoryState(
            entity_id=eid,
            flow_id=hv.flow_id,
            t=datetime.now(timezone.utc),
            hidden_state_ref=ref_path,
            model_version=self.model_version,
        )
