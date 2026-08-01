"""LSTM baseline sequence model implementation (conventional RNN comparison)."""

from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from hdlnn.contracts.encoded import EncodedHypervector
from hdlnn.contracts.trajectory import TrajectoryState
from hdlnn.lnn.interfaces import ISequenceModel
from hdlnn.lnn.utils import load_encoded_vector, save_hidden_state


class LSTMBaseline(ISequenceModel):
    """Conventional PyTorch LSTM sequence model baseline.

    Implements the same ISequenceModel contract as CfCModel/LTCModel for fair evaluation comparison.
    """

    def __init__(
        self,
        input_size: int = 10000,
        hidden_size: int = 64,
        num_layers: int = 1,
        model_version: str | None = None,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_version = model_version or f"lstm_v1_h{self.hidden_size}"

        # Standard PyTorch LSTM module
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

    def forward(
        self, sequence: list[EncodedHypervector], entity_id: str = "default"
    ) -> list[TrajectoryState]:
        """Batch mode: process a full ordered sequence of hypervectors for an entity.

        Args:
            sequence: Ordered list of EncodedHypervector contract objects.
            entity_id: Identifier string for the sequence entity.

        Returns:
            List of TrajectoryState contract instances corresponding to each sequence step.
        """
        if not sequence:
            return []

        step_tensors = [load_encoded_vector(hv).squeeze(0) for hv in sequence]
        x_seq = torch.stack(step_tensors, dim=1)  # Shape (1, N, input_size)

        # Initial hidden state and cell state are zeros
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size)

        out_seq, _ = self.lstm(x_seq, (h0, c0))  # out_seq: (1, N, hidden_size)

        results: list[TrajectoryState] = []
        # Run step-by-step state recording to preserve exact (h_n, c_n) for each step
        hn = h0
        cn = c0
        for i, hv in enumerate(sequence):
            x_step = x_seq[:, i : i + 1, :]
            _, (hn, cn) = self.lstm(x_step, (hn, cn))

            h_output = out_seq[:, i, :]  # Shape (1, hidden_size)

            ref_path = save_hidden_state(
                h_output,
                encoder_version=hv.encoder_version,
                entity_id=entity_id,
                flow_id=hv.flow_id,
            )

            # Persist LSTM (h_n, c_n) pair alongside main trajectory file
            cell_path = Path(ref_path).parent / f"{hv.flow_id}_lstm_cell.npy"
            lstm_state_arr = torch.stack([hn, cn]).detach().cpu().numpy()
            np.save(cell_path, lstm_state_arr)

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
        """Streaming mode: process a single record and return an updated TrajectoryState.

        Args:
            hv: Single EncodedHypervector contract object.
            prev_state: Optional previous TrajectoryState carrying hidden state reference.
            entity_id: Optional entity ID string.

        Returns:
            Updated TrajectoryState contract instance.
        """
        eid = entity_id or (prev_state.entity_id if prev_state else "default")

        if prev_state is not None:
            cell_path = Path(prev_state.hidden_state_ref).parent / f"{prev_state.flow_id}_lstm_cell.npy"
            if cell_path.exists():
                state_arr = np.load(cell_path)
                state_tensor = torch.from_numpy(state_arr).float()
                hn, cn = state_tensor[0], state_tensor[1]
            else:
                hn = torch.zeros(self.num_layers, 1, self.hidden_size)
                cn = torch.zeros(self.num_layers, 1, self.hidden_size)
        else:
            hn = torch.zeros(self.num_layers, 1, self.hidden_size)
            cn = torch.zeros(self.num_layers, 1, self.hidden_size)

        x_step = load_encoded_vector(hv)  # Shape (1, 1, input_size)

        out_step, (hn_next, cn_next) = self.lstm(x_step, (hn, cn))
        h_output = out_step[:, 0, :]  # Shape (1, hidden_size)

        ref_path = save_hidden_state(
            h_output,
            encoder_version=hv.encoder_version,
            entity_id=eid,
            flow_id=hv.flow_id,
        )

        cell_path = Path(ref_path).parent / f"{hv.flow_id}_lstm_cell.npy"
        lstm_state_arr = torch.stack([hn_next, cn_next]).detach().cpu().numpy()
        np.save(cell_path, lstm_state_arr)

        return TrajectoryState(
            entity_id=eid,
            flow_id=hv.flow_id,
            t=datetime.now(timezone.utc),
            hidden_state_ref=ref_path,
            model_version=self.model_version,
        )
