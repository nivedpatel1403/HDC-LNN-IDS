"""CfC (Closed-form Continuous-time) sequence model implementation."""

from datetime import datetime, timezone
import torch
import ncps.torch

from hdlnn.contracts.encoded import EncodedHypervector
from hdlnn.contracts.trajectory import TrajectoryState
from hdlnn.lnn.interfaces import ISequenceModel
from hdlnn.lnn.utils import load_encoded_vector, load_hidden_state, save_hidden_state


class CfCModel(ISequenceModel):
    """Closed-form Continuous-time (CfC) LNN sequence model (ADR-002).

    Maintains continuous-time trajectory states across records per entity.
    """

    def __init__(
        self,
        input_size: int = 10000,
        hidden_size: int = 64,
        model_version: str | None = None,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.model_version = model_version or f"cfc_v1_h{self.hidden_size}"

        # Initialize ncps CfC module
        self.cfc = ncps.torch.CfC(
            input_size=self.input_size,
            units=self.hidden_size,
            return_sequences=True,
        )

    def forward(
        self, sequence: list[EncodedHypervector], entity_id: str = "default"
    ) -> list[TrajectoryState]:
        """Batch mode: process a full ordered sequence of hypervectors for an entity.

        Args:
            sequence: Ordered list of EncodedHypervector contract objects.
            entity_id: Identifier string for the entity (e.g. source IP).

        Returns:
            List of TrajectoryState objects corresponding to each record step.
        """
        if not sequence:
            return []

        # Load and stack all input hypervectors into a batch tensor of shape (1, N, input_size)
        step_tensors = [load_encoded_vector(hv).squeeze(0) for hv in sequence]  # Each (1, input_size)
        x_seq = torch.stack(step_tensors, dim=1)  # Shape (1, N, input_size)

        # Initial hidden state is zero
        hx_initial = torch.zeros(1, self.hidden_size)

        # Forward pass through CfC sequence model
        out_seq, _ = self.cfc(x_seq, hx=hx_initial)  # out_seq: (1, N, hidden_size)

        results: list[TrajectoryState] = []
        for i, hv in enumerate(sequence):
            # Extract step hidden state tensor (1, hidden_size)
            h_step = out_seq[:, i, :]

            # Save hidden state to disk
            ref_path = save_hidden_state(
                h_step,
                encoder_version=hv.encoder_version,
                entity_id=entity_id,
                flow_id=hv.flow_id,
            )

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
            prev_state: Optional previous TrajectoryState containing hidden state path.
            entity_id: Optional entity ID string.

        Returns:
            Updated TrajectoryState contract instance.
        """
        eid = entity_id or (prev_state.entity_id if prev_state else "default")

        # Load previous hidden state or initialize with zeros
        if prev_state is not None:
            hx = load_hidden_state(prev_state.hidden_state_ref)
            if hx.dim() == 1:
                hx = hx.unsqueeze(0)
        else:
            hx = torch.zeros(1, self.hidden_size)

        # Load input hypervector tensor (1, 1, input_size)
        x_step = load_encoded_vector(hv)

        # Compute next hidden state
        _, next_hx = self.cfc(x_step, hx=hx)  # next_hx: (1, hidden_size)

        # Save updated hidden state to disk
        ref_path = save_hidden_state(
            next_hx,
            encoder_version=hv.encoder_version,
            entity_id=eid,
            flow_id=hv.flow_id,
        )

        return TrajectoryState(
            entity_id=eid,
            flow_id=hv.flow_id,
            t=datetime.now(timezone.utc),
            hidden_state_ref=ref_path,
            model_version=self.model_version,
        )
