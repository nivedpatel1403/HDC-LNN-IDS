"""LTC (Liquid Time-Constant) sequence model implementation (Ablation alternative per ADR-002)."""

from datetime import datetime, timezone
import torch
import ncps.torch

from hdlnn.contracts.encoded import EncodedHypervector
from hdlnn.contracts.trajectory import TrajectoryState
from hdlnn.lnn.interfaces import ISequenceModel
from hdlnn.lnn.utils import load_encoded_vector, load_hidden_state, save_hidden_state


class LTCModel(ISequenceModel):
    """Liquid Time-Constant (LTC) sequence model used as an ablation alternative (ADR-002)."""

    def __init__(
        self,
        input_size: int = 10000,
        hidden_size: int = 64,
        model_version: str | None = None,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.model_version = model_version or f"ltc_v1_h{self.hidden_size}"

        # Initialize ncps LTC module
        self.ltc = ncps.torch.LTC(
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
            entity_id: Identifier string for the sequence entity.

        Returns:
            List of TrajectoryState objects corresponding to each record step.
        """
        if not sequence:
            return []

        step_tensors = [load_encoded_vector(hv).squeeze(0) for hv in sequence]
        x_seq = torch.stack(step_tensors, dim=1)

        hx_initial = torch.zeros(1, self.hidden_size)
        out_seq, _ = self.ltc(x_seq, hx=hx_initial)

        results: list[TrajectoryState] = []
        for i, hv in enumerate(sequence):
            h_step = out_seq[:, i, :]

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

        if prev_state is not None:
            hx = load_hidden_state(prev_state.hidden_state_ref)
            while hx.dim() > 2:
                hx = hx.squeeze(0)
            if hx.dim() == 1:
                hx = hx.unsqueeze(0)
        else:
            hx = torch.zeros(1, self.hidden_size)

        x_step = load_encoded_vector(hv)
        out_step, next_hx = self.ltc(x_step, hx=hx)

        ref_path = save_hidden_state(
            out_step,
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
