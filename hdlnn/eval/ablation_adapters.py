"""Ablation adapter modules for HDC-only and HDC-off pipeline modes."""

from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

from hdlnn.contracts.encoded import EncodedHypervector
from hdlnn.contracts.flow import CanonicalFlow
from hdlnn.contracts.trajectory import TrajectoryState
from hdlnn.hdc.interfaces import IEncoder
from hdlnn.lnn.interfaces import ISequenceModel
from hdlnn.lnn.utils import load_encoded_vector, save_hidden_state


class HDCOffEncoder(IEncoder):
    """HDC-off ablation encoder bypassing Hyperdimensional Computing.

    Converts CanonicalFlow features directly into normalized dense feature vectors.
    """

    def __init__(self, dim: int = 10000, encoder_version: str = "hdc_off") -> None:
        self.dim = dim
        self.encoder_version = encoder_version
        self.is_fitted = False
        self.numeric_cols: list[str] = []

    def fit(self, train_flows: list[CanonicalFlow]) -> None:
        """Fit feature index mapping for HDC-off ablation."""
        cols: set[str] = set()
        for f in train_flows:
            cols.update(f.numeric_features.keys())
            cols.update(f.categorical_features.keys())
        self.numeric_cols = sorted(cols)
        self.is_fitted = True

    def encode(self, flow: CanonicalFlow) -> EncodedHypervector:
        """Convert flow into dense feature vector stored as .npy."""
        vec = np.zeros(self.dim, dtype=np.float32)

        idx = 0
        for col in self.numeric_cols:
            if idx >= self.dim:
                break
            if col in flow.numeric_features:
                vec[idx] = float(flow.numeric_features[col])
            elif col in flow.categorical_features:
                # Minimal hash integer representation for categorical features
                val = flow.categorical_features[col]
                vec[idx] = float(abs(hash(val)) % 1000)
            idx += 1

        out_dir = Path("data/encoded") / self.encoder_version
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{flow.flow_id}.npy"
        np.save(out_path, vec)

        return EncodedHypervector(
            flow_id=flow.flow_id,
            encoder_version=self.encoder_version,
            dim=self.dim,
            representation="MAP",
            vector_ref=str(out_path),
            oov_fields=[],
        )


class PassthroughSequenceModel(ISequenceModel):
    """LNN-off ablation model passing raw hypervectors directly to trajectory states."""

    def __init__(self, model_version: str = "hdc_only_passthrough") -> None:
        self.model_version = model_version

    def forward(
        self, sequence: list[EncodedHypervector], entity_id: str = "default"
    ) -> list[TrajectoryState]:
        """Convert each EncodedHypervector directly into a TrajectoryState without LNN accumulation."""
        results: list[TrajectoryState] = []
        for hv in sequence:
            t_val = load_encoded_vector(hv).squeeze(0)  # (1, dim) or (dim,)
            ref_path = save_hidden_state(
                t_val,
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
        """Process single record in passthrough mode."""
        eid = entity_id or (prev_state.entity_id if prev_state else "default")
        t_val = load_encoded_vector(hv).squeeze(0)

        ref_path = save_hidden_state(
            t_val,
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
