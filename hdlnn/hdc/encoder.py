"""Hyperdimensional Computing (HDC) encoder implementing feature binding and bundling."""

from pathlib import Path
import numpy as np
import torch
import torchhd

from hdlnn.contracts.encoded import EncodedHypervector
from hdlnn.contracts.errors import ConfigError, ItemMemoryError
from hdlnn.contracts.flow import CanonicalFlow
from hdlnn.hdc.interfaces import IEncoder
from hdlnn.hdc.item_memory import ItemMemory


class HDCEncoder(IEncoder):
    """HDC Encoder binding categorical/numeric features and bundling into hypervectors.

    Rejects binary representation per ADR-001 (MAP representation supported).
    Saves encoded hypervectors to disk as .npy files per ADR-005.
    """

    def __init__(
        self,
        dim: int = 10000,
        representation: str = "MAP",
        num_levels: int = 100,
        encoder_version: str | None = None,
    ) -> None:
        self.dim = dim
        self.representation = representation

        # Binary HDC is explicitly rejected by ADR-001
        if representation.lower() == "binary":
            raise ConfigError("Binary representation is explicitly rejected per ADR-001.")

        if representation.upper() not in ("MAP", "BIPOLAR"):
            raise ConfigError(
                f"Unsupported HDC representation '{representation}'. Only 'MAP' and 'bipolar' are supported."
            )

        self.num_levels = num_levels
        self.encoder_version = (
            encoder_version or f"hdc_v1_D{self.dim}_{self.representation.lower()}"
        )

        self.item_memory = ItemMemory(
            dim=self.dim,
            representation=self.representation,
            encoder_version=self.encoder_version,
        )

        self.role_vectors: dict[str, torch.Tensor] = {}
        self.level_vectors: dict[str, torch.Tensor] = {}
        self.numeric_ranges: dict[str, tuple[float, float]] = {}
        self.is_fitted = False

    def fit(self, train_flows: list[CanonicalFlow]) -> None:
        """Fit item memory and numeric level hypervectors using train split flows only.

        Args:
            train_flows: List of CanonicalFlow objects from the training split.

        Raises:
            ItemMemoryError: If any flow has split != 'train'.
        """
        for flow in train_flows:
            if flow.split != "train":
                raise ItemMemoryError(
                    f"HDCEncoder.fit() requires all flows to have split='train'. "
                    f"Found flow '{flow.flow_id}' with split='{flow.split}' (P4 violation)."
                )

        # 1. Fit ItemMemory for categorical features
        self.item_memory.fit(train_flows)

        # 2. Gather all feature names for role vector generation
        all_feature_names: set[str] = set()
        numeric_values: dict[str, list[float]] = {}

        for flow in train_flows:
            for cat_name in flow.categorical_features.keys():
                all_feature_names.add(cat_name)

            for num_name, val in flow.numeric_features.items():
                all_feature_names.add(num_name)
                numeric_values.setdefault(num_name, []).append(val)

        # Generate role hypervectors for all features (deterministic via fixed seed if needed)
        for feat_name in all_feature_names:
            if feat_name not in self.role_vectors:
                self.role_vectors[feat_name] = torchhd.random(1, self.dim, vsa="MAP")[0]

        # 3. Generate level hypervectors for numeric features
        for num_name, vals in numeric_values.items():
            if vals:
                min_v = float(min(vals))
                max_v = float(max(vals))
            else:
                min_v, max_v = 0.0, 1.0

            self.numeric_ranges[num_name] = (min_v, max_v)
            self.level_vectors[num_name] = torchhd.level(self.num_levels, self.dim, vsa="MAP")

        self.is_fitted = True

    def encode(self, flow: CanonicalFlow) -> EncodedHypervector:
        """Encode a single CanonicalFlow into an EncodedHypervector contract.

        Stateless per record: all state comes from fitted ItemMemory and level hypervectors.

        Args:
            flow: CanonicalFlow object.

        Returns:
            EncodedHypervector contract model.
        """
        if not self.is_fitted:
            raise ItemMemoryError("HDCEncoder must be fit before calling encode().")

        bound_vectors: list[torch.Tensor] = []
        oov_fields: list[str] = []

        # 1. Process Categorical Features
        for feat_name, val in flow.categorical_features.items():
            base_hv, is_oov = self.item_memory.lookup(feat_name, val)
            if is_oov:
                oov_fields.append(feat_name)

            role_hv = self.role_vectors.get(
                feat_name, torchhd.random(1, self.dim, vsa="MAP")[0]
            )
            bound = torchhd.bind(base_hv, role_hv)
            bound_vectors.append(bound)

        # 2. Process Numeric Features (Clip min/max, Quantize to level hypervectors)
        for feat_name, val in flow.numeric_features.items():
            if feat_name in self.numeric_ranges:
                min_v, max_v = self.numeric_ranges[feat_name]
            else:
                min_v, max_v = 0.0, 1.0
                self.numeric_ranges[feat_name] = (min_v, max_v)
                self.level_vectors[feat_name] = torchhd.level(self.num_levels, self.dim, vsa="MAP")

            # Out-of-range numeric values clip to min/max bin edges (not OOV)
            clipped_val = max(min_v, min(max_v, float(val)))

            if math_isclose(max_v, min_v):
                bin_idx = 0
            else:
                bin_idx = int((clipped_val - min_v) / (max_v - min_v) * (self.num_levels - 1))
                bin_idx = max(0, min(self.num_levels - 1, bin_idx))

            level_hv = self.level_vectors[feat_name][bin_idx]
            role_hv = self.role_vectors.get(
                feat_name, torchhd.random(1, self.dim, vsa="MAP")[0]
            )
            bound = torchhd.bind(level_hv, role_hv)
            bound_vectors.append(bound)

        # 3. Bundle (Multibundle) all per-feature hypervectors
        if bound_vectors:
            stacked = torch.stack(bound_vectors)
            bundled_hv = torchhd.multibundle(stacked)
        else:
            bundled_hv = torch.zeros(self.dim)

        # 4. Save numpy array to disk per ADR-005 (.npy format)
        out_dir = Path("data/encoded") / self.encoder_version
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{flow.flow_id}.npy"

        np_arr = bundled_hv.detach().cpu().numpy()
        np.save(out_path, np_arr)

        representation_str = "MAP" if self.representation.upper() == "MAP" else "bipolar"

        return EncodedHypervector(
            flow_id=flow.flow_id,
            encoder_version=self.encoder_version,
            dim=self.dim,
            representation=representation_str,
            vector_ref=str(out_path),
            oov_fields=oov_fields,
        )


def math_isclose(a: float, b: float, rel_tol: float = 1e-9) -> bool:
    """Helper for floating point equality check."""
    return abs(a - b) <= rel_tol
