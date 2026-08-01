"""Item memory module for managing HDC base and OOV hypervectors."""

from pathlib import Path
from typing import Any
import torch
import torchhd

from hdlnn.contracts.errors import ItemMemoryError
from hdlnn.contracts.flow import CanonicalFlow


class ItemMemory:
    """Stores and manages base hypervectors for categorical feature values.

    Enforces P4 data isolation: fitting is strictly prohibited on non-train splits.
    """

    def __init__(
        self,
        dim: int = 10000,
        representation: str = "MAP",
        encoder_version: str = "hdc_v1",
    ) -> None:
        self.dim = dim
        self.representation = representation.upper()
        self.encoder_version = encoder_version
        self.memory: dict[tuple[str, str], torch.Tensor] = {}
        self.oov_memory: dict[str, torch.Tensor] = {}
        self.is_fitted = False

    def fit(self, train_flows: list[CanonicalFlow]) -> None:
        """Fit ItemMemory by building base hypervectors for categorical features in train split.

        Args:
            train_flows: List of CanonicalFlow objects.

        Raises:
            ItemMemoryError: If any flow in train_flows has split != 'train'.
        """
        for flow in train_flows:
            if flow.split != "train":
                raise ItemMemoryError(
                    f"ItemMemory.fit() requires all flows to have split='train'. "
                    f"Found flow_id '{flow.flow_id}' with split='{flow.split}' (P4 violation)."
                )

        # Gather all unique categorical (feature_name, value) pairs and feature names
        feature_values: dict[str, set[str]] = {}
        for flow in train_flows:
            for feat_name, val in flow.categorical_features.items():
                feature_values.setdefault(feat_name, set()).add(str(val))

        # Generate base hypervectors using torchhd random MAP hypervectors
        for feat_name, values in feature_values.items():
            # Reserve unique OOV hypervector per feature
            if feat_name not in self.oov_memory:
                self.oov_memory[feat_name] = torchhd.random(1, self.dim, vsa="MAP")[0]

            for val in values:
                key = (feat_name, val)
                if key not in self.memory:
                    self.memory[key] = torchhd.random(1, self.dim, vsa="MAP")[0]

        self.is_fitted = True

    def lookup(self, feature_name: str, value: str) -> tuple[torch.Tensor, bool]:
        """Lookup base hypervector for a given feature and value.

        Args:
            feature_name: Categorical feature name.
            value: Feature value string.

        Returns:
            Tuple of (hypervector_tensor, is_oov_bool).
        """
        key = (feature_name, str(value))
        if key in self.memory:
            return self.memory[key], False

        # OOV hit: return feature-specific OOV hypervector if available, else generate fallback
        if feature_name in self.oov_memory:
            return self.oov_memory[feature_name], True

        # Handle unseen feature name gracefully with on-the-fly reserved OOV vector
        new_oov = torchhd.random(1, self.dim, vsa="MAP")[0]
        self.oov_memory[feature_name] = new_oov
        return new_oov, True

    def save(self, path: str) -> None:
        """Serialize item memory state to file path."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "dim": self.dim,
            "representation": self.representation,
            "encoder_version": self.encoder_version,
            "memory": self.memory,
            "oov_memory": self.oov_memory,
            "is_fitted": self.is_fitted,
        }
        torch.save(payload, target_path)

    @classmethod
    def load(cls, path: str) -> "ItemMemory":
        """Deserialize item memory state from file path.

        Raises:
            ItemMemoryError: If file is missing, corrupted, or invalid.
        """
        target_path = Path(path)
        if not target_path.exists():
            raise ItemMemoryError(f"Item memory file not found: '{path}'")

        try:
            payload = torch.load(target_path, weights_only=False)
            if not isinstance(payload, dict) or "memory" not in payload:
                raise ItemMemoryError(f"Corrupted or invalid item memory file at '{path}'")

            item_mem = cls(
                dim=payload.get("dim", 10000),
                representation=payload.get("representation", "MAP"),
                encoder_version=payload.get("encoder_version", "hdc_v1"),
            )
            item_mem.memory = payload["memory"]
            item_mem.oov_memory = payload.get("oov_memory", {})
            item_mem.is_fitted = payload.get("is_fitted", True)
            return item_mem
        except Exception as e:
            if isinstance(e, ItemMemoryError):
                raise
            raise ItemMemoryError(f"Failed to load item memory from '{path}': {e}") from e
