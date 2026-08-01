"""Dataset splitter for deterministic train/val/test flow partitioning and split persistence."""

import json
import math
from pathlib import Path
import random

from hdlnn.contracts.errors import DataValidationError
from hdlnn.contracts.flow import CanonicalFlow


class Splitter:
    """Handles deterministic split assignment and split file persistence."""

    def split(
        self,
        flows: list[CanonicalFlow],
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
    ) -> tuple[list[CanonicalFlow], list[CanonicalFlow], list[CanonicalFlow]]:
        """Partition flows into deterministic train, val, and test splits given a seed.

        Group flows by flow_id so that duplicate flow_ids are assigned to the same split.

        Args:
            flows: List of CanonicalFlow models to partition.
            train_frac: Fraction for training split (default 0.7).
            val_frac: Fraction for validation split (default 0.15).
            test_frac: Fraction for test split (default 0.15).
            seed: Random seed for deterministic shuffling.

        Returns:
            Tuple of (train_flows, val_flows, test_flows).

        Raises:
            DataValidationError: If split fractions do not sum to 1.0.
        """
        total_frac = train_frac + val_frac + test_frac
        if not math.isclose(total_frac, 1.0, abs_tol=1e-5):
            raise DataValidationError(f"Split fractions must sum to 1.0, got {total_frac}")

        # Group flows by flow_id to prevent data leakage across splits
        flow_groups: dict[str, list[CanonicalFlow]] = {}
        for f in flows:
            flow_groups.setdefault(f.flow_id, []).append(f)

        # Sort unique IDs for cross-platform deterministic order before shuffling
        unique_ids = sorted(flow_groups.keys())
        rng = random.Random(seed)
        rng.shuffle(unique_ids)

        total_unique = len(unique_ids)
        n_train = int(total_unique * train_frac)
        n_val = int(total_unique * val_frac)

        train_ids_set = set(unique_ids[:n_train])
        val_ids_set = set(unique_ids[n_train : n_train + n_val])

        train_set: list[CanonicalFlow] = []
        val_set: list[CanonicalFlow] = []
        test_set: list[CanonicalFlow] = []

        for f in flows:
            if f.flow_id in train_ids_set:
                f.split = "train"
                train_set.append(f)
            elif f.flow_id in val_ids_set:
                f.split = "val"
                val_set.append(f)
            else:
                f.split = "test"
                test_set.append(f)

        return train_set, val_set, test_set

    def save_split(
        self,
        flows: list[CanonicalFlow],
        dataset_name: str,
        version: str,
        out_dir: str = "data/splits",
    ) -> str:
        """Persist flat {flow_id: split} mapping to JSON split file.

        Args:
            flows: List of CanonicalFlow objects with assigned .split fields.
            dataset_name: Dataset identifier string (e.g. 'unsw').
            version: Version identifier string (e.g. 'v1').
            out_dir: Target directory path for saved split JSON.

        Returns:
            File path string of saved split JSON file.

        Raises:
            FileExistsError: If split version file already exists.
        """
        target_dir = Path(out_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{dataset_name}_{version}.json"

        if file_path.exists():
            raise FileExistsError(
                f"Split version file already exists at '{file_path}'. Overwriting is prohibited."
            )

        mapping = {f.flow_id: f.split for f in flows if f.split is not None}
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)

        return str(file_path)

    def load_split(
        self,
        dataset_name: str,
        version: str,
        splits_dir: str = "data/splits",
    ) -> dict[str, str]:
        """Load flat {flow_id: split} mapping from JSON split file.

        Args:
            dataset_name: Dataset identifier string.
            version: Version identifier string.
            splits_dir: Directory containing split JSON files.

        Returns:
            Dictionary mapping flow_id to split string ("train"/"val"/"test").

        Raises:
            FileNotFoundError: If split JSON file does not exist.
        """
        file_path = Path(splits_dir) / f"{dataset_name}_{version}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Split file not found at '{file_path}'")

        with open(file_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        return mapping
