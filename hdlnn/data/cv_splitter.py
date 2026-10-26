"""Entity-Disjoint K-Fold Cross-Validation Splitter for HDC-LNN Benchmark.

Ensures rigorous, leakage-free evaluation by partitioning flows across disjoint network entities.
For k folds, rotates through fold assignments such that:
  - Fold i: Test set (never leaked into training)
  - Fold (i + 1) % k: Validation set (used exclusively for reference manifold fitting & threshold calibration)
  - Remaining (k - 2) folds: Training set

Preserves chronological intra-entity sequencing and irregular delta-t computation.
"""

import copy
import random
import logging
from typing import List, Dict, Tuple, Generator, Optional
from hdlnn.contracts.schemas import CanonicalFlow

logger = logging.getLogger(__name__)


def prepare_entity_flows(flows: List[CanonicalFlow]) -> Dict[str, List[CanonicalFlow]]:
    """Groups flows by entity_id, sorts chronologically, and calculates intra-entity delta-t."""
    entity_to_flows: Dict[str, List[CanonicalFlow]] = {}
    for flow in flows:
        entity_to_flows.setdefault(flow.entity_id, []).append(copy.copy(flow))

    for entity_id, entity_flows in entity_to_flows.items():
        entity_flows.sort(key=lambda f: f.timestamp)
        prev_time = None
        for f in entity_flows:
            if prev_time is None:
                f.dt = 0.0
            else:
                f.dt = max(f.timestamp - prev_time, 0.0)
            prev_time = f.timestamp

    return entity_to_flows


def kfold_entity_splits(
    flows: List[CanonicalFlow],
    k: int = 5,
    seed: int = 42,
    threat_entities: Optional[set] = None,
) -> Generator[Tuple[int, List[CanonicalFlow], List[CanonicalFlow], List[CanonicalFlow]], None, None]:
    """Generates k entity-disjoint cross-validation splits.

    Args:
        flows: Full dataset flows.
        k: Number of cross-validation folds (recommended: 5).
        seed: Random seed for deterministic entity assignment.
        threat_entities: Set of threat injection entity IDs that should strictly remain in test.

    Yields:
        Tuple of (fold_idx, train_flows, val_flows, test_flows).
    """
    if k < 2:
        raise ValueError("k must be at least 2 for cross-validation")

    if threat_entities is None:
        threat_entities = {"10.0.0.901", "10.0.0.902"}

    entity_to_flows = prepare_entity_flows(flows)
    unique_entities = sorted(list(entity_to_flows.keys()))
    regular_entities = [e for e in unique_entities if e not in threat_entities]

    rng = random.Random(seed)
    rng.shuffle(regular_entities)

    num_entities = len(regular_entities)
    if num_entities < k:
        logger.warning(
            f"Fewer entities ({num_entities}) than requested folds ({k}). Capping k to {max(2, num_entities)}."
        )
        k = max(2, num_entities)

    # Partition regular entities into k contiguous buckets
    folds: List[List[str]] = [[] for _ in range(k)]
    for idx, entity in enumerate(regular_entities):
        folds[idx % k].append(entity)

    logger.info(
        f"Initialized {k}-Fold Entity-Disjoint Cross-Validation over {num_entities} regular entities "
        f"+ {len(threat_entities.intersection(unique_entities))} threat entities."
    )

    for fold_idx in range(k):
        test_entity_set = set(folds[fold_idx])
        # Always add threat entities to test set
        for te in threat_entities:
            if te in entity_to_flows:
                test_entity_set.add(te)

        val_fold_idx = (fold_idx + 1) % k
        val_entity_set = set(folds[val_fold_idx])

        train_entity_set = set()
        for other_idx in range(k):
            if other_idx != fold_idx and other_idx != val_fold_idx:
                train_entity_set.update(folds[other_idx])

        # If k=2, train and val would overlap if not handled:
        if k == 2:
            train_entity_set = set(folds[val_fold_idx])
            val_entity_set = set(folds[val_fold_idx])

        train_flows: List[CanonicalFlow] = []
        val_flows: List[CanonicalFlow] = []
        test_flows: List[CanonicalFlow] = []

        for entity_id, e_flows in entity_to_flows.items():
            if entity_id in train_entity_set:
                for f in e_flows:
                    f_copy = copy.copy(f)
                    f_copy.split = "train"
                    train_flows.append(f_copy)

            if entity_id in val_entity_set:
                for f in e_flows:
                    f_copy = copy.copy(f)
                    f_copy.split = "val"
                    val_flows.append(f_copy)

            if entity_id in test_entity_set:
                for f in e_flows:
                    f_copy = copy.copy(f)
                    f_copy.split = "test"
                    test_flows.append(f_copy)

        # Sort chronologically
        train_flows.sort(key=lambda f: f.timestamp)
        val_flows.sort(key=lambda f: f.timestamp)
        test_flows.sort(key=lambda f: f.timestamp)

        logger.info(
            f"[Fold {fold_idx + 1}/{k}] "
            f"Train={len(train_flows)} flows ({len(train_entity_set)} entities), "
            f"Val={len(val_flows)} flows ({len(val_entity_set)} entities), "
            f"Test={len(test_flows)} flows ({len(test_entity_set)} entities)"
        )

        yield fold_idx, train_flows, val_flows, test_flows
