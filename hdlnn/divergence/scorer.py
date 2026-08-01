"""Divergence scorer for anomaly detection using calibrated Mahalanobis drift thresholds."""

from pathlib import Path
import numpy as np
import torch

from hdlnn.contracts.decision import AnomalyDecision
from hdlnn.contracts.errors import ManifoldFitError
from hdlnn.contracts.trajectory import TrajectoryState
from hdlnn.divergence.interfaces import IDivergenceScorer
from hdlnn.divergence.manifold import ReferenceManifold


class DivergenceScorer(IDivergenceScorer):
    """Divergence scorer calculating Mahalanobis drift against a reference manifold.

    Calibrates decision threshold to a target False Positive Rate (FPR) on normal trajectories.
    """

    def __init__(
        self,
        target_fpr: float = 0.05,
        manifold_version: str = "manifold_v1_normal_val_only",
        sample_ratio_threshold: float = 5.0,
    ) -> None:
        self.target_fpr = target_fpr
        self.manifold_version = manifold_version
        self.sample_ratio_threshold = sample_ratio_threshold

        self.manifold = ReferenceManifold(
            version=self.manifold_version,
            sample_ratio_threshold=self.sample_ratio_threshold,
        )

        self.threshold: float = 0.0
        self.is_fitted = False

    def fit(self, normal_trajectories: list[TrajectoryState], labels: list[str]) -> None:
        """Fit reference manifold and calibrate decision threshold to target FPR.

        Args:
            normal_trajectories: List of normal TrajectoryState objects.
            labels: List of ground-truth labels (must all be 'normal').

        Raises:
            ManifoldFitError: If fitting fails or any label is non-normal.
        """
        # 1. Fit ReferenceManifold (enforces normal-only rule)
        self.manifold.fit(normal_trajectories, labels)

        # 2. Score normal training trajectories to calibrate threshold
        drift_scores: list[float] = []
        for traj in normal_trajectories:
            arr = np.load(traj.hidden_state_ref)
            hidden_tensor = torch.from_numpy(arr).float()
            score_val = self.manifold.mahalanobis_distance(hidden_tensor)
            drift_scores.append(score_val)

        # 3. Calibrate threshold as (1 - target_fpr) percentile of normal scores (ADR-003)
        pct = (1.0 - self.target_fpr) * 100.0
        self.threshold = float(np.percentile(drift_scores, pct))
        self.is_fitted = True

    def score(
        self,
        trajectory: TrajectoryState,
        ground_truth_label: str | None = None,
        attack_category: str | None = None,
    ) -> AnomalyDecision:
        """Score a single TrajectoryState and return an AnomalyDecision.

        Args:
            trajectory: TrajectoryState object.
            ground_truth_label: Optional label string.
            attack_category: Optional attack category string.

        Returns:
            Populated AnomalyDecision contract object.

        Raises:
            ManifoldFitError: If score is called before fit.
        """
        if not self.is_fitted:
            raise ManifoldFitError("DivergenceScorer must be fit before calling score().")

        path = Path(trajectory.hidden_state_ref)
        if not path.exists():
            raise ManifoldFitError(f"Trajectory hidden state file not found at '{path}'")

        arr = np.load(path)
        hidden_tensor = torch.from_numpy(arr).float()

        drift_score = self.manifold.mahalanobis_distance(hidden_tensor)
        is_anomalous = bool(drift_score > self.threshold)

        return AnomalyDecision(
            entity_id=trajectory.entity_id,
            flow_id=trajectory.flow_id,
            t=trajectory.t,
            drift_score=drift_score,
            threshold=self.threshold,
            is_anomalous=is_anomalous,
            manifold_version=self.manifold.version,
            ground_truth_label=ground_truth_label or "unknown",
            attack_category=attack_category,
        )
