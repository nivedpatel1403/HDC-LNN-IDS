"""Abstract base interface for divergence scorers."""

from abc import ABC, abstractmethod

from hdlnn.contracts.decision import AnomalyDecision
from hdlnn.contracts.trajectory import TrajectoryState


class IDivergenceScorer(ABC):
    """Abstract interface for manifold divergence scoring and anomaly decisions."""

    @abstractmethod
    def fit(self, normal_trajectories: list[TrajectoryState], labels: list[str]) -> None:
        """Fit reference manifold and calibrate decision threshold on normal-only trajectories.

        Args:
            normal_trajectories: List of TrajectoryState objects.
            labels: List of ground-truth labels corresponding 1-to-1 with normal_trajectories.
        """
        pass

    @abstractmethod
    def score(
        self,
        trajectory: TrajectoryState,
        ground_truth_label: str | None = None,
        attack_category: str | None = None,
    ) -> AnomalyDecision:
        """Score a trajectory state and produce an AnomalyDecision.

        Args:
            trajectory: TrajectoryState object to evaluate.
            ground_truth_label: Optional label string ("normal"/"attack" or "unknown").
            attack_category: Optional attack category string if anomalous.

        Returns:
            Populated AnomalyDecision contract model instance.
        """
        pass
