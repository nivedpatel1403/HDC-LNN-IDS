"""Reference manifold module for normal hidden state distribution modeling."""

from pathlib import Path

import numpy as np
import torch

from hdlnn.contracts.errors import ManifoldFitError
from hdlnn.contracts.trajectory import TrajectoryState


class ReferenceManifold:
    """Models the normal trajectory state distribution manifold and computes Mahalanobis distance.

    Enforces P4 data isolation: fitting on non-normal labeled trajectories raises ManifoldFitError.
    """

    def __init__(
        self,
        version: str = "manifold_v1_normal_val_only",
        sample_ratio_threshold: float = 5.0,
    ) -> None:
        self.version = version
        self.sample_ratio_threshold = sample_ratio_threshold
        self.mean: torch.Tensor | None = None
        self.inv_cov: torch.Tensor | None = None
        self.variances: torch.Tensor | None = None
        self.covariance_mode: str = "unfitted"
        self.hidden_size: int = 0
        self.is_fitted = False

    def fit(self, trajectories: list[TrajectoryState], labels: list[str]) -> None:
        """Fit mean and covariance parameters on normal-only trajectory states.

        Args:
            trajectories: List of TrajectoryState contract instances.
            labels: List of ground-truth label strings matching trajectories order.

        Raises:
            ManifoldFitError: If trajectories is empty, lengths mismatch, or any label is not 'normal'.
        """
        if not trajectories:
            raise ManifoldFitError("Cannot fit ReferenceManifold on an empty list of trajectories.")

        if len(trajectories) != len(labels):
            raise ManifoldFitError(
                f"Trajectory count ({len(trajectories)}) does not match label count ({len(labels)})."
            )

        # Enforce strict normal-only leakage prevention rule
        for i, lbl in enumerate(labels):
            if str(lbl).strip().lower() != "normal":
                raise ManifoldFitError(
                    f"Manifold leakage violation: ReferenceManifold fit received non-normal label '{lbl}' "
                    f"at index {i} for flow '{trajectories[i].flow_id}'."
                )

        # Load hidden state tensors from disk
        hidden_states_list: list[torch.Tensor] = []
        for traj in trajectories:
            path = Path(traj.hidden_state_ref)
            if not path.exists():
                raise ManifoldFitError(f"Trajectory hidden state file not found at '{path}'")

            arr = np.load(path)
            t_val = torch.from_numpy(arr).float().squeeze()
            if t_val.dim() == 0:
                t_val = t_val.unsqueeze(0)
            hidden_states_list.append(t_val)

        states_matrix = torch.stack(hidden_states_list, dim=0)  # Shape (N, hidden_size)
        n_samples, self.hidden_size = states_matrix.shape

        self.mean = torch.mean(states_matrix, dim=0)
        min_samples_full = int(self.sample_ratio_threshold * self.hidden_size)

        # Select covariance mode based on sample size threshold (ADR-003)
        if n_samples < min_samples_full or n_samples < 2 or self.hidden_size > 500:
            self.covariance_mode = "diagonal"
            if n_samples >= 2:
                self.variances = torch.var(states_matrix, dim=0, unbiased=True) + 1e-6
            else:
                self.variances = torch.ones(self.hidden_size) * 1e-6
            self.inv_cov = None
        else:
            self.covariance_mode = "full"
            cov = torch.cov(states_matrix.T) + 1e-6 * torch.eye(self.hidden_size)
            self.inv_cov = torch.pinverse(cov)
            self.variances = None

        self.is_fitted = True

    def mahalanobis_distance(self, hidden_state: torch.Tensor) -> float:
        """Compute Mahalanobis distance of a hidden state relative to the reference manifold.

        Args:
            hidden_state: Hidden state tensor of shape (hidden_size,) or (1, hidden_size).

        Returns:
            Mahalanobis distance as a float.

        Raises:
            ManifoldFitError: If manifold is not fitted.
        """
        if not self.is_fitted or self.mean is None:
            raise ManifoldFitError("ReferenceManifold must be fit before calling mahalanobis_distance().")

        h = hidden_state.squeeze()
        if h.dim() == 0:
            h = h.unsqueeze(0)

        delta = h - self.mean

        if self.covariance_mode == "diagonal":
            if self.variances is None:
                raise ManifoldFitError("Corrupted manifold: variances vector is None.")
            sq_dist = torch.sum((delta**2) / self.variances)
        else:  # full covariance mode
            if self.inv_cov is None:
                raise ManifoldFitError("Corrupted manifold: inv_cov matrix is None.")
            sq_dist = torch.matmul(delta.unsqueeze(0), torch.matmul(self.inv_cov, delta.unsqueeze(1))).squeeze()

        dist = torch.sqrt(torch.clamp(sq_dist, min=0.0))
        return float(dist.item())

    def save(self, path: str) -> None:
        """Save manifold state to disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": self.version,
            "sample_ratio_threshold": self.sample_ratio_threshold,
            "mean": self.mean,
            "inv_cov": self.inv_cov,
            "variances": self.variances,
            "covariance_mode": self.covariance_mode,
            "hidden_size": self.hidden_size,
            "is_fitted": self.is_fitted,
        }
        torch.save(payload, target_path)

    @classmethod
    def load(cls, path: str) -> "ReferenceManifold":
        """Load manifold state from disk.

        Raises:
            ManifoldFitError: If file is missing or invalid.
        """
        target_path = Path(path)
        if not target_path.exists():
            raise ManifoldFitError(f"Manifold state file not found at '{path}'")

        try:
            payload = torch.load(target_path, weights_only=False)
            manifold = cls(
                version=payload.get("version", "manifold_v1_normal_val_only"),
                sample_ratio_threshold=payload.get("sample_ratio_threshold", 5.0),
            )
            manifold.mean = payload.get("mean")
            manifold.inv_cov = payload.get("inv_cov")
            manifold.variances = payload.get("variances")
            manifold.covariance_mode = payload.get("covariance_mode", "unfitted")
            manifold.hidden_size = payload.get("hidden_size", 0)
            manifold.is_fitted = payload.get("is_fitted", True)
            return manifold
        except Exception as e:
            if isinstance(e, ManifoldFitError):
                raise
            raise ManifoldFitError(f"Failed to load ReferenceManifold from '{path}': {e}") from e
