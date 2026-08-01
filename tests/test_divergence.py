"""Unit and component tests for L3c Divergence Engine Layer (ReferenceManifold and DivergenceScorer)."""

from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytest
import torch

from hdlnn.contracts.decision import AnomalyDecision
from hdlnn.contracts.errors import ManifoldFitError
from hdlnn.contracts.trajectory import TrajectoryState
from hdlnn.divergence.interfaces import IDivergenceScorer
from hdlnn.divergence.manifold import ReferenceManifold
from hdlnn.divergence.scorer import DivergenceScorer


def create_synthetic_trajectories(
    tmp_path: Path, n_samples: int = 10, hidden_size: int = 16, prefix: str = "traj"
) -> list[TrajectoryState]:
    """Helper to create synthetic TrajectoryState contract objects and persisted numpy state files."""
    traj_dir = tmp_path / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    trajs: list[TrajectoryState] = []

    for i in range(n_samples):
        flow_id = f"{prefix}_{i+1}"
        h_vec = rng.randn(hidden_size).astype(np.float32)
        h_path = traj_dir / f"{flow_id}.npy"
        np.save(h_path, h_vec)

        traj = TrajectoryState(
            entity_id="entity_1",
            flow_id=flow_id,
            t=datetime.now(timezone.utc),
            hidden_state_ref=str(h_path),
            model_version="cfc_v1",
        )
        trajs.append(traj)

    return trajs


@pytest.mark.unit
def test_manifold_fit_leakage_prevention_raises(tmp_path):
    trajectories = create_synthetic_trajectories(tmp_path, n_samples=5, hidden_size=10)
    labels = ["normal", "normal", "attack", "normal", "normal"]

    manifold = ReferenceManifold()
    with pytest.raises(ManifoldFitError, match="Manifold leakage violation"):
        manifold.fit(trajectories, labels)


@pytest.mark.component
def test_manifold_fit_diagonal_vs_full_covariance(tmp_path):
    # Hidden size = 20 -> sample_ratio_threshold * 20 = 100
    # Case A: N = 10 < 100 -> diagonal covariance mode
    trajs_small = create_synthetic_trajectories(tmp_path, n_samples=10, hidden_size=20, prefix="small")
    labels_small = ["normal"] * 10

    manifold_diag = ReferenceManifold(sample_ratio_threshold=5.0)
    manifold_diag.fit(trajs_small, labels_small)

    assert manifold_diag.is_fitted
    assert manifold_diag.covariance_mode == "diagonal"
    assert manifold_diag.variances.dim() == 1  # 1D diagonal variances vector

    # Case B: N = 120 >= 100 -> full covariance mode
    trajs_large = create_synthetic_trajectories(tmp_path, n_samples=120, hidden_size=20, prefix="large")
    labels_large = ["normal"] * 120

    manifold_full = ReferenceManifold(sample_ratio_threshold=5.0)
    manifold_full.fit(trajs_large, labels_large)

    assert manifold_full.is_fitted
    assert manifold_full.covariance_mode == "full"
    assert manifold_full.inv_cov.dim() == 2  # 2D full precision matrix


@pytest.mark.component
def test_divergence_scorer_threshold_calibration_and_scoring(tmp_path):
    normal_trajs = create_synthetic_trajectories(tmp_path, n_samples=20, hidden_size=10, prefix="norm")
    labels = ["normal"] * 20

    scorer = DivergenceScorer(target_fpr=0.05)
    assert isinstance(scorer, IDivergenceScorer)

    scorer.fit(normal_trajs, labels)
    assert scorer.is_fitted
    assert scorer.threshold > 0.0

    # Normal trajectory score
    norm_decision = scorer.score(normal_trajs[0], ground_truth_label="normal")
    assert isinstance(norm_decision, AnomalyDecision)
    assert norm_decision.flow_id == normal_trajs[0].flow_id
    assert norm_decision.threshold == scorer.threshold

    # Anomalous trajectory score (distant vector)
    anomaly_dir = tmp_path / "anom"
    anomaly_dir.mkdir(parents=True, exist_ok=True)
    anom_vec = np.ones(10, dtype=np.float32) * 50.0  # Far away from N(0, I)
    anom_path = anomaly_dir / "anom_1.npy"
    np.save(anom_path, anom_vec)

    anom_traj = TrajectoryState(
        entity_id="entity_1",
        flow_id="anom_1",
        t=datetime.now(timezone.utc),
        hidden_state_ref=str(anom_path),
        model_version="cfc_v1",
    )

    anom_decision = scorer.score(anom_traj, ground_truth_label="attack", attack_category="Exploits")
    assert anom_decision.is_anomalous is True
    assert anom_decision.drift_score > scorer.threshold
    assert anom_decision.ground_truth_label == "attack"


@pytest.mark.unit
def test_manifold_save_load(tmp_path):
    trajs = create_synthetic_trajectories(tmp_path, n_samples=15, hidden_size=10)
    labels = ["normal"] * 15

    manifold = ReferenceManifold()
    manifold.fit(trajs, labels)

    save_path = tmp_path / "manifold.pt"
    manifold.save(str(save_path))
    assert save_path.exists()

    loaded_manifold = ReferenceManifold.load(str(save_path))
    assert loaded_manifold.is_fitted
    assert torch.equal(manifold.mean, loaded_manifold.mean)
    if manifold.inv_cov is not None:
        assert torch.equal(manifold.inv_cov, loaded_manifold.inv_cov)
    if manifold.variances is not None:
        assert torch.equal(manifold.variances, loaded_manifold.variances)


@pytest.mark.component
def test_manifold_fit_high_dim_safety_guard(tmp_path):
    # hidden_size = 501 (> 500 threshold), n_samples = 15
    # High dim guard forces diagonal mode to prevent high-dim LAPACK pinverse CPU hangs
    trajs_high_dim = create_synthetic_trajectories(tmp_path, n_samples=15, hidden_size=501, prefix="high_dim")
    labels = ["normal"] * 15

    manifold = ReferenceManifold(sample_ratio_threshold=0.5)
    manifold.fit(trajs_high_dim, labels)

    assert manifold.is_fitted
    assert manifold.covariance_mode == "diagonal"
    assert manifold.variances.dim() == 1
    assert manifold.inv_cov is None
