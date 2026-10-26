import pytest
import torch
from hdlnn.divergence.manifold import ReferenceManifold

def test_reference_manifold_mahalanobis_calculation():
    hidden_dim = 3
    eps = 1e-4
    manifold = ReferenceManifold(hidden_dim=hidden_dim, eps=eps)

    # Create dummy validation states (normal distribution)
    # Shape: [N, hidden_dim]
    torch.manual_seed(42)
    normal_states = torch.randn(100, hidden_dim) + torch.tensor([1.0, 2.0, 3.0])

    # Fit manifold
    manifold.fit(normal_states)

    assert manifold.is_fit is True
    assert manifold.mean.shape == (hidden_dim,)
    assert manifold.inv_cov.shape == (hidden_dim, hidden_dim)

    # Test distance calculation for a point exactly at the mean (should be ~0.0)
    dist_at_mean = manifold.compute_mahalanobis_distance(manifold.mean)
    assert dist_at_mean.dim() == 1, f"Expected 1-D tensor, got {dist_at_mean.dim()}-D"
    assert dist_at_mean.shape == (1,), f"Expected shape (1,), got {dist_at_mean.shape}"
    assert float(dist_at_mean.item()) == pytest.approx(0.0, abs=1e-5)

    # Test single-point batch (batch_size = 1)
    single_batch = torch.tensor([[1.0, 2.0, 3.0]])
    dist_single = manifold.compute_mahalanobis_distance(single_batch)
    assert dist_single.dim() == 1, f"Expected 1-D tensor for batch 1, got {dist_single.dim()}-D"
    assert dist_single.shape == (1,), f"Expected shape (1,) for batch 1, got {dist_single.shape}"

    # Test distance calculation for a batch of points
    test_points = torch.tensor([
        [1.0, 2.0, 3.0],  # Close to mean
        [10.0, 20.0, 30.0],  # Far from mean
    ])
    distances = manifold.compute_mahalanobis_distance(test_points)
    assert distances.dim() == 1
    assert distances.shape == (2,)
    assert distances[0] < distances[1]

def test_reference_manifold_covariance_regularization():
    hidden_dim = 2
    manifold = ReferenceManifold(hidden_dim=hidden_dim, eps=1e-3)

    # Create collinear states (perfectly correlated, singular covariance)
    # x2 = 2 * x1
    x1 = torch.linspace(-5, 5, 20).unsqueeze(1)
    states = torch.cat([x1, x1 * 2], dim=1)  # [20, 2]

    # Without regularization, linalg.inv would throw RuntimeError or yield NaNs.
    # We verify that with eps regularization, the fit is stable and invert succeeds.
    manifold.fit(states)
    assert manifold.is_fit is True
    assert not torch.isnan(manifold.inv_cov).any()
    assert not torch.isinf(manifold.inv_cov).any()


@pytest.mark.parametrize("cov_type", ["full", "shared", "diagonal"])
def test_reference_manifold_batch_1_shape_invariance_across_modes(cov_type):
    """Verifies that compute_mahalanobis_distance ALWAYS returns a 1D tensor [B] or [1], never 0D."""
    hidden_dim = 4
    torch.manual_seed(42)
    # Create multi-cluster data
    cluster1 = torch.randn(30, hidden_dim) + torch.tensor([0.0, 0.0, 0.0, 0.0])
    cluster2 = torch.randn(30, hidden_dim) + torch.tensor([5.0, 5.0, 5.0, 5.0])
    train_states = torch.cat([cluster1, cluster2], dim=0)

    manifold = ReferenceManifold(hidden_dim=hidden_dim, n_clusters=2, covariance_type=cov_type)
    manifold.fit(train_states)

    # 1. Unbatched 1D vector: [H] -> must return [1]
    unbatched = torch.randn(hidden_dim)
    dist_unbatched = manifold.compute_mahalanobis_distance(unbatched)
    assert dist_unbatched.dim() == 1
    assert dist_unbatched.shape == (1,)

    # 2. Batch size 1: [1, H] -> must return [1]
    batch_1 = torch.randn(1, hidden_dim)
    dist_b1 = manifold.compute_mahalanobis_distance(batch_1)
    assert dist_b1.dim() == 1
    assert dist_b1.shape == (1,)

    # 3. Multi-batch: [4, H] -> must return [4]
    batch_multi = torch.randn(4, hidden_dim)
    dist_bm = manifold.compute_mahalanobis_distance(batch_multi)
    assert dist_bm.dim() == 1
    assert dist_bm.shape == (4,)


def test_reference_manifold_high_dimension_cosine_fallback():
    """Verifies shape consistency when hidden_dim > 1000 (cosine distance branch)."""
    hidden_dim = 1024
    manifold = ReferenceManifold(hidden_dim=hidden_dim)
    states = torch.randn(50, hidden_dim)
    manifold.fit(states)

    # Unbatched [1024] -> [1]
    dist_unb = manifold.compute_mahalanobis_distance(states[0])
    assert dist_unb.dim() == 1
    assert dist_unb.shape == (1,)

    # Batch 1 [1, 1024] -> [1]
    dist_b1 = manifold.compute_mahalanobis_distance(states[:1])
    assert dist_b1.dim() == 1
    assert dist_b1.shape == (1,)

    # Batch 3 [3, 1024] -> [3]
    dist_b3 = manifold.compute_mahalanobis_distance(states[:3])
    assert dist_b3.dim() == 1
    assert dist_b3.shape == (3,)
