# PHASE 5 REPORT: L3c Divergence Engine Layer

## Files created

- `hdlnn/divergence/interfaces.py`
- `hdlnn/divergence/manifold.py`
- `hdlnn/divergence/scorer.py`
- `hdlnn/divergence/__init__.py`
- `tests/test_divergence.py`
- `PHASE5_REPORT.md`

## Confirmation of the leakage-prevention test specifically

The leakage-prevention rule is strictly enforced in `ReferenceManifold.fit()`. If even a single non-normal trajectory label (e.g. `"attack"`) is included in the training batch, `ManifoldFitError` is raised immediately.

### Test Code: `tests/test_divergence.py::test_manifold_fit_leakage_prevention_raises`

```python
def test_manifold_fit_leakage_prevention_raises(tmp_path):
    trajectories = create_synthetic_trajectories(tmp_path, n_samples=5, hidden_size=10)
    labels = ["normal", "normal", "attack", "normal", "normal"]

    manifold = ReferenceManifold()
    with pytest.raises(ManifoldFitError, match="P4 Leakage Violation"):
        manifold.fit(trajectories, labels)
```

### Result

```text
PASSED tests/test_divergence.py::test_manifold_fit_leakage_prevention_raises
```

`ManifoldFitError` raised as expected with message: `P4 Leakage Violation: ReferenceManifold fit received non-normal label 'attack' at index 2 for flow 'traj_flow_2'.`

## Any deviations from this spec (and why)

1. **Covariance Matrix Regularization**: Added `1e-6` ridge regularization (`1e-6 * eye`) to full covariance matrices and diagonal variance vectors prior to inversion to guarantee numerical stability and positive definiteness.
2. **Case-Insensitive Normal Label Check**: `ReferenceManifold.fit()` normalizes label strings via `str(label).strip().lower() == "normal"` to gracefully handle raw casing variations while preserving strict normal-only enforcement.

## Test results

```text
.\.venv\Scripts\pytest.exe
..............................                                           [100%]
30 passed in 9.04s
```

## Open questions / things you weren't sure about

- None. All requirements for `IDivergenceScorer`, `ReferenceManifold`, normal-only leakage prevention, sample-ratio threshold covariance selection, FPR target calibration, and `AnomalyDecision` production are fully implemented and verified.

## Suggested next step

- Proceed to **Phase 6: Baseline Comparative Models (L4 Baseline Layer)** to implement non-HDC and non-LNN comparative baseline models (e.g., standard LSTM/CNN sequence classifiers and static classifiers).
