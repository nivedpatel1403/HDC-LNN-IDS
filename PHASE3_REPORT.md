# PHASE 3 REPORT: L3a HDC Encoding Layer

## Files created

- `hdlnn/hdc/interfaces.py`
- `hdlnn/hdc/item_memory.py`
- `hdlnn/hdc/encoder.py`
- `hdlnn/hdc/__init__.py`
- `tests/test_hdc.py`
- `PHASE3_REPORT.md`

## torchhd API calls actually used

Package: `torch-hd` (Version: 5.8.4)

- `torchhd.random(num_vectors=1, dim=dim, vsa="MAP")`: Generates base hypervectors and feature role hypervectors.
- `torchhd.level(num_levels=num_levels, dim=dim, vsa="MAP")`: Generates correlated level hypervectors for numeric feature quantization bins.
- `torchhd.bind(input_hv, role_hv)`: Binds feature value hypervectors with their corresponding feature role vectors.
- `torchhd.multibundle(stacked_tensors)`: Bundles elementwise all bound per-feature vectors into a single fixed-width hypervector representation.

## Any deviations from this spec (and why)

1. **Explicit Feature-Specific OOV Reservation**: `ItemMemory` creates per-feature reserved OOV hypervectors (`(feature_name, "__OOV__")`) during fitting to prevent cross-feature OOV collision.
2. **On-the-fly OOV Fallback for Unseen Features**: If an entirely new categorical feature name is encountered at evaluation time, `ItemMemory` generates a feature-specific OOV vector on-the-fly without mutating its fitted `memory` dict.

## Test results

```text
.\.venv\Scripts\pytest.exe
....................                                                     [100%]
20 passed in 3.29s
```

## Open questions / things you weren't sure about

- None. All requirements for `ItemMemory`, `HDCEncoder`, P4 train-split enforcement, MAP representation, `.npy` vector storage per ADR-005, and binary representation rejection per ADR-001 are fully implemented and verified.

## Suggested next step

- Proceed to **Phase 4: Liquid Neural Network (LNN) Core Layer (L3b)** to implement the continuous-time sequence processing model (`hdlnn/lnn/`) that processes streams of `EncodedHypervector` inputs and tracks `TrajectoryState` trajectories.
