# PHASE 6 REPORT: Baseline Comparative Models Layer

## Files created

- `hdlnn/baselines/lstm_baseline.py`
- `hdlnn/baselines/cnn_baseline.py`
- `hdlnn/baselines/__init__.py`
- `tests/test_baselines.py`
- `PHASE6_REPORT.md`

## Confirm the P4-message rename was done

- **Housekeeping confirmed**: Renamed error message in `hdlnn/divergence/manifold.py` from `"P4 Leakage Violation"` to `"Manifold leakage violation"` and updated `tests/test_divergence.py`.

## CNN baseline's "hidden state = buffer" design choice

- **Design Explanation**: 1D CNN models lack intrinsic recurrent hidden states. To satisfy the `ISequenceModel` interface without architectural special-casing in downstream evaluation components, `CNNBaseline` maintains a sliding window buffer of recent input `EncodedHypervectors` (up to `window_size`, with left zero-padding when under capacity).
- **Trajectory Persistence**: The unpadded window buffer is stored in a sidecar `.npy` file (`*_cnn_buf.npy`), while `hidden_state_ref` points to the 1D CNN feature embedding tensor of shape `(1, hidden_size)` (`.npy`). This allows `ReferenceManifold` to score `CNNBaseline` trajectories using the exact same code path used for `CfCModel`, `LTCModel`, and `LSTMBaseline`.

## Batch vs streaming equivalence results for LSTM

- **LSTMBaseline Batch vs. Streaming Equivalence**: **PASSED** (Maximum absolute difference between batch `forward()` hidden states and streaming `step()` hidden states across all sequence steps: **0.00e+00**).
- **CNNBaseline Batch vs. Streaming Equivalence**: **PASSED** (Maximum absolute difference: **0.00e+00**).

Batch execution over a full ordered sequence produces exact, bit-identical hidden state output trajectories compared to sequential single-record streaming calls.

## Any deviations from this spec (and why)

1. **LSTM Hidden & Cell State Sidecar Persistence**: `LSTMBaseline` persists PyTorch LSTM cell state `(h_n, c_n)` to `*_lstm_cell.npy` while keeping `hidden_state_ref` pointing directly to output hidden state `h_n` `(1, hidden_size)` for manifold compatibility.

## Test results

```text
.\.venv\Scripts\pytest.exe
...................................                                      [100%]
35 passed in 6.70s
```

## Open questions / things you weren't sure about

- None. Both `LSTMBaseline` and `CNNBaseline` strictly implement `ISequenceModel`, satisfy batch/streaming equivalence, enforce entity state isolation, and output compatible `TrajectoryState` contracts.

## Suggested next step

- Proceed to **Phase 7: Evaluation & Ablation Harness Layer (L4)** to implement `EvaluationHarness`, metric calculators (ROC-AUC, Precision, Recall, F1, Latency), and ablation experiment orchestrators (`hdlnn/eval/`).
