# PHASE 4 REPORT: L3b LNN Core Layer

## Files created

- `hdlnn/lnn/interfaces.py`
- `hdlnn/lnn/utils.py`
- `hdlnn/lnn/cfc_model.py`
- `hdlnn/lnn/ltc_model.py`
- `hdlnn/lnn/__init__.py`
- `tests/test_lnn.py`
- `PHASE4_REPORT.md`

## ncps API calls actually used

Package: `ncps` (Version: 1.0.1)

- `from ncps.torch import CfC`: Instantiates Closed-form Continuous-time (CfC) sequence model (`CfC(input_size=..., units=..., return_sequences=True)`).
- `from ncps.torch import LTC`: Instantiates Liquid Time-Constant (LTC) sequence model (`LTC(input_size=..., units=..., return_sequences=True)`).
- `cfc(x_seq, hx=hx)` / `ltc(x_seq, hx=hx)`: Runs continuous-time forward pass over input sequence tensor with initial or previous hidden state `hx`.

## Confirmation of batch vs streaming equivalence test result specifically

- **CfCModel Batch vs. Streaming Equivalence**: **PASSED** (Maximum absolute difference between batch `forward()` hidden states and streaming `step()` hidden states across all sequence steps: **0.00e+00**).
- **LTCModel Batch vs. Streaming Equivalence**: **PASSED** (Maximum absolute difference: **0.00e+00**).

Batch execution (`forward()`) over a full ordered sequence of `EncodedHypervector` items produces exact, bit-identical hidden state trajectories compared to sequential single-record streaming calls (`step()`).

## Any deviations from this spec (and why)

1. **Shared State I/O Helper Module**: Added `hdlnn/lnn/utils.py` containing `load_encoded_vector`, `save_hidden_state`, and `load_hidden_state` to eliminate code duplication between `CfCModel` and `LTCModel` while preserving baseline-agnostic `ISequenceModel` interface compliance (P3).
2. **Explicit Fallback Entity Identifier**: If `entity_id` is omitted in `step()`, it is automatically inferred from `prev_state.entity_id` if present, or defaults to `"default"`.

## Test results

```text
.\.venv\Scripts\pytest.exe
........................                                                 [100%]
24 passed in 3.88s
```

## Open questions / things you weren't sure about

- None. All requirements for `ISequenceModel`, `CfCModel`, `LTCModel`, per-entity state threading, hidden state path persistence under `data/encoded/<encoder_version>/trajectories/<entity_id>/<flow_id>.npy`, and batch/streaming numerical equivalence are fully implemented and verified.

## Suggested next step

- Proceed to **Phase 5: Divergence & Manifold Drift Layer (L4)** to implement `IManifoldDrift`, Mahalanobis / Cosine divergence measurement, and `AnomalyDecision` production.
