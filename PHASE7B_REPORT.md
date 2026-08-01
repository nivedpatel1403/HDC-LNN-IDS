# PHASE 7b REPORT: L4 Evaluation Harness — Ablation Suite

## Files created/modified

- `hdlnn/common/config.py` (modified: added `hdc_encoder`, `lnn_stage`, `sequence_model_type`, and hyperparameter fields to `Settings`)
- `hdlnn/eval/ablation_adapters.py` (created: `HDCOffEncoder` and `PassthroughSequenceModel`)
- `hdlnn/eval/pipeline_adapter.py` (modified: added self-configuring initialization for HDC-on/off, LNN-on/off, and model type selection)
- `hdlnn/eval/harness.py` (modified: added model checkpoint persistence in `run_experiment`, `ComparisonTable` model, `run_ablation_suite()`, and `run_cross_dataset_transfer()` stub)
- `hdlnn/eval/__init__.py` (modified: exported ablation suite runner and models)
- `configs/hybrid.yaml` (created)
- `configs/hdc_only.yaml` (created)
- `configs/lnn_only.yaml` (created)
- `configs/lstm_baseline.yaml` (created)
- `configs/cnn_baseline.yaml` (created)
- `configs/ltc_ablation.yaml` (created)
- `tests/test_ablation.py` (created)
- `PHASE7B_REPORT.md` (created)

## Checkpoint persistence

- **Path Pattern**: `runs/<run_id>/model_checkpoint.pt`.
- **What is saved**: The sequence model's `state_dict` (e.g. `CfC`, `LTC`, `LSTM`, or `Conv1d` weights).
- **Readout-head Decision & Reasoning**: The linear readout head (`nn.Linear(hidden_size, input_size)`) is used exclusively during the self-supervised training phase as a training scaffold to compute next-step prediction MSE loss. During inference (`step()` / `score()`), only the sequence model's hidden states $h_t$ are passed to `ReferenceManifold` for Mahalanobis drift scoring. Therefore, the readout head is discarded prior to checkpoint saving, ensuring deployed checkpoints contain only active inference parameters.

## HDC-only and HDC-off ablation implementation approach

- **HDC-only Mode (`hdc_encoder="on"`, `lnn_stage="off"`)**: `PassthroughSequenceModel` wraps raw `EncodedHypervectors` into `TrajectoryState` objects without temporal accumulation. The `ReferenceManifold` fits its mean and covariance directly on the $D=10000$ HDC hypervector arrays of normal validation flows, evaluating pure hyperdimensional representation.
- **HDC-off Mode (`hdc_encoder="off"`, `lnn_stage="on"`)**: `HDCOffEncoder` bypasses HDC encoding, converting raw `CanonicalFlow` numeric and categorical features directly into normalized dense feature vectors saved under `data/encoded/hdc_off/<flow_id>.npy`. These dense feature vectors are fed directly into the sequence model, evaluating non-HDC input performance.

## Cross-dataset transfer placeholder confirmation

- **Status**: Structurally present as a standing row in `ComparisonTable.cross_dataset_transfer`.
- **Placeholder Output**:
  ```json
  "cross_dataset_transfer": {
    "status": "not_run: CICIoTLoader not yet implemented (Phase 2 stub)",
    "dataset": "CICIoT2023",
    "f1": null,
    "auroc": null,
    "accuracy": null
  }
  ```
- No synthetic numbers were fabricated; real evaluation is deferred until CICIoT2023 raw data files are available.

## Any deviations from this spec (and why)

1. **Self-Configuring Pipeline Adapter**: `StandardPipeline` accepts an optional `Settings` instance and automatically instantiates the appropriate encoder, sequence model, and divergence scorer based on configuration toggles (`hdc_encoder`, `lnn_stage`, `sequence_model_type`), eliminating manual setup logic in test code.

## Test results

```text
.\.venv\Scripts\pytest.exe
..........................................                               [100%]
42 passed in 11.84s
```

## Open questions / things you weren't sure about

- None. All ablation toggles, 6 named configuration YAMLs, model checkpoint persistence, `ComparisonTable` generation, `runs/ablation_suite_<timestamp>/` JSON/CSV persistence, and cross-dataset transfer placeholder behavior are fully implemented and verified.

## Suggested next step

- Proceed to **Phase 8: Reporting, Visualization & CLI Layer (L5)** to implement `ComparisonTable` markdown/latex renderer (`hdlnn/report/`), command-line interface (`cli.py`), and end-to-end pipeline execution entrypoint (`pipeline.py`).
