# PHASE 7a REPORT: L4 Evaluation Harness Core

## Files created

- `hdlnn/eval/interfaces.py`
- `hdlnn/eval/run_result.py`
- `hdlnn/eval/metrics.py`
- `hdlnn/eval/pipeline_adapter.py`
- `hdlnn/eval/harness.py`
- `hdlnn/eval/__init__.py`
- `tests/test_eval.py`
- `PHASE7A_REPORT.md`

## Training objective chosen for the sequence models

- **Objective**: **Self-Supervised Next-Step Prediction Loss** ($MSE(\hat{x}_{t+1}, x_{t+1})$).
- **Explanation**: In HDC-LNN's anomaly detection framework, sequence models (CfC, LTC, LSTM, CNN) process temporal streams of normal network traffic hypervectors without access to attack labels during training. To optimize the sequence model's hidden state representations before fitting the reference manifold, a linear readout projection head `readout = torch.nn.Linear(hidden_size, input_size)` maps hidden state $h_t$ to predicted next hypervector $\hat{x}_{t+1} = \text{readout}(h_t)$. The model is trained using Adam optimizer over $E$ epochs to minimize Mean Squared Error ($MSE(\hat{x}_{t+1}, x_{t+1})$) across normal training sequences.

## Metrics library used

- **Library**: `scikit-learn` (Version: 1.9.0)
- **Functions Used**: `f1_score`, `precision_score`, `recall_score`, `roc_auc_score`, `roc_curve`.
- **System Metrics**: `psutil` (Version: 7.2.2) for peak RSS process memory measurement.

## Any deviations from this spec (and why)

1. **Explicit Linear Readout Head**: Added a temporary `nn.Linear(hidden_size, input_size)` readout head inside `StandardPipeline._train_sequence_model()` to map sequence model hidden states back to hypervector dimension for MSE next-step prediction loss calculation.

## Test results

```text
.\.venv\Scripts\pytest.exe
........................................                                 [100%]
40 passed in 10.55s
```

## Open questions / things you weren't sure about

- None. All single-run pipeline orchestration, baseline drop-in compatibility, scikit-learn metrics calculation, mandatory device string validation, and 5-artifact persistence (`config.snapshot.yaml`, `git_commit.txt`, `metrics.json`, `decisions.parquet`, `logs/`) under `runs/<run_id>/` are fully implemented and verified.

## Suggested next step

- Proceed to **Phase 7b: L4 Evaluation Harness — Multi-Config Ablation Suite** to implement stage bypass toggles (`hdc_encoder=off`, `lnn_stage=off`), ablation experiment matrix runner (`run_ablation_suite()`), and multi-run comparative metrics aggregation.
