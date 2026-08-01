# PHASE 8 REPORT: L5 Reporting Layer

## Files created

- `hdlnn/report/tables.py`
- `hdlnn/report/plots.py`
- `hdlnn/report/dashboard.py`
- `hdlnn/report/__init__.py`
- `tests/test_report.py`
- `PHASE8_REPORT.md`

## Dashboard approach chosen and run instructions

- **Approach**: Interactive Web Dashboard powered by **Streamlit**.
- **Justification**: Streamlit allows dynamic inspection of experiment run folders, rendering of markdown comparison tables, interactive metric bar chart selection, and live plotting of entity trajectory drift curves directly from disk artifacts.
- **Run Command**:
  ```bash
  streamlit run hdlnn/report/dashboard.py
  ```

## Confirmation that report/ never recomputes metrics

- **Verification Methodology**:
  1. **Source Inspection & Data Flow Audit**: Every function in `hdlnn/report/` exclusively consumes disk artifacts written by `hdlnn/eval/` (`comparison_table.json`, `decisions.parquet`). Table renderers read metric fields directly from dictionary keys, and plotters read pre-computed `drift_score` and `threshold` columns directly from `decisions.parquet`.
  2. **Automated AST & Keyword Code Audit**: Implemented `test_static_code_audit_no_metric_recomputation()` in `tests/test_report.py`. The test scans all Python files in `hdlnn/report/` and asserts that zero metric calculation or manifold fitting symbols (`f1_score`, `precision_score`, `recall_score`, `roc_auc_score`, `roc_curve`, `mahalanobis`, `fit(`) exist inside the package.

## Any deviations from this spec (and why)

- None. All table rendering functions, matplotlib plotters, Streamlit interactive dashboard app, explicit device string attachments, standing OOV rate columns, and metric non-recomputation rules are fully implemented and verified.

## Test results

```text
.\.venv\Scripts\pytest.exe
....................................................                     [100%]
52 passed in 18.86s
```

## Open questions / things you weren't sure about

- None. All 52 unit tests across L1 through L5 passed cleanly.

## Suggested next step

- Proceed to **Phase 9: Pipeline Integration & CLI Layer** to implement `pipeline.py` (main entrypoint orchestrating data loading, encoding, training, evaluation, and reporting) and `cli.py` (Click/argparse CLI providing commands like `hdlnn run`, `hdlnn ablation`, `hdlnn dashboard`, and `hdlnn report`).
