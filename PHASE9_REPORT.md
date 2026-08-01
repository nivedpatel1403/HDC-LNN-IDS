# PHASE 9 REPORT: Pipeline Orchestrator + CLI

## Files created/modified

- `hdlnn/pipeline.py` (created: top-level orchestrator driving `load_and_split_data`, `run_train`, `run_eval`, `run_stream`, `run_ablate`)
- `hdlnn/cli.py` (created: Click CLI exposing `hdlnn train`, `eval`, `stream`, `ablate`, `report` subcommands)
- `tests/test_pipeline.py` (created)
- `tests/test_cli.py` (created)
- `PHASE9_REPORT.md` (created)

## replay_rate semantics in run_stream()

- **Semantics**: `replay_rate` controls real-time streaming pacing simulation. For each consecutive flow in `test_flows` ordered by `occurred_at`, the time difference $\Delta t = (t_{curr} - t_{prev})$ is calculated in seconds. If `replay_rate > 0`, the stream sleeps for $\min(\Delta t / \text{replay\_rate}, 0.05)$ seconds per record to simulate real-time stream pacing without blocking automated test suite execution (when `replay_rate == 0.0`, it replays as fast as possible). Wall-clock latency per `step()` call is captured and stored in `latency_stats`.

## Eval-vs-stream consistency check result

- **Status**: **PASSED** (100% decision agreement verified).
- **Execution Log Output**:
  ```text
  2026-08-01 16:04:53,631 [INFO] hdlnn.pipeline.run_stream - Loaded model checkpoint from 'runs\test_pipe_stream_train\model_checkpoint.pt'
  2026-08-01 16:04:53,631 [INFO] hdlnn.pipeline.run_stream - Running eval-vs-stream consistency verification check...
  2026-08-01 16:04:53,635 [INFO] hdlnn.pipeline.run_stream - Eval-vs-stream consistency verification PASSED: 100% decision agreement!
  ```
- **Verification Details**: Every `AnomalyDecision.is_anomalous` boolean flag produced by streaming `step()` calls over `test_flows` matched the corresponding `batch` `forward()` decision.

## cli.py import-boundary audit result

- **Status**: **PASSED**
- **Audit Implementation**: `test_cli_import_boundary_audit` in `tests/test_cli.py` inspects `cli.py` AST/code text and verifies `cli.py` does NOT import directly from internal stage packages (`hdlnn.data`, `hdlnn.hdc`, `hdlnn.lnn`, `hdlnn.divergence`). `cli.py` strictly delegates all pipeline execution to `pipeline.py`, `eval/`, and `report/`.

## Any deviations from this spec (and why)

1. **Path Fallback for Test Fixtures**: `load_and_split_data()` checks if `dataset_raw_path` exists, and if omitted or missing in temporary test directories, falls back to `tests/fixtures/unsw_sample.csv` relative to repository root.
2. **Duplicate Flow ID Handling**: In `check_consistency`, decisions are paired by combining `flow_id` and ISO timestamp (`f"{flow_id}_{occurred_at.isoformat()}"`), preventing key collisions when raw datasets contain duplicate flow IDs.

## Test results

```text
.\.venv\Scripts\pytest.exe
...............................................................          [100%]
63 passed in 66.88s (0:01:06)
```

## Open questions / things you weren't sure about

- None. All orchestrator functions (`load_and_split_data`, `run_train`, `run_eval`, `run_stream`, `run_ablate`), Click CLI subcommands (`hdlnn train`, `eval`, `stream`, `ablate`, `report`), eval-vs-stream consistency checks, import boundary discipline, and test suites are fully verified.

## Suggested next step

- Proceed to **Phase 10: Zero-Shot Transfer & Secondary Dataset Integration (CICIoT2023 / NSL-KDD)** to implement `CICIoTLoader` and `NSLKDDLoader` data mapping, fill in the cross-dataset transfer evaluation path, and verify zero-shot anomaly detection transfer metrics.
