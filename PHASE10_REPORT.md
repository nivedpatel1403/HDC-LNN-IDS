# PHASE 10 REPORT: Test Suite Tiering & Validation

## Tier coverage map

All 63 test functions across the repository have been categorized and explicitly decorated with custom `@pytest.mark.<tier>` markers defined in `pyproject.toml`:

| Tier | Marker | Count | Key Test File Sources & Coverage Target |
|---|---|---|---|
| **Unit** | `@pytest.mark.unit` | 17 | `test_contracts.py`, `test_common.py`, `test_hdc.py` (ItemMemory OOV/fit/save), `test_divergence.py` (leakage check), `test_eval.py` (metric math checks) |
| **Component** | `@pytest.mark.component` | 16 | `test_hdc.py` (encoder determinism/binding), `test_lnn.py` (CfC/LTC batch vs streaming numerical equivalence & state isolation), `test_divergence.py` (diagonal vs full covariance), `test_baselines.py` (LSTM/CNN batch vs streaming & buffer overflow), `test_ablation_coverage.py` (HDC-only passthrough & HDC-off bypass) |
| **Data-quality** | `@pytest.mark.data_quality` | 5 | `test_data.py` (UNSWLoader schema mapping, DataQualityReport generation, deterministic flow grouping splitter, stub loader exceptions) |
| **Integration** | `@pytest.mark.integration` | 13 | `test_eval.py` (StandardPipeline run & artifact persistence), `test_pipeline.py` (split reuse, train, eval, stream replay with consistency check), `test_cli.py` (train/eval/stream/report CLI commands & import boundary audit), `test_integration_hand_computed.py` (hand-computed metrics verification) |
| **Ablation** | `@pytest.mark.ablation` | 12 | `test_ablation.py` & `test_ablation_coverage.py` (6-config ablation suite execution, OOV rate reporting, checkpoint persistence, cross-dataset transfer stub), `test_report.py` (table rendering, plots, static AST calculation audit), `test_pipeline.py` (`run_ablate`), `test_cli.py` (`hdlnn ablate`) |

---

## Gaps identified and handling strategy

1. **Gap 1: Data-Quality Tier against real dataset samples**:
   - *Status*: **Explicitly Deferred to Phase 11**.
   - *Reasoning*: Data-quality validation against real, full-scale datasets (UNSW-NB15, CICIoT2023, NSL-KDD) requires dataset downloading and raw file preprocessing in Phase 11. Synthetic fixture testing was used for Phase 10 tiering.

2. **Gap 2: Hand-Computed Integration Metrics Verification**:
   - *Status*: **CLOSED with new integration test** (`tests/test_integration_hand_computed.py`).
   - *Reasoning*: Previous integration tests verified execution without error but did not assert pipeline output against hand-calculated mathematical expectations.

3. **Gap 3: Full-Data Ablation Table Regeneration**:
   - *Status*: **Satisfied on synthetic data; full-data run deferred to Phase 11**.
   - *Reasoning*: Multi-config ablation suite execution across all 6 named configs is fully operational and tested on synthetic fixtures; full-data benchmarking occurs in Phase 11/12.

---

## Hand-computed integration test comparison

- **Scenario**: 10 normal train flows, 10 normal validation flows, 8 test flows (4 normal baseline flows + 4 extreme DoS/Exploits attack flows).
- **Hand-Computed Mathematical Expectations**:
  - Ground Truth: 4 Normal, 4 Attack.
  - Classification: TN = 4, FP = 0, TP = 4, FN = 0.
  - Precision = $\frac{TP}{TP + FP} = \frac{4}{4 + 0} = 1.0000$
  - Recall = $\frac{TP}{TP + FN} = \frac{4}{4 + 0} = 1.0000$
  - F1 Score = $2 \times \frac{1.0 \times 1.0}{1.0 + 1.0} = 1.0000$
  - AUROC = $1.0000$
  - FPR @ 95% TPR = $0.0000$

- **Explicit Comparison Table**:

| Metric | Hand-Computed Expected | Actual Pipeline Output | Difference |
|---|---|---|---|
| **Total Decisions** | 8 | 8 | 0 |
| **Anomalies Detected** | 4 | 4 | 0 |
| **Precision** | 1.0000 | 1.0000 | 0.0000 |
| **Recall** | 1.0000 | 1.0000 | 0.0000 |
| **F1 Score** | 1.0000 | 1.0000 | 0.0000 |
| **AUROC** | 1.0000 | 1.0000 | 0.0000 |
| **FPR @ 95% TPR** | 0.0000 | 0.0000 | 0.0000 |

---

## Bugs found via the hand-computed test & disaggregated sequence analysis

1. **Disambiguation of Duplicate Raw Flow IDs**:
   - *Bug*: `UNSWLoader` assigned `unsw_12` to two distinct rows in `unsw_sample.csv`. When saving hidden state `.npy` files or mapping dictionary decisions, row 13 silently overwrote row 12's hidden state artifact.
   - *Fix*: Appended row index `_{idx}` upon duplicate key detection in `UNSWLoader` (`unsw_12_11` vs `unsw_12_12`), guaranteeing 100% unique `flow_id` references across disk artifacts and decision maps.

2. **Sequential Entity State Warmup Transient**:
   - *Behavioral Insight*: When processing sequence steps from initial zero state `hx_0 = 0`, step 1 experiences a transient state initialization jump. Assigning unique entity IDs per independent stream record (`src_ip = 192.168.x.y`) ensured clean state evaluation across independent single-flow records.

---

## Per-tier test run results

### Tier 1: Unit (`pytest -m unit`)
```text
.\.venv\Scripts\pytest.exe -m unit
...................                                                        [100%]
17 passed, 46 deselected in 4.92s
```

### Tier 2: Component (`pytest -m component`)
```text
.\.venv\Scripts\pytest.exe -m component
................                                                         [100%]
16 passed, 47 deselected in 9.61s
```

### Tier 3: Data Quality (`pytest -m data_quality`)
```text
.\.venv\Scripts\pytest.exe -m data_quality
.....                                                                    [100%]
5 passed, 58 deselected in 4.76s
```

### Tier 4: Integration (`pytest -m integration`)
```text
.\.venv\Scripts\pytest.exe -m integration
.............                                                            [100%]
13 passed, 50 deselected in 22.62s
```

### Tier 5: Ablation (`pytest -m ablation`)
```text
.\.venv\Scripts\pytest.exe -m ablation
............                                                             [100%]
12 passed, 51 deselected in 35.49s
```

---

## Full suite results

```text
.\.venv\Scripts\pytest.exe
...............................................................          [100%]
63 passed in 51.54s
```

---

## Suggested next step

- Proceed to **Phase 11 of 12: Real Dataset Acquisition & End-to-End Evaluation** (download/preprocess raw UNSW-NB15, CICIoT2023, and NSL-KDD datasets, execute full multi-config ablation benchmarks, and generate comparative reporting artifacts).
