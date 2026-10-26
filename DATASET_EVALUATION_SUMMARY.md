# Multi-Dataset Benchmark Evaluation Summary

> Comprehensive evaluation of HDC-LNN and baseline architectures across cybersecurity flow telemetry benchmarks under entity-disjoint validation manifold calibration.

## 1. Packaged Benchmark Dataset Evaluation (HDC-LNN)

The following summary reflects execution across the sample benchmark datasets packaged directly within `datasets/` for rapid reproducibility and smoke testing:

| Dataset File | Total Flows | Sample Window | F1 Score | Precision | Recall | AUROC | PR-AUC | Latency (ms/flow) | Throughput (flows/s) | Peak RSS (MB) | Status |
| :--- | ---: | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **UNSW_NB15_testing-set.csv** | 82,332 | 30,000 | **0.9959** | **1.0000** | **0.9919** | 0.9994 | 1.0000 | 0.10 ms | 9,907.1 | 1827.48 MB | Primary Benchmark |
| **kdd_test.csv** | 22,544 | 22,544 | **0.8600** | 0.8329 | 0.8889 | 0.9453 | 0.9512 | 0.10 ms | 9,838.9 | 1782.64 MB | Primary Benchmark |
| **UNSW_NB15_training-set.csv** | 175,341 | 30,000 | 0.0851 | 0.0488 | 0.3333 | 0.9355 | 0.0451 | 0.13 ms | 7441.69 | 5619.12 MB | Single-Class Baseline |
| **ton-iot.csv** | 25 | 25 | 0.7059 | 0.5455 | 1.0000 | 0.5000 | 0.5455 | 0.37 ms | 2701.71 | 343.71 MB | Smoke Verification |

*Note: As detailed in `BENCHMARK_INTEGRITY_AUDIT.md`, `UNSW_NB15_training-set.csv` is a 99.98% benign single-class distribution where F1 is uninformative, and `ton-iot.csv` is a micro-sample (N=25) intended strictly for pipeline sanity testing.*

## 2. Canonical Multi-Model IoT-23 Benchmark Summary

Summary from canonical multi-seed evaluation on full-scale external IoT-23 benchmark captures:

| Dataset Capture | Model | Total Params | F1 Score | Precision | Recall | AUROC | PR-AUC | Throughput (flows/s) | Status |
| :--- | :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **IoT-23 (dataset17.csv)** | HDC-LNN (Ours) | 1.97M | **0.9703** | 0.9632 | 0.9776 | 0.9969 | 0.9985 | **9,994.1** | Canonical Benchmark |
| **IoT-23 (dataset17.csv)** | FT-Transformer | 1.46M | **0.9951** | 0.9905 | 0.9997 | 0.9871 | 0.9984 | 3,545.8 | Tabular Baseline |
| **IoT-23 (dataset17.csv)** | SAINT | 1.49M | **0.9952** | 0.9905 | 1.0000 | 0.9861 | 0.9982 | 3,540.2 | Tabular Baseline |
| **IoT-23 (dataset5.csv)** | HDC-LNN (Ours) | 1.97M | **0.9958** | 0.9967 | 0.9950 | 0.9921 | 0.9961 | **9,790.9** | Canonical Benchmark |
| **IoT-23 (dataset5.csv)** | FT-Transformer | 1.46M | **0.9996** | 0.9992 | 1.0000 | 0.9979 | 0.9994 | 3,519.9 | Tabular Baseline |
| **IoT-23 (dataset5.csv)** | SAINT | 1.49M | **0.9996** | 0.9992 | 1.0000 | 0.9979 | 0.9994 | 3,561.3 | Tabular Baseline |
| **IoT-23 (dataset23.csv)** | HDC-LNN (Ours) | 1.97M | 0.9278 | 0.8653 | 1.0000 | 0.9072 | 0.8446 | **9,696.6** | Canonical Benchmark |
| **IoT-23 (dataset23.csv)** | FT-Transformer | 1.46M | 0.9321 | 0.8728 | 1.0000 | 0.8915 | 0.8709 | 3,522.1 | Tabular Baseline |
| **IoT-23 (dataset23.csv)** | SAINT | 1.49M | 0.4827 | 0.7924 | 0.3471 | 0.8916 | 0.8714 | 3,547.4 | Tabular Baseline |

## 3. Dynamic Execution & Result Isolation

To execute automated batch evaluations across local datasets without altering this canonical document:
```bash
python run_all_datasets.py --quick --output runs/dataset_evaluation_summary.md
```
This isolates local experiment runs to `runs/` while preserving canonical benchmark traceability. Full acquisition details for external captures are documented in [`datasets/SOURCES.md`](datasets/SOURCES.md).