# 5-Fold Cross-Validation Benchmark Summary

- **Dataset**: `UNSW_NB15_testing-set.csv`
- **Folds**: 5 Entity-Disjoint Folds
- **Flow Limit**: 5,000 flows
- **Evaluated Models**: 2 architectures

## Cross-Validation Performance (Mean ± Standard Deviation)

| Model Architecture                   | # Params  | Model Size | F1 Score        | Precision       | Recall          | AUROC           | PR-AUC          | FPR @ 95% TPR   | Latency (ms) | Throughput (f/s) | Peak RSS (MB) |
|:-------------------------------------|-----------|------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|--------------|------------------|--------------:|
| Transformer-Encoder (Self-Attention) | 1,357,136 | 5.24 MB    | 0.9849 ± 0.0042 | 0.9733 ± 0.0072 | 0.9968 ± 0.0033 | 0.9625 ± 0.0234 | 0.9976 ± 0.0017 | 0.1860 ± 0.1065 | 1.26 ms      | 842.3            | 789.5         |
| Mini-GPT (Causal Decoder)            | 1,357,136 | 5.24 MB    | 0.9856 ± 0.0044 | 0.9745 ± 0.0073 | 0.9971 ± 0.0034 | 0.9637 ± 0.0249 | 0.9976 ± 0.0019 | 0.1736 ± 0.0983 | 2.60 ms      | 427.6            | 787.4         |
