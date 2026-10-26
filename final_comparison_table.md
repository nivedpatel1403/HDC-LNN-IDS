# Final Baseline Model Benchmark Comparison

> [!NOTE]
> **FULL-SCALE BENCHMARK (30,000 flows, 4,613 test flows across 41 disjoint entities)**: Full-rank normal validation manifold calibration under strict cybersecurity leakage isolation.

## 1. Sequence & Generative Model Detection Performance

| Model Architecture                   | Dataset / Run ID                          | # Params  | Model Size (MB) | F1 Score | Precision | Recall | AUROC  | PR-AUC | FPR @ 95% TPR | Latency (ms/flow) | Throughput (flows/s) | Peak RSS (MB) |
|:-------------------------------------|-------------------------------------------|-----------|-----------------|----------|-----------|--------|--------|--------|---------------|-------------------|----------------------|--------------:|
| Mamba-2 (Pure PyTorch SSM)           | eval_UNSW_NB15_testing-set_mamba2         | 1,322,704 | 5.05 MB         | 0.9189   | 0.8750    | 0.9675 | 0.9007 | 0.9645 | 0.4324        | 1.51 ms           | 663.61               | 562.07        |
| Mamba-3 (Complex SSM)                | eval_UNSW_NB15_testing-set_mamba3         | 1,328,848 | 5.07 MB         | 0.9482   | 0.9297    | 0.9675 | 0.9802 | 0.9942 | 0.1892        | 5.23 ms           | 191.29               | 502.68        |
| 1D-CNN (Temporal Conv)               | eval_UNSW_NB15_testing-set_cnn            | 2,570,064 | 9.80 MB         | 0.9918   | 1.0000    | 0.9837 | 0.9987 | 0.9996 | 0.0000        | 7.77 ms           | 128.67               | 529.98        |
| Autoencoder (LSTM-AE)                | eval_UNSW_NB15_testing-set_autoencoder    | 3,226,896 | 12.31 MB        | 0.8496   | 0.9320    | 0.7805 | 0.8679 | 0.9626 | 0.8108        | 7.61 ms           | 131.48               | 469.00        |
| HDC-LNN (Ours)                       | eval_UNSW_NB15_testing-set_hdclnn         | 1,971,344 | 7.52 MB         | 1.0000   | 1.0000    | 1.0000 | 1.0000 | 1.0000 | 0.0000        | 1.74 ms           | 573.72               | 549.74        |
| LSTM (Recurrent Baseline)            | eval_UNSW_NB15_testing-set_lstm           | 3,226,896 | 12.31 MB        | 0.8693   | 0.7688    | 1.0000 | 0.4610 | 0.7936 | 1.0000        | 4.80 ms           | 208.37               | 543.56        |
| Transformer-Encoder (Self-Attention) | eval_UNSW_NB15_testing-set_transformerenc | 1,357,136 | 5.24 MB         | 0.9111   | 0.8367    | 1.0000 | 0.8939 | 0.9559 | 0.2973        | 2.10 ms           | 476.35               | 498.17        |
| Mini-GPT (Causal Decoder)            | eval_UNSW_NB15_testing-set_minigpt        | 1,357,136 | 5.24 MB         | 0.9111   | 0.8367    | 1.0000 | 0.9024 | 0.9592 | 0.2703        | 5.20 ms           | 192.32               | 503.37        |

## 2. Encoder Geometric Separability (Empirical Analysis)

| Encoder Architecture     | Centroid Cos Dist | Norm Euclidean Dist | Fisher Ratio | Intra Sim | Inter Sim | Separability Margin |
|:-------------------------|-------------------|---------------------|--------------|-----------|-----------|--------------------:|
| HDC (RecordEncoder)      | 0.2840            | 0.5200              | 0.248735     | 0.5339    | 0.3267    | 0.2072              |
| SAX (Symbolic Aggregate) | 0.2420            | 0.4474              | 0.164670     | 0.4602    | 0.2984    | 0.1618              |
| RFF (Random Fourier)     | 0.2657            | 0.5558              | 0.359254     | 0.6274    | 0.4200    | 0.2074              |

## 3. Honest Architectural Analysis & Relative Ranking

- **Mamba-2 (Pure-PyTorch SSM)** achieves F1: 0.9189, AUROC: 0.9007, Precision: 0.8750, Recall: 0.9675 with throughput of 663.61 flows/sec (latency: 1.51 ms/flow) using selective state-space recurrence.
- **Mamba-3 (Complex SSM & Heavy-Tail)** achieves F1: 0.9482, AUROC: 0.9802, Precision: 0.9297, Recall: 0.9675 with throughput of 191.29 flows/sec (latency: 5.23 ms/flow) utilizing complex-valued state rotations and heavy-tail decay.
- **1D-CNN & LSTM-Autoencoder**: 1D-CNN achieves F1: 0.9918 (AUROC: 0.9987, latency: 7.77 ms/flow), while LSTM-Autoencoder achieves F1: 0.8496 (AUROC: 0.8679) with reconstruction latency of 7.61 ms/flow.
- **HDC-LNN (Ours)** demonstrates throughput: 573.72 flows/sec and latency: 1.74 ms/flow (Precision: 1.0000, Recall: 1.0000, F1: 1.0000, AUROC: 1.0000) under closed-form continuous-time liquid neural dynamics.
- **LSTM (Recurrent Baseline)** achieves F1: 0.8693, AUROC: 0.4610, Precision: 0.7688, Recall: 1.0000 across recurrent state transitions.
- **Transformer-Encoder (Self-Attention)** achieves F1: 0.9111, AUROC: 0.8939, Precision: 0.8367, Recall: 1.0000 with throughput of 476.35 flows/sec (latency: 2.10 ms/flow) using bidirectional multi-head self-attention.
- **Mini-GPT (Causal Decoder)** achieves F1: 0.9111, AUROC: 0.9024, Precision: 0.8367, Recall: 1.0000 with throughput of 192.32 flows/sec (latency: 5.20 ms/flow) using autoregressive causal masking.
