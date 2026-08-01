# HDC-LNN-IDS: Hybrid HDC-LNN Architecture for Real-Time Anomaly Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

## Overview
**HDC-LNN-IDS** is a hybrid framework combining **Hyperdimensional Computing (HDC)** and **Liquid Neural Networks (LNNs)** for real-time network intrusion and anomaly detection. 

The system leverages HDC (Symbolic Hyperdimensional Vector Encoding) for ultra-fast, high-dimensional feature representations of network flow tuples (`src_ip`, `dst_ip`, `src_port`, `dst_port`, `proto`, `state`, `duration`, `bytes`), combined with continuous-time Liquid Neural Network architectures (Closed-form Continuous-depth Networks (**CfC**) and Liquid Time-Constant networks (**LTC**)) for real-time sequence modeling and out-of-distribution anomaly detection via manifold Mahalanobis divergence scoring.

---

## Current Status
- **Development Progress**: Phases 1 through 11 complete (of 12 planned phases).
- **Validation**: Full pipeline contract integrity and end-to-end multi-entity streaming validation verified on the real **UNSW-NB15** dataset (49-feature raw partition specification with genuine IP/port flow sequences across 30 distinct entity trajectories).
- **Test Suite Coverage**: 66 unit, integration, property, and baseline equivalence tests passing cleanly.

---

## Quick Start

### 1. Requirements
- **Python**: 3.10 or higher
- **PyTorch**: 2.0+

### 2. Installation
Clone the repository and install the project in editable mode:
```bash
git clone https://github.com/nivedpatel1403/HDC-LNN-IDS.git
cd HDC-LNN-IDS
pip install -e .
```

### 3. Running Tests
Verify the complete test suite:
```bash
pytest
```

---

## Usage

### Training & Evaluation CLI
Run experiments using predefined configuration files under `configs/`:

```bash
# Run hybrid HDC-CfC model training & evaluation
hdlnn train --config configs/hybrid.yaml

# Run HDC-only baseline
hdlnn train --config configs/hdc_only.yaml

# Run LNN-only baseline
hdlnn train --config configs/lnn_only.yaml

# Run LSTM baseline comparison
hdlnn train --config configs/lstm_baseline.yaml

# Run CNN baseline comparison
hdlnn train --config configs/cnn_baseline.yaml

# Run LTC ablation study
hdlnn train --config configs/ltc_ablation.yaml
```

---

## Architecture & Configuration

Pre-configured YAML models are located in `configs/`:
- `hybrid.yaml`: Full hybrid HDC encoder + CfC liquid neural network stage.
- `hdc_only.yaml`: HDC feature encoder with direct manifold divergence scoring.
- `lnn_only.yaml`: Raw flow features passed directly into CfC liquid neural network.
- `lstm_baseline.yaml`: HDC feature encoder + standard LSTM sequence model.
- `cnn_baseline.yaml`: HDC feature encoder + 1D Convolutional Neural Network baseline.
- `ltc_ablation.yaml`: HDC feature encoder + Liquid Time-Constant (LTC) neural network.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Research Context & Attribution
*(Note: IAMPro / IEEE Computer Society research context attribution details can be specified here by the author)*
