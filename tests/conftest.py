import random
import pytest
import numpy as np
import torch
from typing import List, Dict, Any
from hdlnn.contracts.schemas import CanonicalFlow


@pytest.fixture(autouse=True)
def random_seed():
    """Sets deterministic random seeds for reproducible test execution."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture
def sample_flow_batch() -> List[CanonicalFlow]:
    """Provides a synthetic batch of CanonicalFlow instances for testing."""
    flows = []
    # 5 normal training flows
    for i in range(5):
        flows.append(
            CanonicalFlow(
                entity_id="192.168.1.10",
                timestamp=float(i * 2),
                dt=2.0 if i > 0 else 0.0,
                categorical_fields={"proto": "tcp", "service": "http", "state": "CON"},
                numerical_fields={
                    "dur": 0.1 * (i + 1),
                    "spkts": 10.0 + i,
                    "dpkts": 12.0 + i,
                    "sbytes": 1000.0 * (i + 1),
                    "dbytes": 1500.0 * (i + 1),
                },
                label=0,
                split="train"
            )
        )
    # 2 validation flows (1 normal, 1 anomalous)
    flows.append(
        CanonicalFlow(
            entity_id="192.168.1.10",
            timestamp=10.0,
            dt=2.0,
            categorical_fields={"proto": "tcp", "service": "http", "state": "CON"},
            numerical_fields={"dur": 0.6, "spkts": 15.0, "dpkts": 18.0, "sbytes": 6000.0, "dbytes": 9000.0},
            label=0,
            split="val"
        )
    )
    flows.append(
        CanonicalFlow(
            entity_id="192.168.1.10",
            timestamp=12.0,
            dt=2.0,
            categorical_fields={"proto": "udp", "service": "dns", "state": "INT"},
            numerical_fields={"dur": 0.01, "spkts": 500.0, "dpkts": 0.0, "sbytes": 50000.0, "dbytes": 0.0},
            label=1,
            split="val"
        )
    )
    # 2 test flows (1 normal, 1 anomalous)
    flows.append(
        CanonicalFlow(
            entity_id="192.168.1.10",
            timestamp=14.0,
            dt=2.0,
            categorical_fields={"proto": "tcp", "service": "http", "state": "CON"},
            numerical_fields={"dur": 0.7, "spkts": 16.0, "dpkts": 19.0, "sbytes": 7000.0, "dbytes": 10500.0},
            label=0,
            split="test"
        )
    )
    flows.append(
        CanonicalFlow(
            entity_id="192.168.1.10",
            timestamp=16.0,
            dt=2.0,
            categorical_fields={"proto": "tcp", "service": "private", "state": "REJ"},
            numerical_fields={"dur": 0.0, "spkts": 2.0, "dpkts": 0.0, "sbytes": 120.0, "dbytes": 0.0},
            label=1,
            split="test"
        )
    )
    return flows


@pytest.fixture
def sample_sequence_tensors() -> Dict[str, Any]:
    """Generates standard synthetic tensors for sequence models."""
    batch_size = 4
    seq_len = 6
    input_dim = 64
    hidden_dim = 32

    x = torch.randn(batch_size, seq_len, input_dim)
    dt = torch.rand(batch_size, seq_len) + 0.05
    h0 = torch.zeros(batch_size, hidden_dim)

    return {
        "x": x,
        "dt": dt,
        "h0": h0,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
    }
