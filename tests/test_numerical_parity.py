import pytest
import torch
from hdlnn.lnn.model import LNNSequenceModel

def test_batch_streaming_numerical_parity():
    # Setup hyperparameters
    input_dim = 128
    hidden_dim = 64
    batch_size = 4
    seq_len = 5
    
    # Set PyTorch seed to ensure reproducibility
    torch.manual_seed(42)
    
    # Initialize the model
    model = LNNSequenceModel(input_dim=input_dim, hidden_dim=hidden_dim)
    model.eval()  # Put in evaluation mode

    # Generate random input sequence x and time deltas dt
    x = torch.randn(batch_size, seq_len, input_dim)
    dt = torch.rand(batch_size, seq_len) + 0.1  # Ensure positive dt values

    # Initial state (zeros)
    h0 = torch.zeros(batch_size, hidden_dim)

    # 1. Evaluate batch forward pass
    with torch.no_grad():
        batch_outputs, batch_h_last = model.forward(x, h0=h0, dt=dt)

    # 2. Evaluate streaming step-by-step pass
    streaming_outputs = []
    h_prev = h0.clone()
    
    with torch.no_grad():
        for t in range(seq_len):
            x_t = x[:, t]  # Shape: [batch_size, input_dim]
            dt_t = dt[:, t]  # Shape: [batch_size]
            
            # Step evaluation
            h_next = model.step(x_t, h_prev, dt_t)
            streaming_outputs.append(h_next)
            h_prev = h_next

    # Stack streaming outputs into [batch_size, seq_len, hidden_dim]
    # In step() we get h_next of shape [batch_size, hidden_dim].
    # Stacking them along dim 1 gives [batch_size, seq_len, hidden_dim]
    streaming_outputs_tensor = torch.stack(streaming_outputs, dim=1)

    # 3. Assert absolute numerical parity
    # We compare the outputs at each step and the final hidden state
    assert torch.allclose(batch_outputs, streaming_outputs_tensor, atol=1e-6, rtol=1e-5), (
        "Batch outputs and streaming outputs do not match!"
    )
    
    assert torch.allclose(batch_h_last, h_prev, atol=1e-6, rtol=1e-5), (
        "Final batch hidden state and streaming hidden state do not match!"
    )


def test_batch_cfc_timespan_broadcasting():
    """Verifies that CfC step handles 1D, 2D, and scalar timespans across any batch size (batch != hidden_dim)."""
    input_dim = 64
    hidden_dim = 32
    batch_size = 7  # 7 != 32, which triggers non-singleton dimension 1 mismatch if unpatched
    
    torch.manual_seed(42)
    model = LNNSequenceModel(input_dim=input_dim, hidden_dim=hidden_dim)
    model.eval()

    x = torch.randn(batch_size, input_dim)
    h_prev = torch.randn(batch_size, hidden_dim)

    # 1. 1-D timespan tensor [B]
    dt_1d = torch.rand(batch_size) + 0.1
    h_next_1d = model.step(x, h_prev, dt_1d)
    assert h_next_1d.shape == (batch_size, hidden_dim)
    assert not torch.isnan(h_next_1d).any()

    # 2. 2-D timespan tensor [B, 1]
    dt_2d = dt_1d.unsqueeze(1)
    h_next_2d = model.step(x, h_prev, dt_2d)
    assert h_next_2d.shape == (batch_size, hidden_dim)
    assert torch.allclose(h_next_1d, h_next_2d, atol=1e-6)

    # 3. Scalar float timespan
    dt_scalar = 0.25
    h_next_scalar = model.step(x, h_prev, dt_scalar)
    assert h_next_scalar.shape == (batch_size, hidden_dim)
    assert not torch.isnan(h_next_scalar).any()

    # 4. 0-D scalar tensor
    dt_0d = torch.tensor(0.5)
    h_next_0d = model.step(x, h_prev, dt_0d)
    assert h_next_0d.shape == (batch_size, hidden_dim)
    assert not torch.isnan(h_next_0d).any()


def test_irregular_delta_t_edge_cases():
    """Verifies robustness under zero, negative, and extreme delta-t values."""
    input_dim = 64
    hidden_dim = 32
    torch.manual_seed(42)
    model = LNNSequenceModel(input_dim=input_dim, hidden_dim=hidden_dim)
    model.eval()

    x = torch.randn(2, input_dim)
    h_prev = torch.randn(2, hidden_dim)

    # 1. Zero dt (instantaneous jump condition)
    dt_zero = torch.zeros(2)
    h_zero = model.step(x, h_prev, dt_zero)
    assert h_zero.shape == (2, hidden_dim)
    assert not torch.isnan(h_zero).any()
    assert not torch.isinf(h_zero).any()

    # 2. Negative dt (must clamp to 0.0 defensively without error)
    dt_neg = torch.tensor([-2.5, -0.01])
    h_neg = model.step(x, h_prev, dt_neg)
    assert h_neg.shape == (2, hidden_dim)
    assert torch.allclose(h_zero, h_neg, atol=1e-6), "Negative dt should clamp to 0.0 and match zero-dt state!"

    # 3. Extreme small dt (e.g. 1e-7 s)
    dt_small = torch.tensor([1e-7, 1e-8])
    h_small = model.step(x, h_prev, dt_small)
    assert torch.allclose(h_zero, h_small, atol=1e-4), "Extreme small dt should closely approximate zero dt!"

    # 4. Extreme large dt (e.g. 1e6 s, saturates sigmoid)
    dt_large = torch.tensor([1e6, 1e7])
    h_large = model.step(x, h_prev, dt_large)
    assert h_large.shape == (2, hidden_dim)
    assert not torch.isnan(h_large).any()
    assert not torch.isinf(h_large).any()

    # 5. Batch forward with mixed extreme/negative dt
    x_seq = torch.randn(2, 4, input_dim)
    dt_seq = torch.tensor([
        [-1.0, 0.0, 0.5, 1e5],
        [0.0, -0.2, 1e-6, 1e6]
    ])
    seq_out, h_last = model.forward(x_seq, dt=dt_seq)
    assert seq_out.shape == (2, 4, hidden_dim)
    assert h_last.shape == (2, hidden_dim)
    assert not torch.isnan(seq_out).any()
    assert not torch.isnan(h_last).any()
