"""Transformer-based Sequence Model Baselines for HDC-LNN Benchmark.

Includes:
1. TransformerEncoderBaseline: Standard multi-head self-attention bidirectional encoder.
2. MiniGPTBaseline: Causal autoregressive decoder-only Transformer with masked self-attention.

Both conform to ISequenceModel and support sub-1B parameter configurations suitable for
line-rate network telemetry anomaly detection benchmarks.
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from hdlnn.contracts.interfaces import ISequenceModel

logger = logging.getLogger(__name__)


def compute_model_size(model: nn.Module) -> Dict[str, Any]:
    """Computes exact parameter counts and memory footprint for a PyTorch module."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_bytes = param_bytes + buffer_bytes

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "size_bytes": total_bytes,
        "size_mb": total_bytes / (1024 * 1024),
        "size_gb": total_bytes / (1024 * 1024 * 1024),
    }


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as described in Vaswani et al. (2017)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to input tensor [B, L, D]."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerEncoderBaseline(nn.Module, ISequenceModel):
    """Vanilla Transformer Encoder baseline model.

    Utilizes multi-head self-attention across the temporal flow sequence.
    Conforms strictly to ISequenceModel with streaming step() execution.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        context_len: int = 16,
        proj_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len
        self.proj_dim = proj_dim

        # Input dimension adaptation
        if input_dim != hidden_dim:
            self.in_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.in_proj = nn.Identity()

        self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim, max_len=max(context_len * 4, 256))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(hidden_dim)

        if proj_dim:
            self.projection_head = nn.Linear(hidden_dim, proj_dim)
        else:
            self.projection_head = None

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates batch sequence forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            h0: Unused placeholder for compatibility.
            dt: Unused placeholder for compatibility.

        Returns:
            outputs: [batch_size, seq_len, hidden_dim]
            h_last: [batch_size, hidden_dim]
        """
        x_proj = self.in_proj(x)
        x_pos = self.pos_encoder(x_proj)
        encoded = self.transformer_encoder(x_pos)
        out = self.norm(encoded)
        h_last = out[:, -1, :]
        return out, h_last

    def step(self, x: torch.Tensor, h_prev: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Evaluates a single step by updating the sliding context window.

        Args:
            x: Current input [batch_size, input_dim] or [input_dim]
            h_prev: History tensor [batch_size, context_len * hidden_dim]
            dt: Unused placeholder for compatibility.

        Returns:
            h_next: Updated composite history + last state [batch_size, context_len * hidden_dim]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.size(0)

        history_len = self.context_len - 1
        history_dim = history_len * self.hidden_dim
        expected_state_dim = self.context_len * self.hidden_dim

        if h_prev.dim() == 1:
            h_prev = h_prev.unsqueeze(0)

        if h_prev.size(1) != expected_state_dim:
            history = torch.zeros((batch_size, history_dim), device=x.device, dtype=x.dtype)
        else:
            history = h_prev[:, :history_dim]

        # Project input to hidden_dim
        x_emb = self.in_proj(x)  # [batch_size, hidden_dim]

        # Reshape history to [batch_size, history_len, hidden_dim]
        history_seq = history.view(batch_size, history_len, self.hidden_dim)
        combined_seq = torch.cat([history_seq, x_emb.unsqueeze(1)], dim=1)  # [batch_size, context_len, hidden_dim]

        # Forward pass through Transformer
        x_pos = self.pos_encoder(combined_seq)
        encoded = self.transformer_encoder(x_pos)
        out = self.norm(encoded)
        h_last = out[:, -1, :]  # [batch_size, hidden_dim]

        # Slide history: drop oldest token, append current x_emb
        new_history = combined_seq[:, 1:, :].contiguous().view(batch_size, history_dim)
        h_next = torch.cat([new_history, h_last], dim=-1)
        return h_next

    def predict_next_vector(self, state: torch.Tensor) -> torch.Tensor:
        """Projects hidden representation back to hypervector dimension (proj_dim)."""
        if self.projection_head is None:
            raise ValueError("Projection head not configured in TransformerEncoderBaseline.")

        if state.dim() == 2 and state.size(1) > self.hidden_dim:
            state = state[:, -self.hidden_dim:]
        elif state.dim() == 3 and state.size(-1) > self.hidden_dim:
            state = state[:, :, -self.hidden_dim:]

        return self.projection_head(state)


class MiniGPTBaseline(nn.Module, ISequenceModel):
    """Causal Autoregressive (Decoder-only) Transformer baseline.

    Implements causal triangular masking so each sequence position can only attend
    to current and prior steps, matching GPT-2 architecture at a micro scale (<1B params).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        context_len: int = 16,
        proj_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len
        self.proj_dim = proj_dim

        if input_dim != hidden_dim:
            self.in_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.in_proj = nn.Identity()

        self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim, max_len=max(context_len * 4, 256))

        # Causal decoder layers (using TransformerEncoderLayer with causal mask)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(hidden_dim)

        if proj_dim:
            self.projection_head = nn.Linear(hidden_dim, proj_dim)
        else:
            self.projection_head = None

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generates an upper-triangular causal attention mask."""
        mask = torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)
        return mask

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates causal sequence forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            h0: Unused placeholder for compatibility.
            dt: Unused placeholder for compatibility.
        """
        seq_len = x.size(1)
        x_proj = self.in_proj(x)
        x_pos = self.pos_encoder(x_proj)

        causal_mask = self._generate_causal_mask(seq_len, x.device)
        encoded = self.decoder(x_pos, mask=causal_mask)
        out = self.norm(encoded)
        h_last = out[:, -1, :]
        return out, h_last

    def step(self, x: torch.Tensor, h_prev: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Evaluates a single autoregressive step."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.size(0)

        history_len = self.context_len - 1
        history_dim = history_len * self.hidden_dim
        expected_state_dim = self.context_len * self.hidden_dim

        if h_prev.dim() == 1:
            h_prev = h_prev.unsqueeze(0)

        if h_prev.size(1) != expected_state_dim:
            history = torch.zeros((batch_size, history_dim), device=x.device, dtype=x.dtype)
        else:
            history = h_prev[:, :history_dim]

        x_emb = self.in_proj(x)
        history_seq = history.view(batch_size, history_len, self.hidden_dim)
        combined_seq = torch.cat([history_seq, x_emb.unsqueeze(1)], dim=1)

        x_pos = self.pos_encoder(combined_seq)
        causal_mask = self._generate_causal_mask(self.context_len, x.device)
        encoded = self.decoder(x_pos, mask=causal_mask)
        out = self.norm(encoded)
        h_last = out[:, -1, :]

        new_history = combined_seq[:, 1:, :].contiguous().view(batch_size, history_dim)
        h_next = torch.cat([new_history, h_last], dim=-1)
        return h_next

    def predict_next_vector(self, state: torch.Tensor) -> torch.Tensor:
        """Projects hidden representation back to hypervector dimension (proj_dim)."""
        if self.projection_head is None:
            raise ValueError("Projection head not configured in MiniGPTBaseline.")

        if state.dim() == 2 and state.size(1) > self.hidden_dim:
            state = state[:, -self.hidden_dim:]
        elif state.dim() == 3 and state.size(-1) > self.hidden_dim:
            state = state[:, :, -self.hidden_dim:]

        return self.projection_head(state)
