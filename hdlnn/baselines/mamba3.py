import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from hdlnn.contracts.interfaces import ISequenceModel

logger = logging.getLogger(__name__)


def heavy_tail(x: torch.Tensor) -> torch.Tensor:
    """Heavy-tail activation function as introduced in Mamba-3 (arxiv:2603.15569).
    
    Replaces standard exponential decay with a heavy-tail (rational polynomial) function:
      f(x) = 1 + x        if x >= 0
      f(x) = 1 / (1 - x)  if x < 0
      
    This is strictly positive, continuous at x=0, and maintains power-law tail retention
    for improved long-horizon state memory without exponential vanishing gradients.
    """
    pos = 1.0 + F.relu(x)
    neg = 1.0 / (1.0 + F.relu(-x))
    return torch.where(x >= 0, pos, neg)


class PurePyTorchMamba3SSM(nn.Module):
    """Pure-PyTorch Mamba-3 Selective State-Space block without custom CUDA/Triton dependencies.
    
    Implements Mamba-3 innovations (arxiv:2603.15569):
      1. Complex-valued state transitions h_t in C^{D_inner x N}
      2. Heavy-tail activation for state decay dynamics
      3. Rotary / complex phase modulation for B and C projections
      4. Contractive continuous-to-discrete discretization:
           Delta_A = exp(Delta * (A_real + i * A_imag))
           h_t = Delta_A * h_{t-1} + (Delta * B_t) * x_t
           y_t = Re(h_t * conj(C_t)) + D * x_t
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = dt_rank or math.ceil(self.d_model / 16)

        # 1. In-projection for x and gating branch z
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # 2. Causal 1D Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1
        )

        # 3. Parameter projections (Delta rank, B_real, B_imag, C_real, C_imag)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 4, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize Delta projection & bias
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # 4. Complex Diagonal A parameter:
        # A_real governs decay rate (strictly negative via heavy_tail)
        # A_imag governs rotational frequency (phase modulation)
        A_r = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_real_param = nn.Parameter(torch.log(A_r))

        # Log-spaced frequencies from 0.1*pi to pi (RoPE-style frequency grid)
        freqs = math.pi * torch.linspace(0.1, 1.0, self.d_state).repeat(self.d_inner, 1)
        self.A_imag_param = nn.Parameter(freqs)

        # Direct feedthrough D
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 5. Out-projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def _selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A_complex: torch.Tensor,
        B_complex: torch.Tensor,
        C_complex: torch.Tensor,
        D: torch.Tensor,
        h0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequential selective scan loop with complex-valued state evolution in pure PyTorch.

        Args:
            u: Input activations [B, L, D_inner] (float32)
            delta: Selective step size [B, L, D_inner] (float32)
            A_complex: Continuous diagonal state matrix [D_inner, N] (complex64)
            B_complex: Input projection [B, L, N] (complex64)
            C_complex: State-to-output projection [B, L, N] (complex64)
            D: Feedthrough skip parameter [D_inner] (float32)
            h0: Optional initial state [B, D_inner, N] (complex64)

        Returns:
            Tuple of (y [B, L, D_inner] float32, h_last [B, D_inner, N] complex64)
        """
        batch_size, seq_len, d_inner = u.shape
        d_state = self.d_state

        # Discretize continuous complex A:
        # delta_A = exp(delta * A_complex) -> [B, L, D_inner, N] (complex64)
        # Magnitude is exp(delta * A_real) <= 1; Phase is delta * A_imag
        delta_A = torch.exp(delta.unsqueeze(-1).to(torch.complex64) * A_complex.unsqueeze(0).unsqueeze(0))

        # Discretize B: delta_B_u = delta * u * B -> [B, L, D_inner, N] (complex64)
        delta_u = (delta.unsqueeze(-1) * u.unsqueeze(-1)).to(torch.complex64)
        delta_B_u = delta_u * B_complex.unsqueeze(2)

        # Initialize complex state
        if h0 is not None and h0.shape == (batch_size, d_inner, d_state):
            h = h0.to(torch.complex64)
        else:
            h = torch.zeros((batch_size, d_inner, d_state), device=u.device, dtype=torch.complex64)

        ys = []
        for t in range(seq_len):
            h = delta_A[:, t] * h + delta_B_u[:, t]
            # Output projection: y_t = Re(sum_n(h_t * conj(C_t)))
            C_t = C_complex[:, t].unsqueeze(1)  # [B, 1, N]
            y_t = (h * C_t.conj()).sum(dim=-1).real  # [B, D_inner]
            ys.append(y_t)

        y = torch.stack(ys, dim=1) + u * D.unsqueeze(0).unsqueeze(0)
        return y, h

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs Mamba-3 selective SSM forward pass.

        Args:
            x: Input sequence [B, L, d_model]
            h0: Optional initial complex state [B, d_inner, d_state]

        Returns:
            Tuple of (out [B, L, d_model], state_repr [B, 2 * d_inner * d_state])
        """
        batch_size, seq_len, _ = x.shape

        # 1. Project to [B, L, 2 * d_inner] and split into x_inner and gate z
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        # 2. 1D Causal Conv
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # 3. Project SSM parameters
        ssm_params = self.x_proj(x_conv)
        dt_param, B_real, B_imag, C_real, C_imag = torch.split(
            ssm_params,
            [self.dt_rank, self.d_state, self.d_state, self.d_state, self.d_state],
            dim=-1
        )
        B_complex = torch.complex(B_real, B_imag)
        C_complex = torch.complex(C_real, C_imag)

        # 4. Compute selective step size delta with softplus
        delta = F.softplus(self.dt_proj(dt_param))

        # 5. Form continuous complex diagonal transition matrix A
        # Real part uses heavy-tail decay: strictly negative
        A_real = -heavy_tail(self.A_real_param)
        A_imag = self.A_imag_param
        A_complex = torch.complex(A_real, A_imag)

        # 6. Selective scan with complex state
        y, h_last = self._selective_scan(x_conv, delta, A_complex, B_complex, C_complex, self.D, h0=h0)

        # 7. Gating and out-projection
        y_gated = y * F.silu(z)
        out = self.out_proj(y_gated)

        # 8. Return real-valued representation of final complex state for downstream evaluation/scoring
        # Flatten [B, d_inner, d_state] complex64 into [B, 2 * d_inner * d_state] float32
        h_real = torch.view_as_real(h_last).reshape(batch_size, -1)
        return out, h_real


class Mamba3SequenceModel(nn.Module, ISequenceModel):
    """Mamba-3 Selective State-Space Sequence Model conforming to ISequenceModel.

    Pure-PyTorch implementation running on CPU and GPU without custom CUDA/Triton compilation.
    Integrates heavy-tail decay and complex-valued SSM state recurrence.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        proj_dim: Optional[int] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.d_state = d_state
        self.expand = expand

        # Input adaptation layer
        if input_dim != hidden_dim:
            self.in_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.in_proj = nn.Identity()

        # Core Mamba-3 SSM block
        self.ssm = PurePyTorchMamba3SSM(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # Optional projection head
        if proj_dim:
            self.projection_head = nn.Linear(hidden_dim, proj_dim)
        else:
            self.projection_head = None

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates sequence transitions over time."""
        x_proj = self.in_proj(x)
        out, h_last = self.ssm(x_proj, h0=h0)
        return out, h_last

    def step(self, x: torch.Tensor, h_prev: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Evaluates continuous state transition for a single time step.

        Returns a composite tensor: [batch_size, hidden_dim + 2 * d_inner * d_state]
        where the first hidden_dim slice contains the output feature vector
        and the trailing slice contains the flattened real representation of the complex SSM state.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.size(0)
        x_seq = x.unsqueeze(1)  # [batch_size, 1, input_dim]

        d_inner = self.expand * self.hidden_dim
        # Complex state has 2 floats (real, imag) per element
        ssm_state_size = 2 * d_inner * self.d_state

        if h_prev.dim() == 1:
            h_prev = h_prev.unsqueeze(0)

        if h_prev.size(1) >= ssm_state_size:
            ssm_slice = h_prev[:, -ssm_state_size:]
            # Reshape [B, d_inner, d_state, 2] and convert back to complex64
            h_real_pairs = ssm_slice.view(batch_size, d_inner, self.d_state, 2).contiguous()
            h0 = torch.view_as_complex(h_real_pairs)
        else:
            h0 = None

        x_proj = self.in_proj(x_seq)
        out, h_last = self.ssm(x_proj, h0=h0)

        # Composite state: [out_features, ssm_state_flattened]
        out_feature = out[:, -1, :]  # [batch_size, hidden_dim]
        return torch.cat([out_feature, h_last], dim=-1)

    def predict_next_vector(self, state: torch.Tensor) -> torch.Tensor:
        """Projects hidden state sequence/vector back to hypervector space."""
        if self.projection_head is None:
            raise ValueError("Projection head not configured in Mamba3SequenceModel.")
        return self.projection_head(state)
