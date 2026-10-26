import torch
import ncps.torch

# Save the original forward method of CfCCell
_orig_forward = ncps.torch.CfCCell.forward

def _patched_forward(self, input, hx, ts):
    """Patches ncps CfCCell.forward to correctly handle 1D and scalar timespan (ts) tensor broadcasting
    across any batch size.
    """
    if isinstance(ts, torch.Tensor):
        if ts.ndim == 1:
            ts = ts.unsqueeze(1)
        elif ts.ndim == 0:
            ts = ts.view(1, 1)
        ts = torch.clamp(ts, min=0.0)
    elif isinstance(ts, (float, int)):
        ts = max(0.0, float(ts))
    return _orig_forward(self, input, hx, ts)

# Dynamic monkey patch
ncps.torch.CfCCell.forward = _patched_forward

# Re-export CfCCell
CfCCell = ncps.torch.CfCCell

__all__ = ["CfCCell"]
