"""Global seed management for full reproducibility across random, numpy, and PyTorch."""

import random
import numpy as np


def set_global_seed(seed: int) -> None:
    """Set global seed for Python random, NumPy, and PyTorch (including CUDA if available).

    Args:
        seed: The integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
