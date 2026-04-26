"""Reproducibility helpers for training and evaluation."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, *, cudnn_deterministic: bool = False) -> None:
    """
    Set RNG seeds for Python, NumPy, and PyTorch.

    ``cudnn_deterministic=True`` improves repeatability on GPU at some speed
    cost; keep False for maximum throughput unless reporting strict reruns.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
    torch.backends.cudnn.benchmark = not bool(cudnn_deterministic)
