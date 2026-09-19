"""Reproducibility helpers for training and evaluation.

The canonical entry point is :func:`set_seed`, which fixes every RNG the
pipeline touches and puts cuDNN into deterministic mode. Reviewers of the
GARDIAN paper asked for multi-seed results because reported margins are as
small as 0.1 pp; :mod:`src.common.seeds` builds the per-seed run layout on
top of this module.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, cudnn_deterministic: bool = True) -> int:
    """
    Seed every RNG used by the pipeline and force deterministic kernels.

    Seeds ``random``, ``numpy``, ``torch`` (CPU) and ``torch.cuda`` (all
    devices), sets ``PYTHONHASHSEED``, and configures cuDNN for reproducible
    convolution/reduction algorithm selection.

    Parameters
    ----------
    seed:
        The seed value. Applied verbatim to every RNG.
    cudnn_deterministic:
        When True (the default) sets ``torch.backends.cudnn.deterministic =
        True`` and ``benchmark = False``. Pass False only for throughput runs
        whose numbers are not reported in the paper.

    Returns
    -------
    int
        The seed that was applied, so callers can log it.

    Notes
    -----
    ``PYTHONHASHSEED`` is assigned here so that *child processes* (e.g. the
    ``ProcessPoolExecutor`` workers in ``scripts/10_paper_run.py``) inherit it.
    CPython fixes its own string-hash randomisation at interpreter start-up, so
    setting it in-process does not retroactively change this process's hashing.
    Nothing in the reported pipeline depends on ``str`` hash order -- all
    per-query aggregation is over explicitly sorted keys -- but the variable is
    exported for completeness and for subprocess inheritance.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
    torch.backends.cudnn.benchmark = not bool(cudnn_deterministic)
    return seed


def set_global_seed(seed: int, *, cudnn_deterministic: bool = False) -> int:
    """
    Backwards-compatible alias for :func:`set_seed`.

    Retained so existing call sites keep working. Note the historical default
    of ``cudnn_deterministic=False``; new code should call :func:`set_seed`,
    which defaults to deterministic.
    """
    return set_seed(seed, cudnn_deterministic=cudnn_deterministic)
