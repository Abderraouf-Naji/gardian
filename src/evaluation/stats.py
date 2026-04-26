"""Bootstrap statistics helpers for paper tables."""

from __future__ import annotations

from typing import Sequence, Tuple, Dict

import numpy as np


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Percentile bootstrap on the mean.

    Returns
    -------
    mean, lower, upper
        ``lower`` / ``upper`` are the ``(1-ci)/2`` and ``1-(1-ci)/2`` quantiles of
        the bootstrap distribution of the sample mean.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return 0.0, 0.0, 0.0
    if x.size == 1:
        m = float(x[0])
        return m, m, m

    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    n = x.size
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[i] = float(x[idx].mean())

    alpha = (1.0 - ci) / 2.0
    return float(x.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def format_ci(mean: float, lo: float, hi: float, decimals: int = 4) -> str:
    """Human-readable interval for LaTeX tables."""
    d = decimals
    return f"{mean:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def bootstrap_delta_ci(
    a_values: Sequence[float],
    b_values: Sequence[float],
    *,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Percentile bootstrap CI on paired mean delta: mean(a - b).
    """
    a = np.asarray(a_values, dtype=np.float64)
    b = np.asarray(b_values, dtype=np.float64)
    n = min(a.size, b.size)
    if n == 0:
        return 0.0, 0.0, 0.0
    a = a[:n]
    b = b[:n]
    d = a - b
    if n == 1:
        m = float(d[0])
        return m, m, m
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[i] = float(d[idx].mean())
    alpha = (1.0 - ci) / 2.0
    return float(d.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def paired_randomization_pvalue(
    a_values: Sequence[float],
    b_values: Sequence[float],
    *,
    n_trials: int = 10000,
    seed: int = 42,
) -> float:
    """
    Approximate two-sided paired randomization p-value for mean delta.
    """
    a = np.asarray(a_values, dtype=np.float64)
    b = np.asarray(b_values, dtype=np.float64)
    n = min(a.size, b.size)
    if n == 0:
        return 1.0
    d = (a[:n] - b[:n]).astype(np.float64)
    observed = abs(float(d.mean()))
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_trials):
        flips = rng.choice(np.array([1.0, -1.0]), size=n)
        if abs(float((d * flips).mean())) >= observed:
            hits += 1
    return float((hits + 1) / (n_trials + 1))


def summarize_metric_with_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    mean, lo, hi = bootstrap_mean_ci(values, n_bootstrap=n_bootstrap, ci=ci, seed=seed)
    return {"mean": mean, "ci_low": lo, "ci_high": hi}
