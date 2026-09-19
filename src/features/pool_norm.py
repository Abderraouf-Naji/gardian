"""
Within-pool normalisation of first-stage retrieval scores.

Why this exists
---------------
BM25 and dense similarity live on incompatible scales, and each one's scale
shifts from query to query. Measured on the MedMCQA pool of the BM25+FAISS
back-end: BM25 has a per-query standard deviation of ~3.3 on an unbounded
scale, while FAISS cosine sits in [0, 1] with a per-query sd of ~0.27. A model
handed the raw numbers cannot tell whether a BM25 score of 5.0 is excellent or
poor for the query in front of it.

The consequence was measurable. With raw features a gradient-boosted ranker
scored 0.4475 nDCG@10 on MedMCQA; with the normalisations below it scored
0.4789, and a linear model reached 0.4861 -- above a dev-tuned global-alpha
fusion (0.4746). Normalisation was worth 2.4 to 5.3 nDCG points depending on
back-end, which is larger than any adaptivity effect measured on this data.

All functions here operate on ONE query's candidate pool and are pure: same
input, same output, no global state.
"""

from __future__ import annotations

import numpy as np

# Guards against a degenerate pool where every candidate scores identically.
_EPS = 1e-12

# A candidate found by only one channel of a union pool carries no score from
# the other. The pipeline records that as 0.0, which is a MISSING marker rather
# than a measured value: on real data 43.4% of candidates are exactly 0.0 while
# the smallest genuinely measured dense similarity is 0.347. Treating the marker
# as a real score drags the channel's statistics down and shifts the optimal
# fusion weight (measured: best global alpha moved 0.12 -> 0.44 once the marker
# was handled correctly, and nDCG@10 rose 0.7449 -> 0.7530).
MISSING_SCORE = 0.0


def is_scored(scores: np.ndarray) -> np.ndarray:
    """Boolean mask of candidates this channel actually scored."""
    return np.asarray(scores, dtype=np.float64) != MISSING_SCORE


def normalise_scored_only(
    scores: np.ndarray,
    fn,
    *,
    missing_value: float = 0.0,
) -> np.ndarray:
    """
    Apply a normaliser over the SCORED candidates only.

    Unscored candidates are pinned to ``missing_value`` (0.0, the bottom of the
    normalised range) instead of participating in the statistics. This keeps a
    channel's mean, spread and rank ordering describing what it actually
    retrieved, while still ranking unscored candidates last on that channel.

    Paired with the ``retrieved_by_*`` indicator feature, the model can tell
    "this channel ranked it last" apart from "this channel never saw it".
    """
    x = np.asarray(scores, dtype=np.float64)
    mask = is_scored(x)
    out = np.full(x.shape, float(missing_value), dtype=np.float64)
    if not mask.any():
        return out
    out[mask] = fn(x[mask])
    return out


def impute_missing_at_floor(scores: np.ndarray) -> np.ndarray:
    """
    Replace the missing marker with the channel's lowest observed score.

    A candidate outside a channel's top-k scores *below that channel's cutoff*,
    so the smallest score the channel did return is a far better estimate than
    0.0 -- much closer, and never above the true value by more than the width of
    the unobserved tail.
    """
    x = np.asarray(scores, dtype=np.float64).copy()
    mask = is_scored(x)
    if not mask.any():
        return x
    x[~mask] = float(x[mask].min())
    return x


def minmax_normalise(scores: np.ndarray) -> np.ndarray:
    """
    Rescale a pool's scores to [0, 1].

        x' = (x - min) / (max - min)

    Returns all zeros when the pool is constant, which is the correct neutral
    value: no candidate is preferred on a channel that cannot separate them.
    This is the normalisation used by the global-alpha fusion baseline, so the
    learned model sees exactly the quantity the baseline optimises.
    """
    x = np.asarray(scores, dtype=np.float64)
    lo, hi = float(x.min()), float(x.max())
    span = hi - lo
    if span < _EPS:
        return np.zeros_like(x)
    return (x - lo) / span


def zscore_normalise(scores: np.ndarray) -> np.ndarray:
    """
    Standardise a pool's scores to zero mean and unit variance.

        x' = (x - mean) / sd

    Complements :func:`minmax_normalise`: min-max is bounded but sensitive to a
    single outlying top score, whereas the z-score preserves relative spacing.
    Zeros for a constant pool.
    """
    x = np.asarray(scores, dtype=np.float64)
    sd = float(x.std())
    if sd < _EPS:
        return np.zeros_like(x)
    return (x - float(x.mean())) / sd


def rank_normalise(scores: np.ndarray) -> np.ndarray:
    """
    Position of each candidate in the pool, mapped to [0, 1] with 1 = best.

        x' = 1 - rank / (n - 1)          rank 0 is the highest score

    Completely scale-free, so it is unaffected by the score distribution's shape
    and by outliers. This is the feature that lets one branch express "this
    candidate is top-3 for my channel" independently of the channel's units.
    Ties are broken by the stable ordering of ``argsort``.
    """
    x = np.asarray(scores, dtype=np.float64)
    n = x.size
    if n == 0:
        return x
    if n == 1:
        return np.ones(1, dtype=np.float64)
    order = np.argsort(-x, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return 1.0 - ranks / (n - 1)


def pool_dispersion(scores: np.ndarray) -> float:
    """
    Standard deviation of a channel's scores over the pool.

    A query-level quantity (identical for every candidate of the query) that
    says how sharply this channel separates its own pool. A near-zero value
    means the channel found nothing distinctive for this query.
    """
    x = np.asarray(scores, dtype=np.float64)
    return float(x.std()) if x.size else 0.0


def top_score_gap(scores: np.ndarray) -> float:
    """
    Normalised margin between the best and second-best score in the pool.

        (x_(1) - x_(2)) / (x_(1) - x_(n))

    A channel-confidence signal in [0, 1]: a large gap means the channel is sure
    about its top candidate. Zero for pools with fewer than two candidates or no
    spread.
    """
    x = np.asarray(scores, dtype=np.float64)
    if x.size < 2:
        return 0.0
    s = np.sort(x)[::-1]
    span = float(s[0] - s[-1])
    if span < _EPS:
        return 0.0
    return float((s[0] - s[1]) / span)
