"""
Scale-free, query-level evidence about how the two channels behaved on a pool.

Motivation
----------
The controller that produces the fusion weight alpha sees only the query
embedding (``ControllerMLP.forward(query_emb)``). Alpha is therefore a function
of query semantics alone, fitted on PubMedQA/MedMCQA embeddings and
extrapolated onto a new collection. Two things follow, both measured:

* alpha shifts systematically downward out-of-domain on all four back-ends, and
  the size of that miscalibration predicts the sign of the adaptivity loss;
* the branches' output scales themselves drift out-of-domain (sd ratio doubles
  on BM25+FAISS and SPLADE+++FAISS -- exactly the two back-ends where per-query
  alpha subtracts value), so an alpha calibrated to in-domain scales is invalid
  on a new collection.

Every feature here is invariant to a positive affine rescaling ``x -> a*x + c``
(a > 0) of *either* channel independently: each channel's scores are min-max
normalised within the pool before any shape statistic is taken, agreement
features use ranks only, and the two cross-channel comparisons are expressed as
ratios in [0, 1]. A controller reading these cannot be miscalibrated by a score
distribution shift, because it never sees a raw score.

These are computed from the pool the model is scored on, at both train and eval
time, by the same function -- so there is no train/test skew and no regenerated
data. Note that :meth:`StreamingRankDataset._emit_pool_for_group` subsamples the
pool for the listwise loss; features must be computed on the **full** group
before that subsampling, which is what the call site does.
"""

from __future__ import annotations

import numpy as np

POOL_FEATURE_NAMES: tuple[str, ...] = (
    # --- channel agreement: when the channels rank alike, alpha cannot matter
    "rank_spearman",
    "overlap_at_10",
    "jaccard_at_20",
    # --- sparse channel shape (min-max normalised within pool)
    "sparse_top_gap",
    "sparse_rel_gap12",
    "sparse_dispersion",
    "sparse_entropy",
    "sparse_skew",
    "sparse_top10_mass",
    # --- dense channel shape (min-max normalised within pool)
    "dense_top_gap",
    "dense_rel_gap12",
    "dense_dispersion",
    "dense_entropy",
    "dense_skew",
    "dense_top10_mass",
    # --- pool composition and cross-channel balance
    "sparse_coverage",
    "dense_coverage",
    "pool_size_norm",
    "dispersion_ratio",
    "gap_ratio",
)

POOL_FEATURE_DIM = len(POOL_FEATURE_NAMES)

_EPS = 1e-8


def _minmax(x: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]; a constant channel maps to all-zeros."""
    lo = float(x.min())
    hi = float(x.max())
    if hi - lo < _EPS:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared (Spearman needs tie correction)."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    # average tied ranks
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i + 1
        while j < len(sx) and sx[j] == sx[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def _shape_stats(scores: np.ndarray, scored: np.ndarray) -> tuple[float, ...]:
    """
    Six scale-free shape statistics for one channel.

    Only candidates this channel actually retrieved contribute: a union pool
    leaves ~43% of candidates unscored by one channel, and those carry a
    floor-imputed value that is a marker, not a measurement.
    """
    vals = scores[scored]
    if vals.size < 2:
        return (0.0,) * 6

    v = _minmax(vals.astype(np.float64))
    srt = np.sort(v)[::-1]

    top_gap = float(srt[0] - srt[1])
    # How much of the leader's margin is "the top one is special" vs "the whole
    # head is flat": near 1 when the top document stands alone.
    denom = float(srt[0] - srt[2]) if srt.size >= 3 else float(srt[0] - srt[1])
    rel_gap12 = float(top_gap / denom) if denom > _EPS else 0.0

    dispersion = float(v.std())

    # Softmax entropy over the normalised scores, divided by log(n) so pools of
    # different sizes are comparable. 1.0 = the channel has no opinion.
    e = np.exp(v - v.max())
    p = e / max(float(e.sum()), _EPS)
    entropy = float(-(p * np.log(p + _EPS)).sum() / max(np.log(len(p)), _EPS))

    sd = float(v.std())
    skew = float((((v - v.mean()) / sd) ** 3).mean()) if sd > _EPS else 0.0

    total = float(v.sum())
    top10_mass = float(srt[: min(10, srt.size)].sum() / total) if total > _EPS else 0.0

    return (top_gap, rel_gap12, dispersion, entropy, skew, top10_mass)


def pool_features(
    sparse_scores: np.ndarray,
    dense_scores: np.ndarray,
    sparse_retrieved: np.ndarray,
    dense_retrieved: np.ndarray,
) -> np.ndarray:
    """
    Build the ``POOL_FEATURE_DIM``-vector describing one query's candidate pool.

    Args:
        sparse_scores / dense_scores: raw per-channel scores over the full pool.
        sparse_retrieved / dense_retrieved: 1.0 where that channel actually
            retrieved the candidate (schema dim 7), 0.0 where the stored score
            is a floor-imputed marker.

    Returns:
        float32 array of shape ``(POOL_FEATURE_DIM,)``, invariant to any
        positive affine rescaling of either channel.
    """
    s = np.asarray(sparse_scores, dtype=np.float64).ravel()
    d = np.asarray(dense_scores, dtype=np.float64).ravel()
    s_ok = np.asarray(sparse_retrieved).ravel().astype(bool)
    d_ok = np.asarray(dense_retrieved).ravel().astype(bool)
    n = s.size

    if n == 0:
        return np.zeros(POOL_FEATURE_DIM, dtype=np.float32)

    # --- agreement, computed only where both channels have a real opinion
    both = s_ok & d_ok
    if int(both.sum()) >= 3:
        rs = _rankdata(s[both])
        rd = _rankdata(d[both])
        sd_rs, sd_rd = rs.std(), rd.std()
        if sd_rs > _EPS and sd_rd > _EPS:
            spearman = float(((rs - rs.mean()) * (rd - rd.mean())).mean() / (sd_rs * sd_rd))
        else:
            spearman = 0.0
    else:
        spearman = 0.0

    def _topk(x: np.ndarray, ok: np.ndarray, k: int) -> set:
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            return set()
        return set(idx[np.argsort(x[idx])[::-1][:k]].tolist())

    s10, d10 = _topk(s, s_ok, 10), _topk(d, d_ok, 10)
    overlap10 = len(s10 & d10) / 10.0

    s20, d20 = _topk(s, s_ok, 20), _topk(d, d_ok, 20)
    union20 = len(s20 | d20)
    jaccard20 = len(s20 & d20) / union20 if union20 else 0.0

    sparse_shape = _shape_stats(s, s_ok)
    dense_shape = _shape_stats(d, d_ok)

    sparse_cov = float(s_ok.mean())
    dense_cov = float(d_ok.mean())
    pool_norm = float(n) / 100.0

    # Cross-channel balance as [0,1] shares: scale-free by construction, and
    # 0.5 means "the two channels are equally peaked / equally decisive".
    sd_s, sd_d = sparse_shape[2], dense_shape[2]
    dispersion_ratio = float(sd_s / (sd_s + sd_d)) if (sd_s + sd_d) > _EPS else 0.5
    g_s, g_d = sparse_shape[0], dense_shape[0]
    gap_ratio = float(g_s / (g_s + g_d)) if (g_s + g_d) > _EPS else 0.5

    out = np.array(
        (spearman, overlap10, jaccard20)
        + sparse_shape
        + dense_shape
        + (sparse_cov, dense_cov, pool_norm, dispersion_ratio, gap_ratio),
        dtype=np.float32,
    )
    assert out.size == POOL_FEATURE_DIM, (out.size, POOL_FEATURE_DIM)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def pool_features_from_records(records, sparse_score_col: int = 0,
                               dense_score_col: int = 0) -> np.ndarray:
    """Convenience wrapper over raw rank-JSONL records (full 8+8 vectors)."""
    s = np.fromiter((r["sparse_feats"][sparse_score_col] for r in records),
                    dtype=np.float64, count=len(records))
    d = np.fromiter((r["dense_feats"][dense_score_col] for r in records),
                    dtype=np.float64, count=len(records))
    s_ok = np.fromiter((r["sparse_feats"][7] for r in records),
                       dtype=np.float64, count=len(records))
    d_ok = np.fromiter((r["dense_feats"][7] for r in records),
                       dtype=np.float64, count=len(records))
    return pool_features(s, d, s_ok, d_ok)
