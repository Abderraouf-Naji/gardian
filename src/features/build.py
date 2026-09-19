"""
The single construction of the 8+8 branch feature vectors.

Both feature paths must agree or the model is scored at inference on a
different representation than it was trained on. They had drifted: the offline
builder (``scripts/03_generate_rank_data.py``) produced the full 8+8 layout
declared in :mod:`src.features.schema`, while the live QA path in
``src/evaluation/qa_eval.py`` still produced the pre-revision 3+4 vectors and
appended none of the within-pool normalisations. The 8-dim model then indexed
past the end of a 3-element list, which is how the drift finally surfaced.

The within-pool dims are exactly the ones that cannot be computed per
candidate: min-max, z-score, rank and dispersion are defined over the pool a
query retrieved, and the retrieval indicator records which channel actually
returned the candidate. Those four normalisations carry 77% of tree-attributed
feature gain (docs/CHANGES_SINCE_COOPIS.md §5), so a live path missing them is
not a slightly degraded model, it is a different one.

Everything here is pure: the caller supplies the candidate pool and, for the
embedding-based dense dims, the query/passage embeddings. ``pool_stats``
matches the ``pool_stats`` block written into rank JSONL.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.features.dense_feat import compute_dense_features_with_score
from src.features.pool_norm import (
    impute_missing_at_floor,
    is_scored,
    minmax_normalise,
    normalise_scored_only,
    pool_dispersion,
    rank_normalise,
    top_score_gap,
    zscore_normalise,
)
from src.features.schema import DENSE_FEAT_DIM, SPARSE_FEAT_DIM
from src.features.sparse import compute_sparse_features

# Hybrid families whose sparse channel is BM25 rather than SPLADE++.
_BM25_FAMILIES = frozenset(
    {"bm25", "hybrid_bm25_faiss", "hybrid_bm25_medcpt"}
)
# Families whose dense channel is MedCPT rather than FAISS/PubMedBERT.
_MEDCPT_FAMILIES = frozenset(
    {"medcpt", "hybrid_spladepp_medcpt", "hybrid_bm25_medcpt"}
)


def sparse_signal_scores(
    candidates: Sequence[Dict[str, Any]], retriever_type: str
) -> List[float]:
    """The score the sparse branch consumes, per candidate."""
    if retriever_type in _BM25_FAMILIES:
        return [float(c.get("bm25_score", 0.0)) for c in candidates]
    return [float(c.get("spladepp_score", 0.0)) for c in candidates]


def dense_signal_scores(
    candidates: Sequence[Dict[str, Any]], retriever_type: str
) -> List[float]:
    """
    The score the dense branch consumes, per candidate.

    Kept aligned with the dense retriever of each hybrid family: the FAISS
    score for FAISS hybrids, the MedCPT asymmetric dot product for MedCPT ones.
    """
    out: List[float] = []
    for c in candidates:
        if retriever_type == "bm25":
            out.append(float(c.get("bm25_score", c.get("score", 0.0))))
        elif retriever_type == "spladepp":
            out.append(float(c.get("spladepp_score", c.get("score", 0.0))))
        elif retriever_type in _MEDCPT_FAMILIES:
            out.append(
                float(c.get("medcpt_score", c.get("dense_score", c.get("score", 0.0))))
            )
        else:
            out.append(float(c.get("dense_score", c.get("score", 0.0))))
    return out


def build_branch_features(
    *,
    question: str,
    candidates: Sequence[Dict[str, Any]],
    retriever_type: str,
    idf_table: Optional[Any] = None,
    q_dense: Optional[np.ndarray] = None,
    p_embs: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[List[List[float]], List[List[float]], Dict[str, float]]:
    """
    Build the 8-dim sparse and 8-dim dense vectors for one query's whole pool.

    Parameters
    ----------
    q_dense, p_embs
        Query and passage embeddings for dense dims 1-2 (``mean_abs_diff``,
        ``max_abs_diff``). When either is omitted the pair is scored from the
        retriever's own dense score alone and those two dims are 0.0 -- the
        ``use_dense_scores`` mode of the offline builder.

    Returns
    -------
    ``(sparse_feats, dense_feats, pool_stats)`` with one row per candidate, in
    the candidate order given. Layout is :mod:`src.features.schema`.
    """
    n = len(candidates)
    if n == 0:
        return [], [], {}

    sparse_raw = np.asarray(
        sparse_signal_scores(candidates, retriever_type), dtype=np.float64
    )
    dense_signal = dense_signal_scores(candidates, retriever_type)
    dense_raw = np.asarray(dense_signal, dtype=np.float64)

    # A union pool takes top-k from each channel, so a candidate found by only
    # one carries no score from the other. 0.0 is a MISSING marker, not a
    # measurement: normalise over the scored candidates only, pin unscored ones
    # to the bottom of each range, and impute the raw score at the channel floor.
    sparse_scored = is_scored(sparse_raw)
    dense_scored = is_scored(dense_raw)
    sparse_arr = impute_missing_at_floor(sparse_raw)
    dense_arr = impute_missing_at_floor(dense_raw)

    sparse_minmax = normalise_scored_only(sparse_raw, minmax_normalise)
    sparse_z = normalise_scored_only(sparse_raw, zscore_normalise)
    sparse_rank = rank_normalise(sparse_arr)
    dense_minmax = normalise_scored_only(dense_raw, minmax_normalise)
    dense_z_norm = normalise_scored_only(dense_raw, zscore_normalise)
    dense_rank = rank_normalise(dense_arr)

    sparse_scored_vals = sparse_raw[sparse_scored]
    dense_scored_vals = dense_raw[dense_scored]
    pool_stats: Dict[str, float] = {
        "sparse_dispersion": pool_dispersion(sparse_scored_vals),
        "dense_dispersion": pool_dispersion(dense_scored_vals),
        "sparse_top_gap": top_score_gap(sparse_scored_vals),
        "dense_top_gap": top_score_gap(dense_scored_vals),
        "pool_size": int(n),
        "sparse_coverage": float(sparse_scored.mean()),
        "dense_coverage": float(dense_scored.mean()),
    }

    dense_score_mean = float(np.mean(dense_signal))
    dense_score_std = float(np.std(dense_signal) + 1e-8)
    use_embeddings = q_dense is not None and p_embs is not None

    sparse_out: List[List[float]] = []
    dense_out: List[List[float]] = []
    for i, cand in enumerate(candidates):
        sf = compute_sparse_features(
            query=question,
            passage=cand.get("text", "") or "",
            bm25_score=float(sparse_raw[i]),
            idf_table=idf_table,
        ).tolist()

        if use_embeddings:
            df = compute_dense_features_with_score(
                q_emb=q_dense,
                p_emb=p_embs[i],
                dense_score=dense_signal[i],
                score_mean=dense_score_mean,
                score_std=dense_score_std,
            ).tolist()
        else:
            # [score, 0, 0, z(score)] keeps the base 4-dim shape without an
            # embedding pass, exactly as the offline builder's score-only mode.
            z = (dense_signal[i] - dense_score_mean) / (dense_score_std + 1e-8)
            df = [float(dense_signal[i]), 0.0, 0.0, float(z)]

        # Dim 0 carries the floor-imputed score so no branch sees the 0.0
        # marker as a measurement; dense dim 3 is the pooled z-score.
        sf[0] = float(sparse_arr[i])
        df[0] = float(dense_arr[i])
        df[3] = float(dense_z_norm[i])

        sf = sf + [
            float(sparse_minmax[i]),
            float(sparse_z[i]),
            float(sparse_rank[i]),
            float(pool_stats["sparse_dispersion"]),
            1.0 if sparse_scored[i] else 0.0,
        ]
        df = df + [
            float(dense_minmax[i]),
            float(dense_rank[i]),
            float(pool_stats["dense_dispersion"]),
            1.0 if dense_scored[i] else 0.0,
        ]

        if len(sf) != SPARSE_FEAT_DIM or len(df) != DENSE_FEAT_DIM:
            raise ValueError(
                f"Built {len(sf)}+{len(df)} features, schema declares "
                f"{SPARSE_FEAT_DIM}+{DENSE_FEAT_DIM}. src/features/schema.py is "
                "the contract; update it and the config together."
            )
        sparse_out.append(sf)
        dense_out.append(df)

    return sparse_out, dense_out, pool_stats
