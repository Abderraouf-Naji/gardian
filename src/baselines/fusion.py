"""
Score-fusion baselines: global-alpha, oracle-alpha, oracle re-ranking, RRF.

All of these operate on the first-stage scores already stored in rank JSONL, so
none of them needs a trained model or a GPU. They exist so that any claim about
query-adaptive fusion is stated against the strongest non-adaptive alternative.

Normalisation
-------------
Fusion is performed on **within-pool min-max normalised** scores. Raw BM25 and
dense similarity are on incompatible, query-varying scales: adding them
unnormalised produces a "hybrid" that is really whichever channel happens to
have the larger numbers (measured: the dense channel contributes ~1.5% of the
score spread on BM25+FAISS, and dominates on BM25+MedCPT). ``sum_raw_scores``
reproduces that unnormalised fusion for continuity with the submitted paper, and
is deliberately named so nobody mistakes it for a tuned baseline.
"""

from __future__ import annotations

import collections
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.evaluation.metrics import ndcg_at_k
from src.features.pool_norm import minmax_normalise

# alpha_sparse grid; alpha_dense = 1 - alpha_sparse. Step 0.01 as specified for
# the paper's global-alpha search.
ALPHA_GRID: np.ndarray = np.round(np.arange(0.0, 1.0001, 0.01), 2)


def fuse(
    sparse_scores: Sequence[float],
    dense_scores: Sequence[float],
    alpha: float,
    *,
    normalise: bool = True,
) -> np.ndarray:
    """
    ``alpha * sparse + (1 - alpha) * dense`` on one query's pool.

    With ``normalise=True`` (the default and the only defensible setting) both
    channels are min-max scaled within the pool first.
    """
    s = np.asarray(sparse_scores, dtype=np.float64)
    d = np.asarray(dense_scores, dtype=np.float64)
    if normalise:
        s, d = minmax_normalise(s), minmax_normalise(d)
    return float(alpha) * s + (1.0 - float(alpha)) * d


def sum_raw_scores(
    sparse_scores: Sequence[float], dense_scores: Sequence[float]
) -> np.ndarray:
    """
    Unnormalised ``sparse + dense`` -- the "Sum" row of the submitted paper.

    Kept only for continuity. This is not a tuned baseline: because the two
    channels are on different scales, it is dominated by whichever has the
    larger dynamic range, and that differs by back-end.
    """
    return np.asarray(sparse_scores, dtype=np.float64) + np.asarray(
        dense_scores, dtype=np.float64
    )


def rrf_scores(
    sparse_scores: Sequence[float], dense_scores: Sequence[float], *, k: int = 60
) -> np.ndarray:
    """Reciprocal rank fusion: ``1/(k + rank_sparse) + 1/(k + rank_dense)``."""
    s = np.asarray(sparse_scores, dtype=np.float64)
    d = np.asarray(dense_scores, dtype=np.float64)
    n = s.size

    def ranks(x: np.ndarray) -> np.ndarray:
        order = np.argsort(-x, kind="stable")
        r = np.empty(n, dtype=np.float64)
        r[order] = np.arange(1, n + 1, dtype=np.float64)
        return r

    return 1.0 / (k + ranks(s)) + 1.0 / (k + ranks(d))


def _query_ndcg(
    pids: Sequence[str],
    labels: Sequence[int],
    scores: np.ndarray,
    k: int,
) -> float:
    order = np.argsort(-np.asarray(scores), kind="stable")
    relevant = [p for p, lab in zip(pids, labels) if int(lab) == 1]
    return ndcg_at_k([pids[i] for i in order[:k]], relevant, k)


def alpha_surface(
    pools: Dict[str, Dict[str, Any]],
    *,
    k: int = 10,
    grid: Optional[np.ndarray] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    nDCG@k of every query at every alpha.

    Returns ``(qids, surface)`` where ``surface[i, j]`` is query *i*'s nDCG at
    ``grid[j]``. Every other function here is a reduction of this matrix, so the
    expensive part is computed once.

    ``pools`` maps qid to ``{"pid": [...], "label": [...], "sparse": [...],
    "dense": [...]}``.
    """
    grid = ALPHA_GRID if grid is None else np.asarray(grid, dtype=np.float64)
    qids = [q for q, p in pools.items() if any(int(x) == 1 for x in p["label"])]
    surface = np.zeros((len(qids), grid.size), dtype=np.float64)
    for i, qid in enumerate(qids):
        pool = pools[qid]
        s = minmax_normalise(np.asarray(pool["sparse"], dtype=np.float64))
        d = minmax_normalise(np.asarray(pool["dense"], dtype=np.float64))
        for j, alpha in enumerate(grid):
            surface[i, j] = _query_ndcg(
                pool["pid"], pool["label"], alpha * s + (1.0 - alpha) * d, k
            )
    return qids, surface


def global_alpha_fit(
    dev_pools: Dict[str, Dict[str, Any]],
    *,
    k: int = 10,
    grid: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Grid-search the single alpha maximising mean nDCG@k on the DEV split.

    Returns ``(alpha_sparse, dev_ndcg)``. Fitting on dev and applying at test is
    what makes this an honest baseline rather than an oracle; log the chosen
    alpha per back-end and dataset, as the paper requires.
    """
    grid = ALPHA_GRID if grid is None else np.asarray(grid, dtype=np.float64)
    _, surface = alpha_surface(dev_pools, k=k, grid=grid)
    if surface.size == 0:
        return 0.5, 0.0
    means = surface.mean(axis=0)
    j = int(np.argmax(means))
    return float(grid[j]), float(means[j])


def global_alpha_scores(
    pool: Dict[str, Any], alpha: float
) -> np.ndarray:
    """Apply a fitted global alpha to one query's pool."""
    return fuse(pool["sparse"], pool["dense"], alpha)


def oracle_alpha_per_query(
    pools: Dict[str, Dict[str, Any]],
    *,
    k: int = 10,
    grid: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Per-query best alpha, chosen post hoc: the upper bound on adaptive gain.

    Also returns diagnostics that determine whether that bound is *reachable*.
    On the CoopIS MedMCQA pools the headroom looked large (+0.128 nDCG@10) but
    30.5% of queries had nDCG completely invariant to alpha and 57% of the grid
    tied the per-query maximum, so most of the bound is post-hoc selection over
    ties rather than a learnable property of the query. ``tie_fraction`` and
    ``invariant_fraction`` are reported so that this cannot be overlooked again.
    """
    grid = ALPHA_GRID if grid is None else np.asarray(grid, dtype=np.float64)
    qids, surface = alpha_surface(pools, k=k, grid=grid)
    if surface.size == 0:
        return {"oracle_ndcg": 0.0, "best_alpha": {}, "tie_fraction": None}

    row_max = surface.max(axis=1)
    best_alpha = grid[surface.argmax(axis=1)]
    ties = (surface >= row_max[:, None] - 1e-12).mean(axis=1)
    invariant = np.array([len(np.unique(r)) == 1 for r in surface])
    return {
        "oracle_ndcg": float(row_max.mean()),
        "best_alpha": {q: float(a) for q, a in zip(qids, best_alpha)},
        "best_alpha_mean": float(best_alpha.mean()),
        "best_alpha_std": float(best_alpha.std()),
        "tie_fraction": float(ties.mean()),
        "invariant_fraction": float(invariant.mean()),
        "n_queries": len(qids),
    }


def oracle_rerank_ndcg(
    pools: Dict[str, Dict[str, Any]],
    *,
    k: int = 10,
    include_queries_without_gold: bool = True,
) -> float:
    """
    nDCG@k of a perfect ranking of the pool each query already has.

    The ceiling imposed by first-stage retrieval: no re-ranker can beat it, so
    the gap between it and a system is the part recoverable by better *ranking*,
    while ``1 - pool_recall`` is the part that needs better *retrieval*.

    ``include_queries_without_gold=True`` (the default) scores a query whose
    gold passage was never retrieved as 0.0, because a re-ranker genuinely
    cannot win it. Under this definition the bound equals ``pool_recall`` when
    every query has one gold passage, which is the number the paper should
    quote. Setting it False averages only over winnable queries and yields 1.0
    in the single-gold case -- a true statement about a different population,
    and easy to misread as "the ceiling is perfect".
    """
    scores = []
    for pool in pools.values():
        relevant = [p for p, lab in zip(pool["pid"], pool["label"]) if int(lab) == 1]
        if not relevant:
            if include_queries_without_gold:
                scores.append(0.0)
            continue
        ranked = relevant + [p for p in pool["pid"] if p not in set(relevant)]
        scores.append(ndcg_at_k(ranked, relevant, k))
    return float(np.mean(scores)) if scores else 0.0


def group_alpha_fit(
    dev_pools: Dict[str, Dict[str, Any]],
    groups: Dict[str, str],
    *,
    k: int = 10,
    grid: Optional[np.ndarray] = None,
    min_queries: int = 20,
) -> Dict[str, float]:
    """
    One alpha per group (e.g. per question type), grid-searched on dev.

    Sits between ``global_alpha_fit`` (one alpha for everything) and a
    per-query controller. It answers a specific question: if a single alpha per
    question type already captures whatever a learned controller captures, the
    controller is doing question-type detection rather than genuine per-query
    adaptation.

    Groups with fewer than ``min_queries`` dev queries fall back to the global
    alpha, so a rare group cannot be fitted on noise. The returned mapping
    always contains the key ``"__global__"`` for unseen groups at test time.

    Caveat for this benchmark suite: PubMedQA is 100% yes/no by construction,
    so on both PubMedQA splits every query lands in one group and Group-alpha
    is *identical to Global-alpha by definition*. It is only informative on
    MedMCQA. Report it there and say so, rather than presenting four cells that
    are vacuously equal.
    """
    grid = ALPHA_GRID if grid is None else np.asarray(grid, dtype=np.float64)
    global_alpha, _ = global_alpha_fit(dev_pools, k=k, grid=grid)
    out: Dict[str, float] = {"__global__": global_alpha}

    by_group: Dict[str, Dict[str, Dict[str, Any]]] = collections.defaultdict(dict)
    for qid, pool in dev_pools.items():
        by_group[groups.get(qid, "other")][qid] = pool

    for name, sub in by_group.items():
        if len(sub) < min_queries:
            out[name] = global_alpha
            continue
        out[name], _ = global_alpha_fit(sub, k=k, grid=grid)
    return out


def group_alpha_scores(
    pool: Dict[str, Any], group: str, alphas: Dict[str, float]
) -> np.ndarray:
    """Apply the fitted per-group alpha to one query's pool."""
    return fuse(pool["sparse"], pool["dense"], alphas.get(group, alphas["__global__"]))


def pool_recall(pools: Dict[str, Dict[str, Any]]) -> float:
    """Fraction of queries whose gold passage is present anywhere in the pool."""
    if not pools:
        return 0.0
    hits = [1.0 if any(int(x) == 1 for x in p["label"]) else 0.0 for p in pools.values()]
    return float(np.mean(hits))
