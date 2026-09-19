"""
Retrieval evaluation metrics: Recall@k, MRR@k, nDCG@k.
Also provides evaluate_rank_data() used during training for early stopping.
"""

from __future__ import annotations

import json
import math
import numpy as np
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from pathlib import Path

from src.common.query_emb_cache import QueryEmbStore, load_query_emb_store
from src.features.schema import active_feature_indices


Relevance = Union[Iterable[str], Mapping[str, int]]


def _as_grades(relevant: Relevance) -> Dict[str, int]:
    """
    Normalise a relevance argument to ``{pid: grade}``.

    Accepts a plain iterable of relevant ids (every id gets grade 1, the old
    binary behaviour) or a ``pid -> grade`` mapping from graded qrels. Grades
    <= 0 are dropped so an explicitly-judged-irrelevant entry never counts as
    relevant for recall/hit/MRR.
    """
    if isinstance(relevant, Mapping):
        return {str(pid): int(g) for pid, g in relevant.items() if int(g) > 0}
    return {str(pid): 1 for pid in relevant}


def hit_at_k(ranked_ids: Sequence[str], relevant_ids: Relevance, k: int) -> float:
    """
    Success@k: 1.0 if *any* relevant passage appears in the top-k, else 0.0.

    Averaged over queries this is the "Hit@k" column. It is NOT recall: with a
    single gold passage the two coincide, but with multiple gold passages
    Hit@k saturates at 1.0 while recall does not. Use :func:`recall_at_k` when
    the paper says recall.
    """
    relevant = _as_grades(relevant_ids)
    if not relevant:
        return 0.0
    return 1.0 if any(pid in relevant for pid in ranked_ids[:k]) else 0.0


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Relevance, k: int) -> float:
    """
    Recall@k: |top-k retrieved that are relevant| / |all relevant|.

    The denominator is the size of the judgment set passed in. Pass *full*
    qrels (see :mod:`src.evaluation.qrels`), not the positives that happen to
    be in the candidate pool -- a pool-relative denominator makes recall@100
    identically 1.0 whenever the pool is no larger than 100.
    """
    relevant = _as_grades(relevant_ids)
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant.keys()) / len(relevant)


def mrr_at_k(ranked_ids: Sequence[str], relevant_ids: Relevance, k: int) -> float:
    """Reciprocal rank of the first relevant passage within the top-k, else 0."""
    relevant = _as_grades(relevant_ids)
    for i, rid in enumerate(ranked_ids[:k]):
        if rid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def mrr(ranked_ids: Sequence[str], relevant_ids: Relevance) -> float:
    """Reciprocal rank over the *untruncated* ranking (no cutoff)."""
    relevant = _as_grades(relevant_ids)
    for rank, pid in enumerate(ranked_ids, 1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_ids: Sequence[str],
    relevant_ids: Relevance,
    k: int,
    *,
    gain: str = "linear",
) -> float:
    """
    Graded nDCG@k with linear gain (trec_eval / pytrec_eval ``ndcg_cut``).

        DCG@k  = sum_{i<k} g(rel(ranked_i)) / log2(i + 2)
        IDCG@k = sum_{i<k} g(rel_sorted_i)  / log2(i + 2)
        nDCG@k = DCG@k / IDCG@k     (0.0 when IDCG is 0)

    ``rel`` is 0 for any id absent from the judgments, and the ideal ranking is
    taken over the **full** judgment set -- so on a densely judged collection
    IDCG@k saturates at k perfect documents and the score is comparable to
    published numbers. Passing a bare id list reproduces binary gain.

    ``gain="linear"`` uses g(r) = r, matching ``pytrec_eval``'s ``ndcg_cut``
    and the numbers reported in the paper. ``gain="exp"`` uses g(r) = 2^r - 1
    (Burges et al.) and is provided only for cross-checking against tools that
    default to it; the two disagree wherever grades exceed 1.
    """
    if gain not in ("linear", "exp"):
        raise ValueError(f"gain must be 'linear' or 'exp', got {gain!r}")
    g = (lambda r: float(r)) if gain == "linear" else (lambda r: float(2 ** r - 1))

    grades = _as_grades(relevant_ids)
    if not grades:
        return 0.0

    dcg = sum(
        g(grades.get(pid, 0)) / math.log2(i + 2)
        for i, pid in enumerate(ranked_ids[:k])
        if pid in grades
    )
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g(r) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_rank_data(
    model,
    dev_path: str,
    device: str,
    k: int = 10,
    ablation: Optional[str] = None,
    query_encoder_name: Optional[str] = None,
    query_encoder_device: str = "cpu",
    query_emb_cache_path: Optional[str] = None,
    batch_size: int = 8192,
    dropped_features: Optional[Sequence[str]] = None,
) -> float:
    """
    Compute mean nDCG@k over queries from rank JSONL (early stopping).

    Rank JSONL stores the full 8+8 feature schema; the model may consume a
    subset. ``dropped_features`` must therefore match what the model was
    trained with -- ``None`` means the schema default. The selected width is
    checked against the model's branch widths, because scoring a model with
    silently misaligned columns produces a plausible number that means nothing.
    """

    from collections import defaultdict

    model.eval()
    sparse_cols, dense_cols = active_feature_indices(dropped_features)
    if (len(sparse_cols), len(dense_cols)) != (int(model.sparse_dim), int(model.dense_dim)):
        raise ValueError(
            f"Feature selection does not match the model: dropped_features keeps "
            f"sparse={len(sparse_cols)} dense={len(dense_cols)}, but the model "
            f"expects sparse={model.sparse_dim} dense={model.dense_dim}. Pass the "
            "same model.dropped_features the checkpoint was trained with."
        )
    query_scores: Dict[str, List] = defaultdict(list)
    query_emb_cache = QueryEmbStore.empty(int(model.query_feat_dim))
    query_encoder = None

    if query_emb_cache_path:
        p = Path(query_emb_cache_path)
        if not p.exists() and p.name.endswith("_train_all.pkl"):
            all_cache = p.with_name(p.name.replace("_train_all.pkl", "_all.pkl"))
            if all_cache.exists():
                p = all_cache
        elif p.exists() and p.name.endswith("_train_all.pkl"):
            all_cache = p.with_name(p.name.replace("_train_all.pkl", "_all.pkl"))
            if all_cache.exists():
                p = all_cache
        if p.exists():
            loaded = load_query_emb_store(p, expected_dim=int(model.query_feat_dim))
            if len(loaded):
                query_emb_cache = loaded
                logger.info(
                    f"Loaded query_emb cache for evaluation: {len(query_emb_cache):,} queries"
                )

    def _model_query_dim(rec: Dict) -> int:
        """Query-embedding width the controller expects (independent of the one-hot)."""
        del rec  # kept for call-site symmetry; the model is authoritative
        return int(model.query_feat_dim)

    def _resolve_query_emb(rec: Dict):
        q = rec.get("query_emb")
        if isinstance(q, list):
            return q
        qid = str(rec.get("qid", ""))
        cached = query_emb_cache.get(qid)
        if cached is not None:
            return cached
        question = rec.get("question")
        if not isinstance(question, str) or not question.strip():
            raise KeyError(
                "Missing query_emb and question in dev record; cannot compute query embedding."
            )
        nonlocal query_encoder
        if query_encoder is None:
            if not query_encoder_name:
                raise KeyError(
                    "query_emb missing in dev data and no query_encoder_name was provided."
                )
            query_encoder = SentenceTransformer(query_encoder_name, device=query_encoder_device)
        emb = query_encoder.encode(
            [question],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0].tolist()
        expected = _model_query_dim(rec)
        if len(emb) != expected:
            raise ValueError(
                f"Computed query_emb dim mismatch: got={len(emb)} expected={expected}"
            )
        return query_emb_cache.set(qid, emb)

    # Grouped by query, not streamed flat. Two reasons: the model may
    # standardise each branch within its pool (``normalize_branches``), which is
    # only defined per query; and the controller may consume pool features,
    # which describe the whole pool. ``batch_size`` is now a cap on QUERIES per
    # forward, and pools are padded to the batch maximum with a mask.
    from src.features.pool_features import pool_features_from_records

    records_by_qid: Dict[str, List[Dict]] = defaultdict(list)
    with open(dev_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            records_by_qid[rec["qid"]].append(rec)

    # Pools are capped at ~100 candidates, so a few hundred queries per forward
    # keeps the GPU busy without a large padded tensor.
    queries_per_batch = max(1, int(batch_size) // 128)
    qids = list(records_by_qid)

    with torch.no_grad():
        for start in range(0, len(qids), queries_per_batch):
            chunk = qids[start : start + queries_per_batch]
            pools = [records_by_qid[q] for q in chunk]
            n_max = max(len(p) for p in pools)
            b = len(pools)

            sparse = np.zeros((b, n_max, len(sparse_cols)), dtype=np.float32)
            dense = np.zeros((b, n_max, len(dense_cols)), dtype=np.float32)
            mask = np.zeros((b, n_max), dtype=np.float32)
            qemb = np.zeros((b, int(model.query_feat_dim)), dtype=np.float32)
            pfeats = np.zeros((b, int(model.pool_feat_dim)), dtype=np.float32)

            for i, pool in enumerate(pools):
                n = len(pool)
                sparse[i, :n] = np.asarray(
                    [r["sparse_feats"] for r in pool], dtype=np.float32
                )[:, sparse_cols]
                dense[i, :n] = np.asarray(
                    [r["dense_feats"] for r in pool], dtype=np.float32
                )[:, dense_cols]
                mask[i, :n] = 1.0
                qemb[i] = np.asarray(_resolve_query_emb(pool[0]), dtype=np.float32)
                pfeats[i] = pool_features_from_records(pool)

            out = model(
                sparse_feats=torch.from_numpy(sparse).to(device),
                dense_feats=torch.from_numpy(dense).to(device),
                query_emb=torch.from_numpy(qemb).to(device),
                pool_feats=torch.from_numpy(pfeats).to(device),
                mask=torch.from_numpy(mask).to(device),
                ablation=ablation,
            )
            scores = out[0] if isinstance(out, (tuple, list)) else out
            scores = scores.detach().float().cpu().numpy()

            for i, (qid, pool) in enumerate(zip(chunk, pools)):
                for j, rec in enumerate(pool):
                    query_scores[qid].append(
                        (float(scores[i, j]), rec["pid"], rec["label"])
                    )

    # Score against full graded qrels where they exist. Falling back to in-pool
    # positives makes the denominator pool-relative, which flatters the metric
    # and is not comparable across pool sizes; the fallback is kept only for
    # collections with no qrels source on disk.
    from src.evaluation.qrels import qrels_for_qids

    try:
        qrels = qrels_for_qids(query_scores.keys())
    except (OSError, ValueError):
        qrels = {}

    ndcgs = []

    for qid, scored in query_scores.items():
        if not scored:
            continue

        scored.sort(key=lambda x: x[0], reverse=True)

        ranked = [pid for _, pid, _ in scored]
        relevant = qrels.get(str(qid)) or {pid for _, pid, lbl in scored if int(lbl) >= 1}

        ndcgs.append(ndcg_at_k(ranked, relevant, k))

    result = float(np.mean(ndcgs)) if ndcgs else 0.0

    logger.info(f"Evaluation nDCG@{k}: {result:.4f}")

    return result
