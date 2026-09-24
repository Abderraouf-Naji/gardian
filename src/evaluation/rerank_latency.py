"""
Per-query re-ranking latency, measured on one fixed candidate pool.

Reviewers 2 and 4 asked what the re-ranker costs at inference. Answering that
honestly needs three things the previous measurement did not have:

1. **A common pool.** Every system re-ranks the *same* candidates produced by
   the same first-stage retrieval, so the number is the cost of re-ranking and
   not of retrieval. First-stage cost is shared by all systems and is reported
   separately (``scripts/benchmark_retrieval_efficiency.py``).

2. **The query encoder inside the timed region.** GARDIAN's controller reads a
   768-d PubMedBERT sentence embedding, so a deployed system must run that
   forward pass per query. ``benchmark_retrieval_efficiency.py`` encoded the
   query *outside* ``_timed_ms``, which silently excluded it. With the
   controller off (GARDIAN-Lite) there is no query encoder on the inference
   path at all, and that difference is the point of the comparison -- it cannot
   be shown if the encoder is never counted in either arm.

3. **Device synchronisation.** CUDA kernels are asynchronous, so timing a GPU
   forward without ``torch.cuda.synchronize()`` measures launch latency rather
   than compute.

Reported as median (p50) over per-query samples, after a warmup, because the
first few queries pay one-off allocator and autotuner costs that a deployed
system amortises.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
from loguru import logger

from src.baselines.fusion import rrf_scores
from src.features.pool_features import pool_features_from_records


@contextmanager
def _sync_timer(samples_ms: List[float], device: Optional[str]) -> Iterator[None]:
    """Append the wall-clock duration of the block, synchronising CUDA first."""
    use_cuda = bool(device) and str(device).startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if use_cuda:
            torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)


def latency_stats(samples_ms: Sequence[float]) -> Dict[str, float]:
    """Median/mean/p95 over per-query samples. p50 is the reported figure."""
    if not samples_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "n": 0}
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "std_ms": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "n": int(len(arr)),
    }


def _pool_sizes(pools: Sequence[Sequence[Dict[str, Any]]]) -> Dict[str, float]:
    sizes = np.asarray([len(p) for p in pools], dtype=np.float64)
    if sizes.size == 0:
        return {"mean": 0.0, "min": 0, "max": 0}
    return {
        "mean": float(np.mean(sizes)),
        "min": int(np.min(sizes)),
        "max": int(np.max(sizes)),
    }


def time_rrf(
    pools: Sequence[Sequence[Dict[str, Any]]],
    *,
    rrf_k: int = 60,
    warmup: int,
) -> Dict[str, float]:
    """
    Time reciprocal rank fusion over the pool -- the no-model control.

    Uses column 0 of each branch's feature vector, which is the raw channel
    score (see ``src/features/schema.py``), so the timed input is exactly what
    the evaluation path fuses.
    """
    samples: List[float] = []
    for i, pool in enumerate(pools):
        sparse = [float(r["sparse_feats"][0]) for r in pool]
        dense = [float(r["dense_feats"][0]) for r in pool]
        collect: List[float] = []
        with _sync_timer(collect, device=None):
            fused = rrf_scores(sparse, dense, k=rrf_k)
            np.argsort(-fused, kind="stable")
        if i >= warmup:
            samples.extend(collect)
    return latency_stats(samples)


def time_cross_encoder(
    pools: Sequence[Sequence[Dict[str, Any]]],
    *,
    questions: Sequence[str],
    passage_text: Callable[[Dict[str, Any]], str],
    scorer: Any,
    device: str,
    warmup: int,
) -> Dict[str, float]:
    """
    Time one cross-encoder over each pool: tokenise + forward every pair, then sort.

    This is the full deployed cost: a cross-encoder cannot precompute anything,
    because every score depends jointly on the query and the passage.
    """
    samples: List[float] = []
    for i, (pool, question) in enumerate(zip(pools, questions)):
        passages = [passage_text(r) for r in pool]
        queries = [question] * len(passages)
        collect: List[float] = []
        with torch.no_grad(), _sync_timer(collect, device=device):
            scores = scorer.score_pairs(queries, passages)
            np.argsort(np.asarray(scores, dtype=np.float64))[::-1]
        if i >= warmup:
            samples.extend(collect)
    return latency_stats(samples)


def time_gardian(
    pools: Sequence[Sequence[Dict[str, Any]]],
    *,
    questions: Sequence[str],
    model: Any,
    query_encoder: Any,
    device: str,
    warmup: int,
    encode_query: bool = True,
) -> Dict[str, Any]:
    """
    Time GARDIAN's re-ranking path per query, mirroring the evaluation path
    in :func:`src.evaluation.rank_jsonl_eval.evaluate_all_from_rank_data`:
    one pool per forward, grouped ``(1, N, F)``.

    ``encode_query`` controls whether the PubMedBERT forward pass is inside the
    timed region. It must be True whenever the controller is enabled, because a
    deployed system has no precomputed embedding for an unseen query. With the
    controller disabled (GARDIAN-Lite) the model never reads ``query_emb``, so
    the encoder is genuinely off the inference path and is not timed.

    The breakdown separates the encoder from the ranking head, so the cost the
    controller adds is visible rather than inferred.
    """
    total: List[float] = []
    encode_only: List[float] = []
    rank_only: List[float] = []

    zero_emb = torch.zeros((1, int(getattr(model, "query_feat_dim", 768))), dtype=torch.float32)

    for i, (pool, question) in enumerate(zip(pools, questions)):
        t_total: List[float] = []
        t_encode: List[float] = []
        t_rank: List[float] = []

        with torch.no_grad(), _sync_timer(t_total, device=device):
            if encode_query:
                with _sync_timer(t_encode, device=device):
                    emb = query_encoder.encode(
                        [question],
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                    )
                query_emb = torch.from_numpy(emb).to(device=device, dtype=torch.float32)
            else:
                t_encode.append(0.0)
                query_emb = zero_emb.to(device)

            with _sync_timer(t_rank, device=device):
                sparse_batch = torch.tensor(
                    np.asarray([r["sparse_feats"] for r in pool], dtype=np.float32)
                ).unsqueeze(0).to(device)
                dense_batch = torch.tensor(
                    np.asarray([r["dense_feats"] for r in pool], dtype=np.float32)
                ).unsqueeze(0).to(device)
                pool_feats = torch.from_numpy(
                    pool_features_from_records(pool)
                ).unsqueeze(0).to(device)
                mask = torch.ones(sparse_batch.shape[:2], dtype=torch.float32, device=device)
                scores, _ = model(
                    sparse_feats=sparse_batch,
                    dense_feats=dense_batch,
                    query_emb=query_emb,
                    pool_feats=pool_feats,
                    mask=mask,
                )
                np.argsort(scores.detach().cpu().numpy().flatten())[::-1]

        if i >= warmup:
            total.extend(t_total)
            encode_only.extend(t_encode)
            rank_only.extend(t_rank)

    out: Dict[str, Any] = dict(latency_stats(total))
    out["breakdown"] = {
        "query_encoder": latency_stats(encode_only),
        "ranking_head": latency_stats(rank_only),
        "query_encoder_timed": bool(encode_query),
    }
    return out


def select_timing_pools(
    records_by_qid: Dict[str, List[Dict[str, Any]]],
    *,
    n_queries: int,
    seed: int,
) -> tuple[List[List[Dict[str, Any]]], List[str], List[str]]:
    """Pick a deterministic subset of query pools to time."""
    import random

    qids = sorted(records_by_qid.keys())
    if n_queries > 0 and len(qids) > n_queries:
        qids = sorted(random.Random(seed).sample(qids, n_queries))
    pools = [records_by_qid[q] for q in qids]
    questions = [str(records_by_qid[q][0].get("question") or "") for q in qids]
    return pools, questions, qids


def log_summary(tag: str, stats: Dict[str, Any]) -> None:
    logger.info(
        f"    {tag:<34} p50={stats.get('p50_ms', 0.0):8.2f} ms  "
        f"mean={stats.get('mean_ms', 0.0):8.2f}  "
        f"p95={stats.get('p95_ms', 0.0):8.2f}  n={stats.get('n', 0)}"
    )
