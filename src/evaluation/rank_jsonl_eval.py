"""
Offline evaluation on rank JSONL (BM25 / dense / hybrid / GARDIAN).

Used by ``scripts/05_evaluate_gardian.py`` and ``scripts/paper_run.py`` so
paper tables and CI scripts share one implementation.
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.evaluation.metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k
from src.evaluation.qrels import Qrels, qrels_for_qids
from src.features.pool_features import pool_features_from_records
from src.common.query_emb_cache import load_query_emb_store
from src.common.question_types import normalize_question_type
from src.baselines.fusion import (
    global_alpha_scores,
    group_alpha_scores,
    oracle_alpha_per_query,
    oracle_rerank_ndcg,
    pool_recall,
)
from src.model.gardian import GARDIAN


def load_rank_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    p = pathlib.Path(path)
    if not p.exists():
        logger.warning(f"File not found: {path}")
        return records
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def iter_rank_jsonl_records(path: str) -> Iterator[Dict[str, Any]]:
    """Stream rank JSONL lines without materializing the full file in memory."""
    p = pathlib.Path(path)
    if not p.exists():
        logger.warning(f"File not found: {path}")
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _batch_encode_query_misses(
    need_encode: Dict[str, str],
    *,
    query_encoder_name: str,
    query_encoder_device: str,
    model_query_dim: Optional[int],
    batch_size: int = 128,
) -> Dict[str, List[float]]:
    """Batched SentenceTransformer encode for qids missing query_emb in rank JSONL."""
    if not need_encode:
        return {}
    logger.info(
        f"Encoding {len(need_encode)} unique queries in batches of {batch_size} "
        f"({query_encoder_name!r} on {query_encoder_device!r})..."
    )
    enc = SentenceTransformer(query_encoder_name, device=query_encoder_device)
    pairs = list(need_encode.items())
    qids = [p[0] for p in pairs]
    questions = [p[1] for p in pairs]
    out: Dict[str, List[float]] = {}
    batch_starts = range(0, len(questions), batch_size)
    for start in tqdm(
        batch_starts,
        desc="Query embedding batches",
        leave=False,
        unit="batch",
    ):
        batch_q = questions[start : start + batch_size]
        batch_ids = qids[start : start + batch_size]
        embs = enc.encode(
            batch_q,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        for qid, row in zip(batch_ids, embs):
            emb = row.tolist()
            if model_query_dim is not None and len(emb) != int(model_query_dim):
                raise ValueError(
                    f"Recomputed query_emb dim mismatch during eval: got={len(emb)} expected={model_query_dim}"
                )
            out[qid] = emb
    return out


# Metric primitives live in ``src.evaluation.metrics`` so that this module, the
# question-type breakdown and the trainer's early-stopping all score identically.
# The ``compute_*`` names are kept as the local vocabulary of this module.
compute_mrr = mrr
compute_ndcg = ndcg_at_k
compute_recall = recall_at_k
compute_hit_at_k = hit_at_k


# Metrics aggregated per system in ``metrics_from_score_key`` / GARDIAN scoring.
RANK_METRIC_KEYS = [
    "ndcg@5",
    "ndcg@10",
    "ndcg@20",
    "ndcg@50",
    "ndcg@100",
    "recall@5",
    "recall@10",
    "recall@20",
    "recall@50",
    "recall@100",
    "hit@5",
    "hit@10",
    "hit@20",
    "hit@50",
    "mrr",
]


def _resolve_qrels(qids) -> Tuple[Qrels, bool]:
    """
    Full graded judgments for ``qids``, plus whether coverage is complete.

    Falls back to per-query pool labels only where a query has no entry in the
    qrels sources; that fallback is pool-relative and is reported in ``_meta``
    so a partially-covered run is never mistaken for a comparable number.
    """
    qids = [str(q) for q in qids]
    try:
        qrels = qrels_for_qids(qids)
    except (OSError, ValueError) as exc:
        logger.warning(f"Could not load qrels ({exc}); falling back to pool labels")
        return {}, False
    missing = [q for q in qids if q not in qrels]
    if missing:
        logger.warning(
            f"No qrels for {len(missing)}/{len(qids)} queries "
            f"(e.g. {missing[:3]}); those fall back to pool-relative labels"
        )
    return qrels, not missing


def _relevance_for_query(qid: str, qdata: Dict[str, Any], qrels: Qrels):
    """Graded qrels for ``qid`` when judged, else the in-pool positives."""
    judged = qrels.get(str(qid))
    if judged:
        return judged
    return [c["pid"] for c in qdata["candidates"] if int(c["label"]) >= 1]


def _append_rank_metrics(
    lists: Dict[str, List[float]],
    ranked_ids: List[str],
    relevant_ids: List[str],
) -> None:
    lists["ndcg@5"].append(compute_ndcg(ranked_ids, relevant_ids, 5))
    lists["ndcg@10"].append(compute_ndcg(ranked_ids, relevant_ids, 10))
    lists["ndcg@20"].append(compute_ndcg(ranked_ids, relevant_ids, 20))
    lists["ndcg@50"].append(compute_ndcg(ranked_ids, relevant_ids, 50))
    lists["ndcg@100"].append(compute_ndcg(ranked_ids, relevant_ids, 100))
    lists["recall@5"].append(compute_recall(ranked_ids, relevant_ids, 5))
    lists["recall@10"].append(compute_recall(ranked_ids, relevant_ids, 10))
    lists["recall@20"].append(compute_recall(ranked_ids, relevant_ids, 20))
    lists["recall@50"].append(compute_recall(ranked_ids, relevant_ids, 50))
    lists["recall@100"].append(compute_recall(ranked_ids, relevant_ids, 100))
    lists["hit@5"].append(compute_hit_at_k(ranked_ids, relevant_ids, 5))
    lists["hit@10"].append(compute_hit_at_k(ranked_ids, relevant_ids, 10))
    lists["hit@20"].append(compute_hit_at_k(ranked_ids, relevant_ids, 20))
    lists["hit@50"].append(compute_hit_at_k(ranked_ids, relevant_ids, 50))
    lists["mrr"].append(compute_mrr(ranked_ids, relevant_ids))


def _aggregate_lists(
    lists: Dict[str, List[float]],
) -> Dict[str, float]:
    return {k: float(np.mean(v)) if v else 0.0 for k, v in lists.items()}


def _new_query_bucket() -> Dict[str, Any]:
    return {
        "candidates": [],
        "sparse_scores": [],
        "dense_scores": [],
        "fusion_scores": [],
        "spladepp_scores": [],
        "cross_encoder_scores": [],
        "global_alpha_scores": [],
        "group_alpha_scores": [],
        "oracle_alpha_scores": [],
    }


def _baseline_result_keys(
    retriever_type: str, *, canonical_baseline_keys: bool
) -> Tuple[str, str, str]:
    """Map rank-file retriever_type to metric dict keys for sparse / dense / sum baselines."""
    rt = (retriever_type or "").strip()
    if not canonical_baseline_keys or not rt:
        return "bm25", "dense", "hybrid"
    mapping: Dict[str, Tuple[str, str, str]] = {
        "bm25": ("sparse(bm25)", "dense(none)", "sparse(bm25)"),
        "faiss": ("sparse(none)", "dense(faiss)", "dense(faiss)"),
        "medcpt": ("sparse(none)", "dense(medcpt)", "dense(medcpt)"),
        "spladepp": ("sparse(spladepp)", "dense(none)", "sparse(spladepp)"),
        "hybrid": ("sparse(bm25)", "dense(faiss)", "sum(bm25,faiss)"),
        "hybrid_neural": ("sparse(spladepp)", "dense(medcpt)", "sum(spladepp,medcpt)"),
        "hybrid_bm25_faiss": ("sparse(bm25)", "dense(faiss)", "sum(bm25,faiss)"),
        "hybrid_bm25_medcpt": ("sparse(bm25)", "dense(medcpt)", "sum(bm25,medcpt)"),
        "hybrid_spladepp_faiss": ("sparse(spladepp)", "dense(faiss)", "sum(spladepp,faiss)"),
        "hybrid_spladepp_medcpt": ("sparse(spladepp)", "dense(medcpt)", "sum(spladepp,medcpt)"),
    }
    return mapping.get(rt, ("bm25", "dense", "hybrid"))


def _score_key_has_nonzero_signal(
    queries_data: Dict[str, Dict[str, Any]], key: str
) -> bool:
    for q in queries_data.values():
        for x in q.get(key, []):
            if abs(float(x)) > 1e-15:
                return True
    return False


def _rrf_scores_for_query(
    sparse_scores: List[float],
    dense_scores: List[float],
    *,
    rrf_k: int = 60,
) -> List[float]:
    """Compute per-candidate RRF scores from sparse and dense rankings."""
    if len(sparse_scores) != len(dense_scores):
        raise ValueError("RRF score lists must have equal length.")
    sparse_rank_order = np.argsort(np.asarray(sparse_scores))[::-1]
    dense_rank_order = np.argsort(np.asarray(dense_scores))[::-1]
    sparse_rank = {int(idx): rank + 1 for rank, idx in enumerate(sparse_rank_order)}
    dense_rank = {int(idx): rank + 1 for rank, idx in enumerate(dense_rank_order)}
    out: List[float] = []
    for idx in range(len(sparse_scores)):
        rs = sparse_rank[idx]
        rd = dense_rank[idx]
        out.append((1.0 / (rrf_k + rs)) + (1.0 / (rrf_k + rd)))
    return out


def evaluate_all_from_rank_data(
    rank_data_path: str,
    model: Optional[GARDIAN] = None,
    device: Optional[str] = None,
    *,
    gardian_ablation: Optional[str] = None,
    gardian_fixed_alpha: Optional[float] = None,
    collect_per_query: bool = False,
    query_encoder_name: Optional[str] = None,
    query_encoder_device: str = "cpu",
    query_emb_cache_path: Optional[str] = None,
    expected_query_feat_dim: Optional[int] = None,
    canonical_baseline_keys: bool = False,
    include_standalone_spladepp: bool = False,
    gardian_adaptive_retrieval: bool = False,
    cfg: Optional[Any] = None,
    global_alpha: Optional[float] = None,
    global_alpha_per_query: Optional[Dict[str, float]] = None,
    group_alphas: Optional[Dict[str, float]] = None,
    question_types: Optional[Dict[str, str]] = None,
    include_oracle_alpha: bool = True,
) -> Dict[str, Any]:
    """
    Mean metrics over queries; optionally per-query lists for bootstrap CIs.

    ``gardian_ablation`` is passed to ``GARDIAN.forward(..., ablation=...)``.
    ``gardian_fixed_alpha`` is required when ``gardian_ablation="fixed_alpha"``.

    When ``canonical_baseline_keys`` is True (paper bundle), baseline metric keys
    reflect the actual first-stage channels (e.g. ``sparse(doc2query)``,
    ``dense(biobert)``, ``sum(doc2query,biobert)``) instead of generic
    ``bm25`` / ``dense`` / ``hybrid``.

    When ``gardian_adaptive_retrieval`` is True (``cfg.qa.gardian_adaptive_retrieval``),
  GARDIAN nDCG uses the α-weighted sparse+dense subset per query before fusion
    (same as QA). Baselines (sparse, dense, RRF) still use the full hybrid pool.
    """
    logger.info(f"Loading rank data: {rank_data_path}")
    p = pathlib.Path(rank_data_path)
    if not p.exists():
        logger.warning(f"File not found: {rank_data_path}")
        return {}

    def _resolve_sparse_score(rec: Dict[str, Any]) -> float:
        # Prefer explicit retriever score fields when available.
        retriever_type = str(rec.get("retriever_type", ""))
        if retriever_type in {"faiss", "medcpt"}:
            return 0.0
        if retriever_type in {
            "hybrid",
            "hybrid_bm25_faiss",
            "hybrid_bm25_medcpt",
            "bm25",
        }:
            if rec.get("bm25_score") is not None:
                return float(rec.get("bm25_score", 0.0))
        if retriever_type in {
            "hybrid_neural",
            "hybrid_spladepp_faiss",
            "hybrid_spladepp_medcpt",
            "spladepp",
        }:
            if rec.get("spladepp_score") is not None:
                return float(rec.get("spladepp_score", 0.0))
        sparse_feats = rec.get("sparse_feats") or [0.0]
        return float(sparse_feats[0] if sparse_feats else 0.0)

    def _resolve_dense_score(rec: Dict[str, Any]) -> float:
        if "dense_score" in rec:
            return float(rec["dense_score"])
        retriever_type = str(rec.get("retriever_type", ""))
        if retriever_type in ("bm25", "spladepp"):
            return 0.0
        dense_feats = rec.get("dense_feats") or [0.0]
        return float(dense_feats[0] if dense_feats else 0.0)

    queries: Dict[str, Dict[str, Any]] = defaultdict(_new_query_bucket)
    # Question type per query, read from the rank records themselves. Group-alpha
    # needs this at TEST time; requiring the caller to supply it is how the row
    # silently degenerates into Global-alpha when the mapping is omitted.
    record_question_types: Dict[str, str] = {}
    first_rec: Optional[Dict[str, Any]] = None
    rank_retriever_type = ""
    query_emb_prefill: Dict[str, List[float]] = {}
    pending_q: Dict[str, str] = {}
    want_query_cache = (
        model is not None and device is not None and bool(query_encoder_name)
    )

    for rec in iter_rank_jsonl_records(rank_data_path):
        if first_rec is None:
            first_rec = rec
        rt = str(rec.get("retriever_type", "")).strip()
        if rt and not rank_retriever_type:
            rank_retriever_type = rt
        qid = rec["qid"]
        if qid not in record_question_types:
            record_question_types[qid] = normalize_question_type(rec.get("question_type"))
        sparse_score = _resolve_sparse_score(rec)
        dense_score = _resolve_dense_score(rec)
        fusion_score = float(sparse_score) + float(dense_score)

        queries[qid]["candidates"].append({"pid": rec["pid"], "label": rec["label"]})
        queries[qid]["sparse_scores"].append(sparse_score)
        queries[qid]["dense_scores"].append(dense_score)
        queries[qid]["fusion_scores"].append(fusion_score)
        queries[qid]["spladepp_scores"].append(float(rec.get("spladepp_score", 0.0)))
        queries[qid]["cross_encoder_scores"].append(
            float(rec.get("cross_encoder_score", 0.0))
        )

        if want_query_cache:
            qid_s = str(rec.get("qid", ""))
            qe = rec.get("query_emb")
            if isinstance(qe, list) and qe:
                if qid_s not in query_emb_prefill:
                    query_emb_prefill[qid_s] = qe
                pending_q.pop(qid_s, None)
            else:
                question = rec.get("question")
                if (
                    isinstance(question, str)
                    and question.strip()
                    and qid_s not in query_emb_prefill
                ):
                    pending_q.setdefault(qid_s, question.strip())

    if not queries:
        return {}

    model_query_dim: Optional[int] = (
        int(expected_query_feat_dim) if expected_query_feat_dim is not None else None
    )
    if model_query_dim is None and model is not None:
        # The model reports its own query-embedding width, which is correct whether
        # or not the controller is conditioned on the question type.
        model_query_dim = getattr(model, "query_feat_dim", None)
        if model_query_dim is not None:
            model_query_dim = int(model_query_dim)

    query_emb_cache: Dict[str, List[float]] = {}
    if want_query_cache:
        query_emb_cache = dict(query_emb_prefill)

        # A precomputed cache covers essentially every query in these splits.
        # Without it every evaluation re-encodes the whole split with a BERT
        # forward pass per query -- 18k queries on CPU is ~16 minutes of pure
        # waste against a cache that is already on disk.
        if query_emb_cache_path:
            store = load_query_emb_store(
                query_emb_cache_path, expected_dim=model_query_dim
            )
            if len(store):
                hits = 0
                for qid in pending_q:
                    if qid not in query_emb_cache:
                        v = store.get(qid)
                        if v is not None:
                            query_emb_cache[qid] = (
                                v.tolist() if hasattr(v, "tolist") else list(v)
                            )
                            hits += 1
                logger.info(
                    f"query_emb cache {query_emb_cache_path}: "
                    f"{hits:,}/{len(pending_q):,} queries resolved without encoding"
                )

        need_encode = {k: v for k, v in pending_q.items() if k not in query_emb_cache}
        if need_encode:
            query_emb_cache.update(
                _batch_encode_query_misses(
                    need_encode,
                    query_encoder_name=query_encoder_name,
                    query_encoder_device=query_encoder_device,
                    model_query_dim=model_query_dim,
                )
            )

    query_encoder = None

    def _resolve_query_emb(rec: Dict[str, Any]) -> List[float]:
        query_emb = rec.get("query_emb")
        if isinstance(query_emb, list):
            return query_emb
        qid = str(rec.get("qid", ""))
        if qid in query_emb_cache:
            return query_emb_cache[qid]
        question = rec.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "Missing query_emb and question in rank-data record; cannot recompute for evaluation."
            )
        nonlocal query_encoder
        if query_encoder is None:
            if not query_encoder_name:
                raise ValueError(
                    "query_emb is missing and no query_encoder_name provided for evaluation."
                )
            query_encoder = SentenceTransformer(query_encoder_name, device=query_encoder_device)
        emb = query_encoder.encode(
            [question],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0].tolist()
        if model_query_dim is not None and len(emb) != int(model_query_dim):
            raise ValueError(
                f"Recomputed query_emb dim mismatch during eval: got={len(emb)} expected={model_query_dim}"
            )
        query_emb_cache[qid] = emb
        return emb

    gardian_features: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "sparse_feats": [],
            "dense_feats": [],
            "query_emb": None,
            "candidates": [],
        }
    )

    sparse_key, dense_key, fusion_key = _baseline_result_keys(
        rank_retriever_type, canonical_baseline_keys=canonical_baseline_keys
    )

    records_by_qid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    if model is not None and device is not None:
        for rec in iter_rank_jsonl_records(rank_data_path):
            qid = rec["qid"]
            records_by_qid[qid].append(rec)
            gardian_features[qid]["candidates"].append(
                {"pid": rec["pid"], "label": rec["label"]}
            )
            gardian_features[qid]["sparse_feats"].append(
                torch.tensor(rec["sparse_feats"], dtype=torch.float32)
            )
            gardian_features[qid]["dense_feats"].append(
                torch.tensor(rec["dense_feats"], dtype=torch.float32)
            )
            if gardian_features[qid]["query_emb"] is None:
                query_emb = _resolve_query_emb(rec)
                gardian_features[qid]["query_emb"] = torch.tensor(
                    query_emb, dtype=torch.float32
                )

    for qid, qdata in queries.items():
        qdata["rrf_scores"] = _rrf_scores_for_query(
            qdata["sparse_scores"],
            qdata["dense_scores"],
        )

    qrels, qrels_complete = _resolve_qrels(queries.keys())

    def metrics_from_score_key(
        queries_data: Dict[str, Dict[str, Any]],
        score_key: str,
        collect: bool,
    ) -> Tuple[Dict[str, float], Optional[Dict[str, List[float]]], int]:
        lists: Dict[str, List[float]] = {k: [] for k in RANK_METRIC_KEYS}
        no_positive_queries = 0
        for qid, qdata in queries_data.items():
            scores = qdata[score_key]
            sorted_indices = np.argsort(scores)[::-1]
            ranked_ids = [qdata["candidates"][i]["pid"] for i in sorted_indices]
            relevant_ids = _relevance_for_query(qid, qdata, qrels)
            if not relevant_ids:
                no_positive_queries += 1
            _append_rank_metrics(lists, ranked_ids, relevant_ids)
        means = _aggregate_lists(lists)
        means["mrr@10"] = means["mrr"]
        pq = lists if collect else None
        return means, pq, no_positive_queries

    results: Dict[str, Any] = {}
    query_count = len(queries)
    results["_meta"] = {
        "rank_data_path": rank_data_path,
        "query_count": query_count,
        "retriever_type": rank_retriever_type or None,
        "canonical_baseline_keys": bool(canonical_baseline_keys),
        "qrels": {
            "source": "full_graded" if qrels else "pool_labels",
            "queries_judged": len(qrels),
            "complete_coverage": bool(qrels_complete),
            "mean_relevant_per_query": (
                round(sum(len(v) for v in qrels.values()) / len(qrels), 2) if qrels else None
            ),
            "ndcg_gain": "linear",
        },
        "baseline_keys": {
            "sparse": sparse_key,
            "dense": dense_key,
            "fusion": fusion_key,
        },
    }

    qdict = dict(queries)

    def _emit(name: str, internal_key: str) -> None:
        logger.info(f"  Evaluating {name}...")
        m, pq, no_pos = metrics_from_score_key(queries, internal_key, collect_per_query)
        if collect_per_query and pq is not None:
            m["_per_query"] = pq
        results[name] = m
        results["_meta"][f"{name}_no_positive_queries"] = no_pos

    # Sparse / dense / sum baselines (keys depend on retriever_type when canonical_baseline_keys).
    seen_baseline_labels: set[str] = set()
    for label, internal_key in (
        (sparse_key, "sparse_scores"),
        (dense_key, "dense_scores"),
        (fusion_key, "fusion_scores"),
    ):
        if label in seen_baseline_labels:
            continue
        if "(none)" in label and not _score_key_has_nonzero_signal(qdict, internal_key):
            continue
        seen_baseline_labels.add(label)
        _emit(label, internal_key)

    _emit("rrf", "rrf_scores")

    # ---- score-fusion baselines -------------------------------------------
    # Global-alpha and Group-alpha are fitted on DEV by the caller and applied
    # here, which is what makes them honest baselines rather than oracles.
    # Oracle-alpha picks each query's best alpha using THIS split's labels: it
    # is a diagnostic ceiling on the whole linear-fusion family, never a system
    # to compare against. The gap Oracle-alpha - Global-alpha is the entire
    # maximum payoff of query-adaptive weighting.
    fusion_pools = {
        qid: {
            "pid": [c["pid"] for c in q["candidates"]],
            "label": [c["label"] for c in q["candidates"]],
            "sparse": q["sparse_scores"],
            "dense": q["dense_scores"],
        }
        for qid, q in queries.items()
    }
    results["_meta"]["pool_recall"] = pool_recall(fusion_pools)

    # A single alpha fitted on dev, or -- for collections with no dev split of
    # their own -- one alpha per query fitted by cross-validation on the other
    # folds. The latter is still an honest baseline: no query's alpha is fitted
    # using that query's own labels.
    if global_alpha_per_query:
        for qid, q in queries.items():
            a = global_alpha_per_query.get(qid, global_alpha)
            if a is None:
                continue
            q["global_alpha_scores"] = list(
                global_alpha_scores(fusion_pools[qid], float(a))
            )
        _emit("global_alpha", "global_alpha_scores")
        vals = sorted(set(global_alpha_per_query.values()))
        results["_meta"]["global_alpha_cv"] = {
            "n_queries": len(global_alpha_per_query),
            "distinct_alphas": [float(v) for v in vals],
        }
    elif global_alpha is not None:
        for qid, q in queries.items():
            q["global_alpha_scores"] = list(
                global_alpha_scores(fusion_pools[qid], float(global_alpha))
            )
        _emit("global_alpha", "global_alpha_scores")
        results["_meta"]["global_alpha"] = float(global_alpha)

    if group_alphas:
        qtypes = question_types or record_question_types
        unmapped = sum(1 for qid in queries if qtypes.get(qid) not in group_alphas)
        if unmapped:
            logger.warning(
                f"group_alpha: {unmapped:,}/{len(queries):,} queries have no fitted "
                "group and fall back to the global alpha; for those the row is "
                "Global-alpha, not Group-alpha"
            )
        for qid, q in queries.items():
            q["group_alpha_scores"] = list(
                group_alpha_scores(
                    fusion_pools[qid], qtypes.get(qid, "other"), group_alphas
                )
            )
        _emit("group_alpha", "group_alpha_scores")
        results["_meta"]["group_alphas"] = dict(group_alphas)

    if include_oracle_alpha:
        oracle = oracle_alpha_per_query(fusion_pools, k=10)
        best = oracle.get("best_alpha", {})
        for qid, q in queries.items():
            a = best.get(qid)
            q["oracle_alpha_scores"] = (
                list(global_alpha_scores(fusion_pools[qid], float(a)))
                if a is not None
                else list(q["rrf_scores"])
            )
        _emit("oracle_alpha", "oracle_alpha_scores")
        # The per-query alpha maximises nDCG@k for ONE k. Applying that same
        # alpha at other cutoffs is not a ceiling and can score *below* the
        # system it is supposed to bound (observed on TREC-COVID: oracle
        # nDCG@20 0.6517 < GARDIAN 0.6761). Blank every metric it does not
        # actually bound rather than printing a number that reads as an upper
        # limit but is not one.
        oracle_k = 10
        keep = {f"ndcg@{oracle_k}"}
        results["oracle_alpha"] = {
            k: (v if k in keep else None) for k, v in results["oracle_alpha"].items()
        }
        results["_meta"]["oracle_alpha_valid_cutoff"] = oracle_k
        results["_meta"]["oracle_alpha"] = {
            k: v for k, v in oracle.items() if k != "best_alpha"
        }
        results["_meta"]["oracle_rerank_ndcg@10"] = oracle_rerank_ndcg(
            fusion_pools, k=10
        )
    skip_standalone_spladepp = bool(
        not include_standalone_spladepp
        or (canonical_baseline_keys and sparse_key == "sparse(spladepp)")
    )
    if not skip_standalone_spladepp:
        for name, key in [("spladepp", "spladepp_scores")]:
            if _score_key_has_nonzero_signal(qdict, key):
                logger.info(f"  Evaluating {name}...")
                m, pq, no_pos = metrics_from_score_key(queries, key, collect_per_query)
                if collect_per_query and pq is not None:
                    m["_per_query"] = pq
                results[name] = m
                results["_meta"][f"{name}_no_positive_queries"] = no_pos

    if _score_key_has_nonzero_signal(qdict, "cross_encoder_scores"):
        _emit("cross_encoder", "cross_encoder_scores")

    if model is not None and device is not None and gardian_features:
        # The adaptive *channel budget* is a live-QA retrieval policy (see
        # src/pipeline/gardian_adaptive.py). Offline rank-JSONL evaluation
        # deliberately does NOT apply it: pre-subsetting the pool by
        # (alpha, beta) dropped gold passages that rank high under RRF but low
        # in either single channel, which inflated GARDIAN's retrieval metrics
        # relative to the baselines scored on the full pool. GARDIAN is
        # therefore scored on exactly the same candidates as sparse/dense/RRF;
        # the controller still supplies the per-query fusion weights inside
        # ``GARDIAN.forward``.
        adaptive_requested = bool(gardian_adaptive_retrieval and cfg is not None)
        logger.info(
            f"  Evaluating GARDIAN on the full rank pool (ablation={gardian_ablation!r}; "
            f"adaptive channel budget requested={adaptive_requested}, "
            "not applied offline -- same pool as all baselines)"
        )
        results["_meta"]["gardian_adaptive_retrieval_requested"] = adaptive_requested
        results["_meta"]["gardian_scored_on_full_pool"] = True

        gardian_results: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"candidates": [], "scores": []}
        )
        with torch.no_grad():
            for qid, qdata in tqdm(
                gardian_features.items(),
                desc="    GARDIAN scoring",
                leave=False,
            ):
                if not qdata["candidates"]:
                    continue
                # One pool per forward, kept in grouped (1, N, F) form. The
                # model may standardise each branch within its pool and the
                # controller may read pool features, both of which are defined
                # per query -- a flat batch has no pool to define them over.
                sparse_batch = torch.stack(qdata["sparse_feats"]).unsqueeze(0).to(device)
                dense_batch = torch.stack(qdata["dense_feats"]).unsqueeze(0).to(device)
                query_emb = qdata["query_emb"].unsqueeze(0).to(device)
                pool_feats = torch.from_numpy(
                    pool_features_from_records(records_by_qid[qid])
                ).unsqueeze(0).to(device)
                mask = torch.ones(sparse_batch.shape[:2], dtype=torch.float32, device=device)
                fwd = {
                    "sparse_feats": sparse_batch,
                    "dense_feats": dense_batch,
                    "query_emb": query_emb,
                    "pool_feats": pool_feats,
                    "mask": mask,
                    "ablation": gardian_ablation,
                    "fixed_alpha": gardian_fixed_alpha,
                }
                scores, _ = model(**fwd)
                gardian_results[qid]["candidates"] = qdata["candidates"]
                gardian_results[qid]["scores"] = scores.cpu().numpy().flatten()

        lists: Dict[str, List[float]] = {k: [] for k in RANK_METRIC_KEYS}
        no_positive_queries = 0
        for qid, qdata in gardian_results.items():
            if not qdata["scores"].size:
                continue
            sorted_indices = np.argsort(qdata["scores"])[::-1]
            ranked_ids = [qdata["candidates"][i]["pid"] for i in sorted_indices]
            relevant_ids = _relevance_for_query(qid, qdata, qrels)
            if not relevant_ids:
                no_positive_queries += 1
            _append_rank_metrics(lists, ranked_ids, relevant_ids)

        results["gardian"] = _aggregate_lists(lists)
        results["gardian"]["mrr@10"] = results["gardian"]["mrr"]
        if collect_per_query:
            results["gardian"]["_per_query"] = lists
        results["_meta"]["gardian_no_positive_queries"] = no_positive_queries
    else:
        results["_meta"]["gardian_no_positive_queries"] = query_count

    return results
