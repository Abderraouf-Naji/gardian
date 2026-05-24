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

from src.model.gardian import GARDIAN
from src.pipeline.gardian_adaptive import (
    controller_weights_from_lists,
    subset_rank_records_adaptive,
)


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


def compute_mrr(ranked_ids: List[str], relevant_ids: List[str]) -> float:
    for rank, pid in enumerate(ranked_ids, 1):
        if pid in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_ndcg(ranked_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    dcg = 0.0
    for i, pid in enumerate(ranked_ids[:k]):
        if pid in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)
    idcg = 0.0
    for i in range(min(len(relevant_ids), k)):
        idcg += 1.0 / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


def compute_recall(ranked_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_at_k = set(ranked_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_at_k & relevant_set) / len(relevant_set)


def compute_hit_at_k(ranked_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Success@k: 1 if any relevant doc appears in top-k, else 0 (per query)."""
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    return 1.0 if any(pid in relevant_set for pid in ranked_ids[:k]) else 0.0


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
    collect_per_query: bool = False,
    query_encoder_name: Optional[str] = None,
    query_encoder_device: str = "cpu",
    expected_query_feat_dim: Optional[int] = None,
    canonical_baseline_keys: bool = False,
    include_standalone_spladepp: bool = False,
    gardian_adaptive_retrieval: bool = False,
    cfg: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Mean metrics over queries; optionally per-query lists for bootstrap CIs.

    ``gardian_ablation`` is passed to ``GARDIAN.forward(..., ablation=...)``.

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
    if model_query_dim is None and model is not None and first_rec is not None:
        first_layer = getattr(getattr(model, "controller", None), "net", None)
        try:
            if first_layer is not None and hasattr(first_layer[0], "in_features"):
                qtype_dim = len(first_rec.get("qtype_onehot", []))
                # Controller input is query_feat_dim + n_qtypes (see ControllerMLP).
                model_query_dim = int(first_layer[0].in_features) - int(qtype_dim)
        except Exception:
            model_query_dim = None

    query_emb_cache: Dict[str, List[float]] = {}
    if want_query_cache:
        need_encode = {k: v for k, v in pending_q.items() if k not in query_emb_prefill}
        query_emb_cache = dict(query_emb_prefill)
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
            "kg_feats": [],
            "query_emb": None,
            "qtype_onehot": None,
            "kg_coverage": None,
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
            if not bool(getattr(model, "text_only", True)):
                gardian_features[qid]["kg_feats"].append(
                    torch.tensor(rec["kg_feats"], dtype=torch.float32)
                )
            if gardian_features[qid]["query_emb"] is None:
                query_emb = _resolve_query_emb(rec)
                gardian_features[qid]["query_emb"] = torch.tensor(
                    query_emb, dtype=torch.float32
                )
                gardian_features[qid]["qtype_onehot"] = torch.tensor(
                    rec["qtype_onehot"], dtype=torch.float32
                )
                if not bool(getattr(model, "text_only", True)):
                    gardian_features[qid]["kg_coverage"] = rec.get("kg_coverage", 0.0)

    for qid, qdata in queries.items():
        qdata["rrf_scores"] = _rrf_scores_for_query(
            qdata["sparse_scores"],
            qdata["dense_scores"],
        )

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
            relevant_ids = [c["pid"] for c in qdata["candidates"] if c["label"] == 1]
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
        use_adaptive = bool(gardian_adaptive_retrieval and cfg is not None)
        if use_adaptive:
            logger.info(
                "  GARDIAN adaptive retrieval: query→(α,β)→sparse/dense subset→fusion "
                f"(ablation={gardian_ablation!r})"
            )
        else:
            logger.info(f"  Evaluating GARDIAN (ablation={gardian_ablation!r})...")
        results["_meta"]["gardian_adaptive_retrieval"] = use_adaptive

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
                # Score the FULL rank pool (same candidates as RRF). Pre-subsetting by
                # adaptive_channel_budget dropped high-RRF / low-single-channel gold
                # docs and inflated retrieval metrics vs E2E QA unfairly.
                if use_adaptive and qdata["query_emb"] is not None:
                    pool_records = records_by_qid.get(qid, [])
                    if pool_records:
                        controller_weights_from_lists(
                            model,
                            qdata["query_emb"].cpu().tolist(),
                            qdata["qtype_onehot"].cpu().tolist(),
                            device,
                            ablation=gardian_ablation,
                        )
                batch_size = len(qdata["candidates"])
                sparse_batch = torch.stack(qdata["sparse_feats"]).to(device)
                dense_batch = torch.stack(qdata["dense_feats"]).to(device)
                query_emb = qdata["query_emb"].unsqueeze(0).expand(batch_size, -1).to(device)
                qtype_onehot = qdata["qtype_onehot"].unsqueeze(0).expand(batch_size, -1).to(device)
                fwd = {
                    "sparse_feats": sparse_batch,
                    "dense_feats": dense_batch,
                    "query_emb": query_emb,
                    "qtype_onehot": qtype_onehot,
                    "ablation": gardian_ablation,
                }
                if not bool(getattr(model, "text_only", True)):
                    kg_batch = torch.stack(qdata["kg_feats"]).to(device)
                    kg_coverage = torch.full(
                        (batch_size,),
                        float(qdata.get("kg_coverage") or 0.0),
                        dtype=torch.float32,
                    ).to(device)
                    fwd["kg_feats"] = kg_batch
                    fwd["kg_coverage"] = kg_coverage
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
            relevant_ids = [c["pid"] for c in qdata["candidates"] if c["label"] == 1]
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
