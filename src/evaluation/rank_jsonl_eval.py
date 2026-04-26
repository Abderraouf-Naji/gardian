"""
Offline evaluation on rank JSONL (BM25 / dense / hybrid / GARDIAN).

Used by ``scripts/05_evaluate_gardian.py`` and ``scripts/paper_run.py`` so
paper tables and CI scripts share one implementation.
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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


def _aggregate_lists(
    lists: Dict[str, List[float]],
) -> Dict[str, float]:
    return {k: float(np.mean(v)) if v else 0.0 for k, v in lists.items()}


def _new_query_bucket() -> Dict[str, Any]:
    return {
        "candidates": [],
        "bm25_scores": [],
        "dense_scores": [],
        "hybrid_scores": [],
        "doc2query_scores": [],
    }


def _score_key_has_nonzero_signal(
    queries_data: Dict[str, Dict[str, Any]], key: str
) -> bool:
    for q in queries_data.values():
        for x in q.get(key, []):
            if abs(float(x)) > 1e-15:
                return True
    return False


def evaluate_all_from_rank_data(
    rank_data_path: str,
    model: Optional[GARDIAN] = None,
    device: Optional[str] = None,
    *,
    gardian_ablation: Optional[str] = None,
    collect_per_query: bool = False,
    query_encoder_name: Optional[str] = None,
    query_encoder_device: str = "cpu",
) -> Dict[str, Any]:
    """
    Mean metrics over queries; optionally per-query lists for bootstrap CIs.

    ``gardian_ablation`` is passed to ``GARDIAN.forward(..., ablation=...)``.
    """
    logger.info(f"Loading rank data: {rank_data_path}")
    records = load_rank_jsonl(rank_data_path)
    if not records:
        return {}

    model_query_dim: Optional[int] = None
    if model is not None:
        first_layer = getattr(getattr(model, "controller", None), "net", None)
        try:
            if first_layer is not None and hasattr(first_layer[0], "in_features"):
                qtype_dim = len(records[0].get("qtype_onehot", []))
                model_query_dim = int(first_layer[0].in_features) - int(qtype_dim) - 1
        except Exception:
            model_query_dim = None
    query_encoder = None
    query_emb_cache: Dict[str, List[float]] = {}

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

    queries: Dict[str, Dict[str, Any]] = defaultdict(_new_query_bucket)
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

    for rec in records:
        qid = rec["qid"]
        bm25_score = rec["sparse_feats"][0] if rec.get("sparse_feats") else 0
        dense_score = rec["dense_feats"][0] if rec.get("dense_feats") else 0
        hybrid_score = float(bm25_score) + float(dense_score)

        queries[qid]["candidates"].append({"pid": rec["pid"], "label": rec["label"]})
        queries[qid]["bm25_scores"].append(bm25_score)
        queries[qid]["dense_scores"].append(dense_score)
        queries[qid]["hybrid_scores"].append(hybrid_score)
        queries[qid]["doc2query_scores"].append(float(rec.get("doc2query_score", 0.0)))

        if model is not None and device is not None:
            gardian_features[qid]["candidates"].append(
                {"pid": rec["pid"], "label": rec["label"]}
            )
            gardian_features[qid]["sparse_feats"].append(
                torch.tensor(rec["sparse_feats"], dtype=torch.float32)
            )
            gardian_features[qid]["dense_feats"].append(
                torch.tensor(rec["dense_feats"], dtype=torch.float32)
            )
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
                gardian_features[qid]["kg_coverage"] = rec["kg_coverage"]

    def metrics_from_score_key(
        queries_data: Dict[str, Dict[str, Any]],
        score_key: str,
        collect: bool,
    ) -> Tuple[Dict[str, float], Optional[Dict[str, List[float]]], int]:
        keys = ["ndcg@5", "ndcg@10", "ndcg@20", "recall@5", "recall@20", "mrr"]
        lists: Dict[str, List[float]] = {k: [] for k in keys}
        no_positive_queries = 0
        for qid, qdata in queries_data.items():
            scores = qdata[score_key]
            sorted_indices = np.argsort(scores)[::-1]
            ranked_ids = [qdata["candidates"][i]["pid"] for i in sorted_indices]
            relevant_ids = [c["pid"] for c in qdata["candidates"] if c["label"] == 1]
            if not relevant_ids:
                no_positive_queries += 1
            lists["ndcg@5"].append(compute_ndcg(ranked_ids, relevant_ids, 5))
            lists["ndcg@10"].append(compute_ndcg(ranked_ids, relevant_ids, 10))
            lists["ndcg@20"].append(compute_ndcg(ranked_ids, relevant_ids, 20))
            lists["recall@5"].append(compute_recall(ranked_ids, relevant_ids, 5))
            lists["recall@20"].append(compute_recall(ranked_ids, relevant_ids, 20))
            lists["mrr"].append(compute_mrr(ranked_ids, relevant_ids))
        means = _aggregate_lists(lists)
        means["mrr@10"] = means["mrr"]
        pq = lists if collect else None
        return means, pq, no_positive_queries

    results: Dict[str, Any] = {}
    query_count = len(queries)
    results["_meta"] = {
        "rank_data_path": rank_data_path,
        "query_count": query_count,
    }

    for name, key in [
        ("bm25", "bm25_scores"),
        ("dense", "dense_scores"),
        ("hybrid", "hybrid_scores"),
    ]:
        logger.info(f"  Evaluating {name}...")
        m, pq, no_pos = metrics_from_score_key(queries, key, collect_per_query)
        if collect_per_query and pq is not None:
            m["_per_query"] = pq
        results[name] = m
        results["_meta"][f"{name}_no_positive_queries"] = no_pos

    qdict = dict(queries)
    for name, key in [("doc2query", "doc2query_scores")]:
        if _score_key_has_nonzero_signal(qdict, key):
            logger.info(f"  Evaluating {name}...")
            m, pq, no_pos = metrics_from_score_key(queries, key, collect_per_query)
            if collect_per_query and pq is not None:
                m["_per_query"] = pq
            results[name] = m
            results["_meta"][f"{name}_no_positive_queries"] = no_pos

    if model is not None and device is not None and gardian_features:
        logger.info(
            f"  Evaluating GARDIAN (ablation={gardian_ablation!r})..."
        )
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
                batch_size = len(qdata["candidates"])
                sparse_batch = torch.stack(qdata["sparse_feats"]).to(device)
                dense_batch = torch.stack(qdata["dense_feats"]).to(device)
                kg_batch = torch.stack(qdata["kg_feats"]).to(device)
                query_emb = qdata["query_emb"].unsqueeze(0).expand(batch_size, -1).to(device)
                qtype_onehot = qdata["qtype_onehot"].unsqueeze(0).expand(batch_size, -1).to(device)
                kg_coverage = torch.full(
                    (batch_size,),
                    qdata["kg_coverage"],
                    dtype=torch.float32,
                ).to(device)
                scores, _ = model(
                    sparse_feats=sparse_batch,
                    dense_feats=dense_batch,
                    kg_feats=kg_batch,
                    query_emb=query_emb,
                    qtype_onehot=qtype_onehot,
                    kg_coverage=kg_coverage,
                    ablation=gardian_ablation,
                )
                gardian_results[qid]["candidates"] = qdata["candidates"]
                gardian_results[qid]["scores"] = scores.cpu().numpy().flatten()

        keys = ["ndcg@5", "ndcg@10", "ndcg@20", "recall@5", "recall@20", "mrr"]
        lists: Dict[str, List[float]] = {k: [] for k in keys}
        no_positive_queries = 0
        for qid, qdata in gardian_results.items():
            if not qdata["scores"].size:
                continue
            sorted_indices = np.argsort(qdata["scores"])[::-1]
            ranked_ids = [qdata["candidates"][i]["pid"] for i in sorted_indices]
            relevant_ids = [c["pid"] for c in qdata["candidates"] if c["label"] == 1]
            if not relevant_ids:
                no_positive_queries += 1
            lists["ndcg@5"].append(compute_ndcg(ranked_ids, relevant_ids, 5))
            lists["ndcg@10"].append(compute_ndcg(ranked_ids, relevant_ids, 10))
            lists["ndcg@20"].append(compute_ndcg(ranked_ids, relevant_ids, 20))
            lists["recall@5"].append(compute_recall(ranked_ids, relevant_ids, 5))
            lists["recall@20"].append(compute_recall(ranked_ids, relevant_ids, 20))
            lists["mrr"].append(compute_mrr(ranked_ids, relevant_ids))

        results["gardian"] = _aggregate_lists(lists)
        results["gardian"]["mrr@10"] = results["gardian"]["mrr"]
        if collect_per_query:
            results["gardian"]["_per_query"] = lists
        results["_meta"]["gardian_no_positive_queries"] = no_positive_queries
    else:
        results["_meta"]["gardian_no_positive_queries"] = query_count

    return results
