"""
Retrieval evaluation metrics: Recall@k, MRR@k, nDCG@k.
Also provides evaluate_rank_data() used during training for early stopping.
"""

from __future__ import annotations

import json
import math
import numpy as np
from typing import Dict, List, Optional
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path

from src.common.question_types import normalize_question_type


def recall_at_k(ranked_ids: List[str], relevant_ids: set, k: int) -> float:
    return 1.0 if any(r in relevant_ids for r in ranked_ids[:k]) else 0.0


def mrr_at_k(ranked_ids: List[str], relevant_ids: set, k: int) -> float:
    for i, rid in enumerate(ranked_ids[:k]):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked_ids: List[str], relevant_ids: set, k: int) -> float:
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, r in enumerate(ranked_ids[:k])
        if r in relevant_ids
    )
    idcg = sum(
        1.0 / math.log2(i + 2)
        for i in range(min(len(relevant_ids), k))
    )
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    results: List[Dict],
    cutoffs=(5, 10, 20),
    by_qtype: bool = True
) -> Dict:

    metrics = {k: {"recall": [], "mrr": [], "ndcg": []} for k in cutoffs}
    qtype_metrics = {}

    for item in results:
        ranked = item["ranked_ids"]
        relevant = set(item["relevant_ids"])
        qtype = normalize_question_type(item.get("question_type", "other"))

        if qtype not in qtype_metrics:
            qtype_metrics[qtype] = {k: {"ndcg": []} for k in cutoffs}

        for k in cutoffs:
            r = recall_at_k(ranked, relevant, k)
            m = mrr_at_k(ranked, relevant, k)
            n = ndcg_at_k(ranked, relevant, k)

            metrics[k]["recall"].append(r)
            metrics[k]["mrr"].append(m)
            metrics[k]["ndcg"].append(n)
            qtype_metrics[qtype][k]["ndcg"].append(n)

    summary = {}

    for k in cutoffs:
        summary[f"recall@{k}"] = float(np.mean(metrics[k]["recall"]))
        summary[f"mrr@{k}"] = float(np.mean(metrics[k]["mrr"]))
        summary[f"ndcg@{k}"] = float(np.mean(metrics[k]["ndcg"]))

    if by_qtype:
        for qtype, data in qtype_metrics.items():
            for k in cutoffs:
                vals = data[k]["ndcg"]
                summary[f"ndcg@{k}_{qtype}"] = float(np.mean(vals)) if vals else 0.0

    return summary


def evaluate_rank_data(
    model,
    dev_path: str,
    device: str,
    k: int = 10,
    ablation: Optional[str] = None,
    query_encoder_name: Optional[str] = None,
    query_encoder_device: str = "cpu",
    query_emb_cache_path: Optional[str] = None,
) -> float:
    """Compute mean nDCG@k over queries from rank JSONL (early stopping)."""

    from collections import defaultdict

    model.eval()
    query_scores: Dict[str, List] = defaultdict(list)
    query_emb_cache: Dict[str, List[float]] = {}
    query_encoder = None

    if query_emb_cache_path:
        p = Path(query_emb_cache_path)
        if p.exists():
            try:
                with p.open("rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    query_emb_cache = {str(k): v for k, v in data.items()}
                    logger.info(
                        f"Loaded query_emb cache for evaluation: {len(query_emb_cache):,} queries"
                    )
            except Exception as e:
                logger.warning(f"Could not load query_emb cache {query_emb_cache_path}: {e}")

    def _model_query_dim(rec: Dict) -> int:
        qtype_dim = len(rec.get("qtype_onehot", []))
        in_features = model.controller.net[0].in_features
        return int(in_features) - int(qtype_dim) - 1

    def _resolve_query_emb(rec: Dict) -> List[float]:
        q = rec.get("query_emb")
        if isinstance(q, list):
            return q
        qid = str(rec.get("qid", ""))
        if qid in query_emb_cache:
            return query_emb_cache[qid]
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
        query_emb_cache[qid] = emb
        return emb

    with torch.no_grad(), open(dev_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            t = lambda x: torch.tensor([x], dtype=torch.float32, device=device)
            q_emb = _resolve_query_emb(rec)

            s, _ = model(
                sparse_feats=t(rec["sparse_feats"]),
                dense_feats=t(rec["dense_feats"]),
                kg_feats=t(rec["kg_feats"]),
                query_emb=t(q_emb),
                qtype_onehot=t(rec["qtype_onehot"]),
                kg_coverage=torch.tensor(
                    [rec["kg_coverage"]],
                    dtype=torch.float32,
                    device=device
                ),
                ablation=ablation,
            )

            query_scores[rec["qid"]].append(
                (float(s[0]), rec["pid"], rec["label"])
            )

    ndcgs = []

    for qid, scored in query_scores.items():
        if not scored:
            continue

        scored.sort(key=lambda x: x[0], reverse=True)

        ranked = [pid for _, pid, _ in scored]
        relevant = {pid for _, pid, lbl in scored if lbl == 1}

        ndcgs.append(ndcg_at_k(ranked, relevant, k))

    result = float(np.mean(ndcgs)) if ndcgs else 0.0

    logger.info(f"Evaluation nDCG@{k}: {result:.4f}")

    return result
