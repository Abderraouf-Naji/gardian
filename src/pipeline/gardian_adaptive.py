"""
Query-adaptive retrieval + reranking for GARDIAN (live QA).

Unlike fixed hybrid (50+50 union + RRF with implicit equal channel weight), GARDIAN:

  1. Controller outputs query-specific (α_sparse, α_dense) — not fixed 0.5/0.5.
  2. Retrieves k_sparse = α·cap_sparse and k_dense = β·cap_dense from each index
     (caps from ``retrieval.top_k_bm25`` / ``top_k_faiss``, default 50 each).
  3. Merges to a modest candidate pool (typically tens of passages, not 100).
  4. Reranks with ``gardian_score = α·sparse_branch + β·dense_branch``.
  5. Reader receives only ``qa.top_k_passages`` (e.g. 6) after rerank.

Offline rank JSONL (script 03) may still use fixed 50+50 for training data generation;
live E2E QA uses this module when ``qa.gardian_adaptive_retrieval`` is true.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch


def _resolve_sparse_score(rec: Dict[str, Any]) -> float:
    if "bm25_score" in rec:
        return float(rec.get("bm25_score", 0.0))
    if "spladepp_score" in rec:
        return float(rec.get("spladepp_score", 0.0))
    sparse_feats = rec.get("sparse_feats") or [0.0]
    return float(sparse_feats[0] if sparse_feats else 0.0)


def _resolve_dense_score(rec: Dict[str, Any]) -> float:
    if "dense_score" in rec:
        return float(rec.get("dense_score", 0.0))
    if "medcpt_score" in rec:
        return float(rec.get("medcpt_score", 0.0))
    dense_feats = rec.get("dense_feats") or [0.0]
    return float(dense_feats[0] if dense_feats else 0.0)


def adaptive_channel_budgets(
    alpha_sparse: float,
    alpha_dense: float,
    cfg: Any,
    *,
    min_per_channel: int = 1,
) -> Tuple[int, int]:
    """
    Per-channel retrieval depth for live adaptive QA.

    Default ``proportional``: k_sparse = α·cap_sparse, k_dense = β·cap_dense
    (α,β from controller, caps = top_k_bm25 / top_k_faiss). Pool size adapts per query.

    ``full_caps`` ablation: always retrieve full caps (50+50); α,β only in rerank fusion.
    """
    r = cfg.retrieval
    cap_sparse = int(
        getattr(r, "top_k_bm25", None)
        or getattr(r, "top_k_spladepp", None)
        or 50
    )
    cap_dense = int(
        getattr(r, "top_k_faiss", None)
        or getattr(r, "top_k_medcpt", None)
        or getattr(r, "top_k_dense", None)
        or 50
    )
    mode = str(getattr(r, "adaptive_channel_budget", "proportional") or "proportional").strip().lower()
    if mode in ("full", "full_caps", "fixed", "50+50"):
        return cap_sparse, cap_dense
    s = float(alpha_sparse) + float(alpha_dense) + 1e-8
    k_sparse = max(
        min_per_channel,
        min(cap_sparse, int(round((float(alpha_sparse) / s) * cap_sparse))),
    )
    k_dense = max(
        min_per_channel,
        min(cap_dense, int(round((float(alpha_dense) / s) * cap_dense))),
    )
    return k_sparse, k_dense


def _merge_channel_hits(
    sparse_hits: List[Dict],
    dense_hits: List[Dict],
    *,
    first_score_key: str,
    second_score_key: str,
) -> List[Dict]:
    """Merge sparse + dense hit lists with RRF (same fusion as ``DualHybridRetriever``)."""
    sparse_map = {h["id"]: h for h in sparse_hits}
    dense_map = {h["id"]: h for h in dense_hits}
    sparse_rank = {h["id"]: i + 1 for i, h in enumerate(sparse_hits)}
    dense_rank = {h["id"]: i + 1 for i, h in enumerate(dense_hits)}
    all_ids = list(dict.fromkeys([h["id"] for h in sparse_hits] + [h["id"] for h in dense_hits]))
    rrf_k = 60.0
    out: List[Dict] = []
    for pid in all_ids:
        sh = sparse_map.get(pid, {})
        dh = dense_map.get(pid, {})
        s1 = float(sh.get("score", sh.get(first_score_key, 0.0)))
        s2 = float(dh.get("score", dh.get(second_score_key, 0.0)))
        r1 = sparse_rank.get(pid, 10**9)
        r2 = dense_rank.get(pid, 10**9)
        rrf = (1.0 / (rrf_k + r1)) + (1.0 / (rrf_k + r2))
        row = {
            "id": pid,
            "text": sh.get("text") or dh.get("text", ""),
            first_score_key: s1,
            second_score_key: s2,
            "hybrid_rrf_score": float(rrf),
        }
        if first_score_key == "bm25_score":
            row["bm25_score"] = s1
            row["dense_score"] = s2
        elif first_score_key == "spladepp_score":
            row["spladepp_score"] = s1
            row["medcpt_score"] = s2
        out.append(row)
    out.sort(
        key=lambda x: (
            float(x.get("hybrid_rrf_score", 0.0)),
            float(x.get(first_score_key, 0.0)),
            float(x.get(second_score_key, 0.0)),
        ),
        reverse=True,
    )
    return out


def retrieve_adaptive_candidates_live(
    query: str,
    retriever: Any,
    gardian_model: torch.nn.Module,
    *,
    query_emb: List[float],
    qtype_onehot: List[float],
    cfg: Any,
    device: str = "cuda",
    ablation: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], float, float]:
    """
    Live: α,β from controller → separate sparse/dense retrieve → merged candidate list.

    ``retriever`` must be a ``DualHybridRetriever`` subclass (``.first`` / ``.second``).
    """
    q_t = torch.tensor(query_emb, dtype=torch.float32, device=device)
    qt_t = torch.tensor(qtype_onehot, dtype=torch.float32, device=device)
    weights = gardian_model.controller_weights(q_t, qt_t, ablation=ablation)
    alpha = float(weights[0, 0].item())
    beta = float(weights[0, 1].item())
    k_sparse, k_dense = adaptive_channel_budgets(alpha, beta, cfg)

    first = getattr(retriever, "first", None)
    second = getattr(retriever, "second", None)
    if first is None or second is None:
        raise TypeError(
            "retrieve_adaptive_candidates_live requires DualHybridRetriever (.first / .second)"
        )

    sparse_hits = first.retrieve(query, k_sparse)
    dense_hits = second.retrieve(query, k_dense)
    sk = getattr(retriever, "first_score_key", "bm25_score")
    dk = getattr(retriever, "second_score_key", "dense_score")
    merged = _merge_channel_hits(
        sparse_hits, dense_hits, first_score_key=sk, second_score_key=dk
    )
    return merged, alpha, beta


def subset_rank_records_adaptive(
    candidates: List[Dict[str, Any]],
    alpha_sparse: float,
    alpha_dense: float,
    cfg: Any,
) -> List[Dict[str, Any]]:
    """
    Offline: from a hybrid rank pool, keep top-α_sparse (by sparse score) and
    top-α_dense (by dense score), then union (dedupe by pid).
    """
    k_sparse, k_dense = adaptive_channel_budgets(alpha_sparse, alpha_dense, cfg)
    sparse_sorted = sorted(candidates, key=_resolve_sparse_score, reverse=True)
    dense_sorted = sorted(candidates, key=_resolve_dense_score, reverse=True)
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for r in sparse_sorted[:k_sparse]:
        pid = str(r["pid"])
        if pid not in seen:
            seen.add(pid)
            out.append(r)
    for r in dense_sorted[:k_dense]:
        pid = str(r["pid"])
        if pid not in seen:
            seen.add(pid)
            out.append(r)
    return out


def controller_weights_from_lists(
    gardian_model: torch.nn.Module,
    query_emb: List[float],
    qtype_onehot: List[float],
    device: str,
    *,
    ablation: Optional[str] = None,
) -> Tuple[float, float]:
    q_t = torch.tensor(query_emb, dtype=torch.float32, device=device)
    qt_t = torch.tensor(qtype_onehot, dtype=torch.float32, device=device)
    w = gardian_model.controller_weights(q_t, qt_t, ablation=ablation)
    return float(w[0, 0].item()), float(w[0, 1].item())
