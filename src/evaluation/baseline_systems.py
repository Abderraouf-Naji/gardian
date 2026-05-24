"""Canonical evaluation systems: sparse, dense, hybrid (sum), RRF, GARDIAN."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.common.hybrid_retrievers import SPARSE_DENSE_COMPONENTS

# Reported systems (same candidate pool for hybrid family eval).
EVAL_SYSTEMS: Tuple[str, ...] = ("sparse", "dense", "hybrid", "rrf", "gardian")


def _is_metric_block(v: Any) -> bool:
    return isinstance(v, dict) and "ndcg@10" in v


def _pick_block(raw: Dict[str, Any], keys: List[str]) -> Optional[Dict[str, Any]]:
    for k in keys:
        block = raw.get(k)
        if _is_metric_block(block):
            return block
    return None


def normalize_eval_results(
    raw: Dict[str, Any],
    retriever: str,
    *,
    include_cross_encoder: bool = False,
) -> Dict[str, Any]:
    """
    Map rank_jsonl_eval output to the five paper systems.

    - sparse / dense: first-stage channel scores (from hybrid JSONL or single-retriever files)
    - hybrid: fixed score-sum fusion (sparse_score + dense_score) on the hybrid pool
    - rrf: reciprocal rank fusion on the same pool
    - gardian: learned re-ranker
    """
    parts = SPARSE_DENSE_COMPONENTS.get(retriever, {"sparse": "bm25", "dense": "dense"})
    sp, de = parts["sparse"], parts["dense"]

    sparse = _pick_block(
        raw,
        [
            "sparse",
            "bm25",
            sp,
            f"sparse({sp})",
            "spladepp",
        ],
    )
    dense = _pick_block(
        raw,
        [
            "dense",
            de,
            f"dense({de})",
            "faiss",
            "medcpt",
        ],
    )
    hybrid = _pick_block(
        raw,
        [
            "hybrid",
            f"sum({sp},{de})",
            "fusion",
        ],
    )
    rrf = _pick_block(raw, ["rrf"])
    gardian = _pick_block(raw, ["gardian"])

    out: Dict[str, Any] = {}
    for name, block in zip(EVAL_SYSTEMS, (sparse, dense, hybrid, rrf, gardian)):
        if block is not None:
            out[name] = block

    if include_cross_encoder:
        cross_encoder = _pick_block(raw, ["cross_encoder"])
        if cross_encoder is not None:
            out["cross_encoder"] = cross_encoder

    meta = raw.get("_meta")
    if isinstance(meta, dict):
        out["_meta"] = dict(meta)
        out["_meta"]["eval_systems"] = list(EVAL_SYSTEMS)
        if include_cross_encoder and "cross_encoder" in out:
            out["_meta"]["eval_systems"] = list(EVAL_SYSTEMS) + ["cross_encoder"]
        out["_meta"]["sparse_component"] = sp
        out["_meta"]["dense_component"] = de

    return out


def display_name(system: str, retriever: str) -> str:
    parts = SPARSE_DENSE_COMPONENTS.get(retriever, {"sparse": "sparse", "dense": "dense"})
    if system == "sparse":
        return f"sparse({parts['sparse']})"
    if system == "dense":
        return f"dense({parts['dense']})"
    if system == "hybrid":
        return f"hybrid({parts['sparse']}+{parts['dense']}, sum)"
    if system == "rrf":
        return f"rrf({parts['sparse']}+{parts['dense']})"
    if system == "cross_encoder":
        return "cross_encoder"
    if system == "gardian":
        return "gardian"
    return system
