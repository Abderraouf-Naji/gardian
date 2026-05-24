"""Hybrid retriever families used for rank-data generation, training, and evaluation."""

from __future__ import annotations

from typing import Dict

HYBRID_RETRIEVER_COMBINATIONS: Dict[str, str] = {
    "hybrid_bm25_faiss": "BM25 + FAISS",
    "hybrid_bm25_medcpt": "BM25 + MedCPT",
    "hybrid_spladepp_faiss": "SPLADE++ + FAISS",
    "hybrid_spladepp_medcpt": "SPLADE++ + MedCPT",
}

# One GARDIAN checkpoint per hybrid row (see scripts/04_train_gardian.py).
FOCUS_HYBRID_RETRIEVERS = list(HYBRID_RETRIEVER_COMBINATIONS.keys())

# Single-retriever rank-data used as sparse/dense baselines on the hybrid pool.
SPARSE_DENSE_COMPONENTS: Dict[str, Dict[str, str]] = {
    "hybrid_bm25_faiss": {"sparse": "bm25", "dense": "faiss"},
    "hybrid_bm25_medcpt": {"sparse": "bm25", "dense": "medcpt"},
    "hybrid_spladepp_faiss": {"sparse": "spladepp", "dense": "faiss"},
    "hybrid_spladepp_medcpt": {"sparse": "spladepp", "dense": "medcpt"},
}
