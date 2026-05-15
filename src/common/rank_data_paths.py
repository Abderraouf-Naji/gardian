from __future__ import annotations

from pathlib import Path

# Canonical retriever family names used for files/checkpoints.
RETRIEVER_CANONICAL = {
    "hybrid_bm25_faiss",
    "hybrid_bm25_medcpt",
    "hybrid_spladepp_faiss",
    "hybrid_spladepp_medcpt",
    "bm25",
    "faiss",
    "spladepp",
    "medcpt",
}

# Backward-compatible aliases accepted by CLI and path resolver.
RETRIEVER_ALIASES = {
    "hybrid": "hybrid_bm25_faiss",
    "hybrid_neural": "hybrid_spladepp_medcpt",
}

HYBRID_FAMILIES = {
    "hybrid_bm25_faiss",
    "hybrid_bm25_medcpt",
    "hybrid_spladepp_faiss",
    "hybrid_spladepp_medcpt",
}


def normalize_retriever_name(retriever: str) -> str:
    r = str(retriever).strip()
    return RETRIEVER_ALIASES.get(r, r)


def rank_data_dir(retriever: str) -> Path:
    retriever = normalize_retriever_name(retriever)
    base = Path("data")
    return base / retriever if retriever in HYBRID_FAMILIES else base


def rank_data_file(retriever: str, dataset: str, split: str) -> str:
    retriever = normalize_retriever_name(retriever)
    p = rank_data_dir(retriever) / f"rank_data_{retriever}_{dataset}_{split}.jsonl"
    return str(p)


def rank_data_combined_file(retriever: str, suffix: str) -> str:
    retriever = normalize_retriever_name(retriever)
    p = rank_data_dir(retriever) / f"rank_data_{retriever}_{suffix}.jsonl"
    return str(p)


def legacy_rank_data_file(retriever: str, dataset: str, split: str) -> str:
    return str(Path("data") / f"rank_data_{retriever}_{dataset}_{split}.jsonl")


def resolve_rank_data_file(retriever: str, dataset: str, split: str) -> str:
    r = normalize_retriever_name(retriever)
    new_path = Path(rank_data_file(r, dataset, split))
    if new_path.exists():
        return str(new_path)
    # Try direct legacy for canonical name, then for the original alias.
    old_path = Path(legacy_rank_data_file(r, dataset, split))
    if old_path.exists():
        return str(old_path)
    old_alias = Path(legacy_rank_data_file(retriever, dataset, split))
    if old_alias.exists():
        return str(old_alias)
    return str(new_path)
