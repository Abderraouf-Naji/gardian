"""BM25 / FAISS / MedCPT / SPLADE++ index paths (per-dataset or unified)."""

from __future__ import annotations

import pathlib
from typing import Any, Dict, Optional

# QA dataset name → index folder under data/indices/{bm25,faiss,...}
DATASET_INDEX_KEYS: Dict[str, str] = {
    "pubmedqa_labeled": "pubmedqa_labeled",
    "pubmedqa_artificial": "pubmedqa_artificial",
    "medmcqa": "medmcqa",
    "medqa": "medmcqa",
}

INDICES_ROOT = pathlib.Path("data/indices")


def qa_index_key(dataset_name: str) -> str:
    key = (dataset_name or "").strip().lower()
    if key not in DATASET_INDEX_KEYS:
        raise ValueError(
            f"No per-dataset index mapping for {dataset_name!r}. "
            f"Known: {sorted(DATASET_INDEX_KEYS)}"
        )
    return DATASET_INDEX_KEYS[key]


def resolve_dataset_index_paths(dataset_key: str) -> Dict[str, str]:
    """
    Paths for one benchmark corpus (matches ``scripts/03_generate_rank_data.py``).

    Returns bm25_dir, faiss_index, faiss_meta, spladepp_dir, medcpt_dir.
    """
    dk = qa_index_key(dataset_key) if dataset_key in DATASET_INDEX_KEYS else dataset_key
    root = INDICES_ROOT
    bm25_dir = root / "bm25" / dk
    return {
        "dataset_index_key": dk,
        "bm25_index_dir": str(bm25_dir),
        "bm25_index_pkl": str(bm25_dir / "index.pkl"),
        "faiss_index": str(root / "faiss" / dk / "faiss.index"),
        "faiss_meta": str(root / "faiss" / dk / "faiss_meta.jsonl"),
        "spladepp_dir": str(root / "spladepp" / dk),
        "medcpt_dir": str(root / "medcpt" / dk),
    }


def resolve_retrieval_paths_for_qa(
    cfg: Any,
    *,
    dataset_name: Optional[str] = None,
    use_per_dataset: bool = True,
) -> Dict[str, str]:
    """
    Live QA paths: per-dataset indices when ``dataset_name`` is set and
    ``use_per_dataset`` is true; otherwise unified paths from config.
    """
    if dataset_name and use_per_dataset:
        per = resolve_dataset_index_paths(dataset_name)
        per["index_scope"] = "per_dataset"
        return per
    from src.pipeline.rag_reader import resolve_retrieval_paths

    unified = resolve_retrieval_paths(cfg)
    unified["index_scope"] = "unified"
    unified["dataset_index_key"] = "unified"
    return unified


def assert_dataset_indices_exist(paths: Dict[str, str], *, retriever: str) -> None:
    """Raise FileNotFoundError with a clear message if required artifacts are missing."""
    from src.pipeline.rank_dense_features import uses_faiss_dense, uses_medcpt_dense

    bm25_pkl = pathlib.Path(paths["bm25_index_pkl"])
    if not bm25_pkl.is_file():
        raise FileNotFoundError(f"BM25 index missing for dataset scope: {bm25_pkl}")
    if uses_medcpt_dense(retriever):
        from src.retrieval.medcpt import MedCPTRetriever

        med = pathlib.Path(paths["medcpt_dir"])
        if not MedCPTRetriever.index_ready(str(med)):
            raise FileNotFoundError(f"MedCPT index missing: {med}")
    elif uses_faiss_dense(retriever):
        faiss_p = pathlib.Path(paths["faiss_index"])
        meta_p = pathlib.Path(paths["faiss_meta"])
        if not faiss_p.is_file():
            raise FileNotFoundError(f"FAISS index missing: {faiss_p}")
        if not meta_p.is_file():
            raise FileNotFoundError(f"FAISS meta missing: {meta_p}")
    else:
        spl = pathlib.Path(paths["spladepp_dir"])
        if not spl.is_dir():
            raise FileNotFoundError(f"SPLADE++ index dir missing: {spl}")
