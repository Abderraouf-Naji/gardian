"""On-disk index size reporting for hybrid retrieval families."""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Tuple

from src.common.hybrid_retrievers import FOCUS_HYBRID_RETRIEVERS, SPARSE_DENSE_COMPONENTS
from src.retrieval.index_paths import DATASET_INDEX_KEYS, resolve_dataset_index_paths

PAPER_DATASETS = [
    "pubmedqa_labeled",
    "pubmedqa_artificial",
    "medmcqa",
]


def path_disk_bytes(path: pathlib.Path) -> int:
    """Total bytes for a file or directory tree."""
    p = pathlib.Path(path)
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    total = 0
    for child in p.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _sparse_index_path(paths: Dict[str, str], sparse_component: str) -> pathlib.Path:
    if sparse_component == "bm25":
        return pathlib.Path(paths["bm25_index_dir"])
    return pathlib.Path(paths["spladepp_dir"])


def _dense_index_path(paths: Dict[str, str], dense_component: str) -> pathlib.Path:
    if dense_component == "faiss":
        return pathlib.Path(paths["faiss_index"]).parent
    return pathlib.Path(paths["medcpt_dir"])


def index_size_row(
    retriever: str,
    dataset: str,
) -> Dict[str, Any]:
    """Disk usage for one hybrid family on one benchmark corpus."""
    parts = SPARSE_DENSE_COMPONENTS[retriever]
    paths = resolve_dataset_index_paths(dataset)
    sparse_p = _sparse_index_path(paths, parts["sparse"])
    dense_p = _dense_index_path(paths, parts["dense"])
    sparse_b = path_disk_bytes(sparse_p)
    dense_b = path_disk_bytes(dense_p)
    total_b = sparse_b + dense_b
    return {
        "dataset": dataset,
        "sparse_component": parts["sparse"],
        "dense_component": parts["dense"],
        "sparse_path": str(sparse_p),
        "dense_path": str(dense_p),
        "sparse_disk_bytes": sparse_b,
        "dense_disk_bytes": dense_b,
        "total_disk_bytes": total_b,
        "sparse_disk_mb": round(sparse_b / (1024**2), 2),
        "dense_disk_mb": round(dense_b / (1024**2), 2),
        "total_disk_mb": round(total_b / (1024**2), 2),
        "sparse_disk_gb": round(sparse_b / (1024**3), 3),
        "dense_disk_gb": round(dense_b / (1024**3), 3),
        "total_disk_gb": round(total_b / (1024**3), 3),
    }


def report_all_index_sizes(
    retrievers: List[str] | None = None,
    datasets: List[str] | None = None,
) -> Dict[str, Any]:
    """Nested dict: retriever -> dataset -> size row."""
    retrievers = retrievers or list(FOCUS_HYBRID_RETRIEVERS)
    datasets = datasets or list(PAPER_DATASETS)
    out: Dict[str, Any] = {"datasets": datasets, "retrievers": {}}
    for retriever in retrievers:
        out["retrievers"][retriever] = {}
        for dataset in datasets:
            if dataset not in DATASET_INDEX_KEYS:
                continue
            out["retrievers"][retriever][dataset] = index_size_row(retriever, dataset)
    return out


def format_index_size_table(report: Dict[str, Any]) -> str:
    lines = [
        f"{'Retriever':<28} {'Dataset':<22} {'Sparse MB':>10} {'Dense MB':>10} {'Total MB':>10}",
        "-" * 84,
    ]
    for retriever, ds_block in report.get("retrievers", {}).items():
        for dataset, row in ds_block.items():
            lines.append(
                f"{retriever:<28} {dataset:<22} "
                f"{row['sparse_disk_mb']:>10.1f} {row['dense_disk_mb']:>10.1f} "
                f"{row['total_disk_mb']:>10.1f}"
            )
    return "\n".join(lines)
