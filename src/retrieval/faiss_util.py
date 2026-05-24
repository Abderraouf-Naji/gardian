"""FAISS index access on CPU, native FAISS-GPU, or PyTorch CUDA (no faiss-gpu pip needed)."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from loguru import logger


def faiss_gpu_available() -> bool:
    try:
        import faiss

        return hasattr(faiss, "StandardGpuResources")
    except ImportError:
        return False


def _unwrap_flat_index(index: Any) -> Any:
    """Return underlying IndexFlat* when wrapped (IDMap, etc.)."""
    import faiss

    cur = index
    for _ in range(8):
        if isinstance(cur, faiss.IndexFlat):
            return cur
        if not hasattr(cur, "index"):
            break
        cur = faiss.downcast_index(cur.index)
    return index


def extract_index_vectors(index: Any) -> np.ndarray:
    """Materialize all passage vectors from a built Flat-IP index."""
    import faiss

    flat = _unwrap_flat_index(index)
    ntotal = int(index.ntotal)
    dim = int(index.d)
    if ntotal == 0:
        return np.zeros((0, dim), dtype=np.float32)

    if hasattr(index, "reconstruct_n"):
        try:
            out = index.reconstruct_n(0, ntotal)
            return np.asarray(out, dtype=np.float32).reshape(ntotal, dim)
        except Exception:
            pass

    if isinstance(flat, faiss.IndexFlat) and hasattr(flat, "xb"):
        try:
            xb = faiss.vector_to_array(flat.xb)
            return np.asarray(xb, dtype=np.float32).reshape(ntotal, dim)
        except Exception:
            pass

    out = np.empty((ntotal, dim), dtype=np.float32)
    for i in range(ntotal):
        index.reconstruct(i, out[i])
    return out


class FaissSearchHandle:
    """Unified search + reconstruct API used by DenseRetriever and rank features."""

    def __init__(self, ntotal: int, backend: str) -> None:
        self.ntotal = int(ntotal)
        self.backend = str(backend)

    def search(self, query_vecs: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def reconstruct(self, row: int) -> np.ndarray:
        raise NotImplementedError


class _CpuFaissHandle(FaissSearchHandle):
    def __init__(self, index: Any) -> None:
        super().__init__(int(index.ntotal), "cpu")
        self._index = index

    def search(self, query_vecs: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        q = np.asarray(query_vecs, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        return self._index.search(q, int(top_k))

    def reconstruct(self, row: int) -> np.ndarray:
        return np.asarray(self._index.reconstruct(int(row)), dtype=np.float32)


class _TorchCudaHandle(FaissSearchHandle):
    """Exact Flat inner-product search on GPU via torch.matmul (faiss-cpu only)."""

    def __init__(self, vectors: np.ndarray, gpu_id: int = 0) -> None:
        import torch

        super().__init__(int(vectors.shape[0]), f"torch-cuda:{int(gpu_id)}")
        dev = torch.device(f"cuda:{int(gpu_id)}")
        self._dev = dev
        self._vectors = torch.from_numpy(np.asarray(vectors, dtype=np.float32)).to(dev)
        norms = torch.linalg.norm(self._vectors, dim=1, keepdim=True).clamp_min(1e-8)
        self._vectors = self._vectors / norms

    def search(self, query_vecs: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        import torch

        q = np.asarray(query_vecs, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        qt = torch.from_numpy(q).to(self._dev)
        qt = qt / torch.linalg.norm(qt, dim=1, keepdim=True).clamp_min(1e-8)
        scores = qt @ self._vectors.T
        k = min(int(top_k), int(self.ntotal))
        vals, idxs = torch.topk(scores, k=k, dim=1)
        return vals.detach().cpu().numpy(), idxs.detach().cpu().numpy().astype(np.int64)

    def reconstruct(self, row: int) -> np.ndarray:
        return self._vectors[int(row)].detach().cpu().numpy().astype(np.float32)


class _NativeFaissGpuHandle(FaissSearchHandle):
    def __init__(self, gpu_index: Any, gpu_id: int) -> None:
        super().__init__(int(gpu_index.ntotal), f"faiss-cuda:{int(gpu_id)}")
        self._index = gpu_index

    def search(self, query_vecs: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        q = np.asarray(query_vecs, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        return self._index.search(q, int(top_k))

    def reconstruct(self, row: int) -> np.ndarray:
        return np.asarray(self._index.reconstruct(int(row)), dtype=np.float32)


def open_faiss_index(
    faiss_index_path: str,
    *,
    use_gpu: bool = True,
    gpu_id: int = 0,
) -> FaissSearchHandle:
    """
    Open a FAISS index for search/reconstruct.

    Priority when ``use_gpu`` and CUDA is available:
      1. Native ``faiss-gpu`` / ``faiss-gpu-cu12`` (StandardGpuResources)
      2. PyTorch matmul on GPU (works with ``faiss-cpu`` only)
      3. CPU FAISS
    """
    import faiss

    try:
        import torch
    except ImportError:
        torch = None  # type: ignore

    cpu_index = faiss.read_index(str(faiss_index_path))
    if not use_gpu:
        logger.info(f"FAISS backend: cpu ({cpu_index.ntotal:,} vectors)")
        return _CpuFaissHandle(cpu_index)

    cuda_ok = torch is not None and torch.cuda.is_available()
    if not cuda_ok:
        logger.warning("FAISS GPU requested but torch CUDA unavailable; using CPU FAISS")
        return _CpuFaissHandle(cpu_index)

    if faiss_gpu_available():
        try:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, int(gpu_id), cpu_index)
            logger.info(
                f"FAISS backend: faiss-cuda:{gpu_id} ({gpu_index.ntotal:,} vectors)"
            )
            return _NativeFaissGpuHandle(gpu_index, gpu_id)
        except Exception as exc:
            logger.warning(f"Native FAISS GPU failed ({exc})")

    import os

    if os.environ.get("FAISS_TORCH_CUDA_FALLBACK", "").strip() in ("1", "true", "yes"):
        logger.info(
            f"FAISS backend: torch-cuda:{gpu_id} — loading {cpu_index.ntotal:,} "
            "vectors to GPU (FAISS_TORCH_CUDA_FALLBACK=1)"
        )
        vectors = extract_index_vectors(cpu_index)
        return _TorchCudaHandle(vectors, gpu_id=int(gpu_id))

    logger.warning(
        f"FAISS GPU unavailable; using CPU FAISS ({cpu_index.ntotal:,} vectors). "
        "Set FAISS_TORCH_CUDA_FALLBACK=1 to retry torch matmul on GPU (needs ~9GB+ free VRAM)."
    )
    return _CpuFaissHandle(cpu_index)


# Backward-compatible helper used during migration
def load_faiss_index(
    faiss_index_path: str,
    *,
    use_gpu: bool = True,
    gpu_id: int = 0,
) -> Tuple[Any, str]:
    handle = open_faiss_index(faiss_index_path, use_gpu=use_gpu, gpu_id=gpu_id)
    return handle, handle.backend
