"""Pickle query-embedding caches (qid -> list[float]) with safe atomic writes."""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
from loguru import logger


def normalize_emb_vector(value) -> List[float]:
    """Convert numpy / tensor / list embeddings to a plain picklable list[float]."""
    if value is None:
        raise ValueError("embedding value is None")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("embedding vector is empty")
    return arr.tolist()


def normalize_emb_cache(
    data: Mapping,
    *,
    expected_dim: Optional[int] = None,
) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for k, v in data.items():
        emb = normalize_emb_vector(v)
        if expected_dim is not None and len(emb) != int(expected_dim):
            raise ValueError(
                f"query_emb dim mismatch for qid={k!r}: got {len(emb)} expected {expected_dim}"
            )
        out[str(k)] = emb
    return out


def load_query_emb_cache(
    path: Union[str, Path],
    *,
    expected_dim: Optional[int] = None,
) -> Dict[str, List[float]]:
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        with p.open("rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        logger.warning(f"Could not load query_emb cache {p}: {exc}")
        return {}
    if not isinstance(data, dict):
        logger.warning(f"query_emb cache {p} is not a dict; ignoring")
        return {}
    try:
        return normalize_emb_cache(data, expected_dim=expected_dim)
    except Exception as exc:
        logger.warning(f"query_emb cache {p} failed normalization: {exc}")
        return {}


def save_query_emb_cache(
    path: Union[str, Path],
    cache: Mapping[str, object],
    *,
    expected_dim: Optional[int] = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Persist float32 arrays, not lists of Python floats. A list costs ~24 bytes
    # per dimension and pickle.load must materialise every one of them before a
    # reader can convert, so a 186k-query cache peaked at 6.2 GB to produce
    # 0.53 GB of data. As float32 the file is ~2x smaller and loads at roughly
    # its on-disk size. Readers are unaffected: load_query_emb_cache and
    # load_query_emb_store both pass values through np.asarray.
    payload = {}
    for k, v in cache.items():
        arr = np.asarray(normalize_emb_vector(v), dtype=np.float32).reshape(-1)
        if expected_dim is not None and arr.size != int(expected_dim):
            raise ValueError(
                f"query_emb dim mismatch for qid={k!r}: got {arr.size} expected {expected_dim}"
            )
        payload[str(k)] = arr
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{p.stem}.",
        suffix=".tmp",
        dir=str(p.parent),
    )
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, p)
        logger.info(f"Saved query_emb cache: {len(payload):,} queries -> {p}")
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def merge_query_emb_caches(
    paths: Sequence[Union[str, Path]],
    out_path: Union[str, Path],
    *,
    expected_dim: Optional[int] = None,
) -> int:
    merged: Dict[str, List[float]] = {}
    for path in paths:
        merged.update(load_query_emb_cache(path, expected_dim=expected_dim))
    save_query_emb_cache(out_path, merged, expected_dim=expected_dim)
    return len(merged)


class QueryEmbStore:
    """``qid -> embedding`` lookup backed by one contiguous float32 matrix.

    A ``dict[str, list[float]]`` holds one refcounted Python float per
    dimension: 186k x 768 embeddings cost ~4.5 GB, and any DataLoader worker
    that reads them writes refcounts into those pages, so a forked worker ends
    up with a private copy of the whole cache. The same data is 0.55 GB here,
    and a lookup touches the refcount of a single object (the matrix), so the
    pages stay shared between workers.

    Entries added at runtime land in a small overflow dict, bounded by
    ``extra_max`` (FIFO) so a long run cannot grow without limit.
    """

    __slots__ = ("dim", "_index", "_matrix", "_extra", "_extra_max")

    def __init__(
        self,
        index: Optional[Dict[str, int]] = None,
        matrix: Optional[np.ndarray] = None,
        dim: Optional[int] = None,
        *,
        extra_max: Optional[int] = None,
    ):
        self._index = index or {}
        self._matrix = matrix
        if dim is not None:
            self.dim = int(dim)
        elif matrix is not None:
            self.dim = int(matrix.shape[1])
        else:
            self.dim = 0
        self._extra: Dict[str, np.ndarray] = {}
        self._extra_max = (
            None if extra_max in (None, 0) else max(1, int(extra_max))
        )

    @classmethod
    def empty(
        cls,
        dim: Optional[int] = None,
        *,
        extra_max: Optional[int] = None,
    ) -> "QueryEmbStore":
        return cls(index={}, matrix=None, dim=dim, extra_max=extra_max)

    def __len__(self) -> int:
        return len(self._index) + len(self._extra)

    def __contains__(self, qid: object) -> bool:
        key = str(qid)
        return key in self._index or key in self._extra

    def get(self, qid: str) -> Optional[np.ndarray]:
        """Read-only view of one embedding, or ``None`` when unknown."""
        key = str(qid)
        row = self._index.get(key)
        if row is not None and self._matrix is not None:
            return self._matrix[row]
        return self._extra.get(key)

    def set(self, qid: str, value) -> np.ndarray:
        key = str(qid)
        arr = np.asarray(normalize_emb_vector(value), dtype=np.float32)
        if self.dim and arr.size != self.dim:
            raise ValueError(
                f"query_emb dim mismatch for qid={key!r}: "
                f"got {arr.size} expected {self.dim}"
            )
        if not self.dim:
            self.dim = int(arr.size)
        self._extra[key] = arr
        if self._extra_max:
            while len(self._extra) > self._extra_max:
                self._extra.pop(next(iter(self._extra)))
        return arr

    def items(self):
        """Yield ``(qid, ndarray)`` pairs; safe to hand to :func:`save_query_emb_cache`."""
        if self._matrix is not None:
            for qid, row in self._index.items():
                yield qid, self._matrix[row]
        for qid, arr in self._extra.items():
            yield qid, arr

    @property
    def nbytes(self) -> int:
        base = 0 if self._matrix is None else int(self._matrix.nbytes)
        return base + sum(int(a.nbytes) for a in self._extra.values())


def load_query_emb_store(
    path: Union[str, Path],
    *,
    expected_dim: Optional[int] = None,
    extra_max: Optional[int] = None,
) -> QueryEmbStore:
    """Load a pickled cache straight into a :class:`QueryEmbStore`.

    Values are moved out of the unpickled dict one at a time so the list-of-
    floats representation is never held in full alongside the matrix.
    """
    p = Path(path)
    if not p.is_file():
        return QueryEmbStore.empty(expected_dim, extra_max=extra_max)
    try:
        with p.open("rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        logger.warning(f"Could not load query_emb cache {p}: {exc}")
        return QueryEmbStore.empty(expected_dim, extra_max=extra_max)
    if not isinstance(data, dict):
        logger.warning(f"query_emb cache {p} is not a dict; ignoring")
        return QueryEmbStore.empty(expected_dim, extra_max=extra_max)
    if not data:
        return QueryEmbStore.empty(expected_dim, extra_max=extra_max)

    n = len(data)
    if expected_dim is not None:
        dim = int(expected_dim)
    else:
        dim = int(np.asarray(next(iter(data.values()))).reshape(-1).size)

    matrix = np.empty((n, dim), dtype=np.float32)
    index: Dict[str, int] = {}
    row = 0
    try:
        while data:
            qid, value = data.popitem()
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size != dim:
                raise ValueError(
                    f"query_emb dim mismatch for qid={qid!r}: "
                    f"got {arr.size} expected {dim}"
                )
            matrix[row] = arr
            index[str(qid)] = row
            row += 1
    except Exception as exc:
        logger.warning(f"query_emb cache {p} failed normalization: {exc}")
        return QueryEmbStore.empty(expected_dim, extra_max=extra_max)

    store = QueryEmbStore(index=index, matrix=matrix, dim=dim, extra_max=extra_max)
    logger.info(
        f"Loaded query_emb store: {len(store):,} queries x {dim} "
        f"({store.nbytes / 1024 ** 3:.2f} GB shared) from {p}"
    )
    return store
