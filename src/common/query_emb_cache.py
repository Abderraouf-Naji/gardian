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
    payload = normalize_emb_cache(cache, expected_dim=expected_dim)
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
