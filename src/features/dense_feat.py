"""
Dense semantic feature vector for a (query, passage) pair.

Features (paper Eq. 4):
  [0] cosine similarity
  [1] element-wise mean of |q_emb - p_emb|
  [2] element-wise max  of |q_emb - p_emb|   (scalar summary)
  [3] z-normalised cosine within candidate set
"""

import numpy as np
from typing import List


def compute_dense_features(q_emb: np.ndarray, p_emb: np.ndarray,
                            cosine_mean: float = 0.0,
                            cosine_std: float = 1.0) -> np.ndarray:
    """
    q_emb, p_emb : L2-normalised embeddings (float32)
    cosine_mean/std : statistics over the full candidate set for calibration
    """
    cosine = float(np.dot(q_emb, p_emb))    # fast for unit vectors

    diff   = np.abs(q_emb - p_emb)
    mean_d = float(diff.mean())
    max_d  = float(diff.max())

    # z-normalised cosine
    z_cos  = (cosine - cosine_mean) / (cosine_std + 1e-8)

    return np.array([cosine, mean_d, max_d, z_cos], dtype=np.float32)


def batch_cosines(q_emb: np.ndarray, p_embs: np.ndarray) -> np.ndarray:
    """Compute cosine similarities of one query against many passages."""
    # Both assumed unit-normalised
    return (p_embs @ q_emb).astype(np.float32)