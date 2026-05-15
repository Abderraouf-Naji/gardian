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


def compute_dense_features_with_score(
    q_emb: np.ndarray,
    p_emb: np.ndarray,
    dense_score: float,
    score_mean: float = 0.0,
    score_std: float = 1.0,
) -> np.ndarray:
    """Dense branch features using the active dense retriever score.

    For FAISS hybrids, ``dense_score`` is the PubMedBERT/FAISS similarity. For
    MedCPT hybrids, it is the MedCPT asymmetric query/article dot product. The
    remaining two dimensions keep full passage-embedding distance features so
    we do not collapse the dense branch to a single score-only signal.

    Output:
      [0] active dense retriever score (FAISS or MedCPT)
      [1] mean absolute difference between query and cached passage embedding
      [2] max absolute difference between query and cached passage embedding
      [3] z-normalised active dense score within the candidate set
    """
    diff = np.abs(q_emb - p_emb)
    mean_d = float(diff.mean())
    max_d = float(diff.max())
    z_score = (float(dense_score) - score_mean) / (score_std + 1e-8)
    return np.array([float(dense_score), mean_d, max_d, z_score], dtype=np.float32)


def batch_cosines(q_emb: np.ndarray, p_embs: np.ndarray) -> np.ndarray:
    """Compute cosine similarities of one query against many passages."""
    # Both assumed unit-normalised
    return (p_embs @ q_emb).astype(np.float32)