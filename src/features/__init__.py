"""
Feature extraction for the sparse and dense branches.

  schema     the authoritative per-dimension layout of both feature vectors
  sparse     lexical features (BM25/SPLADE++ score, IDF overlap, Jaccard)
  dense_feat semantic features (dense score, embedding distances)
  pool_norm  within-pool normalisation shared by both channels

``schema.assert_feature_dims`` is the guard that keeps the config, the rank data
and these modules in agreement.
"""

from src.features.schema import (
    DENSE_FEATURE_NAMES,
    DENSE_FEAT_DIM,
    SPARSE_FEATURE_NAMES,
    SPARSE_FEAT_DIM,
    assert_feature_dims,
)

__all__ = [
    "SPARSE_FEATURE_NAMES",
    "DENSE_FEATURE_NAMES",
    "SPARSE_FEAT_DIM",
    "DENSE_FEAT_DIM",
    "assert_feature_dims",
]
