"""
Sparse lexical feature vector for a (query, passage) pair.

Features (paper Eq. 3):
  [0] BM25 score (pre-computed by retriever)
  [1] IDF-weighted overlap over content tokens
  [2] Jaccard similarity over content word sets

Note: this repo's sparse branch is implemented as a 3-dim vector:
`[bm25_score, idf_overlap, jaccard]`.
"""

import math, re
from typing import List, Dict
import numpy as np

STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","to","of","in","for",
    "on","with","at","by","from","this","that","these","those",
    "and","or","not","but","if","then","as","it","its","their",
}


def _content_tokens(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def compute_sparse_features(query: str, passage: str,
                             bm25_score: float,
                             idf_table: Dict[str, float] = None) -> np.ndarray:
    """
    Returns a 3-dim float32 vector matching the paper's sparse feature branch.
    """
    q_toks = set(_content_tokens(query))
    p_toks = set(_content_tokens(passage))

    # (1) Token overlap
    overlap_tokens = q_toks & p_toks
    overlap_count  = float(len(overlap_tokens))

    # (2) IDF-weighted overlap
    if idf_table and overlap_tokens:
        idf_overlap = sum(idf_table.get(t, 1.0) for t in overlap_tokens)
    else:
        # Fallback: use log(1 + count) approximation
        idf_overlap = math.log1p(overlap_count)

    # (3) Jaccard
    union = q_toks | p_toks
    jaccard = overlap_count / len(union) if union else 0.0

    return np.array([bm25_score, idf_overlap, jaccard], dtype=np.float32)