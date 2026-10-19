"""
Corpus IDF table for the sparse branch's IDF-weighted overlap feature.

The offline rank-data builder (scripts/03_generate_rank_data.py) and the live
end-to-end QA path (scripts/06_end_to_end_qa.py) must compute this feature
the same way. Before this module existed only the offline builder built the
table, and the live path passed ``idf_table=None``, so the feature silently
fell back to ``log(1 + overlap count)``. That changed the BM25+FAISS
end-to-end ranking (MedMCQA Hit@10 57.8 live vs 61.9 offline).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Dict

from loguru import logger
from tqdm import tqdm


def build_idf_table(corpus_jsonl: str) -> Dict[str, float]:
    """Build the IDF table from a passage corpus (one JSON object per line)."""
    logger.info(f"Building IDF table from {corpus_jsonl} ...")
    df: Counter = Counter()
    n_docs = 0

    with open(corpus_jsonl, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="IDF corpus scan"):
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec.get("text", "")
            toks = set(re.findall(r"[a-z0-9]+", text.lower()))
            for t in toks:
                df[t] += 1
            n_docs += 1

    idf = {}
    for tok, freq in df.items():
        idf[tok] = math.log((n_docs + 1) / (freq + 1)) + 1.0

    logger.success(f"Built IDF table with {len(idf)} terms over {n_docs:,} docs")
    return idf
