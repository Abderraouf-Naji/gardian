"""
Generate ranking training data (JSONL) for GARDIAN.

For each training query:
  1. Run hybrid retrieval → candidate pool (bounded by ``max_candidates``, usually
     from ``cfg.retrieval.candidate_pool_size``, e.g. 100).
  2. Compute sparse / dense features for every candidate (KG optional)
  3. Label each candidate 1 (positive) if its id is in gold_passage_ids,
     else 0 (negative). If no gold found, fall back to BM25-top-1 as pseudo-positive.
  4. Serialise to JSONL.
"""

import json, pathlib
from typing import List, Dict
import numpy as np
from tqdm import tqdm
from loguru import logger

from src.retrieval.hybrid import HybridRetriever
from src.kg.linker         import EntityLinker
from src.features.sparse   import compute_sparse_features
from src.features.dense_feat import compute_dense_features, batch_cosines
from src.features.kg_feat import compute_kg_features
from src.common.question_types import normalize_question_type, qtype_onehot


def generate_rank_data(queries: List[Dict],
                       hybrid: HybridRetriever,
                       encoder,          # SentenceTransformer
                       linker: EntityLinker,
                       kg,               # nx.DiGraph
                       out_path: str,
                       max_candidates: int = 100):
    """
    queries : list of {id, question, gold_passage_ids, question_type}
    Writes one JSONL per query-candidate pair.
    """
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as fout:
        for item in tqdm(queries, desc=f"Rank-data → {out_path}"):
            qid       = item["id"]
            question  = item["question"]
            gold_ids  = set(item.get("gold_passage_ids", []))
            qtype     = normalize_question_type(item.get("question_type", "other"))

            # ── Hybrid retrieval ──────────────────────────
            candidates = hybrid.retrieve(question)[:max_candidates]
            if not candidates:
                continue

            # ── Encode query ──────────────────────────────
            q_emb = encoder.encode([question], normalize_embeddings=True)[0]

            # ── Encode all passages ───────────────────────
            p_texts = [c["text"] for c in candidates]
            p_embs  = encoder.encode(p_texts, normalize_embeddings=True,
                                     show_progress_bar=False)

            # Cosine statistics for z-normalisation
            cosines     = batch_cosines(q_emb, p_embs)
            cos_mean    = float(cosines.mean())
            cos_std     = float(cosines.std()) + 1e-8

            # ── KG entity linking for query ───────────────
            q_entities  = linker.link(question)
            kg_coverage = 1.0 if q_entities else 0.0

            # ── Pseudo-label if no gold found ─────────────
            if not gold_ids:
                sorted_by_bm25 = sorted(candidates, key=lambda x: x["bm25_score"], reverse=True)
                if sorted_by_bm25:
                    gold_ids = {sorted_by_bm25[0]["id"]}

            # ── Per-candidate features ────────────────────
            for i, cand in enumerate(candidates):
                p_entities = linker.link(cand["text"])
                label      = 1 if cand["id"] in gold_ids else 0

                sparse_feats = compute_sparse_features(
                    question, cand["text"], cand["bm25_score"]
                ).tolist()

                dense_feats = compute_dense_features(
                    q_emb, p_embs[i], cos_mean, cos_std
                ).tolist()

                kg_feats = compute_kg_features(
                    q_entities, p_entities, kg
                ).tolist()

                record = {
                    "qid":           qid,
                    "pid":           cand["id"],
                    "label":         label,
                    "sparse_feats":  sparse_feats,
                    "dense_feats":   dense_feats,
                    "kg_feats":      kg_feats,
                    "query_emb":     q_emb.tolist(),
                    "qtype_onehot":  qtype_onehot(qtype),
                    "kg_coverage":   kg_coverage,
                }
                fout.write(json.dumps(record) + "\n")

    logger.success(f"Rank data written → {out_path}")