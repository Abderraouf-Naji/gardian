"""
Script 03– Generate ranking data for GARDIAN offline training.

Datasets supported:
  - PubMedQA (artificial + labeled)
  - MedMCQA

What this script does
---------------------
For each query in each split:
1. Run retrieval (bm25 / faiss / spladev3 / colbert / hybrid families)
2. Build candidate pool from unified OR per-corpus indices
3. Compute sparse, dense, and KG features for every candidate
4. Label candidate positive ONLY if passage ID is in gold_passage_ids
5. Skip queries with no gold passages
6. Save JSONL records for training/dev ranking

Usage:
    # Hybrid (BM25 + FAISS)
    python scripts/03_generate_rank_data.py --retriever hybrid

    # Hybrid neural (SPLADEv3 + ColBERT)
    python scripts/03_generate_rank_data.py --retriever hybrid_neural
    
    # With unified index
    python scripts/03_generate_rank_data.py --mode unified --retriever hybrid
    
Neural first-stage baselines (SPLADEv3 / ColBERT+SPLADEv3) require one rank JSONL per
``--retriever``; evaluation (``05_evaluate_gardian.py`` / ``rank_jsonl_eval``)
adds SPLADEv3 columns whenever ``spladev3_score`` is non-zero (hybrid runs zero-fill).
"""

import json
import math
import os
import pathlib
import random
import sys
import hashlib
import platform
import pickle
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, ".")

from src.features.dense_feat import batch_cosines, compute_dense_features
from src.features.kg_feat import (
    build_degree_lookup,
    build_node_set,
    build_query_kg_cache,
    compute_kg_features,
)
from src.features.sparse import compute_sparse_features
from src.kg.builder import load_kg
from src.kg.linker import EntityLinker
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import (
    DualHybridRetriever,
    HybridBm25FaissRetriever,
    HybridSpladev3ColbertRetriever,
)
from src.retrieval.spladev3 import SpladeV3Retriever
from src.retrieval.colbert import ColBERTRetriever
from src.common.question_types import normalize_question_type, qtype_onehot
from src.common.rank_data_paths import normalize_retriever_name, rank_data_file


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Dataset configurations
DATASETS = {
    "pubmedqa_artificial": {
        "train": "data/pubmedqa_artificial_train.jsonl",
        "dev": "data/pubmedqa_artificial_dev.jsonl",
        "test": "data/pubmedqa_artificial_test.jsonl",
        "corpus": "pubmedqa_artificial",
        "type": "training"
    },
    "pubmedqa_labeled": {
        "eval": "data/pubmedqa_labeled_eval.jsonl",
        "corpus": "pubmedqa_labeled",
        "type": "evaluation"
    },
    "medmcqa": {
        "train": "data/medmcqa_train.jsonl",
        "dev": "data/medmcqa_dev.jsonl",
        "test": "data/medmcqa_test.jsonl",
        "corpus": "medmcqa",
        "type": "training"
    }
}

# Unified corpus
UNIFIED_CORPUS = {
    "corpus": "unified",
    "description": "All sources combined"
}

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_file(path: str) -> str:
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def validate_query_record(rec: dict, dataset_name: str, idx: int) -> dict:
    if "id" not in rec:
        raise ValueError(f"{dataset_name} record #{idx} missing 'id'")
    if "question" not in rec:
        raise ValueError(f"{dataset_name} record #{idx} missing 'question'")
    if "gold_passage_ids" not in rec:
        raise ValueError(
            f"{dataset_name} record #{idx} missing 'gold_passage_ids'. "
            "This script requires real relevance labels."
        )
    rec["question_type"] = normalize_question_type(rec.get("question_type", "other"))
    return rec

def build_idf_table(corpus_jsonl: str) -> Dict[str, float]:
    """Build IDF table from corpus for sparse features."""
    import re
    
    logger.info(f"Building IDF table from {corpus_jsonl} ...")
    df = Counter()
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

def deduplicate_candidates(candidates: List[Dict], max_candidates: int) -> List[Dict]:
    seen = set()
    out = []
    for c in candidates:
        pid = c["id"]
        if pid in seen:
            continue
        seen.add(pid)
        out.append(c)
        if len(out) >= max_candidates:
            break
    return out


def query_cache_file(retriever: str, dataset: str, split: str) -> str:
    r = normalize_retriever_name(retriever)
    return str(pathlib.Path("data") / f"query_emb_cache_{r}_{dataset}_{split}.pkl")


def merge_query_caches(src_paths: List[str], out_path: str) -> int:
    merged: Dict[str, List[float]] = {}
    for p in src_paths:
        pp = pathlib.Path(p)
        if not pp.is_file():
            continue
        try:
            with pp.open("rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict):
                merged.update({str(k): v for k, v in obj.items()})
        except Exception as e:
            logger.warning(f"Skipping bad query cache {pp}: {e}")
    if not merged:
        return 0
    op = pathlib.Path(out_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    with op.open("wb") as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Merged query cache -> {op} ({len(merged):,} qids)")
    return len(merged)

# -----------------------------------------------------------------------------
# Retriever Factory
# -----------------------------------------------------------------------------

def _index_paths(dataset_key: str) -> Dict[str, pathlib.Path]:
    root = pathlib.Path("data/indices")
    return {
        "bm25_dir": root / "bm25" / dataset_key,
        "faiss_path": root / "faiss" / dataset_key / "faiss.index",
        "faiss_meta": root / "faiss" / dataset_key / "faiss_meta.jsonl",
        "spladev3_dir": root / "spladev3" / dataset_key,
        "colbert_dir": root / "colbert" / dataset_key,
    }


def get_hybrid_retriever(dataset_key: str, cfg) -> HybridBm25FaissRetriever:
    """Create hybrid retriever (BM25 + FAISS)."""
    paths = _index_paths(dataset_key)
    bm25_dir = paths["bm25_dir"]
    faiss_path = paths["faiss_path"]
    meta_path = paths["faiss_meta"]
    
    if not (bm25_dir / "index.pkl").exists():
        raise FileNotFoundError(f"BM25 index not found: {bm25_dir}")
    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
    
    bm25 = BM25Retriever(index_dir=str(bm25_dir))
    dense = DenseRetriever(
        faiss_index_path=str(faiss_path),
        meta_path=str(meta_path),
        encoder_name=cfg.encoder.model_name,
        batch_size=int(cfg.encoder.batch_size),
        max_length=int(cfg.encoder.max_length),
    )
    
    return HybridBm25FaissRetriever(
        bm25=bm25,
        dense=dense,
        top_k_bm25=int(cfg.retrieval.top_k_bm25),
        top_k_dense=int(cfg.retrieval.get("top_k_faiss", cfg.retrieval.get("top_k_dense", 50))),
    )


def get_bm25_retriever(dataset_key: str, cfg) -> BM25Retriever:
    paths = _index_paths(dataset_key)
    bm25_dir = paths["bm25_dir"]
    if not (bm25_dir / "index.pkl").exists():
        raise FileNotFoundError(f"BM25 index not found: {bm25_dir}")
    return BM25Retriever(index_dir=str(bm25_dir))


def get_faiss_retriever(dataset_key: str, cfg) -> DenseRetriever:
    paths = _index_paths(dataset_key)
    faiss_path = paths["faiss_path"]
    meta_path = paths["faiss_meta"]
    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
    return DenseRetriever(
        faiss_index_path=str(faiss_path),
        meta_path=str(meta_path),
        encoder_name=cfg.encoder.model_name,
        batch_size=int(cfg.encoder.batch_size),
        max_length=int(cfg.encoder.max_length),
    )


def get_spladev3_retriever(dataset_key: str, cfg) -> SpladeV3Retriever:
    paths = _index_paths(dataset_key)
    splade_dir = paths["spladev3_dir"]
    if not (splade_dir / "spladev3_index.pt").exists():
        raise FileNotFoundError(f"SPLADEv3 index not found: {splade_dir / 'spladev3_index.pt'}")
    return SpladeV3Retriever(
        index_path=str(splade_dir),
        model_name=str(cfg.retrieval.get("spladev3_encoder", "naver/splade-v3-distilbert")),
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=int(cfg.retrieval.get("spladev3_batch_size", cfg.encoder.batch_size)),
        max_length=int(cfg.retrieval.get("spladev3_max_length", cfg.encoder.max_length)),
    )


def get_colbert_retriever(dataset_key: str, cfg) -> ColBERTRetriever:
    paths = _index_paths(dataset_key)
    colbert_dir = paths["colbert_dir"]
    if not ColBERTRetriever.index_ready(str(colbert_dir)):
        raise FileNotFoundError(f"Native ColBERT index not found/ready: {colbert_dir}")
    return ColBERTRetriever(
        index_path=str(colbert_dir),
        model_name=str(cfg.retrieval.get("colbert_encoder", "colbert-ir/colbertv2.0")),
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=int(cfg.retrieval.get("colbert_batch_size", cfg.encoder.batch_size)),
        max_length=int(cfg.retrieval.get("colbert_max_length", cfg.encoder.max_length)),
    )


def get_hybrid_spladev3_colbert_retriever(dataset_key: str, cfg) -> DualHybridRetriever:
    """Create hybrid retriever (SPLADEv3 + ColBERT)."""
    paths = _index_paths(dataset_key)
    if not (paths["spladev3_dir"] / "spladev3_index.pt").exists():
        raise FileNotFoundError(f"SPLADEv3 index not found: {paths['spladev3_dir'] / 'spladev3_index.pt'}")
    if not ColBERTRetriever.index_ready(str(paths["colbert_dir"])):
        raise FileNotFoundError(f"Native ColBERT index not found/ready: {paths['colbert_dir']}")
    spladev3 = get_spladev3_retriever(dataset_key, cfg)
    colbert = get_colbert_retriever(dataset_key, cfg)
    return HybridSpladev3ColbertRetriever(
        spladev3=spladev3,
        colbert=colbert,
        top_k_spladev3=int(cfg.retrieval.get("top_k_spladev3", 50)),
        top_k_colbert=int(cfg.retrieval.get("top_k_colbert", 50)),
    )


def get_retriever(dataset_key: str, cfg, retriever_type: str = "hybrid_bm25_faiss"):
    """Factory function to get the appropriate retriever."""
    retriever_type = normalize_retriever_name(retriever_type)
    if retriever_type == "bm25":
        return get_bm25_retriever(dataset_key, cfg)
    elif retriever_type == "faiss":
        return get_faiss_retriever(dataset_key, cfg)
    elif retriever_type == "spladev3":
        return get_spladev3_retriever(dataset_key, cfg)
    elif retriever_type == "colbert":
        return get_colbert_retriever(dataset_key, cfg)
    elif retriever_type == "hybrid_bm25_faiss":
        return get_hybrid_retriever(dataset_key, cfg)
    elif retriever_type == "hybrid_spladev3_colbert":
        return get_hybrid_spladev3_colbert_retriever(dataset_key, cfg)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

# -----------------------------------------------------------------------------
# Query Processing
# -----------------------------------------------------------------------------

def compute_query_bundle(question: str, qtype: str, encoder, linker) -> Dict:
    """Compute query embeddings and KG entities."""
    q_emb = encoder.encode(
        [question],
        batch_size=1,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )[0]
    q_entities = linker.link(question)
    kg_coverage = 1.0 if len(q_entities) > 0 else 0.0

    return {
        "query_emb": q_emb,
        "q_entities": q_entities,
        "qtype_onehot": qtype_onehot(qtype),
        "kg_coverage": kg_coverage,
    }

def process_queries(
    queries: List[Dict],
    out_path: str,
    retriever,
    retriever_type: str,
    encoder,
    linker,
    kg,
    degree_lookup: Dict,
    node_set: frozenset,
    idf_table: Optional[Dict[str, float]] = None,
    max_candidates: int = 400,
    max_queries: Optional[int] = None,
    expected_query_feat_dim: Optional[int] = None,
    lean_records: bool = False,
    query_cache_out_path: Optional[str] = None,
    # Default to the correctness-first path:
    # we need the passage embedding to compute mean/max |q_emb - p_emb|.
    # The "dense_from_retriever_scores" fast path intentionally zero-fills
    # those two dims and is only safe for speed ablations.
    dense_from_retriever_scores: bool = False,
    exact_distance_features: bool = False,
    kg_max_path: int = 4,
    kg_refine_top_n: int = 0,
) -> None:
    """Process queries and generate ranking data."""
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if max_queries and len(queries) > max_queries:
        queries = queries[:max_queries]
        logger.info(f"Limited to {max_queries} queries for testing")

    total_queries = 0
    total_records = 0
    total_pos = 0
    total_neg = 0
    empty_queries = 0
    skipped_no_gold = 0
    no_positive_in_pool = 0
    qtype_counter = Counter()
    passage_entity_cache: Dict[str, List] = {}
    # Quick scale diagnostics to detect feature dominance / collapse.
    kg_l1_sum = 0.0
    kg_feat_abs_sum = np.zeros(6, dtype=np.float64)
    dense_l1_sum = 0.0
    sparse_l1_sum = 0.0
    diag_samples = 0
    qid_emb_cache: Dict[str, List[float]] = {}

    with open(out_path, "w", encoding="utf-8") as fout:
        for item in tqdm(queries, desc=f"Generating {pathlib.Path(out_path).name}"):
            qid = item["id"]
            question = item["question"]
            qtype = normalize_question_type(item.get("question_type", "other"))
            gold_ids = set(item.get("gold_passage_ids", []))

            if not gold_ids:
                skipped_no_gold += 1
                continue

            qtype_counter[qtype] += 1
            total_queries += 1

            # Retrieve candidates
            top_k = max_candidates * 2  # Retrieve more for dedup
            candidates = retriever.retrieve(question, top_k=top_k)
            candidates = deduplicate_candidates(candidates, max_candidates=max_candidates)

            if not candidates:
                empty_queries += 1
                logger.warning(f"No candidates returned for qid={qid}")
                continue

            # Compute query bundle
            qb = compute_query_bundle(question, qtype, encoder, linker)
            q_emb = qb["query_emb"]
            if qid not in qid_emb_cache:
                qid_emb_cache[str(qid)] = q_emb.tolist()
            if expected_query_feat_dim is not None and int(q_emb.shape[0]) != int(expected_query_feat_dim):
                raise ValueError(
                    f"query_emb dim mismatch for qid={qid}: got={int(q_emb.shape[0])} "
                    f"expected={int(expected_query_feat_dim)}"
                )
            q_entities = qb["q_entities"]
            qtype_oh = qb["qtype_onehot"]
            kg_coverage = qb["kg_coverage"]

            # Build query KG caches:
            # - base: always cheap (no BFS distance map)
            # - exact: optional per-query BFS map used only for top-N candidates
            query_kg_cache_base = build_query_kg_cache(
                q_entities,
                kg,
                node_set=node_set,
                compute_distances=False,
                max_path=int(kg_max_path),
            )
            use_refine = (
                int(kg_refine_top_n) > 0
                and retriever_type in {
                    "hybrid_bm25_faiss",
                    "hybrid_spladev3_colbert",
                }
            )
            query_kg_cache_exact = None
            if bool(exact_distance_features) or use_refine:
                query_kg_cache_exact = build_query_kg_cache(
                    q_entities,
                    kg,
                    node_set=node_set,
                    compute_distances=True,
                    max_path=int(kg_max_path),
                )

            # Fast dense features from retriever scores (recommended for large full runs):
            # avoids re-encoding up to 200 passage texts per query.
            use_dense_scores = dense_from_retriever_scores and retriever_type in {
                "bm25",
                "spladev3",
                "faiss",
                "colbert",
                "hybrid_bm25_faiss",
                "hybrid_spladev3_colbert",
            }
            p_embs = None
            dense_scores_raw: List[float] = []
            if use_dense_scores:
                for cand in candidates:
                    if retriever_type == "bm25":
                        dense_scores_raw.append(float(cand.get("bm25_score", cand.get("score", 0.0))))
                    elif retriever_type == "spladev3":
                        dense_scores_raw.append(float(cand.get("spladev3_score", cand.get("score", 0.0))))
                    elif retriever_type in {"colbert", "hybrid_spladev3_colbert"}:
                        dense_scores_raw.append(float(cand.get("colbert_score", cand.get("dense_score", cand.get("score", 0.0)))))
                    else:
                        dense_scores_raw.append(float(cand.get("dense_score", cand.get("score", 0.0))))
                cos_mean = float(np.mean(dense_scores_raw))
                cos_std = float(np.std(dense_scores_raw) + 1e-8)
            else:
                # Original dense feature path: encode passages with sentence encoder.
                p_texts = [c.get("text", "") for c in candidates]
                p_embs = encoder.encode(
                    p_texts,
                    batch_size=512,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                cosines = batch_cosines(q_emb, p_embs)
                cos_mean = float(np.mean(cosines))
                cos_std = float(np.std(cosines) + 1e-8)

            # Process each candidate
            has_positive_in_pool = any(c["id"] in gold_ids for c in candidates)
            if not has_positive_in_pool:
                no_positive_in_pool += 1
            for i, cand in enumerate(candidates):
                pid = cand["id"]
                p_text = cand.get("text", "")
                
                # Extract scores based on retriever type
                bm25_score = 0.0
                dense_score = 0.0
                spladev3_score = 0.0
                
                if retriever_type == "bm25":
                    bm25_score = float(cand.get("score", cand.get("bm25_score", 0.0)))
                elif retriever_type == "faiss":
                    dense_score = float(cand.get("score", cand.get("dense_score", 0.0)))
                elif retriever_type == "colbert":
                    dense_score = float(cand.get("score", cand.get("colbert_score", 0.0)))
                elif retriever_type == "hybrid_bm25_faiss":
                    bm25_score = float(cand.get("bm25_score", 0.0))
                    dense_score = float(cand.get("dense_score", 0.0))
                elif retriever_type == "hybrid_spladev3_colbert":
                    spladev3_score = float(cand.get("spladev3_score", 0.0))
                    dense_score = float(cand.get("colbert_score", 0.0))
                elif retriever_type == "spladev3":
                    spladev3_score = float(cand.get("score", cand.get("spladev3_score", 0.0)))
                
                # Get or compute passage entities
                if pid not in passage_entity_cache:
                    passage_entity_cache[pid] = linker.link(p_text)
                p_entities = passage_entity_cache[pid]

                # Compute features (use appropriate scores)
                sparse_signal_score = (
                    bm25_score
                    if retriever_type in ("bm25", "hybrid_bm25_faiss")
                    else spladev3_score
                )
                sparse_feats = compute_sparse_features(
                    query=question,
                    passage=p_text,
                    bm25_score=sparse_signal_score,
                    idf_table=idf_table,
                ).tolist()

                if use_dense_scores:
                    # [score, 0, 0, z(score)] keeps expected 4-dim shape with
                    # retriever-consistent signal at a fraction of the runtime.
                    if retriever_type == "bm25":
                        ds = bm25_score
                    elif retriever_type == "spladev3":
                        ds = spladev3_score
                    else:
                        ds = dense_score
                    z = (ds - cos_mean) / (cos_std + 1e-8)
                    dense_feats = [float(ds), 0.0, 0.0, float(z)]
                else:
                    dense_feats = compute_dense_features(
                        q_emb=q_emb,
                        p_emb=p_embs[i],
                        cosine_mean=cos_mean,
                        cosine_std=cos_std,
                    ).tolist()

                use_exact_for_candidate = bool(exact_distance_features) or (
                    use_refine and i < int(kg_refine_top_n)
                )
                kg_feats = compute_kg_features(
                    q_entities=q_entities,
                    p_entities=p_entities,
                    G=kg,
                    max_path=int(kg_max_path),
                    query_cache=(
                        query_kg_cache_exact if use_exact_for_candidate and query_kg_cache_exact is not None
                        else query_kg_cache_base
                    ),
                    degree_lookup=degree_lookup,
                    node_set=node_set,
                ).tolist()
                # Track branch feature scales for troubleshooting.
                sf = np.asarray(sparse_feats, dtype=np.float32)
                df = np.asarray(dense_feats, dtype=np.float32)
                kf = np.asarray(kg_feats, dtype=np.float32)
                sparse_l1_sum += float(np.mean(np.abs(sf)))
                dense_l1_sum += float(np.mean(np.abs(df)))
                kg_l1_sum += float(np.mean(np.abs(kf)))
                if kf.shape[0] == 6:
                    kg_feat_abs_sum += np.abs(kf).astype(np.float64)
                diag_samples += 1

                label = 1 if pid in gold_ids else 0

                rec = {
                    "qid": qid,
                    "pid": pid,
                    "question": question,
                    "question_type": qtype,
                    "label": label,
                    "gold_passage_ids": list(gold_ids),
                    "sparse_feats": sparse_feats,
                    "dense_feats": dense_feats,
                    "kg_feats": kg_feats,
                    "qtype_onehot": qtype_oh,
                    "kg_coverage": kg_coverage,
                    "retriever_type": retriever_type,
                    "has_positive_in_pool": has_positive_in_pool,
                }
                # Store only score channels relevant to current retriever.
                # Avoid writing constant zero score columns across massive files.
                if retriever_type in ("bm25", "hybrid_bm25_faiss"):
                    rec["bm25_score"] = bm25_score
                if retriever_type in (
                    "faiss",
                    "hybrid_bm25_faiss",
                    "colbert",
                    "hybrid_spladev3_colbert",
                ):
                    rec["dense_score"] = dense_score
                if retriever_type in ("spladev3", "hybrid_spladev3_colbert"):
                    rec["spladev3_score"] = spladev3_score
                _ = lean_records  # kept for CLI compatibility

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_records += 1
                if label == 1:
                    total_pos += 1
                else:
                    total_neg += 1

    logger.success(
        f"Saved {out_path} | retriever={retriever_type} | queries={total_queries} | "
        f"records={total_records:,} | pos={total_pos:,} | neg={total_neg:,} | "
        f"empty_queries={empty_queries} | skipped_no_gold={skipped_no_gold} | "
        f"no_positive_in_pool={no_positive_in_pool}"
    )
    logger.info(f"Question type distribution: {dict(qtype_counter)}")
    if diag_samples > 0:
        logger.info(
            "Feature scale diagnostics | "
            f"sparse_mean_abs={sparse_l1_sum / diag_samples:.6f} | "
            f"dense_mean_abs={dense_l1_sum / diag_samples:.6f} | "
            f"kg_mean_abs={kg_l1_sum / diag_samples:.6f}"
        )
        logger.info(
            "KG per-dimension mean|abs| = "
            + ", ".join(f"f{i}={v / diag_samples:.6f}" for i, v in enumerate(kg_feat_abs_sum))
        )
    if query_cache_out_path:
        cp = pathlib.Path(query_cache_out_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        with cp.open("wb") as f:
            pickle.dump(qid_emb_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Wrote query cache -> {cp} ({len(qid_emb_cache):,} qids)")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate ranking data for GARDIAN")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["unified", "per-corpus"],
        default="per-corpus",
        help="Retrieval mode: unified (single index) or per-corpus (routed)"
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=[
            "bm25",
            "faiss",
            "spladev3",
            "colbert",
            "hybrid_bm25_faiss",
            "hybrid_spladev3_colbert",
            # backward-compatible aliases
            "hybrid",
            "hybrid_neural",
            "all",
        ],
        default="all",
        help=(
            "Retriever type: bm25, faiss, spladev3, colbert, "
            "hybrid_bm25_faiss, hybrid_spladev3_colbert, or all. "
            "Aliases accepted: hybrid, hybrid_neural."
        )
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum number of queries to process (for testing)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["pubmedqa_artificial", "pubmedqa_labeled", "medmcqa", "all"],
        default="all",
        help="Dataset to process"
    )
    parser.add_argument(
        "--lean-records",
        action="store_true",
        help=(
            "Write compact rank records (drop repeated metadata fields like "
            "gold_passage_ids/retriever_type/has_positive_in_pool)."
        ),
    )
    parser.add_argument(
        "--no-write-query-cache",
        action="store_true",
        help="Disable writing query cache pkl files during rank-data generation.",
    )
    parser.add_argument(
        "--dense-from-retriever-scores",
        action="store_true",
        help=(
            "Use fast dense feature shortcut from retriever scores ([score,0,0,z]). "
            "Default is correctness mode: full dense features from query/passage embeddings."
        ),
    )
    parser.add_argument(
        "--no-dense-from-retriever-scores",
        action="store_true",
        help=(
            "Deprecated alias kept for compatibility. "
            "Default already uses full dense features."
        ),
    )
    parser.add_argument(
        "--kg-refine-top-n",
        type=int,
        default=None,
        help=(
            "Compute exact KG distance features only for top-N retrieved candidates per query "
            "(0 disables; ignored when kg.exact_distance_features=true). "
            "Good quality/speed trade-off for production."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for encoder/retriever components. Default auto.",
    )
    
    args = parser.parse_args()
    
    cfg = OmegaConf.load("configs/base.yaml")
    set_all_seeds(int(cfg.seed))
    configured_query_feat_dim = int(cfg.model.query_feat_dim)

    # Load KG and components
    logger.info("Loading KG and lexical index ...")
    kg, lex = load_kg(cfg.paths.kg_graph, cfg.paths.kg_lexical_idx)
    linker = EntityLinker(
        lexical_index=lex,
        max_entities=int(cfg.kg.max_entities_per_text),
    )

    logger.info("Pre-computing KG degree lookup and node set ...")
    degree_lookup = build_degree_lookup(kg)
    node_set = build_node_set(kg)
    logger.success(f"KG ready | degree_lookup={len(degree_lookup):,} nodes | node_set={len(node_set):,}")

    # Load encoder
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda requested but torch.cuda.is_available() is False in this venv."
        )
    logger.info(f"Loading encoder on {device} ...")
    encoder = SentenceTransformer(cfg.encoder.model_name, device=device)
    if device == "cuda":
        encoder = encoder.half()

    # Use the encoder's real output dimension as source of truth for rank-data.
    # This avoids hard failures when config.model.query_feat_dim is stale.
    encoder_query_feat_dim = int(encoder.get_embedding_dimension())
    if encoder_query_feat_dim != configured_query_feat_dim:
        logger.warning(
            "query_feat_dim mismatch: cfg.model.query_feat_dim="
            f"{configured_query_feat_dim} but encoder outputs {encoder_query_feat_dim}. "
            f"Using encoder dimension ({encoder_query_feat_dim}) for rank-data generation."
        )
    expected_query_feat_dim = encoder_query_feat_dim

    max_candidates = int(cfg.retrieval.candidate_pool_size)

    retriever_types = (
        [
            "bm25",
            "faiss",
            "spladev3",
            "colbert",
            "hybrid_bm25_faiss",
            "hybrid_spladev3_colbert",
        ]
        if args.retriever == "all"
        else [normalize_retriever_name(args.retriever)]
    )

    for retriever_type in retriever_types:
        kg_refine_top_n = (
            int(args.kg_refine_top_n)
            if args.kg_refine_top_n is not None
            else int(getattr(cfg.kg, "refine_top_n", 0) or 0)
        )
        cache_paths_all: List[str] = []
        cache_paths_train: List[str] = []
        if args.mode == "unified":
            logger.info("=" * 72)
            logger.info(f"UNIFIED MODE - Retriever: {retriever_type.upper()}")
            logger.info("=" * 72)

            unified_corpus = UNIFIED_CORPUS["corpus"]
            unified_jsonl = "data/indices/unified/corpus_unified.jsonl"

            # Build IDF table from unified corpus
            idf_table = build_idf_table(unified_jsonl)

            # Get unified retriever
            try:
                retriever = get_retriever(unified_corpus, cfg, retriever_type)
            except FileNotFoundError as e:
                logger.warning(f"Skipping unified mode for retriever={retriever_type} - {e}")
                continue

            # Process datasets
            for dataset_name, dataset_config in DATASETS.items():
                if args.dataset != "all" and dataset_name != args.dataset:
                    continue

                for split in ["train", "dev", "eval", "test"]:
                    if split in dataset_config:
                        in_path = dataset_config[split]
                        if not os.path.exists(in_path):
                            logger.info(f"Skipping {dataset_name}_{split} - file not found")
                            continue

                        out_path = rank_data_file(retriever_type, dataset_name, split)
                        queries = read_jsonl(in_path)
                        queries = [
                            validate_query_record(rec, f"{dataset_name}_{split}", i)
                            for i, rec in enumerate(queries)
                        ]

                        if not queries:
                            continue

                        logger.info(f"Processing {dataset_name}_{split} ({len(queries)} queries)")
                        process_queries(
                            queries=queries,
                            out_path=out_path,
                            retriever=retriever,
                            retriever_type=retriever_type,
                            encoder=encoder,
                            linker=linker,
                            kg=kg,
                            degree_lookup=degree_lookup,
                            node_set=node_set,
                            idf_table=idf_table,
                            max_candidates=max_candidates,
                            max_queries=args.max_queries,
                            expected_query_feat_dim=expected_query_feat_dim,
                            lean_records=bool(args.lean_records),
                            query_cache_out_path=(
                                None
                                if args.no_write_query_cache
                                else query_cache_file(retriever_type, dataset_name, split)
                            ),
                            dense_from_retriever_scores=bool(args.dense_from_retriever_scores),
                            exact_distance_features=bool(getattr(cfg.kg, "exact_distance_features", False)),
                            kg_max_path=int(getattr(cfg.kg, "max_path_length", 4)),
                            kg_refine_top_n=kg_refine_top_n,
                        )
                        if not args.no_write_query_cache:
                            cp = query_cache_file(retriever_type, dataset_name, split)
                            cache_paths_all.append(cp)
                            if split == "train":
                                cache_paths_train.append(cp)

        else:  # per-corpus mode
            logger.info("=" * 72)
            logger.info(f"PER-CORPUS MODE - Retriever: {retriever_type.upper()}")
            logger.info("=" * 72)

            for dataset_name, dataset_config in DATASETS.items():
                if args.dataset != "all" and dataset_name != args.dataset:
                    continue

                corpus_dir = dataset_config.get("corpus")
                if not corpus_dir:
                    logger.warning(f"Skipping {dataset_name} - missing corpus key")
                    continue

                # Get corpus JSONL for IDF
                corpus_jsonl = None
                if dataset_name == "pubmedqa_artificial":
                    corpus_jsonl = "data/corpus_pubmedqa_artificial.jsonl"
                elif dataset_name == "pubmedqa_labeled":
                    corpus_jsonl = "data/corpus_pubmedqa_labeled.jsonl"
                elif dataset_name == "medmcqa":
                    corpus_jsonl = "data/corpus_medmcqa.jsonl"

                if not corpus_jsonl or not os.path.exists(corpus_jsonl):
                    logger.warning(f"Corpus JSONL not found for {dataset_name}")
                    continue

                # Build IDF table for this corpus
                idf_table = build_idf_table(corpus_jsonl)

                # Get retriever for this corpus
                try:
                    retriever = get_retriever(corpus_dir, cfg, retriever_type)
                except FileNotFoundError as e:
                    logger.warning(f"Skipping {dataset_name} - {e}")
                    continue

                # Process splits for this dataset
                for split in ["train", "dev", "eval", "test"]:
                    if split in dataset_config:
                        in_path = dataset_config[split]
                        if not os.path.exists(in_path):
                            continue

                        out_path = rank_data_file(retriever_type, dataset_name, split)
                        queries = read_jsonl(in_path)
                        queries = [
                            validate_query_record(rec, f"{dataset_name}_{split}", i)
                            for i, rec in enumerate(queries)
                        ]

                        if not queries:
                            continue

                        logger.info(f"Processing {dataset_name}_{split} ({len(queries)} queries)")
                        process_queries(
                            queries=queries,
                            out_path=out_path,
                            retriever=retriever,
                            retriever_type=retriever_type,
                            encoder=encoder,
                            linker=linker,
                            kg=kg,
                            degree_lookup=degree_lookup,
                            node_set=node_set,
                            idf_table=idf_table,
                            max_candidates=max_candidates,
                            max_queries=args.max_queries,
                            expected_query_feat_dim=expected_query_feat_dim,
                            lean_records=bool(args.lean_records),
                            query_cache_out_path=(
                                None
                                if args.no_write_query_cache
                                else query_cache_file(retriever_type, dataset_name, split)
                            ),
                            dense_from_retriever_scores=bool(args.dense_from_retriever_scores),
                            exact_distance_features=bool(getattr(cfg.kg, "exact_distance_features", False)),
                            kg_max_path=int(getattr(cfg.kg, "max_path_length", 4)),
                            kg_refine_top_n=kg_refine_top_n,
                        )
                        if not args.no_write_query_cache:
                            cp = query_cache_file(retriever_type, dataset_name, split)
                            cache_paths_all.append(cp)
                            if split == "train":
                                cache_paths_train.append(cp)

        if not args.no_write_query_cache:
            rname = normalize_retriever_name(retriever_type)
            merge_query_caches(
                cache_paths_all,
                f"data/query_emb_cache_{rname}_all.pkl",
            )
            merge_query_caches(
                cache_paths_train,
                f"data/query_emb_cache_{rname}_train_all.pkl",
            )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/03_generate_rank_data.py",
        "args": vars(args),
        "seed": int(cfg.seed),
        "encoder_model": str(cfg.encoder.model_name),
        "expected_query_feat_dim": expected_query_feat_dim,
        "platform": platform.platform(),
        "python_version": sys.version,
        "files": {},
    }
    tracked_inputs = [
        "configs/base.yaml",
        "data/pubmedqa_artificial_train.jsonl",
        "data/pubmedqa_artificial_dev.jsonl",
        "data/pubmedqa_artificial_test.jsonl",
        "data/pubmedqa_labeled_eval.jsonl",
        "data/medmcqa_train.jsonl",
        "data/medmcqa_dev.jsonl",
        "data/medmcqa_test.jsonl",
    ]
    for p in tracked_inputs:
        manifest["files"][p] = {
            "exists": os.path.exists(p),
            "sha256": sha256_file(p) if os.path.exists(p) else "",
        }
    manifest_path = pathlib.Path(cfg.paths.results_dir) / "rank_data_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote rank-data manifest -> {manifest_path}")

    logger.success("=" * 72)
    logger.success(f"Rank-data generation complete! (retriever: {args.retriever})")
    logger.success("=" * 72)

if __name__ == "__main__":
    main()