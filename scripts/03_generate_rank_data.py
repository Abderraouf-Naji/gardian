"""
Script 03– Generate ranking data for GARDIAN offline training.

Datasets supported:
  - PubMedQA (artificial + labeled)
  - MedMCQA

What this script does
---------------------
For each query in each split:
1. Run retrieval (bm25 / faiss / biobert / doc2query / hybrid / hybrid_neural)
2. Build candidate pool from unified OR per-corpus indices
3. Compute sparse, dense, and KG features for every candidate
4. Label candidate positive ONLY if passage ID is in gold_passage_ids
5. Skip queries with no gold passages
6. Save JSONL records for training/dev ranking

Usage:
    # Hybrid (BM25 + FAISS)
    python scripts/03_generate_rank_data.py --retriever hybrid

    # Hybrid neural (BioBERT + Doc2Query)
    python scripts/03_generate_rank_data.py --retriever hybrid_neural
    
    # Doc2Query only
    python scripts/03_generate_rank_data.py --retriever doc2query
    
    # With unified index
    python scripts/03_generate_rank_data.py --mode unified --retriever hybrid
    
Neural first-stage baselines (Doc2Query / BioBERT+Doc2Query) require one rank JSONL per
``--retriever``; evaluation (``05_evaluate_gardian.py`` / ``rank_jsonl_eval``)
adds Doc2Query columns whenever ``doc2query_score`` is non-zero (hybrid runs zero-fill).
"""

import json
import math
import os
import pathlib
import random
import sys
import hashlib
import platform
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
from src.retrieval.biobert import BioBERTRetriever
from src.retrieval.hybrid import HybridBm25FaissRetriever, HybridBioBertDoc2QueryRetriever
from src.retrieval.doc2query import Doc2QueryRetriever
from src.common.question_types import normalize_question_type, qtype_onehot


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

# -----------------------------------------------------------------------------
# Retriever Factory
# -----------------------------------------------------------------------------

def _index_paths(dataset_key: str) -> Dict[str, pathlib.Path]:
    root = pathlib.Path("data/indices")
    return {
        "bm25_dir": root / "bm25" / dataset_key,
        "faiss_path": root / "faiss" / dataset_key / "faiss.index",
        "faiss_meta": root / "faiss" / dataset_key / "faiss_meta.jsonl",
        "biobert_dir": root / "biobert" / dataset_key,
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
        top_k_dense=int(cfg.retrieval.get("top_k_faiss", cfg.retrieval.get("top_k_dense", 200))),
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


def get_biobert_retriever(dataset_key: str, cfg) -> BioBERTRetriever:
    paths = _index_paths(dataset_key)
    biobert_dir = paths["biobert_dir"]
    if not (biobert_dir / "biobert_index.pt").exists():
        raise FileNotFoundError(f"BioBERT index not found: {biobert_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return BioBERTRetriever(
        index_path=str(biobert_dir),
        checkpoint=cfg.retrieval.get("biobert_checkpoint", "dmis-lab/biobert-v1.1"),
        device=device,
        batch_size=int(cfg.retrieval.get("biobert_batch_size", 32)),
        max_length=int(cfg.retrieval.get("biobert_max_length", 512)),
    )


def get_hybrid_neural_retriever(dataset_key: str, cfg) -> HybridBioBertDoc2QueryRetriever:
    """Create hybrid retriever (BioBERT + Doc2Query)."""
    paths = _index_paths(dataset_key)
    biobert_dir = paths["biobert_dir"]

    if not (biobert_dir / "biobert_index.pt").exists():
        raise FileNotFoundError(f"BioBERT index not found: {biobert_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    biobert = BioBERTRetriever(
        index_path=str(biobert_dir),
        checkpoint=cfg.retrieval.get("biobert_checkpoint", "dmis-lab/biobert-v1.1"),
        device=device,
        batch_size=int(cfg.retrieval.get("biobert_batch_size", 32)),
        max_length=int(cfg.retrieval.get("biobert_max_length", 512)),
    )
    bm25 = BM25Retriever(index_dir=str(paths["bm25_dir"]))
    doc2query = Doc2QueryRetriever(
        bm25=bm25,
        model_name=str(cfg.retrieval.get("doc2query_model", "doc2query/msmarco-t5-base-v1")),
        device=device,
        num_expansions=int(cfg.retrieval.get("doc2query_num_expansions", 4)),
        max_new_tokens=int(cfg.retrieval.get("doc2query_max_new_tokens", 24)),
    )
    return HybridBioBertDoc2QueryRetriever(
        biobert=biobert,
        doc2query=doc2query,
        top_k_biobert=int(cfg.retrieval.get("top_k_biobert", cfg.retrieval.get("top_k_dense", 200))),
        top_k_doc2query=int(cfg.retrieval.get("top_k_doc2query", cfg.retrieval.get("top_k_bm25", 200))),
    )

def get_doc2query_retriever(dataset_key: str, cfg) -> Doc2QueryRetriever:
    """Create standalone Doc2Query retriever over BM25 index."""
    bm25_dir = _index_paths(dataset_key)["bm25_dir"]
    if not (bm25_dir / "index.pkl").exists():
        raise FileNotFoundError(f"BM25 index not found: {bm25_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bm25 = BM25Retriever(index_dir=str(bm25_dir))
    return Doc2QueryRetriever(
        bm25=bm25,
        model_name=str(cfg.retrieval.get("doc2query_model", "doc2query/msmarco-t5-base-v1")),
        device=device,
        num_expansions=int(cfg.retrieval.get("doc2query_num_expansions", 4)),
        max_new_tokens=int(cfg.retrieval.get("doc2query_max_new_tokens", 24)),
    )

def get_retriever(dataset_key: str, cfg, retriever_type: str = "hybrid"):
    """Factory function to get the appropriate retriever."""
    if retriever_type == "bm25":
        return get_bm25_retriever(dataset_key, cfg)
    elif retriever_type == "faiss":
        return get_faiss_retriever(dataset_key, cfg)
    elif retriever_type == "biobert":
        return get_biobert_retriever(dataset_key, cfg)
    elif retriever_type == "hybrid":
        return get_hybrid_retriever(dataset_key, cfg)
    elif retriever_type == "hybrid_neural":
        return get_hybrid_neural_retriever(dataset_key, cfg)
    elif retriever_type == "doc2query":
        return get_doc2query_retriever(dataset_key, cfg)
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
    qtype_counter = Counter()
    passage_entity_cache: Dict[str, List] = {}

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
            if expected_query_feat_dim is not None and int(q_emb.shape[0]) != int(expected_query_feat_dim):
                raise ValueError(
                    f"query_emb dim mismatch for qid={qid}: got={int(q_emb.shape[0])} "
                    f"expected={int(expected_query_feat_dim)}"
                )
            q_entities = qb["q_entities"]
            qtype_oh = qb["qtype_onehot"]
            kg_coverage = qb["kg_coverage"]

            # Build query KG cache
            query_kg_cache = build_query_kg_cache(q_entities, kg, node_set=node_set)

            # Encode passage texts (for dense features)
            p_texts = [c.get("text", "") for c in candidates]
            p_embs = encoder.encode(
                p_texts,
                batch_size=128,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # Compute cosine statistics
            cosines = batch_cosines(q_emb, p_embs)
            cos_mean = float(np.mean(cosines))
            cos_std = float(np.std(cosines) + 1e-8)

            # Process each candidate
            has_positive_in_pool = any(c["id"] in gold_ids for c in candidates)
            for i, cand in enumerate(candidates):
                pid = cand["id"]
                p_text = cand.get("text", "")
                
                # Extract scores based on retriever type
                bm25_score = 0.0
                dense_score = 0.0
                doc2query_score = 0.0
                
                if retriever_type == "bm25":
                    bm25_score = float(cand.get("score", cand.get("bm25_score", 0.0)))
                elif retriever_type == "faiss":
                    dense_score = float(cand.get("score", cand.get("dense_score", 0.0)))
                elif retriever_type == "biobert":
                    dense_score = float(cand.get("biobert_score", 0.0))
                elif retriever_type == "hybrid":
                    bm25_score = float(cand.get("bm25_score", 0.0))
                    dense_score = float(cand.get("dense_score", 0.0))
                elif retriever_type == "hybrid_neural":
                    dense_score = float(cand.get("biobert_score", 0.0))
                    doc2query_score = float(cand.get("doc2query_score", 0.0))
                elif retriever_type == "doc2query":
                    doc2query_score = float(cand.get("doc2query_score", 0.0))
                
                # Get or compute passage entities
                if pid not in passage_entity_cache:
                    passage_entity_cache[pid] = linker.link(p_text)
                p_entities = passage_entity_cache[pid]

                # Compute features (use appropriate scores)
                sparse_feats = compute_sparse_features(
                    query=question,
                    passage=p_text,
                    bm25_score=bm25_score if retriever_type in ("bm25", "hybrid") else doc2query_score,
                    idf_table=idf_table,
                ).tolist()

                dense_feats = compute_dense_features(
                    q_emb=q_emb,
                    p_emb=p_embs[i],
                    cosine_mean=cos_mean,
                    cosine_std=cos_std,
                ).tolist()

                kg_feats = compute_kg_features(
                    q_entities=q_entities,
                    p_entities=p_entities,
                    G=kg,
                    query_cache=query_kg_cache,
                    degree_lookup=degree_lookup,
                    node_set=node_set,
                ).tolist()

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
                if retriever_type in ("bm25", "hybrid"):
                    rec["bm25_score"] = bm25_score
                if retriever_type in ("faiss", "hybrid", "biobert", "hybrid_neural"):
                    rec["dense_score"] = dense_score
                if retriever_type in ("doc2query", "hybrid_neural"):
                    rec["doc2query_score"] = doc2query_score
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
        f"empty_queries={empty_queries} | skipped_no_gold={skipped_no_gold}"
    )
    logger.info(f"Question type distribution: {dict(qtype_counter)}")

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
        choices=["bm25", "faiss", "biobert", "hybrid", "hybrid_neural", "doc2query", "all"],
        default="all",
        help="Retriever type: bm25, faiss, biobert, hybrid, hybrid_neural, doc2query, or all"
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    retriever_types = ["bm25", "faiss", "biobert", "doc2query"] if args.retriever == "all" else [args.retriever]

    for retriever_type in retriever_types:
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

                        out_path = f"data/rank_data_{retriever_type}_{dataset_name}_{split}.jsonl"
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
                        )

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

                        out_path = f"data/rank_data_{retriever_type}_{dataset_name}_{split}.jsonl"
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