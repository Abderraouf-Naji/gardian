"""
Script 03 – Generate ranking data for GARDIAN offline training.

Datasets supported:
  - PubMedQA (artificial + labeled)
  - MedMCQA

What this script does
---------------------
For each query in each split:
1. Run retrieval (bm25 / faiss / spladepp / medcpt / hybrid families)
2. Build candidate pool from unified OR per-corpus indices
3. Compute sparse, dense, and KG features for every candidate
4. Label candidate positive ONLY if passage ID is in gold_passage_ids
5. Skip queries with no gold passages
6. Save JSONL records for training/dev ranking

Usage:
    # The 4 hybrid families used in the paper:
    python scripts/03_generate_rank_data.py --retriever hybrid_bm25_faiss
    python scripts/03_generate_rank_data.py --retriever hybrid_bm25_medcpt
    python scripts/03_generate_rank_data.py --retriever hybrid_spladepp_faiss
    python scripts/03_generate_rank_data.py --retriever hybrid_spladepp_medcpt

    # Run all four sequentially:
    python scripts/03_generate_rank_data.py --retriever all

    # Single-retriever baselines (kept for ablations only, not training):
    python scripts/03_generate_rank_data.py --retriever bm25
    python scripts/03_generate_rank_data.py --retriever faiss
    python scripts/03_generate_rank_data.py --retriever spladepp
    python scripts/03_generate_rank_data.py --retriever medcpt

Each hybrid record carries both retrievers' raw scores (e.g. ``bm25_score`` and
``dense_score`` for ``hybrid_bm25_faiss``), so single-retriever baselines can
be evaluated by reading the relevant score column from the hybrid rank-data
without re-running the slow encoding pass.
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
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, ".")

from src.common.query_emb_cache import save_query_emb_cache
from src.features.dense_feat import compute_dense_features_with_score
from src.features.sparse import compute_sparse_features
from src.pipeline.rank_dense_features import (
    FaissPassageEmbeddingLookup,
    MedCPTFeatureEncoder,
    dense_embedding_pair_for_candidates,
    uses_faiss_dense,
    uses_medcpt_dense,
)
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import (
    DualHybridRetriever,
    HybridBm25FaissRetriever,
    HybridBm25MedcptRetriever,
    HybridSpladePPFaissRetriever,
    HybridSpladePPMedcptRetriever,
)
from src.retrieval.spladepp import SpladePPRetriever
from src.retrieval.medcpt import MedCPTRetriever
from src.common.question_types import normalize_question_type, qtype_onehot
from src.common.rank_data_paths import normalize_retriever_name, rank_data_file
from src.retrieval.faiss_util import faiss_gpu_available


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

def _faiss_kwargs(cfg, device: str, args) -> Dict:
    use_gpu = bool(cfg.retrieval.get("faiss_use_gpu", True))
    if getattr(args, "no_faiss_gpu", False):
        use_gpu = False
    if getattr(args, "faiss_gpu", False):
        use_gpu = True
    return {
        "device": device,
        "use_faiss_gpu": use_gpu and device == "cuda",
        "faiss_gpu_id": int(cfg.retrieval.get("faiss_gpu_id", 0)),
    }

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


def _lru_touch(od: OrderedDict, key: str, max_entries: Optional[int]) -> None:
    if max_entries and max_entries > 0 and key in od:
        od.move_to_end(key)


def _lru_put(
    od: OrderedDict,
    key: str,
    value: List,
    max_entries: Optional[int],
) -> None:
    if key in od:
        del od[key]
    od[key] = value
    if max_entries and max_entries > 0:
        while len(od) > max_entries:
            od.popitem(last=False)

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


def merge_query_caches(
    src_paths: List[str],
    out_path: str,
    *,
    expected_dim: Optional[int] = None,
) -> int:
    merged: Dict[str, List[float]] = {}
    for p in src_paths:
        from src.common.query_emb_cache import load_query_emb_cache

        merged.update(load_query_emb_cache(p, expected_dim=expected_dim))
    if not merged:
        return 0
    save_query_emb_cache(out_path, merged, expected_dim=expected_dim)
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
        "spladepp_dir": root / "spladepp" / dataset_key,
        "medcpt_dir": root / "medcpt" / dataset_key,
    }


def get_hybrid_retriever(dataset_key: str, cfg, **dense_kw) -> HybridBm25FaissRetriever:
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
        **dense_kw,
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


def get_faiss_retriever(dataset_key: str, cfg, **dense_kw) -> DenseRetriever:
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
        **dense_kw,
    )


def get_spladepp_retriever(dataset_key: str, cfg, device: str = "cuda") -> SpladePPRetriever:
    paths = _index_paths(dataset_key)
    splade_dir = paths["spladepp_dir"]
    if not SpladePPRetriever.index_ready(str(splade_dir)):
        raise FileNotFoundError(f"SPLADE++ index not found/ready: {splade_dir}")
    return SpladePPRetriever(
        index_path=str(splade_dir),
        model_name=str(
            cfg.retrieval.get(
                "spladepp_encoder", "naver/splade-cocondenser-ensembledistil"
            )
        ),
        device=device,
        batch_size=int(cfg.retrieval.get("spladepp_batch_size", 64)),
        max_length=int(cfg.retrieval.get("spladepp_max_length", 256)),
    )


def get_medcpt_retriever(dataset_key: str, cfg, device: str = "cuda") -> MedCPTRetriever:
    paths = _index_paths(dataset_key)
    medcpt_dir = paths["medcpt_dir"]
    if not MedCPTRetriever.index_ready(str(medcpt_dir)):
        raise FileNotFoundError(f"MedCPT index not found/ready: {medcpt_dir}")
    return MedCPTRetriever(
        index_path=str(medcpt_dir),
        article_encoder=str(
            cfg.retrieval.get("medcpt_article_encoder", "ncbi/MedCPT-Article-Encoder")
        ),
        query_encoder=str(
            cfg.retrieval.get("medcpt_query_encoder", "ncbi/MedCPT-Query-Encoder")
        ),
        device=device,
        batch_size=int(cfg.retrieval.get("medcpt_batch_size", 256)),
        max_length=int(cfg.retrieval.get("medcpt_max_length", 512)),
    )


def get_hybrid_spladepp_medcpt_retriever(
    dataset_key: str, cfg, *, device: str = "cuda"
) -> DualHybridRetriever:
    """Create hybrid retriever (SPLADE++ + MedCPT)."""
    paths = _index_paths(dataset_key)
    if not SpladePPRetriever.index_ready(str(paths["spladepp_dir"])):
        raise FileNotFoundError(f"SPLADE++ index not found/ready: {paths['spladepp_dir']}")
    if not MedCPTRetriever.index_ready(str(paths["medcpt_dir"])):
        raise FileNotFoundError(f"MedCPT index not found/ready: {paths['medcpt_dir']}")
    spladepp = get_spladepp_retriever(dataset_key, cfg, device=device)
    medcpt = get_medcpt_retriever(dataset_key, cfg, device=device)
    return HybridSpladePPMedcptRetriever(
        spladepp=spladepp,
        medcpt=medcpt,
        top_k_spladepp=int(cfg.retrieval.get("top_k_spladepp", 50)),
        top_k_medcpt=int(cfg.retrieval.get("top_k_medcpt", 50)),
    )


def get_hybrid_bm25_medcpt_retriever(
    dataset_key: str, cfg, *, device: str = "cuda"
) -> DualHybridRetriever:
    """Create hybrid retriever (BM25 + MedCPT)."""
    paths = _index_paths(dataset_key)
    if not (paths["bm25_dir"] / "index.pkl").exists():
        raise FileNotFoundError(f"BM25 index not found: {paths['bm25_dir']}")
    if not MedCPTRetriever.index_ready(str(paths["medcpt_dir"])):
        raise FileNotFoundError(f"MedCPT index not found/ready: {paths['medcpt_dir']}")
    bm25 = get_bm25_retriever(dataset_key, cfg)
    medcpt = get_medcpt_retriever(dataset_key, cfg, device=device)
    return HybridBm25MedcptRetriever(
        bm25=bm25,
        medcpt=medcpt,
        top_k_bm25=int(cfg.retrieval.top_k_bm25),
        top_k_medcpt=int(cfg.retrieval.get("top_k_medcpt", 50)),
    )


def get_hybrid_spladepp_faiss_retriever(
    dataset_key: str, cfg, **dense_kw
) -> DualHybridRetriever:
    """Create hybrid retriever (SPLADE++ + FAISS)."""
    paths = _index_paths(dataset_key)
    if not SpladePPRetriever.index_ready(str(paths["spladepp_dir"])):
        raise FileNotFoundError(f"SPLADE++ index not found/ready: {paths['spladepp_dir']}")
    if not paths["faiss_path"].exists():
        raise FileNotFoundError(f"FAISS index not found: {paths['faiss_path']}")
    spladepp = get_spladepp_retriever(
        dataset_key, cfg, device=str(dense_kw.get("device", "cuda"))
    )
    dense = get_faiss_retriever(dataset_key, cfg, **dense_kw)
    return HybridSpladePPFaissRetriever(
        spladepp=spladepp,
        dense=dense,
        top_k_spladepp=int(cfg.retrieval.get("top_k_spladepp", 50)),
        top_k_dense=int(
            cfg.retrieval.get("top_k_faiss", cfg.retrieval.get("top_k_dense", 50))
        ),
    )


def get_retriever(
    dataset_key: str,
    cfg,
    retriever_type: str = "hybrid_bm25_faiss",
    *,
    device: str = "cuda",
    dense_kw: Optional[Dict] = None,
):
    """Factory function to get the appropriate retriever."""
    retriever_type = normalize_retriever_name(retriever_type)
    dk = dict(dense_kw or {})
    if retriever_type == "bm25":
        return get_bm25_retriever(dataset_key, cfg)
    elif retriever_type == "faiss":
        return get_faiss_retriever(dataset_key, cfg, **dk)
    elif retriever_type == "spladepp":
        return get_spladepp_retriever(dataset_key, cfg, device=device)
    elif retriever_type == "medcpt":
        return get_medcpt_retriever(dataset_key, cfg, device=device)
    elif retriever_type == "hybrid_bm25_faiss":
        return get_hybrid_retriever(dataset_key, cfg, **dk)
    elif retriever_type == "hybrid_bm25_medcpt":
        return get_hybrid_bm25_medcpt_retriever(dataset_key, cfg, device=device)
    elif retriever_type == "hybrid_spladepp_faiss":
        return get_hybrid_spladepp_faiss_retriever(dataset_key, cfg, **dk)
    elif retriever_type == "hybrid_spladepp_medcpt":
        return get_hybrid_spladepp_medcpt_retriever(dataset_key, cfg, device=device)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

# -----------------------------------------------------------------------------
# Query Processing
# -----------------------------------------------------------------------------

def compute_query_bundle(question: str, qtype: str, encoder) -> Dict:
    """PubMedBERT query embedding for the GARDIAN controller (768-d)."""
    q_emb = encoder.encode(
        [question],
        batch_size=1,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )[0]
    return {
        "query_emb": q_emb,
        "qtype_onehot": qtype_onehot(qtype),
    }

def process_queries(
    queries: List[Dict],
    out_path: str,
    retriever,
    retriever_type: str,
    encoder,
    idf_table: Optional[Dict[str, float]] = None,
    max_candidates: int = 400,
    max_queries: Optional[int] = None,
    expected_query_feat_dim: Optional[int] = None,
    lean_records: bool = False,
    query_cache_out_path: Optional[str] = None,
    dense_from_retriever_scores: bool = False,
    faiss_lookup: Optional[FaissPassageEmbeddingLookup] = None,
    medcpt_encoder: Optional[MedCPTFeatureEncoder] = None,
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
            qb = compute_query_bundle(question, qtype, encoder)
            q_emb = qb["query_emb"]
            if query_cache_out_path and qid not in qid_emb_cache:
                qid_emb_cache[str(qid)] = q_emb.tolist()
            if expected_query_feat_dim is not None and int(q_emb.shape[0]) != int(expected_query_feat_dim):
                raise ValueError(
                    f"query_emb dim mismatch for qid={qid}: got={int(q_emb.shape[0])} "
                    f"expected={int(expected_query_feat_dim)}"
                )
            qtype_oh = qb["qtype_onehot"]

            # Active dense score used by the dense branch. For FAISS hybrids it
            # is the FAISS/PubMedBERT score; for MedCPT hybrids it is the
            # MedCPT asymmetric query/article dot product. This keeps the dense
            # branch aligned with the dense retriever in each hybrid family.
            use_dense_scores = dense_from_retriever_scores and retriever_type in {
                "bm25",
                "spladepp",
                "faiss",
                "medcpt",
                "hybrid_bm25_faiss",
                "hybrid_bm25_medcpt",
                "hybrid_spladepp_faiss",
                "hybrid_spladepp_medcpt",
            }
            active_dense_scores: List[float] = []
            for cand in candidates:
                if retriever_type == "bm25":
                    active_dense_scores.append(float(cand.get("bm25_score", cand.get("score", 0.0))))
                elif retriever_type == "spladepp":
                    active_dense_scores.append(float(cand.get("spladepp_score", cand.get("score", 0.0))))
                elif retriever_type in {"medcpt", "hybrid_spladepp_medcpt", "hybrid_bm25_medcpt"}:
                    active_dense_scores.append(
                        float(cand.get("medcpt_score", cand.get("dense_score", cand.get("score", 0.0))))
                    )
                else:
                    active_dense_scores.append(float(cand.get("dense_score", cand.get("score", 0.0))))
            dense_score_mean = float(np.mean(active_dense_scores))
            dense_score_std = float(np.std(active_dense_scores) + 1e-8)

            q_dense = None
            p_embs: List[np.ndarray] = []
            if not use_dense_scores:
                q_dense, p_embs = dense_embedding_pair_for_candidates(
                    retriever_type=retriever_type,
                    question=question,
                    candidates=candidates,
                    pubmedbert_encoder=encoder,
                    faiss_lookup=faiss_lookup,
                    medcpt_encoder=medcpt_encoder,
                )

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
                spladepp_score = 0.0
                
                if retriever_type == "bm25":
                    bm25_score = float(cand.get("score", cand.get("bm25_score", 0.0)))
                elif retriever_type == "faiss":
                    dense_score = float(cand.get("score", cand.get("dense_score", 0.0)))
                elif retriever_type == "medcpt":
                    dense_score = float(cand.get("score", cand.get("medcpt_score", 0.0)))
                elif retriever_type == "hybrid_bm25_faiss":
                    bm25_score = float(cand.get("bm25_score", 0.0))
                    dense_score = float(cand.get("dense_score", 0.0))
                elif retriever_type == "hybrid_bm25_medcpt":
                    bm25_score = float(cand.get("bm25_score", 0.0))
                    dense_score = float(cand.get("medcpt_score", 0.0))
                elif retriever_type == "hybrid_spladepp_faiss":
                    spladepp_score = float(cand.get("spladepp_score", 0.0))
                    dense_score = float(cand.get("dense_score", 0.0))
                elif retriever_type == "hybrid_spladepp_medcpt":
                    spladepp_score = float(cand.get("spladepp_score", 0.0))
                    dense_score = float(cand.get("medcpt_score", 0.0))
                elif retriever_type == "spladepp":
                    spladepp_score = float(cand.get("score", cand.get("spladepp_score", 0.0)))
                
                # Compute features (use appropriate scores)
                sparse_signal_score = (
                    bm25_score
                    if retriever_type in ("bm25", "hybrid_bm25_faiss", "hybrid_bm25_medcpt")
                    else spladepp_score
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
                    ds = active_dense_scores[i]
                    z = (ds - dense_score_mean) / (dense_score_std + 1e-8)
                    dense_feats = [float(ds), 0.0, 0.0, float(z)]
                else:
                    dense_feats = compute_dense_features_with_score(
                        q_emb=q_dense,
                        p_emb=p_embs[i],
                        dense_score=active_dense_scores[i],
                        score_mean=dense_score_mean,
                        score_std=dense_score_std,
                    ).tolist()

                sf = np.asarray(sparse_feats, dtype=np.float32)
                df = np.asarray(dense_feats, dtype=np.float32)
                sparse_l1_sum += float(np.mean(np.abs(sf)))
                dense_l1_sum += float(np.mean(np.abs(df)))
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
                    "qtype_onehot": qtype_oh,
                    "retriever_type": retriever_type,
                    "has_positive_in_pool": has_positive_in_pool,
                }
                # Store only score channels relevant to current retriever.
                # Avoid writing constant zero score columns across massive files.
                if retriever_type in (
                    "bm25",
                    "hybrid_bm25_faiss",
                    "hybrid_bm25_medcpt",
                ):
                    rec["bm25_score"] = bm25_score
                if retriever_type in (
                    "faiss",
                    "hybrid_bm25_faiss",
                    "hybrid_spladepp_faiss",
                    "medcpt",
                    "hybrid_bm25_medcpt",
                    "hybrid_spladepp_medcpt",
                ):
                    rec["dense_score"] = dense_score
                if retriever_type in (
                    "spladepp",
                    "hybrid_spladepp_faiss",
                    "hybrid_spladepp_medcpt",
                ):
                    rec["spladepp_score"] = spladepp_score
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
            f"dense_mean_abs={dense_l1_sum / diag_samples:.6f}"
        )
    if query_cache_out_path and qid_emb_cache:
        save_query_emb_cache(
            query_cache_out_path,
            qid_emb_cache,
            expected_dim=expected_query_feat_dim,
        )

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
            "spladepp",
            "medcpt",
            "hybrid_bm25_faiss",
            "hybrid_bm25_medcpt",
            "hybrid_spladepp_faiss",
            "hybrid_spladepp_medcpt",
            # backward-compatible aliases
            "hybrid",
            "hybrid_neural",
            "all",
        ],
        default="all",
        help=(
            "Retriever type: bm25, faiss, spladepp, medcpt, "
            "hybrid_bm25_faiss, hybrid_bm25_medcpt, hybrid_spladepp_faiss, "
            "hybrid_spladepp_medcpt, or all (iterates the 4 hybrid families). "
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
        "--exact-kg-distances",
        action="store_true",
        help="Force exact KG shortest-path distance features for this run.",
    )
    parser.add_argument(
        "--no-exact-kg-distances",
        action="store_true",
        help="Disable exact KG shortest-path distance features for this run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for encoder/retriever components. Default auto.",
    )
    parser.add_argument(
        "--faiss-gpu",
        action="store_true",
        help="Force FAISS index search/reconstruct on GPU (needs faiss-gpu package).",
    )
    parser.add_argument(
        "--no-faiss-gpu",
        action="store_true",
        help="Keep FAISS on CPU even when --device cuda.",
    )
    parser.add_argument(
        "--text-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip KG feature computation (default: configs/base.yaml model.text_only).",
    )
    parser.add_argument(
        "--passage-entity-cache-max",
        type=int,
        default=150_000,
        help=(
            "Max unique passage IDs to retain for entity-linking cache (LRU). "
            "Reduces RAM on large runs; evicted passages are re-linked if seen again. "
            "Use 0 for unlimited (previous behavior)."
        ),
    )

    args = parser.parse_args()
    
    cfg = OmegaConf.load("configs/base.yaml")
    set_all_seeds(int(cfg.seed))
    configured_query_feat_dim = int(cfg.model.query_feat_dim)

    if args.text_only is False:
        raise ValueError(
            "KG rank-data generation was removed. Omit --no-text-only (text-only is default)."
        )
    logger.info("Text-only rank generation (no KG features).")

    logger.info(
        "Encoder config | controller query_emb: "
        f"{cfg.encoder.model_name} (dim={configured_query_feat_dim}, "
        f"batch={cfg.encoder.batch_size}, max_len={cfg.encoder.max_length}) | "
        f"SPLADE++={cfg.retrieval.spladepp_encoder} | "
        f"MedCPT query={cfg.retrieval.medcpt_query_encoder} | "
        f"MedCPT article={cfg.retrieval.medcpt_article_encoder}"
    )

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda requested but torch.cuda.is_available() is False in this venv."
        )
    dense_kw = _faiss_kwargs(cfg, device, args)
    if dense_kw["use_faiss_gpu"] and device == "cuda":
        if faiss_gpu_available():
            logger.info(
                f"FAISS: native GPU if available (gpu_id={dense_kw['faiss_gpu_id']})"
            )
        else:
            logger.info(
                f"FAISS: torch CUDA fallback on gpu_id={dense_kw['faiss_gpu_id']} "
                "(faiss-cpu only — no faiss-gpu pip package needed)"
            )
    if not args.dense_from_retriever_scores:
        logger.info(
            "Tip: add --dense-from-retriever-scores for much faster rank-data generation "
            "(skips per-candidate FAISS reconstruct; dense_feats use [score,0,0,z])."
        )

    logger.info(f"Loading PubMedBERT encoder (fp32, matches FAISS index) on {device} ...")
    encoder = SentenceTransformer(cfg.encoder.model_name, device=device)

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

    # ``--retriever all`` iterates the four hybrid families used in the paper:
    #   (1) BM25 + FAISS              (classical sparse + neural dense)
    #   (2) BM25 + MedCPT             (classical sparse + biomedical dense)
    #   (3) SPLADE++ + FAISS          (learned sparse + neural dense)
    #   (4) SPLADE++ + MedCPT         (learned sparse + biomedical dense)
    # Single retrievers still work explicitly for ablations.
    retriever_types = (
        [
            "hybrid_bm25_faiss",
            "hybrid_bm25_medcpt",
            "hybrid_spladepp_faiss",
            "hybrid_spladepp_medcpt",
        ]
        if args.retriever == "all"
        else [normalize_retriever_name(args.retriever)]
    )

    for retriever_type in retriever_types:
        medcpt_encoder = None
        if uses_medcpt_dense(retriever_type):
            logger.info("Loading MedCPT encoders for dense-branch distance features ...")
            medcpt_encoder = MedCPTFeatureEncoder(
                article_encoder=str(cfg.retrieval.medcpt_article_encoder),
                query_encoder=str(cfg.retrieval.medcpt_query_encoder),
                device=device,
                batch_size=int(cfg.retrieval.medcpt_batch_size),
                max_length=int(cfg.retrieval.medcpt_max_length),
                fp16=(device == "cuda"),
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
                retriever = get_retriever(
                    unified_corpus,
                    cfg,
                    retriever_type,
                    device=device,
                    dense_kw=dense_kw,
                )
            except FileNotFoundError as e:
                logger.warning(f"Skipping unified mode for retriever={retriever_type} - {e}")
                continue

            faiss_lookup = None
            if uses_faiss_dense(retriever_type):
                upaths = _index_paths(unified_corpus)
                if upaths["faiss_path"].exists():
                    faiss_lookup = FaissPassageEmbeddingLookup(
                        str(upaths["faiss_path"]),
                        str(upaths["faiss_meta"]),
                        use_faiss_gpu=dense_kw["use_faiss_gpu"],
                        faiss_gpu_id=dense_kw["faiss_gpu_id"],
                    )

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
                            faiss_lookup=faiss_lookup,
                            medcpt_encoder=medcpt_encoder,
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
                    retriever = get_retriever(
                        corpus_dir,
                        cfg,
                        retriever_type,
                        device=device,
                        dense_kw=dense_kw,
                    )
                except FileNotFoundError as e:
                    logger.warning(f"Skipping {dataset_name} - {e}")
                    continue

                faiss_lookup = None
                if uses_faiss_dense(retriever_type):
                    dpaths = _index_paths(corpus_dir)
                    if dpaths["faiss_path"].exists():
                        faiss_lookup = FaissPassageEmbeddingLookup(
                            str(dpaths["faiss_path"]),
                            str(dpaths["faiss_meta"]),
                            use_faiss_gpu=dense_kw["use_faiss_gpu"],
                            faiss_gpu_id=dense_kw["faiss_gpu_id"],
                        )

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
                            faiss_lookup=faiss_lookup,
                            medcpt_encoder=medcpt_encoder,
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
                expected_dim=expected_query_feat_dim,
            )
            merge_query_caches(
                cache_paths_train,
                f"data/query_emb_cache_{rname}_train_all.pkl",
                expected_dim=expected_query_feat_dim,
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