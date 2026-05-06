"""BM25 retriever using bm25s (fast, pure Python, no Lucene)."""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional

from loguru import logger
import bm25s


class BM25Retriever:
    """
    BM25 retriever with disk persistence support.
    Can load from saved index or build from corpus.
    """

    def __init__(self, index_dir: Optional[str] = None, corpus_jsonl: Optional[str] = None):
        """
        Either load from saved index directory OR build from corpus file.
        
        Args:
            index_dir: Directory containing saved BM25 index (index.pkl, metadata.pkl)
            corpus_jsonl: Path to corpus JSONL file to build index from
        """
        if index_dir and Path(index_dir).exists():
            self._load_from_disk(index_dir)
        elif corpus_jsonl:
            self._build_from_corpus(corpus_jsonl)
        else:
            raise ValueError("Either index_dir or corpus_jsonl must be provided")

    def _load_from_disk(self, index_dir: str):
        """Load BM25 index and metadata from disk."""
        index_dir_path = Path(index_dir)
        index_path = index_dir_path / "index.pkl"
        metadata_path = index_dir_path / "metadata.pkl"
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        logger.info(f"Loading BM25 index from {index_dir}")
        
        with open(index_path, "rb") as f:
            self.bm25 = pickle.load(f)
        
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
            self.doc_ids = metadata["doc_ids"]
            self.docs = metadata["docs"]
        
        logger.success(f"BM25 index loaded ({len(self.doc_ids):,} passages)")

    def _build_from_corpus(self, corpus_jsonl: str):
        """Build BM25 index from corpus file."""
        self.docs: List[str] = []
        self.doc_ids: List[str] = []

        logger.info(f"Loading corpus from {corpus_jsonl} …")
        with open(corpus_jsonl, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                self.doc_ids.append(rec["id"])
                # Combine title and text for better retrieval
                title = rec.get("title", "") or ""
                text = rec.get("text", "") or ""
                combined = f"{title}. {text}".strip(". ") if title else text
                self.docs.append(combined)
        logger.info(f"  Loaded {len(self.docs):,} passages")

        logger.info("Building BM25 index in memory with bm25s …")
        # Tokenize corpus once
        corpus_tokens = bm25s.tokenize(self.docs, stopwords="en")
        # Use lucene method for good default performance
        self.bm25 = bm25s.BM25(method="lucene")
        self.bm25.index(corpus_tokens)
        logger.success("BM25 index ready")

    def save(self, index_dir: str):
        """Save BM25 index and metadata to disk."""
        index_dir_path = Path(index_dir)
        index_dir_path.mkdir(parents=True, exist_ok=True)
        
        index_path = index_dir_path / "index.pkl"
        metadata_path = index_dir_path / "metadata.pkl"
        
        # Save BM25 index
        with open(index_path, "wb") as f:
            pickle.dump(self.bm25, f)
        
        # Save metadata
        metadata = {
            "doc_ids": self.doc_ids,
            "docs": self.docs,
            "num_docs": len(self.docs),
            "method": "lucene"
        }
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        
        logger.info(f"BM25 index saved to {index_dir}")

    def retrieve(self, query: str, top_k: int = 50) -> List[Dict]:
        """Retrieve top-k passages for a query."""
        query_tokens = bm25s.tokenize(query, stopwords="en")
        doc_idxs, scores = self.bm25.retrieve(query_tokens, k=top_k)
        doc_idxs = doc_idxs[0]
        scores = scores[0]

        results: List[Dict] = []
        for idx, score in zip(doc_idxs, scores):
            idx = int(idx)
            if idx < 0:
                continue
            results.append(
                {
                    "id": self.doc_ids[idx],
                    "score": float(score),
                    "text": self.docs[idx],
                }
            )
        return results

    def batch_retrieve(self, queries: List[str], top_k: int = 50) -> List[List[Dict]]:
        """Batch retrieve for multiple queries."""
        q_tokens = bm25s.tokenize(queries, stopwords="en")
        all_idxs, all_scores = self.bm25.retrieve(q_tokens, k=top_k)

        out: List[List[Dict]] = []
        for doc_idxs, scores in zip(all_idxs, all_scores):
            results: List[Dict] = []
            for idx, score in zip(doc_idxs, scores):
                idx = int(idx)
                if idx < 0:
                    continue
                results.append(
                    {
                        "id": self.doc_ids[idx],
                        "score": float(score),
                        "text": self.docs[idx],
                    }
                )
            out.append(results)
        return out


def build_bm25_index(corpus_jsonl: str, index_dir: str):
    """
    Build and save BM25 index from corpus JSONL.
    This is the main function called by 01_build_index.py.
    """
    logger.info(f"Building BM25 index from {corpus_jsonl}")
    retriever = BM25Retriever(corpus_jsonl=corpus_jsonl)
    retriever.save(index_dir)
    logger.success(f"BM25 index saved to {index_dir}")