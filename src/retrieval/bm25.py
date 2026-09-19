"""BM25 retriever using bm25s (fast, pure Python, no Lucene)."""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional

from loguru import logger
import bm25s

try:
    import Stemmer as _pystemmer
except ImportError:  # pragma: no cover - PyStemmer is a hard dep of the BM25 path
    _pystemmer = None

# BEIR's BM25 baseline stems both sides with Porter. bm25s applies no stemmer by
# default, which is the single largest gap between our BM25 and the published
# TREC-COVID number (nDCG@10 .563 unstemmed vs .656 published). ``k1``/``b`` stay
# at the bm25s "lucene" defaults: sweeping them to BEIR's (0.9/0.4) measured
# *worse* here (.578), so only the stemmer is changed.
DEFAULT_STEMMER = "porter"
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def _make_stemmer(name: Optional[str]):
    """Build a bm25s-compatible stemmer callable, or ``None`` for no stemming."""
    if not name or name == "none":
        return None
    if _pystemmer is None:
        raise ImportError(
            f"stemmer={name!r} requested but PyStemmer is not installed "
            "(pip install PyStemmer)"
        )
    return _pystemmer.Stemmer(name)


class BM25Retriever:
    """
    BM25 retriever with disk persistence support.
    Can load from saved index or build from corpus.
    """

    def __init__(
        self,
        index_dir: Optional[str] = None,
        corpus_jsonl: Optional[str] = None,
        *,
        stemmer: Optional[str] = DEFAULT_STEMMER,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ):
        """
        Either load from saved index directory OR build from corpus file.
        
        Args:
            index_dir: Directory containing saved BM25 index (index.pkl, metadata.pkl)
            corpus_jsonl: Path to corpus JSONL file to build index from
            stemmer: PyStemmer algorithm name, or "none". Ignored when loading an
                index -- the stemmer recorded at build time wins, because querying
                a stemmed index with unstemmed tokens silently loses recall.
            k1, b: BM25 parameters (build-time only, same reasoning).
        """
        self.stemmer_name = stemmer
        self.k1 = float(k1)
        self.b = float(b)
        self._stemmer = _make_stemmer(stemmer)

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

        # The index defines the token space; adopt its analyzer rather than the
        # constructor's, so query tokenisation can never drift from the index.
        self.stemmer_name = metadata.get("stemmer", "none")
        self.k1 = float(metadata.get("k1", DEFAULT_K1))
        self.b = float(metadata.get("b", DEFAULT_B))
        self._stemmer = _make_stemmer(self.stemmer_name)
        if self.stemmer_name in (None, "none"):
            logger.warning(
                f"BM25 index at {index_dir} was built without a stemmer; "
                "rebuild it to match the BEIR baseline (see DEFAULT_STEMMER)."
            )

        logger.success(
            f"BM25 index loaded ({len(self.doc_ids):,} passages | "
            f"stemmer={self.stemmer_name} k1={self.k1} b={self.b})"
        )

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

        logger.info(
            f"Building BM25 index in memory with bm25s "
            f"(stemmer={self.stemmer_name} k1={self.k1} b={self.b}) …"
        )
        # Tokenize corpus once, with the same analyzer retrieve() will use.
        corpus_tokens = bm25s.tokenize(self.docs, stopwords="en", stemmer=self._stemmer)
        # Use lucene method for good default performance
        self.bm25 = bm25s.BM25(method="lucene", k1=self.k1, b=self.b)
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
            "method": "lucene",
            "stemmer": self.stemmer_name or "none",
            "k1": self.k1,
            "b": self.b,
        }
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        
        logger.info(f"BM25 index saved to {index_dir}")

    def retrieve(self, query: str, top_k: int = 50) -> List[Dict]:
        """Retrieve top-k passages for a query."""
        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=self._stemmer)
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
        q_tokens = bm25s.tokenize(
            queries, stopwords="en", stemmer=self._stemmer, show_progress=False
        )
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


def build_bm25_index(
    corpus_jsonl: str,
    index_dir: str,
    *,
    stemmer: Optional[str] = DEFAULT_STEMMER,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
):
    """
    Build and save BM25 index from corpus JSONL.
    This is the main function called by 01_build_index.py.
    """
    logger.info(f"Building BM25 index from {corpus_jsonl}")
    retriever = BM25Retriever(corpus_jsonl=corpus_jsonl, stemmer=stemmer, k1=k1, b=b)
    retriever.save(index_dir)
    logger.success(f"BM25 index saved to {index_dir}")