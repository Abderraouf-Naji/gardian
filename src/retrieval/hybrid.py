"""Hybrid retrievers: union of two retrievers, deduplicated by passage id."""

from typing import Dict, List, Optional
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .biobert import BioBERTRetriever
from .doc2query import Doc2QueryRetriever


class HybridRetriever:
    """
    Combine BM25 and Dense results into a candidate pool.

    The paper uses the union of top-200 BM25 and top-200 Dense
    results, deduplicated by passage id, giving ~300-400 candidates.
    """

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever,
                 top_k_bm25: int = 200, top_k_dense: int = 200):
        self.bm25 = bm25
        self.dense = dense
        self.top_k_bm25 = top_k_bm25
        self.top_k_dense = top_k_dense

    def retrieve(self, query: str, top_k: Optional[int] = None, **kwargs) -> List[Dict]:
        """
        Returns deduplicated candidate list.
        Each entry: {id, text, bm25_score, dense_score}

        ``top_k`` / extra kwargs are accepted for API compatibility with
        ``scripts/03_generate_rank_data.py`` (ignored; internal top_k_bm25 /
        top_k_dense are used).
        """
        _ = top_k, kwargs
        bm25_hits = {h["id"]: h for h in self.bm25.retrieve(query, self.top_k_bm25)}
        dense_hits = {h["id"]: h for h in self.dense.retrieve(query, self.top_k_dense)}
        all_ids = set(bm25_hits) | set(dense_hits)

        results = []
        for pid in all_ids:
            bh = bm25_hits.get(pid, {})
            dh = dense_hits.get(pid, {})
            results.append({
                "id": pid,
                "text": bh.get("text") or dh.get("text", ""),
                "bm25_score": bh.get("score", 0.0),
                "dense_score": dh.get("score", 0.0),
            })
        return results

    def batch_retrieve(self, queries: List[str]) -> List[List[Dict]]:
        """Batch retrieve for multiple queries."""
        return [self.retrieve(q) for q in queries]


class DualHybridRetriever:
    """
    Generic two-retriever union with score preservation.

    Useful for:
      - BM25 + FAISS (lexical + dense)
      - BioBERT + Doc2Query (neural + lexical expansion)
    """

    def __init__(
        self,
        first,
        second,
        first_score_key: str,
        second_score_key: str,
        top_k_first: int = 200,
        top_k_second: int = 200,
    ):
        self.first = first
        self.second = second
        self.first_score_key = first_score_key
        self.second_score_key = second_score_key
        self.top_k_first = top_k_first
        self.top_k_second = top_k_second

    def _extract_score(self, hit: Dict, key: str) -> float:
        if key in hit:
            return float(hit.get(key, 0.0))
        # Backward compatibility for retrievers that return generic "score".
        return float(hit.get("score", 0.0))

    def retrieve(self, query: str, top_k: Optional[int] = None, **kwargs) -> List[Dict]:
        _ = top_k, kwargs
        first_hits = {h["id"]: h for h in self.first.retrieve(query, self.top_k_first)}
        second_hits = {h["id"]: h for h in self.second.retrieve(query, self.top_k_second)}
        all_ids = set(first_hits) | set(second_hits)

        results = []
        for pid in all_ids:
            h1 = first_hits.get(pid, {})
            h2 = second_hits.get(pid, {})
            results.append(
                {
                    "id": pid,
                    "text": h1.get("text") or h2.get("text", ""),
                    self.first_score_key: self._extract_score(h1, self.first_score_key),
                    self.second_score_key: self._extract_score(h2, self.second_score_key),
                }
            )
        return results

    def batch_retrieve(self, queries: List[str]) -> List[List[Dict]]:
        return [self.retrieve(q) for q in queries]


class HybridBm25FaissRetriever(DualHybridRetriever):
    """Union of BM25 + FAISS retrievers."""

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        top_k_bm25: int = 200,
        top_k_dense: int = 200,
    ):
        super().__init__(
            first=bm25,
            second=dense,
            first_score_key="bm25_score",
            second_score_key="dense_score",
            top_k_first=top_k_bm25,
            top_k_second=top_k_dense,
        )


class HybridBioBertDoc2QueryRetriever(DualHybridRetriever):
    """Union of BioBERT + Doc2Query retrievers."""

    def __init__(
        self,
        biobert: BioBERTRetriever,
        doc2query: Doc2QueryRetriever,
        top_k_biobert: int = 200,
        top_k_doc2query: int = 200,
    ):
        super().__init__(
            first=biobert,
            second=doc2query,
            first_score_key="biobert_score",
            second_score_key="doc2query_score",
            top_k_first=top_k_biobert,
            top_k_second=top_k_doc2query,
        )