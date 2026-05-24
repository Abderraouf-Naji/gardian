"""Dense-branch embeddings for rank-data generation and live GARDIAN inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger


def uses_faiss_dense(retriever_type: str) -> bool:
    r = str(retriever_type)
    return r in ("faiss", "hybrid_bm25_faiss", "hybrid_spladepp_faiss")


def uses_medcpt_dense(retriever_type: str) -> bool:
    r = str(retriever_type)
    return r in ("medcpt", "hybrid_bm25_medcpt", "hybrid_spladepp_medcpt")


def passage_text_for_encoding(candidate: Dict[str, Any], meta: Optional[Dict] = None) -> str:
    """Match FAISS/MedCPT index strings: title + text when title exists."""
    text = (candidate.get("text") or "").strip()
    title = (candidate.get("title") or "").strip()
    if meta:
        text = text or (meta.get("text") or "").strip()
        title = title or (meta.get("title") or "").strip()
    if title:
        return f"{title}. {text}".strip(". ")
    return text


class FaissPassageEmbeddingLookup:
    """Reconstruct L2-normalized passage vectors from a built FAISS index."""

    def __init__(
        self,
        faiss_index_path: str,
        meta_path: str,
        *,
        use_faiss_gpu: bool = True,
        faiss_gpu_id: int = 0,
    ) -> None:
        from src.retrieval.faiss_util import open_faiss_index

        self._faiss = open_faiss_index(
            str(faiss_index_path),
            use_gpu=use_faiss_gpu,
            gpu_id=faiss_gpu_id,
        )
        self._faiss_backend = self._faiss.backend
        self._pid_to_row: Dict[str, int] = {}
        with Path(meta_path).open("r", encoding="utf-8") as f:
            for row, line in enumerate(f):
                if not line.strip():
                    continue
                rec = json.loads(line)
                pid = rec.get("id")
                if pid is not None:
                    self._pid_to_row[str(pid)] = row
        logger.info(
            f"FAISS passage lookup: {len(self._pid_to_row):,} ids from {faiss_index_path} "
            f"(faiss={self._faiss_backend})"
        )

    def get_embedding(self, pid: str) -> Optional[np.ndarray]:
        row = self._pid_to_row.get(str(pid))
        if row is None:
            return None
        try:
            vec = self._faiss.reconstruct(int(row)).astype(np.float32)
        except Exception:
            return None
        n = float(np.linalg.norm(vec))
        if n > 1e-8:
            vec /= n
        return vec

    def get_embeddings(
        self,
        candidates: Sequence[Dict[str, Any]],
        *,
        fallback_encoder=None,
    ) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        missing: List[Tuple[int, str]] = []
        for i, cand in enumerate(candidates):
            pid = str(cand.get("id", ""))
            vec = self.get_embedding(pid) if pid else None
            if vec is not None:
                out.append(vec)
            else:
                out.append(np.zeros(1, dtype=np.float32))
                if fallback_encoder is not None:
                    missing.append((i, passage_text_for_encoding(cand)))
        if missing and fallback_encoder is not None:
            texts = [t for _, t in missing]
            embs = fallback_encoder.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            for (idx, _), emb in zip(missing, embs):
                out[idx] = np.asarray(emb, dtype=np.float32)
        return out


class MedCPTFeatureEncoder:
    """MedCPT query/article encoders for dense-branch distance features (no index required)."""

    def __init__(
        self,
        *,
        article_encoder: str,
        query_encoder: str,
        device: str = "cuda",
        batch_size: int = 256,
        max_length: int = 512,
        fp16: bool = True,
    ) -> None:
        from src.retrieval.medcpt import MedCPTRetriever

        self._rt = MedCPTRetriever(
            index_path="/dev/null",
            article_encoder=article_encoder,
            query_encoder=query_encoder,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            gpu_corpus=False,
            fp16=fp16,
        )

    def encode_query(self, question: str) -> np.ndarray:
        return self._rt._encode_query(question).numpy().astype(np.float32)

    def encode_passages(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []
        embs = self._rt._encode_articles(texts)
        return [embs[i].numpy().astype(np.float32) for i in range(len(texts))]


def dense_embedding_pair_for_candidates(
    *,
    retriever_type: str,
    question: str,
    candidates: Sequence[Dict[str, Any]],
    pubmedbert_encoder,
    faiss_lookup: Optional[FaissPassageEmbeddingLookup] = None,
    medcpt_encoder: Optional[MedCPTFeatureEncoder] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
  Return (q_dense, list[p_dense]) in the space used for dense_feats[1:2].

  Controller ``query_emb`` stays PubMedBERT; this pair is only for |q-p| features.
  """
    if uses_medcpt_dense(retriever_type):
        if medcpt_encoder is None:
            raise ValueError("MedCPT encoder required for MedCPT hybrid rank features")
        q = medcpt_encoder.encode_query(question)
        texts = [passage_text_for_encoding(c) for c in candidates]
        return q, medcpt_encoder.encode_passages(texts)

    q = pubmedbert_encoder.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )[0].astype(np.float32)

    if uses_faiss_dense(retriever_type) and faiss_lookup is not None:
        p_list = faiss_lookup.get_embeddings(
            candidates, fallback_encoder=pubmedbert_encoder
        )
        return q, p_list

    texts = [passage_text_for_encoding(c) for c in candidates]
    p_embs = pubmedbert_encoder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return q, [p.astype(np.float32) for p in p_embs]
