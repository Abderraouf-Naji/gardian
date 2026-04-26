"""Dense dual-encoder retriever using FAISS for ANN search."""

import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from loguru import logger


class DenseRetriever:
    """
    Encode queries with a biomedical sentence encoder and search
    a pre-built FAISS index.
    """

    def __init__(
        self,
        faiss_index_path: str,
        meta_path: str,
        encoder_name: str,
        batch_size: int = 64,
        max_length: int = 512,
        device: str = "cuda",  # Will auto-fallback to cpu on Windows if cuda not available
    ):
        import faiss
        from sentence_transformers import SentenceTransformer
        import torch

        # Check CUDA availability on Windows
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"

        self.encoder = SentenceTransformer(encoder_name, device=device)
        self.index = faiss.read_index(faiss_index_path)
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device

        # meta: list of {id, text, ...} in index order
        self.meta: List[Dict] = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.meta.append(json.loads(line))

        if len(self.meta) != self.index.ntotal:
            logger.warning(
                f"Meta rows ({len(self.meta)}) != index.ntotal ({self.index.ntotal})"
            )

        logger.info(f"Dense index loaded ({self.index.ntotal} vectors) on {device}")

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        vecs = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype="float32")

    def retrieve(self, query: str, top_k: int = 200) -> List[Dict]:
        """Retrieve top-k passages for a query."""
        q_vec = self._encode([query])
        scores, indices = self.index.search(q_vec, top_k)
        results: List[Dict] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.meta[idx]
            results.append(
                {
                    "id": meta["id"],
                    "score": float(score),
                    "text": meta.get("text", ""),
                    "title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                }
            )
        return results

    def batch_retrieve(self, queries: List[str], top_k: int = 200) -> List[List[Dict]]:
        """Batch retrieve for multiple queries."""
        q_vecs = self._encode(queries)
        all_scores, all_indices = self.index.search(q_vecs, top_k)
        out: List[List[Dict]] = []

        for scores, indices in zip(all_scores, all_indices):
            results: List[Dict] = []
            for score, idx in zip(scores, indices):
                if idx == -1:
                    continue
                meta = self.meta[idx]
                results.append(
                    {
                        "id": meta["id"],
                        "score": float(score),
                        "text": meta.get("text", ""),
                        "title": meta.get("title", ""),
                        "source": meta.get("source", ""),
                    }
                )
            out.append(results)
        return out


def build_faiss_index(
    corpus_jsonl: str,
    faiss_path: str,
    meta_path: str,
    encoder_name: str,
    batch_size: int = 64,
    device: str = "cuda",
):
    """
    Encode all corpus passages and write a FAISS Flat-IP index.
    
    corpus_jsonl records are expected to have at least:
      {"id": str, "text": str, "title": str, "source": str}
    """
    import faiss
    from sentence_transformers import SentenceTransformer
    import torch

    # Check CUDA availability on Windows
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    encoder = SentenceTransformer(encoder_name, device=device)

    texts: List[str] = []
    meta_rows: List[Dict] = []

    logger.info("Reading corpus …")
    with open(corpus_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            pid = rec["id"]
            title = rec.get("title", "") or ""
            text = rec.get("text", "") or ""
            source = rec.get("source", "")

            # This is the exact string we index on; keep it consistent with retrieval.
            combined = f"{title}. {text}".strip(". ") if title else text

            texts.append(combined)
            meta_rows.append(
                {
                    "id": pid,
                    "title": title,
                    "text": text,
                    "source": source,
                }
            )

    logger.info(f"Encoding {len(texts):,} passages on {device}…")
    vecs = encoder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    vecs = np.asarray(vecs, dtype="float32")

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    Path(faiss_path).parent.mkdir(parents=True, exist_ok=True)
    Path(meta_path).parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, faiss_path)
    logger.info(f"FAISS index written → {faiss_path}")

    with open(meta_path, "w", encoding="utf-8") as f:
        for row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.success(f"FAISS meta written → {meta_path}")