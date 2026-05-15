"""MedCPT retriever (NCBI biomedical dense, asymmetric Q/A encoders).

MedCPT is a contrastively-pretrained biomedical retriever from NCBI with a
dedicated *Article* encoder for the corpus side and a dedicated *Query*
encoder for the user-question side (asymmetric encoding). For ranking we use
the [CLS] embedding of each side as a single vector and score with a plain
dot product, matching the official ``ncbi/MedCPT-*-Encoder`` model card.

Index layout (one folder per corpus, e.g. ``data/indices/medcpt/medmcqa/``):

    medcpt_index.npy        float16 [N, D] embeddings array
    medcpt_meta.jsonl       per-row metadata: id, title, text, source

At query time:
  * Encode the query once with ``ncbi/MedCPT-Query-Encoder``.
  * Optionally hold the corpus matrix on GPU (fp16) for fast dot products.
  * Top-k by descending dot-product score.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from loguru import logger
from transformers import AutoModel, AutoTokenizer


class MedCPTRetriever:
    """Dense retriever wrapping the asymmetric NCBI MedCPT encoders."""

    INDEX_NAME = "medcpt_index.npy"
    META_NAME = "medcpt_meta.jsonl"

    def __init__(
        self,
        index_path: str,
        article_encoder: str = "ncbi/MedCPT-Article-Encoder",
        query_encoder: str = "ncbi/MedCPT-Query-Encoder",
        device: str = "cuda",
        batch_size: int = 256,
        max_length: int = 512,
        gpu_corpus: bool = True,
        fp16: bool = True,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.index_path = Path(index_path)
        self.index_file = self.index_path / self.INDEX_NAME
        self.meta_file = self.index_path / self.META_NAME
        self.article_encoder_name = str(article_encoder)
        self.query_encoder_name = str(query_encoder)
        self.device = device
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.gpu_corpus = bool(gpu_corpus) and device == "cuda"
        self.fp16 = bool(fp16) and device == "cuda"

        # Lazily loaded so building the corpus side does not pay for the
        # query encoder weights and vice-versa.
        self._a_tokenizer: Optional[AutoTokenizer] = None
        self._a_model: Optional[AutoModel] = None
        self._q_tokenizer: Optional[AutoTokenizer] = None
        self._q_model: Optional[AutoModel] = None

        self.meta: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None             # [N, D] fp16
        self.embeddings_gpu: Optional[torch.Tensor] = None       # [N, D] fp16 if GPU

        if self.index_file.exists() and self.meta_file.exists():
            self._load_index()

    @classmethod
    def index_ready(cls, index_path: str) -> bool:
        p = Path(index_path)
        return (p / cls.INDEX_NAME).exists() and (p / cls.META_NAME).exists()

    # ------------------------------------------------------------------
    # Encoders
    # ------------------------------------------------------------------

    def _load_article_encoder(self) -> None:
        if self._a_model is not None:
            return
        self._a_tokenizer = AutoTokenizer.from_pretrained(
            self.article_encoder_name, use_fast=True
        )
        # See note in spladepp._ensure_model — prefer safetensors so the
        # CVE-2025-32434 guard in transformers >= 5.8 doesn't block loading.
        self._a_model = AutoModel.from_pretrained(
            self.article_encoder_name, use_safetensors=True
        )
        self._a_model.to(self.device).eval()

    def _load_query_encoder(self) -> None:
        if self._q_model is not None:
            return
        self._q_tokenizer = AutoTokenizer.from_pretrained(
            self.query_encoder_name, use_fast=True
        )
        self._q_model = AutoModel.from_pretrained(
            self.query_encoder_name, use_safetensors=True
        )
        self._q_model.to(self.device).eval()

    @torch.no_grad()
    def _encode_articles(self, texts: List[str]) -> torch.Tensor:
        self._load_article_encoder()
        assert self._a_tokenizer is not None and self._a_model is not None
        toks = self._a_tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            if self.fp16
            else torch.amp.autocast(device_type=self.device, enabled=False)
        )
        with amp_ctx:
            out = self._a_model(**toks).last_hidden_state[:, 0, :]  # [B, D] CLS
        return out.float().cpu()

    @torch.no_grad()
    def _encode_query(self, text: str) -> torch.Tensor:
        self._load_query_encoder()
        assert self._q_tokenizer is not None and self._q_model is not None
        toks = self._q_tokenizer(
            [text],
            truncation=True,
            padding=True,
            max_length=64,  # query side is short; matches MedCPT model card
            return_tensors="pt",
        ).to(self.device)
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            if self.fp16
            else torch.amp.autocast(device_type=self.device, enabled=False)
        )
        with amp_ctx:
            out = self._q_model(**toks).last_hidden_state[:, 0, :]  # [1, D]
        return out.float().cpu()[0]

    # ------------------------------------------------------------------
    # Index build / load
    # ------------------------------------------------------------------

    def build_index(
        self,
        corpus_jsonl: str,
        output_dir: str,
        overwrite: bool = False,
    ) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        idx_file = out / self.INDEX_NAME
        meta_file = out / self.META_NAME
        if idx_file.exists() and meta_file.exists() and not overwrite:
            logger.info(f"MedCPT index exists at {idx_file}, skipping")
            return

        texts: List[str] = []
        meta: List[Dict] = []
        with open(corpus_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                title = rec.get("title", "") or ""
                text = rec.get("text", "") or ""
                # MedCPT was contrastively trained on (title, abstract) pairs.
                combined = f"{title}. {text}".strip(". ") if title else text
                texts.append(combined)
                meta.append(
                    {
                        "id": rec["id"],
                        "title": title,
                        "text": text,
                        "source": rec.get("source", ""),
                    }
                )

        n = len(texts)
        logger.info(
            f"Encoding MedCPT corpus ({n:,} passages) on {self.device} "
            f"(article_encoder={self.article_encoder_name}, fp16={self.fp16}) ..."
        )
        chunks: List[np.ndarray] = []
        for i in range(0, n, self.batch_size):
            embs = self._encode_articles(texts[i : i + self.batch_size])
            chunks.append(embs.numpy().astype(np.float16))
            if (i // self.batch_size) % 50 == 0 and i > 0:
                logger.info(f"encoded {i:,}/{n:,}")
        embeddings = np.concatenate(chunks, axis=0).astype(np.float16)

        np.save(idx_file, embeddings)
        with open(meta_file, "w", encoding="utf-8") as f:
            for m in meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        self.index_path = out
        self.index_file = idx_file
        self.meta_file = meta_file
        self.meta = meta
        self.embeddings = embeddings
        self._maybe_move_corpus_to_gpu()
        logger.success(
            f"MedCPT index saved -> {idx_file} "
            f"(N={n:,}, D={int(embeddings.shape[1])})"
        )

    def _load_index(self) -> None:
        # ``np.save`` adds the ``.npy`` extension; we already include it above.
        self.embeddings = np.load(self.index_file).astype(np.float16)
        self.meta = []
        with open(self.meta_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.meta.append(json.loads(line))
        logger.info(
            f"Loaded MedCPT index "
            f"(N={len(self.meta):,}, D={int(self.embeddings.shape[1])}) "
            f"from {self.index_file}"
        )
        self._maybe_move_corpus_to_gpu()

    def _maybe_move_corpus_to_gpu(self) -> None:
        self.embeddings_gpu = None
        if not self.gpu_corpus or self.embeddings is None:
            return
        try:
            self.embeddings_gpu = torch.from_numpy(self.embeddings).to(
                "cuda", dtype=torch.float16
            )
            logger.info(
                f"MedCPT corpus on GPU "
                f"({self.embeddings_gpu.element_size() * self.embeddings_gpu.numel() / (1024**3):.2f} GB fp16)"
            )
        except Exception as e:  # pragma: no cover
            logger.warning(f"Could not move MedCPT corpus to GPU, staying on CPU: {e}")
            self.embeddings_gpu = None

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @torch.no_grad()
    def retrieve(self, query: str, top_k: int = 50) -> List[Dict]:
        if self.embeddings is None:
            raise FileNotFoundError(
                f"MedCPT index not loaded: {self.index_file}. Build it first."
            )

        q = self._encode_query(query)  # [D] fp32 cpu
        if self.embeddings_gpu is not None:
            q_gpu = q.to(self.embeddings_gpu.device, dtype=torch.float16)
            scores = self.embeddings_gpu @ q_gpu  # [N]
            scores = scores.float().cpu()
        else:
            scores = torch.from_numpy(self.embeddings.astype(np.float32)) @ q  # [N]

        n_docs = int(scores.shape[0])
        k = min(int(top_k), n_docs)
        if k == 0:
            return []
        vals, idxs = torch.topk(scores, k=k)
        out: List[Dict] = []
        for s, i in zip(vals.tolist(), idxs.tolist()):
            m = self.meta[int(i)]
            out.append(
                {
                    "id": m["id"],
                    "text": m.get("text", ""),
                    "title": m.get("title", ""),
                    "source": m.get("source", ""),
                    "medcpt_score": float(s),
                    "score": float(s),
                }
            )
        return out
