"""Real SPLADE++ retriever (sparse-MLM, vocabulary space).

This implementation is the canonical SPLADE formulation: each text is mapped
to a sparse non-negative vector over the model's WordPiece vocabulary by

    s = max_{j over tokens}  log(1 + ReLU(MLM_logits_j))

i.e. the masked-LM logits are saturated with ``log(1 + ReLU(.))`` and then
max-pooled over the input sequence (with attention-mask zeroing). This gives
an interpretable lexical representation that mostly contains 0s, plus a few
hundred non-zero "term weights".

The default checkpoint is ``naver/splade-cocondenser-ensembledistil``. We refer
to this branch as SPLADE++ throughout the codebase: it is open, stable, and a
standard learned-sparse baseline for retrieval papers.

Index layout (one folder per corpus, e.g. ``data/indices/spladepp/medmcqa/``):

    spladepp_index.pt   torch.save({...}) with sparse CSR tensors + meta

Storage format inside ``spladepp_index.pt``::

    {
        "model_name":      str,
        "vocab_size":      int (V),
        "csr_indptr":      torch.LongTensor[N+1],
        "csr_indices":     torch.LongTensor[nnz],
        "csr_values":      torch.FloatTensor[nnz],
        "meta":            list[dict] of length N,
    }

Retrieval at query time: encode the query into the same V-dim sparse vector,
then compute the dot product against every passage row of the CSR matrix. We
use a contiguous SciPy-free implementation in PyTorch to keep the runtime
zero-dependency and GPU-friendly when the index fits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from loguru import logger
from transformers import AutoModelForMaskedLM, AutoTokenizer


class SpladePPRetriever:
    """Real SPLADE retriever (sparse-MLM, max-pool, vocabulary-space)."""

    INDEX_NAME = "spladepp_index.pt"

    def __init__(
        self,
        index_path: str,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        device: str = "cuda",
        batch_size: int = 64,
        max_length: int = 256,
        fp16: bool = True,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.index_path = Path(index_path)
        self.index_file = self.index_path / self.INDEX_NAME
        self.model_name = str(model_name)
        self.device = device
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.fp16 = bool(fp16) and device == "cuda"

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForMaskedLM] = None

        self.vocab_size: Optional[int] = None
        self.meta: List[Dict] = []
        self._csr_indptr: Optional[torch.Tensor] = None    # [N+1]
        self._csr_indices: Optional[torch.Tensor] = None   # [nnz]
        self._csr_values: Optional[torch.Tensor] = None    # [nnz]
        self._csr_row_ids: Optional[torch.Tensor] = None    # [nnz]

        if self.index_file.exists():
            self._load_index()

    @classmethod
    def index_ready(cls, index_path: str) -> bool:
        return (Path(index_path) / cls.INDEX_NAME).exists()

    def _ensure_model(self) -> None:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, use_fast=True
            )
        if self.model is None:
            # ``use_safetensors=True`` avoids transformers' new CVE-2025-32434
            # guard that blocks ``torch.load`` of ``pytorch_model.bin`` when the
            # local torch build is < 2.6. The SPLADE++ checkpoint already ships
            # a ``model.safetensors`` file on the Hub, so this is a zero-cost
            # switch.
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.model_name, use_safetensors=True
            )
            self.model.to(self.device).eval()
            self.vocab_size = int(self.model.config.vocab_size)

    @torch.no_grad()
    def _encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Return dense [B, V] tensor (still on GPU). Caller sparsifies."""
        assert self.tokenizer is not None and self.model is not None
        toks = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            if self.fp16
            else torch.amp.autocast(device_type="cuda", enabled=False)
            if self.device == "cuda"
            else torch.amp.autocast(device_type="cpu", enabled=False)
        )
        with amp_ctx:
            out = self.model(**toks).logits  # [B, L, V]
        relu_log = torch.log1p(torch.relu(out))            # log(1 + ReLU(.))
        mask = toks["attention_mask"].unsqueeze(-1).to(relu_log.dtype)  # [B, L, 1]
        masked = relu_log * mask
        sparse_vec, _ = masked.max(dim=1)                  # [B, V] max-pool over tokens
        return sparse_vec.float()

    # ------------------------------------------------------------------
    # Index build
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
        if idx_file.exists() and not overwrite:
            logger.info(f"SPLADE++ index exists at {idx_file}, skipping")
            return

        self._ensure_model()
        assert self.vocab_size is not None

        texts: List[str] = []
        meta: List[Dict] = []
        with open(corpus_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                title = rec.get("title", "") or ""
                text = rec.get("text", "") or ""
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
        V = int(self.vocab_size)
        logger.info(
            f"Encoding SPLADE++ corpus ({n:,} passages, V={V}) "
            f"on {self.device} (fp16={self.fp16}) ..."
        )

        # Streaming CSR accumulation (avoids holding [N, V] dense in RAM).
        indptr_list: List[int] = [0]
        indices_chunks: List[torch.Tensor] = []
        values_chunks: List[torch.Tensor] = []
        nnz_running = 0
        running_doc_nnz = 0

        for i in range(0, n, self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            sparse_vec = self._encode_batch(batch_texts)  # [b, V] on device
            # Move to CPU early so the GPU isn't choked on the long tail.
            sparse_vec_cpu = sparse_vec.cpu()
            for row in sparse_vec_cpu:
                nz_mask = row > 0
                nz_idx = nz_mask.nonzero(as_tuple=False).squeeze(1)  # [nnz_row]
                nz_val = row[nz_idx]
                indices_chunks.append(nz_idx.to(torch.int64))
                values_chunks.append(nz_val.to(torch.float32))
                nnz_running += int(nz_idx.numel())
                running_doc_nnz += int(nz_idx.numel())
                indptr_list.append(nnz_running)

            done = i + len(batch_texts)
            if done % (self.batch_size * 50) == 0 or done >= n:
                avg_nnz = running_doc_nnz / max(done, 1)
                logger.info(
                    f"encoded {done:,}/{n:,} (avg nnz/doc so far: {avg_nnz:.1f})"
                )

        csr_indptr = torch.tensor(indptr_list, dtype=torch.int64)
        csr_indices = torch.cat(indices_chunks) if indices_chunks else torch.empty(0, dtype=torch.int64)
        csr_values = torch.cat(values_chunks) if values_chunks else torch.empty(0, dtype=torch.float32)
        avg = nnz_running / max(n, 1)
        approx_mb = (csr_indices.numel() * 8 + csr_values.numel() * 4) / (1024 * 1024)
        logger.info(
            f"SPLADE++: nnz={nnz_running:,} ({avg:.1f} per doc) "
            f"-> ~{approx_mb:.1f} MB (int64+fp32)"
        )

        torch.save(
            {
                "model_name": self.model_name,
                "vocab_size": V,
                "csr_indptr": csr_indptr,
                "csr_indices": csr_indices,
                "csr_values": csr_values,
                "meta": meta,
            },
            idx_file,
        )
        self.index_path = out
        self.index_file = idx_file
        self.meta = meta
        self.vocab_size = V
        self._csr_indptr = csr_indptr
        self._csr_indices = csr_indices
        self._csr_values = csr_values
        row_lengths = self._csr_indptr[1:] - self._csr_indptr[:-1]
        self._csr_row_ids = torch.repeat_interleave(
            torch.arange(int(row_lengths.numel()), dtype=torch.int64),
            row_lengths,
        )
        logger.success(f"SPLADE++ index saved -> {idx_file}")

    # ------------------------------------------------------------------
    # Index load + retrieve
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        data = torch.load(self.index_file, map_location="cpu")
        self.model_name = str(data.get("model_name", self.model_name))
        self.vocab_size = int(data["vocab_size"])
        self._csr_indptr = data["csr_indptr"].to(torch.int64)
        self._csr_indices = data["csr_indices"].to(torch.int64)
        self._csr_values = data["csr_values"].to(torch.float32)
        row_lengths = self._csr_indptr[1:] - self._csr_indptr[:-1]
        self._csr_row_ids = torch.repeat_interleave(
            torch.arange(int(row_lengths.numel()), dtype=torch.int64),
            row_lengths,
        )
        self.meta = list(data["meta"])
        logger.info(
            f"Loaded SPLADE++ index ({len(self.meta):,} docs, V={self.vocab_size}) "
            f"from {self.index_file}"
        )

    @torch.no_grad()
    def _encode_query_dense(self, query: str) -> torch.Tensor:
        """Return dense [V] tensor for a single query (sparse-friendly)."""
        self._ensure_model()
        return self._encode_batch([query])[0]

    @torch.no_grad()
    def retrieve(self, query: str, top_k: int = 50) -> List[Dict]:
        if (
            self._csr_indptr is None
            or self._csr_indices is None
            or self._csr_values is None
            or self._csr_row_ids is None
            or self.vocab_size is None
        ):
            raise FileNotFoundError(
                f"SPLADE++ index not loaded: {self.index_file}. Build it first."
            )

        q = self._encode_query_dense(query).cpu()  # [V] dense fp32
        # Score per document = sum_{j in nz_doc} q[j] * v_doc[j]
        # Implemented vectorised: gather q at csr_indices, weight by csr_values,
        # then scatter into document rows. Row ids are precomputed at load time.
        gathered = q.index_select(0, self._csr_indices) * self._csr_values  # [nnz]
        n_docs = int(self._csr_indptr.numel() - 1)
        scores = torch.zeros(n_docs, dtype=torch.float32)
        scores.scatter_add_(0, self._csr_row_ids, gathered)

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
                    "spladepp_score": float(s),
                    "score": float(s),
                }
            )
        return out
