"""Native ColBERTv2 retriever via RAGatouille engine."""

import json
import os
from pathlib import Path
from typing import Dict, List

from loguru import logger
import torch


class ColBERTRetriever:
    """Wrap RAGatouille ColBERTv2 indexing and retrieval behind repo API."""

    INDEX_NAME = "colbert_index"
    READY_MARKER = "colbert_index_ready.json"
    META_FILE = "colbert_meta.jsonl"

    def __init__(
        self,
        index_path: str,
        model_name: str = "colbert-ir/colbertv2.0",
        device: str = "cuda",
        batch_size: int = 128,  
        max_length: int = 256,
    ):
        self.index_path = Path(index_path)
        self.model_name = model_name
        self.device = device
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.meta_by_id: Dict[str, Dict] = {}
        self.ready_file = self.index_path / self.READY_MARKER
        self.meta_file = self.index_path / self.META_FILE
        self.engine_index_path: str | None = None
        self._rag = None

        if self.ready_file.exists() and self.meta_file.exists():
            self._load_runtime()

    @classmethod
    def index_ready(cls, index_path: str) -> bool:
        p = Path(index_path)
        return (p / cls.READY_MARKER).exists() and (p / cls.META_FILE).exists()

    def _load_runtime(self) -> None:
        self._ensure_cuda_home()
        n_gpu = self._resolve_n_gpu()
        try:
            from ragatouille import RAGPretrainedModel
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "RAGatouille is required for native ColBERTv2. "
                "Install with: pip install ragatouille"
            ) from e

        self._rag = RAGPretrainedModel.from_pretrained(
            self.model_name, index_root=str(self.index_path), n_gpu=n_gpu
        )
        if self.ready_file.exists():
            try:
                info = json.loads(self.ready_file.read_text(encoding="utf-8"))
                self.engine_index_path = str(info.get("engine_index_path") or "")
            except Exception:
                self.engine_index_path = None
        # Load index built with .index(...); ragatouille expects a concrete index path.
        load_path = self.engine_index_path or str(self.index_path / self.INDEX_NAME)
        self._rag.from_index(load_path)
        self.meta_by_id.clear()
        with open(self.meta_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                self.meta_by_id[str(rec["id"])] = rec
        logger.info(f"Loaded native ColBERTv2 index at {self.index_path}")

    def build_index(self, corpus_jsonl: str, output_dir: str, overwrite: bool = False) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ready_file = out / self.READY_MARKER
        meta_file = out / self.META_FILE
        if ready_file.exists() and meta_file.exists() and not overwrite:
            logger.info(f"Native ColBERT index exists at {out}, skipping")
            return

        self._ensure_cuda_home()
        n_gpu = self._resolve_n_gpu()
        try:
            from ragatouille import RAGPretrainedModel
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "RAGatouille is required for native ColBERTv2. "
                "Install with: pip install ragatouille"
            ) from e

        docs: List[str] = []
        doc_ids: List[str] = []
        meta_rows: List[Dict] = []
        with open(corpus_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                pid = str(rec["id"])
                title = rec.get("title", "") or ""
                text = rec.get("text", "") or ""
                combined = f"{title}. {text}".strip(". ") if title else text
                docs.append(combined)
                doc_ids.append(pid)
                meta_rows.append(
                    {
                        "id": pid,
                        "title": title,
                        "text": text,
                        "source": rec.get("source", ""),
                    }
                )

        logger.info(f"Building native ColBERTv2 index ({len(docs):,} passages) ...")
        rag = RAGPretrainedModel.from_pretrained(
            self.model_name, index_root=str(out), n_gpu=n_gpu
        )
        engine_index_path = rag.index(
            collection=docs,
            document_ids=doc_ids,
            index_name=self.INDEX_NAME,
            max_document_length=int(self.max_length),
            overwrite_index=bool(overwrite),
        )

        with open(meta_file, "w", encoding="utf-8") as f:
            for row in meta_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(ready_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": self.model_name,
                    "index_name": self.INDEX_NAME,
                    "engine_index_path": str(engine_index_path),
                    "n_docs": len(meta_rows),
                },
                f,
                indent=2,
            )
        self.index_path = out
        self.ready_file = ready_file
        self.meta_file = meta_file
        self.engine_index_path = str(engine_index_path)
        logger.success(f"Native ColBERTv2 index saved -> {out}")
        self._load_runtime()

    @staticmethod
    def _ensure_cuda_home() -> None:
        """Set CUDA_HOME when CUDA toolkit is installed but env var is missing."""
        if os.environ.get("CUDA_HOME"):
            return
        candidates = (
            "/usr/local/cuda",
            "/usr/local/cuda-12.4",
            "/usr/local/cuda-12.3",
            "/usr/local/cuda-12.2",
            "/usr/local/cuda-12.1",
            "/usr/local/cuda-12.0",
            "/usr/local/cuda-11.8",
        )
        for path in candidates:
            if Path(path).exists():
                os.environ["CUDA_HOME"] = path
                logger.info(f"Detected CUDA_HOME={path}")
                return
        logger.warning(
            "CUDA_HOME is not set and no common CUDA install path was found. "
            "ColBERT extension build may fail."
        )

    def _resolve_n_gpu(self) -> int:
        """
        Decide whether ColBERT should run in GPU mode.
        GPU mode needs CUDA toolkit (CUDA_HOME) for extension compilation.
        """
        wants_cuda = str(self.device).lower().startswith("cuda")
        if not wants_cuda:
            return 0
        if not torch.cuda.is_available():
            return 0
        cuda_home = os.environ.get("CUDA_HOME", "")
        if cuda_home and Path(cuda_home).exists():
            return 1
        logger.warning(
            "Falling back to CPU ColBERT indexing (n_gpu=0) because CUDA toolkit "
            "(CUDA_HOME) is unavailable. Install CUDA toolkit to enable GPU ColBERT indexing."
        )
        return 0

    def retrieve(self, query: str, top_k: int = 50) -> List[Dict]:
        if self._rag is None:
            raise FileNotFoundError(
                f"Native ColBERT index not loaded: {self.index_path}. "
                "Build first with scripts/01_build_spladev3_colbert_indices.py --only colbert."
            )
        hits = self._rag.search(query=query, k=int(top_k))
        out: List[Dict] = []
        for h in hits:
            pid = str(h.get("document_id") or h.get("id") or "")
            meta = self.meta_by_id.get(pid, {})
            out.append(
                {
                    "id": pid or meta.get("id", ""),
                    "text": meta.get("text", h.get("content", "")),
                    "title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                    "colbert_score": float(h.get("score", 0.0)),
                    "score": float(h.get("score", 0.0)),
                }
            )
        return out
