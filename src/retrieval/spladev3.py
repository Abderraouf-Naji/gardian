"""SPLADEv3 retriever with its own on-disk torch index."""

import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModel, AutoTokenizer


class SpladeV3Retriever:
    def __init__(
        self,
        index_path: str,
        model_name: str = "naver/splade-v3-distilbert",
        device: str = "cuda",
        batch_size: int = 128,
        max_length: int = 256,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.index_path = Path(index_path)
        self.index_file = self.index_path / "spladev3_index.pt"
        self.model_name = model_name
        self.device = device
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.meta: List[Dict] = []
        self.embeddings: torch.Tensor | None = None
        if self.index_file.exists():
            self._load_index()

    def _encode(self, texts: List[str]) -> torch.Tensor:
        rows: List[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                toks = self.tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                out = self.model(**toks).last_hidden_state
                mask = toks["attention_mask"].unsqueeze(-1).float()
                pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                pooled = F.normalize(pooled, p=2, dim=1)
                rows.append(pooled.cpu())
        return torch.cat(rows, dim=0)

    def build_index(self, corpus_jsonl: str, output_dir: str, overwrite: bool = False) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        idx_file = out / "spladev3_index.pt"
        if idx_file.exists() and not overwrite:
            logger.info(f"SPLADEv3 index exists at {idx_file}, skipping")
            return

        texts: List[str] = []
        self.meta = []
        with open(corpus_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                title = rec.get("title", "") or ""
                text = rec.get("text", "") or ""
                combined = f"{title}. {text}".strip(". ") if title else text
                texts.append(combined)
                self.meta.append(
                    {
                        "id": rec["id"],
                        "title": title,
                        "text": text,
                        "source": rec.get("source", ""),
                    }
                )
        logger.info(f"Encoding SPLADEv3 corpus ({len(texts):,} passages) on {self.device}...")
        self.embeddings = self._encode(texts).half()
        torch.save(
            {
                "model_name": self.model_name,
                "embeddings": self.embeddings,
                "meta": self.meta,
            },
            idx_file,
        )
        self.index_path = out
        self.index_file = idx_file
        logger.success(f"SPLADEv3 index saved -> {idx_file}")

    def _load_index(self) -> None:
        data = torch.load(self.index_file, map_location="cpu")
        self.embeddings = data["embeddings"].float()
        self.meta = data["meta"]
        logger.info(f"Loaded SPLADEv3 index ({len(self.meta):,} docs) from {self.index_file}")

    def retrieve(self, query: str, top_k: int = 50) -> List[Dict]:
        if self.embeddings is None:
            raise FileNotFoundError(f"SPLADEv3 index not loaded: {self.index_file}")
        q = self._encode([query]).float()[0]
        scores = torch.mv(self.embeddings, q)
        k = min(int(top_k), int(scores.shape[0]))
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
                    "spladev3_score": float(s),
                    "score": float(s),
                }
            )
        return out
