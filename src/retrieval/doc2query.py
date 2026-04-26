"""Doc2Query retriever built on top of BM25 index."""

from typing import Dict, List

import torch
from loguru import logger
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .bm25 import BM25Retriever


class Doc2QueryRetriever:
    """BM25 retrieval with query expansion using a Doc2Query T5 model."""

    def __init__(
        self,
        bm25: BM25Retriever,
        model_name: str = "doc2query/msmarco-t5-base-v1",
        device: str = "cuda",
        num_expansions: int = 4,
        max_new_tokens: int = 24,
    ):
        self.bm25 = bm25
        self.model_name = model_name
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.num_expansions = int(num_expansions)
        self.max_new_tokens = int(max_new_tokens)

        logger.info(f"Loading Doc2Query model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _expand_query(self, query: str) -> str:
        if not query.strip() or self.num_expansions <= 0:
            return query
        with torch.inference_mode():
            inputs = self.tokenizer(
                [query],
                return_tensors="pt",
                truncation=True,
                max_length=128,
            ).to(self.device)
            outs = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=max(2, self.num_expansions),
                num_return_sequences=self.num_expansions,
                max_new_tokens=self.max_new_tokens,
                early_stopping=True,
            )
            expansions = self.tokenizer.batch_decode(outs, skip_special_tokens=True)
        expansions = [e.strip() for e in expansions if e.strip()]
        if not expansions:
            return query
        return query + " " + " ".join(expansions)

    def retrieve(self, query: str, top_k: int = 200) -> List[Dict]:
        expanded = self._expand_query(query)
        hits = self.bm25.retrieve(expanded, top_k=top_k)
        out: List[Dict] = []
        for h in hits:
            score = float(h.get("score", 0.0))
            out.append(
                {
                    "id": h.get("id"),
                    "text": h.get("text", ""),
                    "doc2query_score": score,
                    "score": score,
                }
            )
        return out

