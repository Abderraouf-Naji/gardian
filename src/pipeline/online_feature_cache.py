"""Fast online feature-cache helpers for GARDIAN inference.

The rank-data generator is allowed to do expensive per-pair work offline.  The
online path is different: passage-side features should be looked up from
artifacts that are already loaded into memory.

This module provides the two expensive passage-side caches needed by
``scripts/08_ask_gardian.py``:

* PubMedBERT passage embeddings are reconstructed from the already-built FAISS
  index instead of re-encoding passage text for every user question.
* Passage KG entities are cached by passage id after the first entity-linking
  call in a long-running process.

Query-side work still happens per request: encode the user question once and
link query entities once.  That is cheap compared with re-encoding 100 passages.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from src.kg.linker import EntityLinker


class OnlinePassageFeatureCache:
    """Online cache for passage embeddings and linked KG entities.

    Parameters
    ----------
    embedding_index_path:
        FAISS index containing the same PubMedBERT passage embeddings used by
        the dense feature branch.
    embedding_meta_path:
        JSONL metadata in the same row order as the FAISS index.
    linker:
        Entity linker backed by ``data/kg/default/umls_kg_lex.pkl``.
    encoder:
        Optional fallback encoder. Used only when a candidate id is missing
        from the FAISS meta or the FAISS index cannot reconstruct vectors.
    max_entity_cache:
        Bounded in-memory passage-id -> CUI-list cache. Set <=0 for unbounded.
    """

    def __init__(
        self,
        *,
        embedding_index_path: str,
        embedding_meta_path: str,
        linker: EntityLinker,
        encoder=None,
        max_entity_cache: int = 200_000,
    ) -> None:
        self.linker = linker
        self.encoder = encoder
        self._pid_to_row: Dict[str, int] = {}
        self._faiss_index = None
        self._entity_cache: OrderedDict[str, List[str]] = OrderedDict()
        self._entity_cache_max = None if int(max_entity_cache) <= 0 else int(max_entity_cache)

        idx_path = Path(embedding_index_path)
        meta_path = Path(embedding_meta_path)
        if idx_path.is_file() and meta_path.is_file():
            try:
                import faiss

                self._faiss_index = faiss.read_index(str(idx_path))
                with meta_path.open("r", encoding="utf-8") as f:
                    for row, line in enumerate(f):
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        pid = rec.get("id")
                        if pid is not None:
                            self._pid_to_row[str(pid)] = row
                logger.info(
                    "Online feature cache: loaded FAISS embedding map "
                    f"({len(self._pid_to_row):,} passage ids)"
                )
            except Exception as exc:
                logger.warning(f"Could not initialize online FAISS embedding cache: {exc}")
                self._faiss_index = None
                self._pid_to_row = {}
        else:
            logger.warning(
                "Online feature cache disabled for embeddings: missing "
                f"{idx_path} or {meta_path}"
            )

    def get_passage_entities(self, candidate: Dict) -> List[str]:
        """Return cached linked KG entities for one candidate passage."""
        pid = str(candidate.get("id", ""))
        if pid and pid in self._entity_cache:
            if self._entity_cache_max is not None:
                self._entity_cache.move_to_end(pid)
            return self._entity_cache[pid]

        entities = self.linker.link(candidate.get("text", "") or "")
        if pid:
            self._entity_cache[pid] = entities
            if self._entity_cache_max is not None:
                while len(self._entity_cache) > self._entity_cache_max:
                    self._entity_cache.popitem(last=False)
        return entities

    def _embedding_from_faiss(self, pid: str) -> Optional[np.ndarray]:
        if self._faiss_index is None:
            return None
        row = self._pid_to_row.get(str(pid))
        if row is None:
            return None
        try:
            vec = self._faiss_index.reconstruct(int(row))
        except Exception:
            return None
        return np.asarray(vec, dtype=np.float32)

    def get_passage_embeddings(self, candidates: List[Dict]) -> np.ndarray:
        """Return PubMedBERT embeddings for candidates.

        Most candidates are served by FAISS ``reconstruct``.  Any misses are
        batch-encoded once with the fallback encoder so online inference remains
        correct even if artifacts are incomplete.
        """
        out: List[Optional[np.ndarray]] = []
        miss_positions: List[int] = []
        miss_texts: List[str] = []

        for i, cand in enumerate(candidates):
            vec = self._embedding_from_faiss(str(cand.get("id", "")))
            out.append(vec)
            if vec is None:
                miss_positions.append(i)
                miss_texts.append(cand.get("text", "") or "")

        if miss_positions:
            if self.encoder is None:
                raise RuntimeError(
                    "Online passage embedding cache had misses and no fallback encoder is configured."
                )
            logger.warning(
                "Online passage embedding cache misses: "
                f"{len(miss_positions)}/{len(candidates)}; fallback re-encoding these passages."
            )
            embs = self.encoder.encode(
                miss_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            embs = np.asarray(embs, dtype=np.float32)
            for pos, emb in zip(miss_positions, embs):
                out[pos] = emb

        return np.asarray(out, dtype=np.float32)
