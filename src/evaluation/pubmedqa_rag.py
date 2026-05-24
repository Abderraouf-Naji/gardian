"""PubMedQA-specific RAG modes and passage helpers for honest QA evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def is_pubmedqa_dataset(dataset: str) -> bool:
    ds = (dataset or "").strip().lower()
    return ds in ("pubmedqa", "pubmedqa_labeled", "pubmedqa_artificial")


def resolve_pubmedqa_rag_mode(cfg: Any, override: Optional[str] = None) -> str:
    """
    ``gold_context`` — standard PubMedQA: reader sees the labeled abstract(s) only.
    ``open_domain`` — retrieve from per-dataset indices (tests retrieval + reader).
    """
    if override is not None:
        mode = str(override).strip().lower()
    else:
        mode = str(getattr(getattr(cfg, "qa", None), "pubmedqa_rag_mode", "open_domain")).strip().lower()
    if mode not in ("gold_context", "open_domain"):
        raise ValueError(f"pubmedqa_rag_mode must be gold_context or open_domain, got {mode!r}")
    return mode


def build_gold_passage_lookup(
    questions: Sequence[Dict[str, Any]],
    corpus_paths: Sequence[str],
    scan_fn,
) -> Dict[str, str]:
    """Load passage text for all ``gold_passage_ids`` referenced in ``questions``."""
    want: set = set()
    for item in questions:
        for pid in item.get("gold_passage_ids") or []:
            if isinstance(pid, str) and pid:
                want.add(pid)
    if not want:
        return {}
    out: Dict[str, str] = {}
    remaining = set(want)
    for path in corpus_paths:
        if not remaining:
            break
        chunk = scan_fn(path, remaining)
        out.update(chunk)
        remaining -= set(chunk.keys())
    return out


def gold_context_passages(
    item: Dict[str, Any],
    corpus_lookup: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Passages for standard PubMedQA RAG (labeled abstract chunks)."""
    rows: List[Dict[str, Any]] = []
    for pid in item.get("gold_passage_ids") or []:
        if not isinstance(pid, str) or not pid:
            continue
        text = (corpus_lookup.get(pid) or "").strip()
        if not text:
            continue
        rows.append(
            {
                "id": pid,
                "text": text,
                "bm25_score": 1.0,
                "dense_score": 1.0,
                "hybrid_rrf_score": 1.0,
            }
        )
    return rows
