"""Citation metrics for RAG QA (PubMedQA only)."""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.pipeline.rag.parser import extract_citations


def citation_metrics_applicable(reader_task: str, dataset: str) -> bool:
    rt = (reader_task or "").strip().lower()
    ds = (dataset or "").strip().lower()
    if rt == "mcq" or ds in ("medmcqa", "medqa"):
        return False
    return ds in ("pubmedqa", "pubmedqa_labeled", "pubmedqa_artificial") or rt == "yesno"


def citation_precision(cited_idxs: List[str], passages: List[dict], gold_ids: List[str]) -> Optional[float]:
    if not cited_idxs:
        return None
    gold_set = set(gold_ids)
    correct = 0
    for idx_str in cited_idxs:
        try:
            idx = int(idx_str) - 1
            if 0 <= idx < len(passages) and passages[idx]["id"] in gold_set:
                correct += 1
        except ValueError:
            pass
    return correct / len(cited_idxs)


def citation_recall(cited_idxs: List[str], passages: List[dict], gold_ids: List[str]) -> float:
    gold_set = set(gold_ids)
    if not gold_set:
        return 0.0
    cited_gold = set()
    for idx_str in cited_idxs:
        try:
            idx = int(idx_str) - 1
            if 0 <= idx < len(passages):
                pid = passages[idx]["id"]
                if pid in gold_set:
                    cited_gold.add(pid)
        except ValueError:
            continue
    return len(cited_gold) / len(gold_set)


def unsupported_claim_rate(
    cited_idxs: List[str], passages: List[dict], gold_ids: List[str]
) -> Optional[float]:
    if not cited_idxs:
        # No [P#] markers to audit → 0 unsupported citations (not "missing metric").
        return 0.0
    gold_set = set(gold_ids)
    unsupported = 0
    for idx_str in cited_idxs:
        try:
            idx = int(idx_str) - 1
            if not (0 <= idx < len(passages)) or passages[idx]["id"] not in gold_set:
                unsupported += 1
        except ValueError:
            unsupported += 1
    return unsupported / len(cited_idxs)


def _unique_citation_idxs(cited_idxs: List[str], answer_text: str = "") -> List[str]:
    """Prefer unique passage tags; fall back to deduping the provided list."""
    if answer_text:
        return extract_citations(answer_text, unique=True)
    seen: set[str] = set()
    out: List[str] = []
    for idx in cited_idxs:
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def compute_citation_metrics(
    cited_idxs: List[str],
    passages: List[dict],
    gold_ids: List[str],
    *,
    reader_task: str,
    dataset: str,
    answer_text: str = "",
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not citation_metrics_applicable(reader_task, dataset):
        return None, None, None
    if not gold_ids:
        return None, None, None
    unique = _unique_citation_idxs(cited_idxs, answer_text)
    if not unique:
        return None, 0.0, 0.0
    return (
        citation_precision(unique, passages, gold_ids),
        citation_recall(unique, passages, gold_ids),
        unsupported_claim_rate(unique, passages, gold_ids),
    )
