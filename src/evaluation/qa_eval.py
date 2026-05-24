"""End-to-end QA evaluation helpers for controlled RAG comparisons."""

import pickle
import re
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from src.common.question_types import normalize_question_type, qtype_onehot
from src.features.dense_feat import compute_dense_features_with_score
from src.features.sparse import compute_sparse_features
from src.pipeline.rank_dense_features import (
    dense_embedding_pair_for_candidates,
)
from src.common.query_emb_cache import load_query_emb_cache
from src.pipeline.gardian_adaptive import (
    adaptive_channel_budgets,
    controller_weights_from_lists,
    retrieve_adaptive_candidates_live,
    subset_rank_records_adaptive,
)
from src.evaluation.pubmedqa_rag import (
    gold_context_passages,
    is_pubmedqa_dataset,
    resolve_pubmedqa_rag_mode,
)
from src.pipeline.rag.metrics import compute_citation_metrics
from src.pipeline.rag.parser import extract_citations
from src.pipeline.rag.reader_types import normalize_system_name
from src.pipeline.rag_reader import (
    format_reader_context,
    retrieve_hybrid_candidates,
    run_reader_llm_only_block,
    run_reader_rag_block,
)


def _gardian_text_only(cfg: Any = None, model: Optional[torch.nn.Module] = None) -> bool:
    if model is not None:
        return bool(getattr(model, "text_only", True))
    if cfg is not None:
        return bool(getattr(getattr(cfg, "model", None), "text_only", True))
    return True


def format_context(passages: List[Dict], top_k: int = 5) -> str:
    """Backward-compatible alias for :func:`format_reader_context`."""
    return format_reader_context(passages, top_k=top_k, max_chars_per_passage=600)


def _looks_like_mcq(item: Dict[str, Any]) -> bool:
    """True when the row carries explicit answer options."""
    opts = item.get("options")
    if isinstance(opts, dict) and any(str(v).strip() for v in opts.values()):
        return True
    if isinstance(opts, list) and any(str(v).strip() for v in opts):
        return True
    if str(item.get("answer_letter") or "").strip():
        return True
    return str(item.get("dataset") or "").strip().lower() in {"medmcqa", "medqa"}


def _format_options_block(options: Any) -> List[str]:
    """Return stable A./B./C. option lines for dict or list option payloads."""
    if isinstance(options, dict):
        keys = sorted(options.keys(), key=lambda x: str(x).strip().upper())
        out = []
        for key in keys:
            text = options.get(key)
            if text is not None and str(text).strip():
                out.append(f"  {str(key).strip().upper()}. {str(text).strip()}")
        return out
    if isinstance(options, list):
        out = []
        for idx, text in enumerate(options):
            if text is not None and str(text).strip():
                out.append(f"  {chr(ord('A') + idx)}. {str(text).strip()}")
        return out
    return []


def format_qa_question_for_reader(item: Dict[str, Any]) -> str:
    """
    Build the question string shown to the reader.

    MedMCQA rows store the stem in ``question`` and options separately; without
    expanding A/B/C/D the model often answers ``I don't know``.
    """
    q = (item.get("question") or "").strip()
    if _looks_like_mcq(item):
        option_lines = _format_options_block(item.get("options"))
        lines = ["Question type: multiple choice", q]
        if option_lines:
            lines.extend(["", "Options:", *option_lines])
        lines.extend(["", "Choose one option letter and copy its text in the final answer line."])
        return "\n".join(lines)
    qt = str(item.get("question_type") or "").strip().lower()
    if qt == "yesno":
        return (
            "Question type: yes/no/maybe\n"
            f"{q}\n\n"
            "Answer with a final line exactly: Answer: yes, Answer: no, or Answer: maybe."
        )
    return q


def reader_task_for_item(item: Dict[str, Any]) -> str:
    """``yesno`` | ``mcq`` | ``open`` — selects reader prompt templates."""
    if _looks_like_mcq(item):
        return "mcq"
    qt = str(item.get("question_type") or "").strip().lower()
    if qt == "yesno":
        return "yesno"
    return "open"


def _passage_line_text(r: Dict[str, Any], lookup: Dict[str, str]) -> str:
    t = r.get("text")
    if isinstance(t, str) and t.strip():
        return t
    pid = r.get("pid")
    if isinstance(pid, str) and pid:
        return str(lookup.get(pid) or "")
    return ""


def _resolve_sparse_score(rec: Dict[str, Any]) -> float:
    """Sparse channel score for BM25 or SPLADE++ rank rows."""
    if "bm25_score" in rec:
        return float(rec.get("bm25_score", 0.0))
    if "spladepp_score" in rec:
        return float(rec.get("spladepp_score", 0.0))
    sparse_feats = rec.get("sparse_feats") or [0.0]
    return float(sparse_feats[0] if sparse_feats else 0.0)


def _resolve_dense_score(rec: Dict[str, Any]) -> float:
    """Dense channel score for FAISS or MedCPT rank rows."""
    if "dense_score" in rec:
        return float(rec.get("dense_score", 0.0))
    if "medcpt_score" in rec:
        return float(rec.get("medcpt_score", 0.0))
    dense_feats = rec.get("dense_feats") or [0.0]
    return float(dense_feats[0] if dense_feats else 0.0)


def _rrf_fused_candidates(
    candidates: List[Dict[str, Any]],
    text_lookup: Dict[str, str],
    *,
    rrf_k: int = 60,
) -> List[Tuple[float, Dict[str, Any]]]:
    """Scale-free hybrid ranking from sparse and dense ranks."""
    sparse_scores = [_resolve_sparse_score(r) for r in candidates]
    dense_scores = [_resolve_dense_score(r) for r in candidates]
    sparse_order = np.argsort(np.asarray(sparse_scores, dtype=np.float64))[::-1]
    dense_order = np.argsort(np.asarray(dense_scores, dtype=np.float64))[::-1]
    sparse_rank = {int(idx): rank + 1 for rank, idx in enumerate(sparse_order)}
    dense_rank = {int(idx): rank + 1 for rank, idx in enumerate(dense_order)}

    out: List[Tuple[float, Dict[str, Any]]] = []
    for idx, r in enumerate(candidates):
        score = (1.0 / (rrf_k + sparse_rank[idx])) + (1.0 / (rrf_k + dense_rank[idx]))
        out.append(
            (
                float(score),
                {
                    "id": r["pid"],
                    "text": _passage_line_text(r, text_lookup),
                    "bm25_score": float(r.get("bm25_score", 0.0)),
                    "spladepp_score": float(r.get("spladepp_score", 0.0)),
                    "dense_score": _resolve_dense_score(r),
                    "medcpt_score": float(r.get("medcpt_score", 0.0)),
                    "hybrid_rrf_score": float(score),
                },
            )
        )
    # Match ``DualHybridRetriever`` tie-break: RRF, then sparse signal, then dense signal.
    out.sort(
        key=lambda t: (
            t[0],
            float(t[1].get("bm25_score", 0.0)) + float(t[1].get("spladepp_score", 0.0)),
            float(t[1].get("dense_score", 0.0)) + float(t[1].get("medcpt_score", 0.0)),
        ),
        reverse=True,
    )
    return out


# Lazy encoder only when a qid is missing from the precomputed pickle (e.g. eval-only splits).
_ST_ENCODER: Dict[str, Any] = {"key": None, "model": None}
_QUERY_EMB_FALLBACK_WARNED = False


def _encode_query_emb_st_fallback(first: Dict[str, Any], cfg: Any, device: str) -> List[float]:
    """Encode ``question`` with ``cfg.encoder.model_name`` (matches rank-data pipeline)."""
    global _ST_ENCODER, _QUERY_EMB_FALLBACK_WARNED
    question = (first.get("question") or "").strip()
    if not question:
        raise ValueError("Cannot encode query_emb: missing question on rank row.")
    if not _QUERY_EMB_FALLBACK_WARNED:
        logger.warning(
            "At least one qid is missing from query_emb cache; encoding with "
            f"{cfg.encoder.model_name!r} on-the-fly. To avoid this, build a pickle that "
            "includes all splits you evaluate (scripts/12_precompute_query_cache.py on the "
            "corresponding rank JSONL, then merge or pass the wider file via --query-emb-cache)."
        )
        _QUERY_EMB_FALLBACK_WARNED = True
    key = str(cfg.encoder.model_name)
    if _ST_ENCODER["key"] != key or _ST_ENCODER["model"] is None:
        from sentence_transformers import SentenceTransformer

        _ST_ENCODER["key"] = key
        _ST_ENCODER["model"] = SentenceTransformer(key, device=device)
    vec = _ST_ENCODER["model"].encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )[0]
    return [float(x) for x in np.asarray(vec, dtype=np.float32)]


def _normalize_query_emb_cache_paths(
    path: Optional[Union[str, Sequence[str]]],
) -> List[str]:
    if path is None:
        return []
    if isinstance(path, (list, tuple)):
        return [str(p).strip() for p in path if str(p).strip()]
    return [p.strip() for p in str(path).split(",") if p.strip()]


def _load_query_emb_cache_file(path: str) -> Dict[str, List[float]]:
    """Load a single ``qid -> query_emb`` pickle."""
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        with p.open("rb") as f:
            data = pickle.load(f)
    except Exception as e:
        logger.warning(f"Could not load query_emb cache {path}: {e}")
        return {}
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items()}
    logger.warning(f"Query_emb cache {path} is not a dict; ignoring.")
    return {}


def _load_query_emb_cache(paths: Optional[Union[str, Sequence[str]]]) -> Dict[str, List[float]]:
    """
    Load and merge ``qid -> query_emb`` pickles (``scripts/12_precompute_query_cache.py``).

    Multiple paths: comma-separated string, or a sequence of paths. Later entries override
    earlier ones on duplicate ``qid`` keys (typically disjoint train vs eval caches).
    """
    merged: Dict[str, List[float]] = {}
    for p in _normalize_query_emb_cache_paths(paths):
        chunk = _load_query_emb_cache_file(p)
        if chunk:
            merged.update(chunk)
    return merged


def _resolve_query_emb_vector(
    candidates: List[Dict[str, Any]],
    q_emb_by_qid: Dict[str, List[float]],
    cfg: Any,
    device: str,
    *,
    allow_encode_on_cache_miss: bool = True,
) -> List[float]:
    """
    Prefer inline ``query_emb`` on rank rows, else ``q_emb_by_qid[qid]`` from pickle cache.

    If the qid is still missing (typical when the cache was built from train-only rank JSONL
    but you evaluate on dev/eval/test), optionally encode ``question`` with ``cfg.encoder``.
    """
    first = candidates[0]
    qe = first.get("query_emb")
    if isinstance(qe, list) and len(qe) > 0:
        return [float(x) for x in qe]
    qid = str(first.get("qid", ""))
    if qid in q_emb_by_qid:
        return [float(x) for x in q_emb_by_qid[qid]]
    if allow_encode_on_cache_miss:
        return _encode_query_emb_st_fallback(first, cfg, device)
    raise KeyError(
        f"No query_emb for qid={qid!r}: rank JSONL rows omit query_emb and qid is missing "
        f"from query_emb cache. Run scripts/12_precompute_query_cache.py on the rank file(s) "
        f"or pass a merged cache via --query-emb-cache."
    )


def _bootstrap_ci(values: List[float], n_bootstrap: int = 2000, seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap mean and 95% CI; entries ``None`` are skipped (e.g. N/A metrics)."""
    values = [v for v in values if v is not None]
    if not values:
        return 0.0, 0.0, 0.0
    x = np.asarray(values, dtype=np.float64)
    if x.size == 1:
        m = float(x[0])
        return m, m, m
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    n = x.size
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[i] = float(x[idx].mean())
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _bootstrap_ci_optional(
    values: List[Optional[float]],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Optional[Tuple[float, float, float]]:
    """Like :func:`_bootstrap_ci` but returns ``None`` when every value is N/A."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    return _bootstrap_ci(present, n_bootstrap, seed)


def _pack_ci(triple: Optional[Tuple[float, float, float]]) -> Optional[Dict[str, float]]:
    """Expose bootstrap stats as ``{mean, ci95_low, ci95_high}`` plus legacy list."""
    if triple is None:
        return None
    mean, lo, hi = triple
    return {
        "mean": float(mean),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "ci95": [float(mean), float(lo), float(hi)],
    }


def _citation_metrics_applicable(reader_task: str, dataset: str) -> bool:
    """
    Citation precision/recall are defined for PubMedQA-style evidence passages.

    MedMCQA uses ``mcq_*_explanation`` gold ids that usually are not the passages
  the model cites in ``[P#]`` slots (retrieved explanations from other items), so
    citation scores are misleading — use ``answer_accuracy`` only.
    """
    rt = (reader_task or "open").strip().lower()
    ds = (dataset or "").strip().lower()
    if rt == "mcq" or ds in ("medmcqa", "medqa"):
        return False
    return ds in ("pubmedqa", "pubmedqa_labeled", "pubmedqa_artificial") or rt == "yesno"


def _gold_evidence_in_reader_context(
    top_passages: List[Dict[str, Any]],
    gold_ids: List[str],
) -> Optional[float]:
    """
    Fraction of ``gold_passage_ids`` whose text appears in the reader's top-k.

    Diagnostic for RQ4: links retrieval rerank quality to citation/accuracy outcomes.
    Returns ``None`` when there is no gold set (e.g. some MedMCQA rows).
    """
    gold_set = {g for g in gold_ids if g}
    if not gold_set:
        return None
    shown = {str(p.get("id", "")) for p in top_passages if p.get("id")}
    return len(gold_set & shown) / len(gold_set)


def _compute_citation_metrics(
    cited_idxs: List[str],
    passages: List[Dict],
    gold_ids: List[str],
    *,
    reader_task: str,
    dataset: str,
    answer_text: str = "",
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    return compute_citation_metrics(
        cited_idxs,
        passages,
        gold_ids,
        reader_task=reader_task,
        dataset=dataset,
        answer_text=answer_text,
    )


def _aggregate_system_metrics(
    rows: List[Dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    """Per-system aggregate with explicit bootstrap CI fields."""
    acc_ci = _bootstrap_ci([r["accuracy"] for r in rows], bootstrap_samples, bootstrap_seed)
    cit_p_ci = _bootstrap_ci_optional(
        [r.get("citation_precision") for r in rows], bootstrap_samples, bootstrap_seed
    )
    cit_r_ci = _bootstrap_ci_optional(
        [r.get("citation_recall") for r in rows], bootstrap_samples, bootstrap_seed
    )
    uns_ci = _bootstrap_ci_optional(
        [r.get("unsupported_claim_rate") for r in rows], bootstrap_samples, bootstrap_seed
    )
    sup_vals = [
        (1.0 - float(r["unsupported_claim_rate"]))
        for r in rows
        if r.get("unsupported_claim_rate") is not None
    ]
    sup_ci = _bootstrap_ci_optional(sup_vals, bootstrap_samples, bootstrap_seed)
    gold_ctx_ci = _bootstrap_ci_optional(
        [r.get("gold_evidence_in_context_rate") for r in rows],
        bootstrap_samples,
        bootstrap_seed,
    )
    out: Dict[str, Any] = {
        "n_questions": len(rows),
        "answer_accuracy": list(acc_ci),
        "answer_accuracy_ci": _pack_ci(acc_ci),
        "citation_precision": list(cit_p_ci) if cit_p_ci else None,
        "citation_precision_ci": _pack_ci(cit_p_ci),
        "citation_recall": list(cit_r_ci) if cit_r_ci else None,
        "citation_recall_ci": _pack_ci(cit_r_ci),
        "supported_citation_rate": list(sup_ci) if sup_ci else None,
        "supported_citation_rate_ci": _pack_ci(sup_ci),
        "unsupported_claim_rate": list(uns_ci) if uns_ci else None,
        "unsupported_claim_rate_ci": _pack_ci(uns_ci),
        "gold_evidence_in_context_rate": list(gold_ctx_ci) if gold_ctx_ci else None,
        "gold_evidence_in_context_rate_ci": _pack_ci(gold_ctx_ci),
    }
    return out


def _citation_recall(cited_idxs: List[str], passages: List[Dict], gold_ids: List[str]) -> float:
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


def _unsupported_claim_rate(cited_idxs: List[str], passages: List[Dict], gold_ids: List[str]) -> float:
    """
    Fraction of **citation markers** that point to a non-gold passage (or out-of-range).

    When the model emits no ``[P#]`` tags, this is ``0.0`` (no attributed claims to audit),
    not ``1.0`` — the latter made every no-citation answer look maximally "unsupported".
    """
    if not cited_idxs:
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
    return unsupported / max(1, len(cited_idxs))


def _enrich_live_candidates_for_gardian(
    *,
    question: str,
    candidates: List[Dict[str, Any]],
    qtype: str,
    retriever_name: str,
    encoder,
    faiss_lookup=None,
    medcpt_encoder=None,
) -> Tuple[List[float], List[float]]:
    """Attach sparse/dense branch features; return (qtype_onehot, controller query_emb)."""
    qtype_oh = qtype_onehot(normalize_question_type(qtype))
    q_emb = encoder.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )[0]
    q_dense, p_embs = dense_embedding_pair_for_candidates(
        retriever_type=retriever_name,
        question=question,
        candidates=candidates,
        pubmedbert_encoder=encoder,
        faiss_lookup=faiss_lookup,
        medcpt_encoder=medcpt_encoder,
    )
    dense_scores = [
        float(c.get("medcpt_score", c.get("dense_score", c.get("score", 0.0))))
        for c in candidates
    ]
    dense_mean = float(np.mean(dense_scores))
    dense_std = float(np.std(dense_scores) + 1e-8)
    for i, cand in enumerate(candidates):
        cand["sparse_feats"] = compute_sparse_features(
            query=question,
            passage=cand.get("text", ""),
            bm25_score=float(cand.get("bm25_score", cand.get("spladepp_score", 0.0))),
            idf_table=None,
        ).tolist()
        cand["dense_feats"] = compute_dense_features_with_score(
            q_emb=q_dense,
            p_emb=p_embs[i],
            dense_score=dense_scores[i],
            score_mean=dense_mean,
            score_std=dense_std,
        ).tolist()
    return qtype_oh, q_emb.tolist()


def _live_passage_sort_key(pair: Tuple[float, Dict[str, Any]]) -> Tuple[float, float, float]:
    score, d = pair
    sp = float(d.get("bm25_score", 0.0)) + float(d.get("spladepp_score", 0.0))
    den = float(d.get("dense_score", 0.0)) + float(d.get("medcpt_score", 0.0))
    return (score, sp, den)


def _candidate_id(c: Dict[str, Any]) -> str:
    return str(c.get("id") or c.get("pid") or "")


def _live_sparse_score(c: Dict[str, Any]) -> float:
    return float(c.get("bm25_score", c.get("spladepp_score", 0.0)))


def _live_dense_score(c: Dict[str, Any]) -> float:
    return float(c.get("dense_score", c.get("medcpt_score", c.get("score", 0.0))))


def _reader_passage_row(
    c: Dict[str, Any],
    *,
    text_lookup: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    pid = _candidate_id(c)
    text = c.get("text")
    if (not isinstance(text, str) or not text.strip()) and text_lookup is not None:
        text = text_lookup.get(pid, "")
    return {
        "id": pid,
        "text": str(text or ""),
        "bm25_score": float(c.get("bm25_score", 0.0)),
        "spladepp_score": float(c.get("spladepp_score", 0.0)),
        "dense_score": float(c.get("dense_score", c.get("medcpt_score", 0.0))),
        "medcpt_score": float(c.get("medcpt_score", 0.0)),
        "hybrid_rrf_score": float(c.get("hybrid_rrf_score", 0.0)),
    }


def _hybrid_balanced_top_passages(
    pool: List[Dict[str, Any]],
    k: int,
    *,
    text_lookup: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Top-k for hybrid reader: half from sparse ranking, half from dense (deduped)."""
    if k <= 0 or not pool:
        return []
    k_sparse = (k + 1) // 2
    k_dense = k - k_sparse
    by_sparse = sorted(pool, key=_live_sparse_score, reverse=True)
    by_dense = sorted(pool, key=_live_dense_score, reverse=True)
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []

    for c in by_sparse:
        pid = _candidate_id(c)
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append(_reader_passage_row(c, text_lookup=text_lookup))
        if len(out) >= k_sparse:
            break

    dense_added = 0
    for c in by_dense:
        pid = _candidate_id(c)
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append(_reader_passage_row(c, text_lookup=text_lookup))
        dense_added += 1
        if dense_added >= k_dense:
            break

    if len(out) < k:
        by_rrf = sorted(
            pool,
            key=lambda c: float(c.get("hybrid_rrf_score", 0.0)),
            reverse=True,
        )
        for c in by_rrf:
            pid = _candidate_id(c)
            if not pid or pid in seen:
                continue
            seen.add(pid)
            out.append(_reader_passage_row(c, text_lookup=text_lookup))
            if len(out) >= k:
                break
    return out[:k]


def _resolve_reader_top_k(cfg: Any, reader_task: str) -> int:
    """Effective passages fed to the reader (task-specific caps)."""
    k = int(getattr(cfg.qa, "top_k_passages", 10) or 10)
    rt = (reader_task or "open").strip().lower()
    if rt == "yesno":
        cap = int(getattr(cfg.qa, "yesno_top_k_passages", 0) or 0)
        if cap > 0:
            k = min(k, cap)
    elif rt == "mcq":
        cap = int(getattr(cfg.qa, "mcq_top_k_passages", 0) or 0)
        if cap > 0:
            k = min(k, cap)
    return max(1, k)


def _hybrid_use_balanced_top_k(cfg: Any) -> bool:
    """RRF top-k (retrieval-aligned) unless explicitly using balanced hybrid pool."""
    if bool(getattr(cfg.qa, "rq4_align_retrieval_top_k", True)):
        return False
    return bool(getattr(cfg.qa, "hybrid_balanced_top_k", False))


def _select_reader_passages(
    system: str,
    *,
    candidates: List[Dict[str, Any]],
    scored: List[Tuple[float, Dict[str, Any]]],
    k: int,
    cfg: Any,
    gardian_ranked: Optional[List[Dict[str, Any]]] = None,
    text_lookup: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Passages shown to the reader LLM.

    - **gardian**: top-$k$ by ``gardian_score`` (matches offline reranked Hit@$k$).
    - **hybrid**: top-$k$ by RRF when ``rq4_align_retrieval_top_k`` (default), else
      balanced sparse/dense slots if ``hybrid_balanced_top_k``.
    - **sparse/dense**: top-$k$ by channel score.
    """
    if k <= 0:
        return []
    if system == "gardian":
        pool = gardian_ranked or []
        if not pool:
            return []
        ordered = sorted(
            pool, key=lambda c: float(c.get("gardian_score", 0.0)), reverse=True
        )
        return [
            _reader_passage_row(c, text_lookup=text_lookup) for c in ordered[:k]
        ]
    if system == "hybrid" and _hybrid_use_balanced_top_k(cfg):
        return _hybrid_balanced_top_passages(
            candidates, k, text_lookup=text_lookup
        )
    return [p for _, p in scored[:k]]


def evaluate_qa_live(
    questions: List[Dict[str, Any]],
    *,
    systems: List[str],
    cfg: Any,
    device: str,
    retriever: Optional[Any],
    gardian_model: Optional[torch.nn.Module],
    tokenizer,
    reader_model,
    encoder,
    retriever_name: str,
    faiss_lookup=None,
    medcpt_encoder=None,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 42,
    top_candidates: Optional[int] = None,
    gardian_adaptive_retrieval: bool = False,
    pubmedqa_rag_mode: Optional[str] = None,
    gold_passage_lookup: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    """
    QA eval with live hybrid retrieval → GARDIAN rerank → RAG reader.

    Retrieval indices are per-dataset when configured in ``06_end_to_end_qa.py``
    (``qa.use_per_dataset_indices``); each benchmark's own corpus unless ``--unified-indices``.

    When ``gardian_adaptive_retrieval`` is True, retrieval uses the controller budget
    (α_sparse, α_dense) for **all** RAG systems on that question (fair shared pool).

    ``pubmedqa_rag_mode=gold_context`` (PubMedQA only): reader sees labeled abstract(s)
    from ``gold_passage_lookup`` — the standard PubMedQA setting, not open-domain retrieval.
    """
    if "doc2query" in systems:
        logger.warning("doc2query is not supported with live retrieval; skipping that system.")
        systems = [s for s in systems if s != "doc2query"]
    systems = [normalize_system_name(s) for s in systems]

    pool_k = int(top_candidates or getattr(cfg.retrieval, "candidate_pool_size", 100))
    reader_max_input = int(cfg.qa.get("reader_max_input_length", 2048) or 2048)
    passage_max_chars = int(cfg.qa.get("max_chars_per_passage", 600) or 600)
    yesno_compact = bool(getattr(cfg.qa, "rag_yesno_compact", False))
    pmqa_mode = resolve_pubmedqa_rag_mode(cfg, pubmedqa_rag_mode)
    gold_lookup = gold_passage_lookup or {}
    needs_rank = any(s in systems for s in ("sparse", "dense", "hybrid", "gardian"))
    if needs_rank:
        pool_mode = (
            "adaptive_live"
            if (
                gardian_adaptive_retrieval
                and gardian_model is not None
                and not bool(getattr(cfg.qa, "rq4_full_union_pool", False))
            )
            else "full_union_rrf"
        )
        logger.info(
            f"QA reader top-k={int(getattr(cfg.qa, 'top_k_passages', 10) or 10)} | "
            f"hybrid_balanced={_hybrid_use_balanced_top_k(cfg)} | "
            f"rq4_align_retrieval={bool(getattr(cfg.qa, 'rq4_align_retrieval_top_k', True))} | "
            f"pool={pool_mode}"
        )

    per_system_rows: Dict[str, List[Dict[str, Any]]] = {s: [] for s in systems}
    dataset_names = sorted({str(q.get("dataset") or "dataset") for q in questions})
    progress_desc = (
        f"QA live {dataset_names[0]}"
        if len(dataset_names) == 1
        else f"QA live {len(dataset_names)} datasets"
    )

    for item in tqdm(questions, desc=progress_desc, unit="question"):
        qid = item.get("id")
        gold_answer = item.get("answer", "")
        gold_ids = item.get("gold_passage_ids", [])
        dataset = item.get("dataset", "")
        q_reader = format_qa_question_for_reader(item)
        r_task = reader_task_for_item(item)
        qtype = str(item.get("question_type") or "other")

        if "llm_only" in systems:
            answer = run_reader_llm_only_block(
                question=q_reader,
                tokenizer=tokenizer,
                reader_model=reader_model,
                device=device,
                max_new_tokens=int(cfg.qa.max_new_tokens),
                max_input_length=reader_max_input,
                question_type=item.get("question_type"),
                reader_task=r_task,
            )
            per_system_rows["llm_only"].append(
                {
                    "qid": qid,
                    "system": "llm_only",
                    "answer": answer,
                    "accuracy": _check_accuracy(
                        answer,
                        gold_answer,
                        dataset,
                        gold_letter=item.get("answer_letter"),
                    ),
                    "citation_precision": None,
                    "citation_recall": None,
                    "unsupported_claim_rate": None,
                }
            )

        if not needs_rank:
            continue

        question_text = (item.get("question") or "").strip()
        use_gold_context = is_pubmedqa_dataset(dataset) and pmqa_mode == "gold_context"
        retrieval_meta: Optional[Dict[str, Any]] = None
        if use_gold_context:
            candidates = gold_context_passages(item, gold_lookup)
            if not candidates:
                logger.warning(f"No gold-context passages for qid={qid!r}; skipping RAG systems")
                continue
            retrieval_meta = {"mode": "gold_context", "pool_size": len(candidates)}
        elif retriever is not None:
            use_adaptive_pool = bool(
                gardian_adaptive_retrieval
                and gardian_model is not None
                and not bool(getattr(cfg.qa, "rq4_full_union_pool", False))
            )
            if use_adaptive_pool:
                qtype_oh = qtype_onehot(normalize_question_type(qtype))
                q_emb = encoder.encode(
                    [question_text],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )[0].tolist()
                candidates, alpha_ret, beta_ret = retrieve_adaptive_candidates_live(
                    question_text,
                    retriever,
                    gardian_model,
                    query_emb=q_emb,
                    qtype_onehot=qtype_oh,
                    cfg=cfg,
                    device=device,
                )
                if not candidates:
                    continue
                k_sparse, k_dense = adaptive_channel_budgets(alpha_ret, beta_ret, cfg)
                retrieval_meta = {
                    "pool_mode": "adaptive_live",
                    "alpha_sparse": float(alpha_ret),
                    "alpha_dense": float(beta_ret),
                    "k_sparse": int(k_sparse),
                    "k_dense": int(k_dense),
                    "pool_size": len(candidates),
                    "cap_sparse": int(
                        getattr(cfg.retrieval, "top_k_bm25", None)
                        or getattr(cfg.retrieval, "top_k_spladepp", None)
                        or 50
                    ),
                    "cap_dense": int(
                        getattr(cfg.retrieval, "top_k_faiss", None)
                        or getattr(cfg.retrieval, "top_k_medcpt", None)
                        or 50
                    ),
                    "adaptive_channel_budget": str(
                        getattr(cfg.retrieval, "adaptive_channel_budget", "full_caps")
                    ),
                }
            else:
                candidates = retrieve_hybrid_candidates(
                    question_text,
                    retriever,
                    top_k=pool_k,
                )
                if not candidates:
                    continue
                retrieval_meta = {
                    "pool_mode": "full_union_rrf",
                    "pool_size": len(candidates),
                    "cap_sparse": int(
                        getattr(cfg.retrieval, "top_k_bm25", None)
                        or getattr(cfg.retrieval, "top_k_spladepp", None)
                        or 50
                    ),
                    "cap_dense": int(
                        getattr(cfg.retrieval, "top_k_faiss", None)
                        or getattr(cfg.retrieval, "top_k_medcpt", None)
                        or 50
                    ),
                    "adaptive_channel_budget": "fixed_union_rrf",
                }
        else:
            logger.warning(f"No retriever for qid={qid!r}; skipping RAG systems")
            continue

        gardian_ranked: List[Dict[str, Any]] = []
        alpha_triplet = None
        if "gardian" in systems and gardian_model is not None:
            qtype_oh, q_emb = _enrich_live_candidates_for_gardian(
                question=question_text,
                candidates=candidates,
                qtype=qtype,
                retriever_name=retriever_name,
                encoder=encoder,
                faiss_lookup=faiss_lookup,
                medcpt_encoder=medcpt_encoder,
            )
            gardian_ranked = gardian_model.rerank(
                candidates=candidates,
                query_features={
                    "query_emb": q_emb,
                    "qtype_onehot": qtype_oh,
                },
                device=device,
            )
            if gardian_ranked:
                w = gardian_ranked[0]
                alpha_triplet = [
                    float(w.get("sparse_alfa", 0.0)),
                    float(w.get("dense_alfa", 0.0)),
                ]
                if not _gardian_text_only(cfg, gardian_model):
                    alpha_triplet.append(float(w.get("kg_alfa", 0.0)))

        for system in systems:
            if system == "llm_only":
                continue
            if system == "gardian":
                if not gardian_ranked:
                    continue
                scored = [
                    (float(c.get("gardian_score", 0.0)), c) for c in gardian_ranked
                ]
            elif system == "hybrid":
                scored = [
                    (
                        float(c.get("hybrid_rrf_score", 0.0)),
                        {
                            "id": c["id"],
                            "text": c.get("text", ""),
                            "bm25_score": float(c.get("bm25_score", 0.0)),
                            "dense_score": float(c.get("dense_score", 0.0)),
                            "hybrid_rrf_score": float(c.get("hybrid_rrf_score", 0.0)),
                        },
                    )
                    for c in candidates
                ]
            elif system == "sparse":
                scored = [
                    (
                        float(c.get("bm25_score", c.get("spladepp_score", 0.0))),
                        {
                            "id": c["id"],
                            "text": c.get("text", ""),
                            "bm25_score": float(c.get("bm25_score", 0.0)),
                            "spladepp_score": float(c.get("spladepp_score", 0.0)),
                        },
                    )
                    for c in candidates
                ]
            elif system == "dense":
                scored = [
                    (
                        float(c.get("dense_score", c.get("medcpt_score", c.get("score", 0.0)))),
                        {
                            "id": c["id"],
                            "text": c.get("text", ""),
                            "dense_score": float(c.get("dense_score", 0.0)),
                            "medcpt_score": float(c.get("medcpt_score", 0.0)),
                        },
                    )
                    for c in candidates
                ]
            else:
                continue

            scored = sorted(scored, key=_live_passage_sort_key, reverse=True)
            k_reader = _resolve_reader_top_k(cfg, r_task)
            top_passages = _select_reader_passages(
                system,
                candidates=candidates,
                scored=scored,
                k=k_reader,
                cfg=cfg,
                gardian_ranked=gardian_ranked if system == "gardian" else None,
            )
            if not top_passages:
                continue

            answer = run_reader_rag_block(
                question=q_reader,
                passages_top_k=top_passages,
                tokenizer=tokenizer,
                reader_model=reader_model,
                device=device,
                top_k_passages=k_reader,
                max_new_tokens=int(cfg.qa.max_new_tokens),
                max_input_length=reader_max_input,
                max_chars_per_passage=passage_max_chars,
                question_type=item.get("question_type"),
                reader_task=r_task,
                yesno_compact=yesno_compact,
                cfg=cfg,
                alpha_sparse=(alpha_triplet[0] if (system == "gardian" and alpha_triplet) else None),
                alpha_dense=(alpha_triplet[1] if (system == "gardian" and alpha_triplet) else None),
                alpha_kg=(
                    alpha_triplet[2]
                    if (system == "gardian" and alpha_triplet and len(alpha_triplet) > 2)
                    else None
                ),
                include_signal_features=bool(
                    getattr(cfg.qa, "reader_include_signal_features", False)
                ),
                use_react=bool(getattr(cfg.qa, "reader_react", False)),
                react_max_steps=int(getattr(cfg.qa, "reader_react_max_steps", 6)),
                react_tokens_per_step=(
                    int(cfg.qa.reader_react_tokens_per_step)
                    if cfg.qa.get("reader_react_tokens_per_step") is not None
                    else None
                ),
            )
            cited_idxs = extract_citations(answer)
            cit_p, cit_r, uns = _compute_citation_metrics(
                cited_idxs,
                top_passages,
                gold_ids,
                reader_task=r_task,
                dataset=dataset,
                answer_text=answer,
            )
            row: Dict[str, Any] = {
                "qid": qid,
                "system": system,
                "answer": answer,
                "passage_ids": [str(p.get("id", "")) for p in top_passages],
                "accuracy": _check_accuracy(
                    answer,
                    gold_answer,
                    dataset,
                    gold_letter=item.get("answer_letter"),
                ),
                "citation_precision": cit_p,
                "citation_recall": cit_r,
                "unsupported_claim_rate": uns,
                "gold_evidence_in_context_rate": _gold_evidence_in_reader_context(
                    top_passages, gold_ids
                ),
            }
            if system == "gardian" and alpha_triplet is not None:
                row["sparse_alfa"] = alpha_triplet[0]
                row["dense_alfa"] = alpha_triplet[1]
                if len(alpha_triplet) > 2:
                    row["kg_alfa"] = alpha_triplet[2]
                    row["fusion_formula"] = (
                        "score = alpha_sparse*sparse + alpha_dense*dense + alpha_kg*kg"
                    )
                else:
                    row["fusion_formula"] = "score = alpha_sparse*sparse + alpha_dense*dense"
            if retrieval_meta is not None:
                row["retrieval"] = dict(retrieval_meta)
            per_system_rows[system].append(row)

    aggregate: Dict[str, Any] = {
        "_ci_format": "[mean, ci95_low, ci95_high] — see also *_ci objects with mean/ci95_low/ci95_high",
    }
    for system, rows in per_system_rows.items():
        if not rows:
            continue
        aggregate[system] = _aggregate_system_metrics(
            rows,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
    logger.info(f"QA live evaluation systems completed: {list(aggregate.keys())}")
    return aggregate, per_system_rows


def evaluate_qa_from_rank_records(
    questions: List[Dict[str, Any]],
    rank_records: List[Dict[str, Any]],
    *,
    systems: List[str],
    gardian_model: Optional[torch.nn.Module],
    tokenizer,
    reader_model,
    cfg,
    device: str = "cuda",
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 42,
    passage_text_by_pid: Optional[Dict[str, str]] = None,
    query_emb_cache_path: Optional[Union[str, Sequence[str]]] = None,
    allow_query_emb_encode_on_cache_miss: bool = True,
    gardian_adaptive_retrieval: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    """
    Controlled QA eval using pre-generated rank records for each query.

    ``systems`` can include: llm_only, bm25, dense, hybrid, doc2query, gardian.

    When ``gardian_adaptive_retrieval`` is True, GARDIAN uses controller weights to
    build an α-weighted sparse+dense subset from the rank pool before fusion (query-first).

    ``llm_only`` does not require a non-empty candidate pool; other systems need
    rank JSONL candidates for the corresponding ``qid``.

    ``passage_text_by_pid``: optional map for rank rows that omit ``text`` (reader
    context).

    ``query_emb_cache_path``: optional pickle(s) ``qid -> query_emb`` (see
    ``scripts/12_precompute_query_cache.py``). Pass a comma-separated string or a
    list of paths to merge caches (e.g. train + eval); later files override earlier
    on duplicate qids. Used when rank rows omit ``query_emb`` (typical compact JSONL).

    ``allow_query_emb_encode_on_cache_miss``: if True (default), encode the question
    with ``cfg.encoder.model_name`` when a qid is missing from the pickle (e.g. eval
    questions not present in ``*_train_all.pkl``).
    """
    by_qid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in rank_records:
        by_qid[rec["qid"]].append(rec)

    per_system_rows: Dict[str, List[Dict[str, Any]]] = {s: [] for s in systems}

    needs_rank_pool = any(
        s in systems for s in ("bm25", "dense", "hybrid", "doc2query", "gardian")
    )
    text_lookup: Dict[str, str] = dict(passage_text_by_pid or {})
    q_emb_paths = _normalize_query_emb_cache_paths(query_emb_cache_path)
    q_emb_by_qid = _load_query_emb_cache(query_emb_cache_path)
    if q_emb_by_qid and "gardian" in systems:
        src = ", ".join(q_emb_paths) if q_emb_paths else "?"
        logger.info(f"QA eval: loaded {len(q_emb_by_qid):,} query embeddings from [{src}]")

    reader_max_input = int(cfg.qa.get("reader_max_input_length", 2048) or 2048)
    passage_max_chars = int(cfg.qa.get("max_chars_per_passage", 600) or 600)
    if bool(getattr(cfg.qa, "reader_react", False)) and needs_rank_pool:
        logger.info(
            "QA eval: Self-RAG–inspired ReAct reader enabled "
            f"(qa.reader_react=true, max_steps={int(cfg.qa.get('reader_react_max_steps', 6) or 6)})"
        )

    dataset_names = sorted({str(q.get("dataset") or "dataset") for q in questions})
    progress_desc = (
        f"QA {dataset_names[0]}"
        if len(dataset_names) == 1
        else f"QA {len(dataset_names)} datasets"
    )
    for item in tqdm(questions, desc=progress_desc, unit="question"):
        qid = item.get("id")
        candidates = by_qid.get(qid, [])
        if needs_rank_pool and not candidates:
            continue
        gold_answer = item.get("answer", "")
        gold_ids = item.get("gold_passage_ids", [])
        dataset = item.get("dataset", "")
        q_reader = format_qa_question_for_reader(item)
        r_task = reader_task_for_item(item)

        scored_cache: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {}
        alpha_triplet = None
        if "gardian" in systems and gardian_model is not None and candidates:
            gardian_pool = candidates
            if gardian_adaptive_retrieval:
                qvec_pre = _resolve_query_emb_vector(
                    candidates,
                    q_emb_by_qid,
                    cfg,
                    device,
                    allow_encode_on_cache_miss=allow_query_emb_encode_on_cache_miss,
                )
                qoh_pre = candidates[0]["qtype_onehot"]
                alpha, beta = controller_weights_from_lists(
                    gardian_model,
                    qvec_pre,
                    qoh_pre,
                    device,
                )
                gardian_pool = subset_rank_records_adaptive(candidates, alpha, beta, cfg)
                if not gardian_pool:
                    gardian_pool = candidates

            with torch.no_grad():
                sparse = torch.tensor([r["sparse_feats"] for r in gardian_pool], dtype=torch.float32, device=device)
                dense = torch.tensor([r["dense_feats"] for r in gardian_pool], dtype=torch.float32, device=device)
                qvec = _resolve_query_emb_vector(
                    gardian_pool,
                    q_emb_by_qid,
                    cfg,
                    device,
                    allow_encode_on_cache_miss=allow_query_emb_encode_on_cache_miss,
                )
                query_emb = torch.tensor(qvec, dtype=torch.float32, device=device).unsqueeze(0).expand(
                    len(gardian_pool), -1
                )
                qtype = torch.tensor(gardian_pool[0]["qtype_onehot"], dtype=torch.float32, device=device).unsqueeze(0).expand(len(gardian_pool), -1)
                fwd = {
                    "sparse_feats": sparse,
                    "dense_feats": dense,
                    "query_emb": query_emb,
                    "qtype_onehot": qtype,
                    "return_breakdown": True,
                }
                if not _gardian_text_only(cfg, gardian_model):
                    fwd["kg_feats"] = torch.tensor(
                        [r["kg_feats"] for r in gardian_pool], dtype=torch.float32, device=device
                    )
                    fwd["kg_coverage"] = torch.full(
                        (len(gardian_pool),),
                        float(gardian_pool[0].get("kg_coverage", 0.0)),
                        dtype=torch.float32,
                        device=device,
                    )
                scores, weights, breakdown = gardian_model(**fwd)
                sparse_contrib = breakdown["sparse_contrib"].detach().cpu().tolist()
                dense_contrib = breakdown["dense_contrib"].detach().cpu().tolist()
                scored_cache["gardian"] = [
                    (
                        float(s),
                        {
                            "id": r["pid"],
                            "text": _passage_line_text(r, text_lookup),
                            "gardian_score": float(s),
                            "sparse_contribution": float(sc),
                            "dense_contribution": float(dc),
                            "bm25_score": float(r.get("bm25_score", 0.0)),
                            "dense_score": float(r.get("dense_score", 0.0)),
                            "doc2query_score": float(r.get("doc2query_score", 0.0)),
                            "adaptive_retrieval": bool(gardian_adaptive_retrieval),
                        },
                    )
                    for s, r, sc, dc in zip(
                        scores.detach().cpu().tolist(),
                        gardian_pool,
                        sparse_contrib,
                        dense_contrib,
                    )
                ]
                if weights.shape[0] > 0:
                    alpha_triplet = [float(x) for x in weights[0].detach().cpu().tolist()]

        for system in systems:
            if system == "llm_only":
                answer = run_reader_llm_only_block(
                    question=q_reader,
                    tokenizer=tokenizer,
                    reader_model=reader_model,
                    device=device,
                    max_new_tokens=int(cfg.qa.max_new_tokens),
                    max_input_length=reader_max_input,
                    question_type=item.get("question_type"),
                    reader_task=r_task,
                )
                top_passages: List[Dict[str, Any]] = []
                # No passages in this baseline — passage-level citation metrics are N/A.
                row = {
                    "qid": qid,
                    "system": system,
                    "answer": answer,
                    "accuracy": _check_accuracy(
                        answer,
                        gold_answer,
                        dataset,
                        gold_letter=item.get("answer_letter"),
                    ),
                    "citation_precision": None,
                    "citation_recall": None,
                    "unsupported_claim_rate": None,
                }
                per_system_rows[system].append(row)
                continue

            if system == "gardian":
                scored = scored_cache.get("gardian", [])
            elif system == "bm25":
                scored = [
                    (
                        _resolve_sparse_score(r),
                        {
                            "id": r["pid"],
                            "text": _passage_line_text(r, text_lookup),
                            "bm25_score": float(r.get("bm25_score", 0.0)),
                            "spladepp_score": float(r.get("spladepp_score", 0.0)),
                            "dense_score": _resolve_dense_score(r),
                        },
                    )
                    for r in candidates
                ]
            elif system == "dense":
                scored = [
                    (
                        _resolve_dense_score(r),
                        {
                            "id": r["pid"],
                            "text": _passage_line_text(r, text_lookup),
                            "bm25_score": float(r.get("bm25_score", 0.0)),
                            "spladepp_score": float(r.get("spladepp_score", 0.0)),
                            "dense_score": _resolve_dense_score(r),
                        },
                    )
                    for r in candidates
                ]
            elif system == "hybrid":
                scored = _rrf_fused_candidates(candidates, text_lookup)
            elif system == "doc2query":
                scored = [
                    (
                        float(r.get("doc2query_score", 0.0)),
                        {
                            "id": r["pid"],
                            "text": _passage_line_text(r, text_lookup),
                            "doc2query_score": float(r.get("doc2query_score", 0.0)),
                        },
                    )
                    for r in candidates
                ]
            else:
                continue

            def _qa_passage_sort_key(
                pair: Tuple[float, Dict[str, Any]],
            ) -> Tuple[float, float, float]:
                score, d = pair
                sp = float(d.get("bm25_score", 0.0)) + float(d.get("spladepp_score", 0.0))
                den = float(d.get("dense_score", 0.0)) + float(d.get("medcpt_score", 0.0))
                return (score, sp, den)

            scored = sorted(scored, key=_qa_passage_sort_key, reverse=True)
            k_reader = _resolve_reader_top_k(cfg, r_task)
            gardian_ranked = [p for _, p in scored] if system == "gardian" else None
            top_passages = _select_reader_passages(
                system,
                candidates=candidates,
                scored=scored,
                k=k_reader,
                cfg=cfg,
                gardian_ranked=gardian_ranked,
                text_lookup=text_lookup,
            )
            if not top_passages:
                continue
            answer = run_reader_rag_block(
                question=q_reader,
                passages_top_k=top_passages,
                tokenizer=tokenizer,
                reader_model=reader_model,
                device=device,
                top_k_passages=k_reader,
                max_new_tokens=int(cfg.qa.max_new_tokens),
                max_input_length=reader_max_input,
                max_chars_per_passage=passage_max_chars,
                question_type=(
                    item.get("question_type")
                    or (candidates[0].get("question_type") if candidates else None)
                    or "other"
                ),
                reader_task=r_task,
                alpha_sparse=(alpha_triplet[0] if (system == "gardian" and alpha_triplet is not None) else None),
                alpha_dense=(alpha_triplet[1] if (system == "gardian" and alpha_triplet is not None) else None),
                alpha_kg=(
                    alpha_triplet[2]
                    if (
                        system == "gardian"
                        and alpha_triplet is not None
                        and len(alpha_triplet) > 2
                    )
                    else None
                ),
                include_signal_features=True,
                use_react=bool(getattr(cfg.qa, "reader_react", False)),
                react_max_steps=int(getattr(cfg.qa, "reader_react_max_steps", 6)),
                react_tokens_per_step=(
                    int(cfg.qa.reader_react_tokens_per_step)
                    if cfg.qa.get("reader_react_tokens_per_step") is not None
                    else None
                ),
            )
            cited_idxs = extract_citations(answer)
            cit_p, cit_r, uns = _compute_citation_metrics(
                cited_idxs,
                top_passages,
                gold_ids,
                reader_task=r_task,
                dataset=dataset,
                answer_text=answer,
            )
            row = {
                "qid": qid,
                "system": system,
                "answer": answer,
                "passage_ids": [str(p.get("id", "")) for p in top_passages],
                "accuracy": _check_accuracy(
                    answer,
                    gold_answer,
                    dataset,
                    gold_letter=item.get("answer_letter"),
                ),
                "citation_precision": cit_p,
                "citation_recall": cit_r,
                "unsupported_claim_rate": uns,
                "gold_evidence_in_context_rate": _gold_evidence_in_reader_context(
                    top_passages, gold_ids
                ),
            }
            if system == "gardian" and alpha_triplet is not None:
                row["sparse_alfa"] = alpha_triplet[0]
                row["dense_alfa"] = alpha_triplet[1]
                if len(alpha_triplet) > 2:
                    row["kg_alfa"] = alpha_triplet[2]
                    row["fusion_formula"] = (
                        "score = alpha_sparse*sparse + alpha_dense*dense + alpha_kg*kg"
                    )
                else:
                    row["fusion_formula"] = "score = alpha_sparse*sparse + alpha_dense*dense"
                row["top_passage_contributions"] = [
                    {
                        "pid": p.get("id"),
                        "sparse_contribution": float(p.get("sparse_contribution", 0.0)),
                        "dense_contribution": float(p.get("dense_contribution", 0.0)),
                        "kg_contribution": float(p.get("kg_contribution", 0.0)),
                    }
                    for p in top_passages
                ]
            per_system_rows[system].append(row)

    aggregate: Dict[str, Any] = {
        "_ci_format": "[mean, ci95_low, ci95_high] — see also *_ci objects with mean/ci95_low/ci95_high",
    }
    for system, rows in per_system_rows.items():
        if not rows:
            continue
        aggregate[system] = _aggregate_system_metrics(
            rows,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
    logger.info(f"QA evaluation systems completed: {list(aggregate.keys())}")
    return aggregate, per_system_rows


def _pubmedqa_label_in_prediction(pred: str, gold: str) -> bool:
    """
    Whether ``pred`` supports gold label ``yes`` / ``no`` / ``maybe``.

    Uses whole-word / explicit-label matching only — substring checks like ``"no" in pred``
    are wrong because ``"know"`` contains ``"no"`` (e.g. false positives on ``I don't know``).

    If the model follows the prompt and puts the verdict on the **last non-empty line**
    (``yes`` / ``no`` / ``maybe``), that line wins over incidental words in the body.
    """
    pred_raw = pred.strip()
    pred_l = pred_raw.lower()
    gold = gold.strip().lower()
    if gold not in ("yes", "no", "maybe"):
        return False
    if re.search(rf"(?i)\blabel\s*:\s*{re.escape(gold)}\b", pred_l):
        return True
    # "The answer: no." / "Answer: yes" (common Flan-T5 pattern; last explicit verdict wins).
    ans_spans = list(
        re.finditer(r"(?i)\b(?:the\s+)?answer\s*:\s*(yes|no|maybe)\b", pred_l)
    )
    if ans_spans:
        return ans_spans[-1].group(1).lower() == gold
    lines = [ln.strip() for ln in pred_raw.splitlines() if ln.strip()]
    if lines:
        last = lines[-1].strip().lower()
        m = re.match(r"^(yes|no|maybe)[\s.!?,;:]*$", last)
        if m:
            return m.group(1) == gold
        m2 = re.match(r"^(?:(?:the\s+)?answer|label)\s*:\s*(yes|no|maybe)[\s.!?,;:]*$", last, re.I)
        if m2:
            return m2.group(1).lower() == gold
    return bool(re.search(rf"(?i)\b{re.escape(gold)}\b", pred_l))


def _check_accuracy(
    pred: str,
    gold: str,
    dataset: str,
    *,
    gold_letter: Optional[str] = None,
) -> float:
    pred_raw = pred.strip()
    pred_l = pred_raw.lower()
    gold_l = gold.strip().lower()
    # JSONL writers use pubmedqa_labeled / pubmedqa_artificial; keep legacy "pubmedqa".
    if dataset in ("pubmedqa", "pubmedqa_labeled", "pubmedqa_artificial"):
        return 1.0 if _pubmedqa_label_in_prediction(pred_raw, gold_l) else 0.0
    # MedMCQA rows use dataset == "medmcqa"; keep legacy "medqa".
    if dataset in ("medqa", "medmcqa"):
        gl = (gold_letter or "").strip().upper()
        if gl and len(gl) == 1 and gl.isalpha():
            if re.search(rf"(?i)answer\s*:\s*{re.escape(gl)}\s*[—\-–]", pred_raw):
                return 1.0
            if re.search(rf"(?i)answer\s*:\s*{re.escape(gl)}\b", pred_raw):
                return 1.0
        g = gold.strip()
        if g:
            return 1.0 if re.search(rf"(?i)\b{re.escape(g)}\b", pred_raw) else 0.0
        return 0.0
    return 0.0


def _citation_precision(cited_idxs: List[str],
                        passages: List[Dict],
                        gold_ids: List[str]) -> Optional[float]:
    if not cited_idxs:
        return None
    gold_set = set(gold_ids)
    correct  = 0
    for idx_str in cited_idxs:
        try:
            idx = int(idx_str) - 1    # P1 → index 0
            if 0 <= idx < len(passages) and passages[idx]["id"] in gold_set:
                correct += 1
        except ValueError:
            pass
    return correct / len(cited_idxs)