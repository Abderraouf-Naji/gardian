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
from src.features.kg_feat import build_degree_lookup, build_node_set, build_query_kg_cache, compute_kg_features
from src.features.sparse import compute_sparse_features
from src.pipeline.rag_reader import (
    format_reader_context,
    retrieve_hybrid_candidates,
    run_reader_llm_only_block,
    run_reader_rag_block,
)


def format_context(passages: List[Dict], top_k: int = 5) -> str:
    """Backward-compatible alias for :func:`format_reader_context`."""
    return format_reader_context(passages, top_k=top_k, max_chars_per_passage=600)


def extract_citations(answer_text: str) -> List[str]:
    """Parse [P1], [P2] citations from answer text."""
    return re.findall(r"\[P(\d+)\]", answer_text)


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


def _compute_citation_metrics(
    cited_idxs: List[str],
    passages: List[Dict],
    gold_ids: List[str],
    *,
    reader_task: str,
    dataset: str,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not _citation_metrics_applicable(reader_task, dataset):
        return None, None, None
    if not gold_ids:
        return None, None, None
    return (
        _citation_precision(cited_idxs, passages, gold_ids),
        _citation_recall(cited_idxs, passages, gold_ids),
        _unsupported_claim_rate(cited_idxs, passages, gold_ids),
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
    out: Dict[str, Any] = {
        "n_questions": len(rows),
        "answer_accuracy": list(acc_ci),
        "answer_accuracy_ci": _pack_ci(acc_ci),
        "citation_precision": list(cit_p_ci) if cit_p_ci else None,
        "citation_precision_ci": _pack_ci(cit_p_ci),
        "citation_recall": list(cit_r_ci) if cit_r_ci else None,
        "citation_recall_ci": _pack_ci(cit_r_ci),
        "unsupported_claim_rate": list(uns_ci) if uns_ci else None,
        "unsupported_claim_rate_ci": _pack_ci(uns_ci),
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
    cfg: Any,
    device: str,
    encoder,
    feature_cache,
    kg,
    linker,
    degree_lookup,
    node_set,
) -> Tuple[List[float], List[float], float]:
    """Attach branch feature vectors; return (qtype_onehot, query_emb, kg_coverage)."""
    qtype_oh = qtype_onehot(normalize_question_type(qtype))
    q_emb = encoder.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )[0]
    p_embs = feature_cache.get_passage_embeddings(candidates)
    dense_scores = [
        float(c.get("medcpt_score", c.get("dense_score", c.get("score", 0.0))))
        for c in candidates
    ]
    dense_mean = float(np.mean(dense_scores))
    dense_std = float(np.std(dense_scores) + 1e-8)
    q_entities = linker.link(question)
    kg_coverage = 1.0 if q_entities else 0.0
    query_kg_cache = build_query_kg_cache(
        q_entities,
        kg,
        node_set=node_set,
        compute_distances=bool(getattr(cfg.kg, "exact_distance_features", False)),
        max_path=int(getattr(cfg.kg, "max_path_length", 4)),
    )
    for i, cand in enumerate(candidates):
        p_entities = feature_cache.get_passage_entities(cand)
        cand["sparse_feats"] = compute_sparse_features(
            query=question,
            passage=cand.get("text", ""),
            bm25_score=float(cand.get("bm25_score", cand.get("spladepp_score", 0.0))),
            idf_table=None,
        ).tolist()
        cand["dense_feats"] = compute_dense_features_with_score(
            q_emb=q_emb,
            p_emb=p_embs[i],
            dense_score=dense_scores[i],
            score_mean=dense_mean,
            score_std=dense_std,
        ).tolist()
        cand["kg_feats"] = compute_kg_features(
            q_entities=q_entities,
            p_entities=p_entities,
            G=kg,
            max_path=int(getattr(cfg.kg, "max_path_length", 4)),
            query_cache=query_kg_cache,
            degree_lookup=degree_lookup,
            node_set=node_set,
        ).tolist()
    return qtype_oh, q_emb.tolist(), kg_coverage


def _live_passage_sort_key(pair: Tuple[float, Dict[str, Any]]) -> Tuple[float, float, float]:
    score, d = pair
    sp = float(d.get("bm25_score", 0.0)) + float(d.get("spladepp_score", 0.0))
    den = float(d.get("dense_score", 0.0)) + float(d.get("medcpt_score", 0.0))
    return (score, sp, den)


def evaluate_qa_live(
    questions: List[Dict[str, Any]],
    *,
    systems: List[str],
    cfg: Any,
    device: str,
    retriever: Any,
    gardian_model: Optional[torch.nn.Module],
    tokenizer,
    reader_model,
    encoder,
    feature_cache,
    kg,
    linker,
    degree_lookup,
    node_set,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 42,
    top_candidates: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    """
    QA eval with live hybrid retrieval (unified BM25 + FAISS) → GARDIAN rerank → RAG reader.

    Matches ``scripts/11_run_gardian_server.py``: retrieve, fuse features, ``gardian.rerank``,
    then ``run_reader_rag_block`` on the top-k GARDIAN-ordered passages (hybrid baseline uses
    RRF order from the retriever).
    """
    if "doc2query" in systems:
        logger.warning("doc2query is not supported with live retrieval; skipping that system.")
        systems = [s for s in systems if s != "doc2query"]

    pool_k = int(top_candidates or getattr(cfg.retrieval, "candidate_pool_size", 100))
    top_k_reader = int(cfg.qa.top_k_passages)
    reader_max_input = int(cfg.qa.get("reader_max_input_length", 2048) or 2048)
    passage_max_chars = int(cfg.qa.get("max_chars_per_passage", 600) or 600)
    needs_rank = any(s in systems for s in ("bm25", "dense", "hybrid", "gardian"))

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

        candidates = retrieve_hybrid_candidates(
            (item.get("question") or "").strip(),
            retriever,
            top_k=pool_k,
        )
        if not candidates:
            continue

        gardian_ranked: List[Dict[str, Any]] = []
        alpha_triplet = None
        if "gardian" in systems and gardian_model is not None:
            qtype_oh, q_emb, kg_cov = _enrich_live_candidates_for_gardian(
                question=(item.get("question") or "").strip(),
                candidates=candidates,
                qtype=qtype,
                cfg=cfg,
                device=device,
                encoder=encoder,
                feature_cache=feature_cache,
                kg=kg,
                linker=linker,
                degree_lookup=degree_lookup,
                node_set=node_set,
            )
            gardian_ranked = gardian_model.rerank(
                candidates=candidates,
                query_features={
                    "query_emb": q_emb,
                    "qtype_onehot": qtype_oh,
                    "kg_coverage": kg_cov,
                },
                device=device,
            )
            if gardian_ranked:
                w = gardian_ranked[0]
                alpha_triplet = [
                    float(w.get("sparse_alfa", 0.0)),
                    float(w.get("dense_alfa", 0.0)),
                    float(w.get("kg_alfa", 0.0)),
                ]

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
            elif system == "bm25":
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
            top_passages = [p for _, p in scored[:top_k_reader]]
            if not top_passages:
                continue

            answer = run_reader_rag_block(
                question=q_reader,
                passages_top_k=top_passages,
                tokenizer=tokenizer,
                reader_model=reader_model,
                device=device,
                top_k_passages=top_k_reader,
                max_new_tokens=int(cfg.qa.max_new_tokens),
                max_input_length=reader_max_input,
                max_chars_per_passage=passage_max_chars,
                question_type=item.get("question_type"),
                reader_task=r_task,
                alpha_sparse=(alpha_triplet[0] if (system == "gardian" and alpha_triplet) else None),
                alpha_dense=(alpha_triplet[1] if (system == "gardian" and alpha_triplet) else None),
                alpha_kg=(alpha_triplet[2] if (system == "gardian" and alpha_triplet) else None),
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
            )
            row: Dict[str, Any] = {
                "qid": qid,
                "system": system,
                "answer": answer,
                "accuracy": _check_accuracy(
                    answer,
                    gold_answer,
                    dataset,
                    gold_letter=item.get("answer_letter"),
                ),
                "citation_precision": cit_p,
                "citation_recall": cit_r,
                "unsupported_claim_rate": uns,
            }
            if system == "gardian" and alpha_triplet is not None:
                row["sparse_alfa"] = alpha_triplet[0]
                row["dense_alfa"] = alpha_triplet[1]
                row["kg_alfa"] = alpha_triplet[2]
                row["fusion_formula"] = "score = alpha_sparse*sparse + alpha_dense*dense + alpha_kg*kg"
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
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    """
    Controlled QA eval using pre-generated rank records for each query.

    ``systems`` can include: llm_only, bm25, dense, hybrid, doc2query, gardian.

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
            with torch.no_grad():
                sparse = torch.tensor([r["sparse_feats"] for r in candidates], dtype=torch.float32, device=device)
                dense = torch.tensor([r["dense_feats"] for r in candidates], dtype=torch.float32, device=device)
                kg = torch.tensor([r["kg_feats"] for r in candidates], dtype=torch.float32, device=device)
                qvec = _resolve_query_emb_vector(
                    candidates,
                    q_emb_by_qid,
                    cfg,
                    device,
                    allow_encode_on_cache_miss=allow_query_emb_encode_on_cache_miss,
                )
                query_emb = torch.tensor(qvec, dtype=torch.float32, device=device).unsqueeze(0).expand(
                    len(candidates), -1
                )
                qtype = torch.tensor(candidates[0]["qtype_onehot"], dtype=torch.float32, device=device).unsqueeze(0).expand(len(candidates), -1)
                kg_coverage = torch.full((len(candidates),), float(candidates[0]["kg_coverage"]), dtype=torch.float32, device=device)
                scores, weights, breakdown = gardian_model(
                    sparse_feats=sparse,
                    dense_feats=dense,
                    kg_feats=kg,
                    query_emb=query_emb,
                    qtype_onehot=qtype,
                    kg_coverage=kg_coverage,
                    return_breakdown=True,
                )
                sparse_contrib = breakdown["sparse_contrib"].detach().cpu().tolist()
                dense_contrib = breakdown["dense_contrib"].detach().cpu().tolist()
                kg_contrib = breakdown["kg_contrib"].detach().cpu().tolist()
                scored_cache["gardian"] = [
                    (
                        float(s),
                        {
                            "id": r["pid"],
                            "text": _passage_line_text(r, text_lookup),
                            "gardian_score": float(s),
                            "sparse_contribution": float(sc),
                            "dense_contribution": float(dc),
                            "kg_contribution": float(kc),
                            "bm25_score": float(r.get("bm25_score", 0.0)),
                            "dense_score": float(r.get("dense_score", 0.0)),
                            "doc2query_score": float(r.get("doc2query_score", 0.0)),
                        },
                    )
                    for s, r, sc, dc, kc in zip(
                        scores.detach().cpu().tolist(),
                        candidates,
                        sparse_contrib,
                        dense_contrib,
                        kg_contrib,
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
            top_passages = [p for _, p in scored[: int(cfg.qa.top_k_passages)]]
            if not top_passages:
                continue
            answer = run_reader_rag_block(
                question=q_reader,
                passages_top_k=top_passages,
                tokenizer=tokenizer,
                reader_model=reader_model,
                device=device,
                top_k_passages=int(cfg.qa.top_k_passages),
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
                alpha_kg=(alpha_triplet[2] if (system == "gardian" and alpha_triplet is not None) else None),
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
            )
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
                "citation_precision": cit_p,
                "citation_recall": cit_r,
                "unsupported_claim_rate": uns,
            }
            if system == "gardian" and alpha_triplet is not None:
                row["sparse_alfa"] = alpha_triplet[0]
                row["dense_alfa"] = alpha_triplet[1]
                row["kg_alfa"] = alpha_triplet[2]
                row["fusion_formula"] = "score = alpha_sparse*sparse + alpha_dense*dense + alpha_kg*kg"
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
                        gold_ids: List[str]) -> float:
    if not cited_idxs:
        return 0.0
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