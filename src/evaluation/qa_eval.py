"""End-to-end QA evaluation helpers for controlled RAG comparisons."""

import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from loguru import logger

from src.pipeline.rag_reader import format_reader_context, run_reader_rag_block


def format_context(passages: List[Dict], top_k: int = 5) -> str:
    """Backward-compatible alias for :func:`format_reader_context`."""
    return format_reader_context(passages, top_k=top_k, max_chars_per_passage=600)


def extract_citations(answer_text: str) -> List[str]:
    """Parse [P1], [P2] citations from answer text."""
    return re.findall(r"\[P(\d+)\]", answer_text)


def _bootstrap_ci(values: List[float], n_bootstrap: int = 2000, seed: int = 42) -> Tuple[float, float, float]:
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
    if not cited_idxs:
        return 1.0
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
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    """
    Controlled QA eval using pre-generated rank records for each query.

    ``systems`` can include: bm25, dense, hybrid, doc2query, gardian.
    """
    by_qid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in rank_records:
        by_qid[rec["qid"]].append(rec)

    per_system_rows: Dict[str, List[Dict[str, Any]]] = {s: [] for s in systems}

    for item in questions:
        qid = item.get("id")
        candidates = by_qid.get(qid, [])
        if not candidates:
            continue
        gold_answer = item.get("answer", "")
        gold_ids = item.get("gold_passage_ids", [])
        dataset = item.get("dataset", "")

        scored_cache: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {}
        alpha_triplet = None
        if "gardian" in systems and gardian_model is not None:
            with torch.no_grad():
                sparse = torch.tensor([r["sparse_feats"] for r in candidates], dtype=torch.float32, device=device)
                dense = torch.tensor([r["dense_feats"] for r in candidates], dtype=torch.float32, device=device)
                kg = torch.tensor([r["kg_feats"] for r in candidates], dtype=torch.float32, device=device)
                query_emb = torch.tensor(candidates[0]["query_emb"], dtype=torch.float32, device=device).unsqueeze(0).expand(len(candidates), -1)
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
                            "text": r.get("text", ""),
                            "sparse_contribution": float(sc),
                            "dense_contribution": float(dc),
                            "kg_contribution": float(kc),
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
            if system == "gardian":
                scored = scored_cache.get("gardian", [])
            elif system == "bm25":
                scored = [(float(r.get("bm25_score", 0.0)), {"id": r["pid"], "text": r.get("text", "")}) for r in candidates]
            elif system == "dense":
                scored = [(float(r.get("dense_score", 0.0)), {"id": r["pid"], "text": r.get("text", "")}) for r in candidates]
            elif system == "hybrid":
                scored = [
                    (float(r.get("bm25_score", 0.0)) + float(r.get("dense_score", 0.0)), {"id": r["pid"], "text": r.get("text", "")})
                    for r in candidates
                ]
            elif system == "doc2query":
                scored = [(float(r.get("doc2query_score", 0.0)), {"id": r["pid"], "text": r.get("text", "")}) for r in candidates]
            else:
                continue

            scored = sorted(scored, key=lambda x: x[0], reverse=True)
            top_passages = [p for _, p in scored[: int(cfg.qa.top_k_passages)]]
            if not top_passages:
                continue
            answer = run_reader_rag_block(
                question=item["question"],
                passages_top_k=top_passages,
                tokenizer=tokenizer,
                reader_model=reader_model,
                device=device,
                top_k_passages=int(cfg.qa.top_k_passages),
                max_new_tokens=int(cfg.qa.max_new_tokens),
                question_type=candidates[0].get("question_type", "other"),
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
            row = {
                "qid": qid,
                "system": system,
                "answer": answer,
                "accuracy": _check_accuracy(answer, gold_answer, dataset),
                "citation_precision": _citation_precision(cited_idxs, top_passages, gold_ids),
                "citation_recall": _citation_recall(cited_idxs, top_passages, gold_ids),
                "unsupported_claim_rate": _unsupported_claim_rate(cited_idxs, top_passages, gold_ids),
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

    aggregate: Dict[str, Any] = {}
    for system, rows in per_system_rows.items():
        if not rows:
            continue
        aggregate[system] = {
            "n_questions": len(rows),
            "answer_accuracy": _bootstrap_ci([r["accuracy"] for r in rows], bootstrap_samples, bootstrap_seed),
            "citation_precision": _bootstrap_ci([r["citation_precision"] for r in rows], bootstrap_samples, bootstrap_seed),
            "citation_recall": _bootstrap_ci([r["citation_recall"] for r in rows], bootstrap_samples, bootstrap_seed),
            "unsupported_claim_rate": _bootstrap_ci([r["unsupported_claim_rate"] for r in rows], bootstrap_samples, bootstrap_seed),
        }
    logger.info(f"QA evaluation systems completed: {list(aggregate.keys())}")
    return aggregate, per_system_rows


def _check_accuracy(pred: str, gold: str, dataset: str) -> float:
    pred = pred.strip().lower()
    gold = gold.strip().lower()
    if dataset == "pubmedqa":
        # yes/no/maybe exact match
        for label in ["yes", "no", "maybe"]:
            if label in pred and label == gold:
                return 1.0
        return 0.0
    elif dataset == "medqa":
        # Check if correct option letter/text appears first in answer
        return 1.0 if gold in pred else 0.0
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