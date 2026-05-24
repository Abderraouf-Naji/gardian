"""Per-question-type retrieval metrics for paper / RQ analysis (text-only)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from loguru import logger

from src.common.question_types import (
    ORDERED_QUESTION_TYPES,
    QTYPE_TO_IDX,
    normalize_question_type,
)
from src.evaluation.rank_jsonl_eval import (
    _batch_encode_query_misses,
    compute_mrr,
    compute_ndcg,
    compute_recall,
    iter_rank_jsonl_records,
)
from src.evaluation.stats import bootstrap_delta_ci, paired_randomization_pvalue
from src.model.gardian import GARDIAN
from src.pipeline.gardian_adaptive import (
    controller_weights_from_lists,
    subset_rank_records_adaptive,
)

# RQ: benefits across clinical reasoning types (MedMCQA-heavy); yesno reported for PubMedQA.
RESEARCH_QUESTION_TYPES: Tuple[str, ...] = (
    "diagnosis",
    "treatment",
    "mechanism",
    "contraindication",
    "factoid",
)

PUBMEDQA_QTYPES: Tuple[str, ...] = ("yesno",)

METRIC_KEYS = [
    "ndcg@5",
    "ndcg@10",
    "ndcg@20",
    "ndcg@50",
    "ndcg@100",
    "recall@5",
    "recall@20",
    "recall@50",
    "recall@100",
    "mrr",
]


def _aggregate_lists(lists: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: float(np.mean(v)) if v else 0.0 for k, v in lists.items()}


def _metrics_for_query_subset(
    queries: Dict[str, Dict[str, Any]],
    qids: Set[str],
    score_key: str,
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    lists: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
    for qid in qids:
        if qid not in queries:
            continue
        qdata = queries[qid]
        scores = qdata.get(score_key, [])
        if not scores:
            continue
        sorted_indices = np.argsort(scores)[::-1]
        ranked_ids = [qdata["candidates"][i]["pid"] for i in sorted_indices]
        relevant_ids = [c["pid"] for c in qdata["candidates"] if c["label"] == 1]
        if not relevant_ids:
            continue
        lists["ndcg@5"].append(compute_ndcg(ranked_ids, relevant_ids, 5))
        lists["ndcg@10"].append(compute_ndcg(ranked_ids, relevant_ids, 10))
        lists["ndcg@20"].append(compute_ndcg(ranked_ids, relevant_ids, 20))
        lists["ndcg@50"].append(compute_ndcg(ranked_ids, relevant_ids, 50))
        lists["ndcg@100"].append(compute_ndcg(ranked_ids, relevant_ids, 100))
        lists["recall@5"].append(compute_recall(ranked_ids, relevant_ids, 5))
        lists["recall@20"].append(compute_recall(ranked_ids, relevant_ids, 20))
        lists["recall@50"].append(compute_recall(ranked_ids, relevant_ids, 50))
        lists["recall@100"].append(compute_recall(ranked_ids, relevant_ids, 100))
        lists["mrr"].append(compute_mrr(ranked_ids, relevant_ids))
    means = _aggregate_lists(lists)
    means["mrr@10"] = means.get("mrr", 0.0)
    return means, lists


def _build_query_buckets(
    rank_data_path: str,
    gardian_features: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    query_emb_prefill: Optional[Dict[str, List[float]]] = None,
    pending_questions: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, List[Dict[str, Any]]]]:
    """One pass over rank JSONL: scores + labels + question_type per qid."""
    from src.evaluation.rank_jsonl_eval import _new_query_bucket, _rrf_scores_for_query

    def _resolve_sparse_score(rec: Dict[str, Any]) -> float:
        rt = str(rec.get("retriever_type", ""))
        if rt in {"faiss", "medcpt"}:
            return 0.0
        if rt in {"hybrid", "hybrid_bm25_faiss", "hybrid_bm25_medcpt", "bm25"}:
            if rec.get("bm25_score") is not None:
                return float(rec.get("bm25_score", 0.0))
        if rt in {"hybrid_neural", "hybrid_spladepp_faiss", "hybrid_spladepp_medcpt", "spladepp"}:
            if rec.get("spladepp_score") is not None:
                return float(rec.get("spladepp_score", 0.0))
        sparse_feats = rec.get("sparse_feats") or [0.0]
        return float(sparse_feats[0] if sparse_feats else 0.0)

    def _resolve_dense_score(rec: Dict[str, Any]) -> float:
        if "dense_score" in rec:
            return float(rec["dense_score"])
        rt = str(rec.get("retriever_type", ""))
        if rt in ("bm25", "spladepp"):
            return 0.0
        dense_feats = rec.get("dense_feats") or [0.0]
        return float(dense_feats[0] if dense_feats else 0.0)

    queries: Dict[str, Dict[str, Any]] = defaultdict(_new_query_bucket)
    qid_to_qtype: Dict[str, str] = {}
    records_by_qid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for rec in iter_rank_jsonl_records(rank_data_path):
        qid = str(rec["qid"])
        records_by_qid[qid].append(rec)
        if qid not in qid_to_qtype:
            qtype = normalize_question_type(rec.get("question_type"))
            qid_to_qtype[qid] = qtype
        sparse_score = _resolve_sparse_score(rec)
        dense_score = _resolve_dense_score(rec)
        queries[qid]["candidates"].append({"pid": rec["pid"], "label": rec["label"]})
        queries[qid]["sparse_scores"].append(sparse_score)
        queries[qid]["dense_scores"].append(dense_score)
        queries[qid]["fusion_scores"].append(float(sparse_score) + float(dense_score))
        if gardian_features is not None:
            gf = gardian_features[qid]
            gf["candidates"].append({"pid": rec["pid"], "label": rec["label"]})
            gf["sparse_feats"].append(
                torch.tensor(rec["sparse_feats"], dtype=torch.float32)
            )
            gf["dense_feats"].append(
                torch.tensor(rec["dense_feats"], dtype=torch.float32)
            )
            if gf["qtype_onehot"] is None and isinstance(rec.get("qtype_onehot"), list):
                gf["qtype_onehot"] = torch.tensor(
                    rec["qtype_onehot"], dtype=torch.float32
                )
            if gf["query_emb"] is None:
                qe = rec.get("query_emb")
                if isinstance(qe, list) and qe:
                    if query_emb_prefill is not None:
                        query_emb_prefill[qid] = qe
                    gf["query_emb"] = torch.tensor(qe, dtype=torch.float32)
                elif pending_questions is not None:
                    question = rec.get("question")
                    if isinstance(question, str) and question.strip():
                        pending_questions.setdefault(qid, question.strip())

    for qid, qdata in queries.items():
        qdata["rrf_scores"] = _rrf_scores_for_query(
            qdata["sparse_scores"],
            qdata["dense_scores"],
        )

    return dict(queries), qid_to_qtype, dict(records_by_qid)


def _attach_gardian_scores(
    queries: Dict[str, Dict[str, Any]],
    model: GARDIAN,
    device: str,
    gardian_features: Dict[str, Dict[str, Any]],
    *,
    records_by_qid: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    cfg: Optional[Any] = None,
    gardian_adaptive_retrieval: bool = False,
) -> None:
    with torch.no_grad():
        for qid, qdata in gardian_features.items():
            if not qdata["candidates"]:
                continue
            if (
                gardian_adaptive_retrieval
                and cfg is not None
                and records_by_qid
                and qdata["query_emb"] is not None
            ):
                pool = records_by_qid.get(qid, [])
                if pool:
                    alpha, beta = controller_weights_from_lists(
                        model,
                        qdata["query_emb"].cpu().tolist(),
                        qdata["qtype_onehot"].cpu().tolist(),
                        device,
                    )
                    subset = subset_rank_records_adaptive(pool, alpha, beta, cfg)
                    if subset:
                        qdata = {
                            "sparse_feats": [
                                torch.tensor(r["sparse_feats"], dtype=torch.float32)
                                for r in subset
                            ],
                            "dense_feats": [
                                torch.tensor(r["dense_feats"], dtype=torch.float32)
                                for r in subset
                            ],
                            "query_emb": qdata["query_emb"],
                            "qtype_onehot": qdata["qtype_onehot"],
                            "candidates": [
                                {"pid": r["pid"], "label": r["label"]} for r in subset
                            ],
                        }
            batch_size = len(qdata["candidates"])
            sparse_batch = torch.stack(qdata["sparse_feats"]).to(device)
            dense_batch = torch.stack(qdata["dense_feats"]).to(device)
            query_emb = qdata["query_emb"].unsqueeze(0).expand(batch_size, -1).to(device)
            qtype_onehot = qdata["qtype_onehot"].unsqueeze(0).expand(batch_size, -1).to(device)
            scores, _ = model(
                sparse_feats=sparse_batch,
                dense_feats=dense_batch,
                query_emb=query_emb,
                qtype_onehot=qtype_onehot,
                ablation=None,
            )
            queries[qid]["gardian_scores"] = scores.cpu().numpy().flatten().tolist()


def evaluate_by_question_type(
    rank_data_path: str,
    model: GARDIAN,
    device: str,
    retriever: str,
    *,
    focus_qtypes: Sequence[str] = RESEARCH_QUESTION_TYPES,
    include_yesno: bool = True,
    randomization_trials: int = 10000,
    seed: int = 42,
    query_encoder_name: Optional[str] = None,
    query_encoder_device: str = "cpu",
    expected_query_feat_dim: Optional[int] = None,
    cfg: Optional[Any] = None,
    gardian_adaptive_retrieval: bool = False,
) -> Dict[str, Any]:
    """
    Macro-average retrieval metrics per question type for sparse, dense, hybrid, rrf, gardian.

    Returns structure suitable for paper bundle ``question_type_analysis``.
    """
    gardian_features: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "sparse_feats": [],
            "dense_feats": [],
            "query_emb": None,
            "qtype_onehot": None,
            "candidates": [],
        }
    )
    query_emb_prefill: Dict[str, List[float]] = {}
    pending_questions: Dict[str, str] = {}
    queries, qid_to_qtype, records_by_qid = _build_query_buckets(
        rank_data_path,
        gardian_features,
        query_emb_prefill=query_emb_prefill,
        pending_questions=pending_questions,
    )
    if not queries:
        return {}

    qtypes_present: Dict[str, int] = defaultdict(int)
    for qt in qid_to_qtype.values():
        qtypes_present[qt] += 1

    report_types = list(focus_qtypes)
    if include_yesno and qtypes_present.get("yesno", 0) > 0:
        if "yesno" not in report_types:
            report_types.append("yesno")

    if query_encoder_name and pending_questions:
        need = {k: v for k, v in pending_questions.items() if k not in query_emb_prefill}
        encoded = _batch_encode_query_misses(
            need,
            query_encoder_name=query_encoder_name,
            query_encoder_device=query_encoder_device,
            model_query_dim=expected_query_feat_dim,
        )
        query_emb_prefill.update(encoded)

    for qid, gf in gardian_features.items():
        if gf["query_emb"] is not None:
            continue
        qe = query_emb_prefill.get(qid)
        if not qe:
            raise ValueError(
                f"Missing query_emb for qid={qid!r}; provide query_encoder_name or embed in rank JSONL."
            )
        gf["query_emb"] = torch.tensor(qe, dtype=torch.float32)
        if gf["qtype_onehot"] is None:
            gf["qtype_onehot"] = torch.zeros(len(ORDERED_QUESTION_TYPES), dtype=torch.float32)
            idx = QTYPE_TO_IDX.get(qid_to_qtype.get(qid, "other"), QTYPE_TO_IDX["other"])
            gf["qtype_onehot"][idx] = 1.0

    _attach_gardian_scores(
        queries,
        model,
        device,
        gardian_features,
        records_by_qid=records_by_qid,
        cfg=cfg,
        gardian_adaptive_retrieval=gardian_adaptive_retrieval,
    )

    score_keys = {
        "sparse": "sparse_scores",
        "dense": "dense_scores",
        "hybrid": "fusion_scores",
        "rrf": "rrf_scores",
        "gardian": "gardian_scores",
    }

    by_type: Dict[str, Any] = {}
    for qtype in report_types:
        qids = {qid for qid, qt in qid_to_qtype.items() if qt == qtype}
        if not qids:
            continue
        systems_out: Dict[str, Any] = {}
        per_query_store: Dict[str, Dict[str, List[float]]] = {}
        for system, sk in score_keys.items():
            means, pq = _metrics_for_query_subset(queries, qids, sk)
            systems_out[system] = means
            per_query_store[system] = pq

        sig: Dict[str, Any] = {}
        g_pq = per_query_store.get("gardian", {}).get("ndcg@10", [])
        for baseline in ("hybrid", "rrf", "sparse", "dense"):
            b_pq = per_query_store.get(baseline, {}).get("ndcg@10", [])
            if len(g_pq) != len(b_pq) or not g_pq:
                continue
            d_mean, d_lo, d_hi = bootstrap_delta_ci(
                g_pq, b_pq, n_bootstrap=500, seed=seed
            )
            pval = paired_randomization_pvalue(
                g_pq, b_pq, n_trials=randomization_trials, seed=seed
            )
            sig[f"gardian_minus_{baseline}_ndcg10"] = {
                "delta_mean": d_mean,
                "delta_ci95": [d_lo, d_hi],
                "paired_randomization_pvalue": pval,
                "baseline": baseline,
            }

        best_baseline = "hybrid"
        best_ndcg = -1.0
        for b in ("sparse", "dense", "hybrid", "rrf"):
            v = float(systems_out.get(b, {}).get("ndcg@10", 0.0))
            if v > best_ndcg:
                best_ndcg = v
                best_baseline = b
        gardian_ndcg = float(systems_out.get("gardian", {}).get("ndcg@10", 0.0))
        by_type[qtype] = {
            "n_queries": len(qids),
            "systems": systems_out,
            "significance": sig,
            "gardian_ndcg@10": gardian_ndcg,
            "best_baseline": best_baseline,
            "best_baseline_ndcg@10": best_ndcg,
            "gardian_minus_best_baseline_ndcg10": gardian_ndcg - best_ndcg,
        }

    return {
        "by_type": by_type,
        "qtype_counts": dict(qtypes_present),
        "research_focus_types": list(focus_qtypes),
        "all_types_in_config": list(ORDERED_QUESTION_TYPES),
    }


def run_qtype_analysis_for_retriever(
    project_root: str,
    retriever: str,
    cfg_path: str,
    dataset_splits: List[Tuple[str, str]],
    device: str,
    query_encoder_name: str,
    randomization_trials: int,
    seed: int,
    cuda_visible_devices: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run question-type breakdown on each dataset split (text-only GARDIAN)."""
    import os
    import pathlib
    import sys

    from omegaconf import OmegaConf

    from src.evaluation.paper_bundle import build_paper_model

    sys.path.insert(0, project_root)
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    if device == "cuda":
        compute_device = "cuda:0"
    else:
        compute_device = device

    cfg = OmegaConf.load(cfg_path)
    gardian_adaptive = bool(getattr(cfg.qa, "gardian_adaptive_retrieval", False))
    try:
        model, _expected_qdim = build_paper_model(cfg, compute_device, retriever)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and str(compute_device).startswith("cuda"):
            logger.warning("CUDA OOM loading model for qtype analysis; using CPU.")
            model, _expected_qdim = build_paper_model(cfg, "cpu", retriever)
            compute_device = "cpu"
        else:
            raise

    from src.common.rank_data_paths import resolve_rank_data_file

    out: Dict[str, Dict[str, Any]] = {}
    for ds_name, split in dataset_splits:
        rank_path = resolve_rank_data_file(retriever, ds_name, split)
        if not pathlib.Path(rank_path).exists():
            logger.warning(f"Qtype analysis skip missing: {rank_path}")
            continue
        logger.info(f"Question-type breakdown | {retriever} | {ds_name} | {rank_path}")
        out[ds_name] = evaluate_by_question_type(
            str(rank_path),
            model,
            compute_device,
            retriever,
            randomization_trials=randomization_trials,
            seed=seed,
            query_encoder_name=query_encoder_name,
            query_encoder_device=(
                compute_device if str(compute_device).startswith("cuda") else "cpu"
            ),
            expected_query_feat_dim=_expected_qdim,
            cfg=cfg,
            gardian_adaptive_retrieval=gardian_adaptive,
        )
    return out


def print_question_type_summary(qtype_block: Dict[str, Dict[str, Any]]) -> None:
    """Stdout table: GARDIAN nDCG@10 vs best baseline per question type."""
    print("\n" + "=" * 96)
    print("QUESTION-TYPE BREAKDOWN (GARDIAN full vs sparse / dense / hybrid / rrf)")
    print("=" * 96)
    for retriever, ds_map in qtype_block.items():
        if not isinstance(ds_map, dict):
            continue
        print(f"\n--- {retriever} ---")
        for ds_name, payload in ds_map.items():
            if not isinstance(payload, dict):
                continue
            counts = payload.get("qtype_counts", {})
            print(f"\n  {ds_name}  (counts: {counts})")
            by_type = payload.get("by_type", {})
            if not by_type:
                print("    (no per-type buckets)")
                continue
            print(
                f"    {'type':<18} {'n':>6} {'GARDIAN':>10} {'best':>10} "
                f"{'best_sys':>10} {'Δ nDCG@10':>12}"
            )
            for qtype in list(RESEARCH_QUESTION_TYPES) + ["yesno", "other"]:
                block = by_type.get(qtype)
                if not block:
                    continue
                g = float(block.get("gardian_ndcg@10", 0.0))
                b = float(block.get("best_baseline_ndcg@10", 0.0))
                bs = str(block.get("best_baseline", "?"))
                delta = float(block.get("gardian_minus_best_baseline_ndcg10", g - b))
                print(
                    f"    {qtype:<18} {block.get('n_queries', 0):>6} "
                    f"{g:>10.4f} {b:>10.4f} {bs:>10} {delta:>+12.4f}"
                )
    print("=" * 96 + "\n")
