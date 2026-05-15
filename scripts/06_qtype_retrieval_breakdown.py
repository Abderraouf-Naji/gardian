"""
Per–question-type retrieval breakdown for GARDIAN vs baselines.

Uses the same rank JSONL format as training/eval (see scripts/03_generate_rank_data.py):
each line includes qid, question_type, candidates features, labels, etc.

For each question type, reports mean nDCG@k, Recall@k, and MRR (same definitions
as scripts/05_evaluate_gardian.py) so you can see where adaptive fusion helps.

Usage:
    python scripts/06_qtype_retrieval_breakdown.py
    python scripts/06_qtype_retrieval_breakdown.py --rank-data data/rank_data_medmcqa_test.jsonl
    python scripts/06_qtype_retrieval_breakdown.py --ndcg-k 10 --out results/qtype_breakdown.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, ".")

from src.common.question_types import (
    ORDERED_QUESTION_TYPES,
    assert_cfg_question_types,
    normalize_question_type,
)
from src.common.rank_data_paths import resolve_rank_data_file
from src.evaluation.stats import bootstrap_mean_ci
from src.model.gardian import GARDIAN, build_gardian_from_model_cfg

torch.set_float32_matmul_precision("high")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    p = pathlib.Path(path)
    if not p.exists():
        logger.warning(f"File not found: {path}")
        return records
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_model(cfg, device: str, retriever: str) -> GARDIAN:
    ckpt_path = pathlib.Path(cfg.paths.results_dir) / f"gardian_best_{retriever}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_model_cfg = ckpt.get("cfg", {}).get("model") if isinstance(ckpt.get("cfg"), dict) else None
    model = build_gardian_from_model_cfg(ckpt_model_cfg or cfg.model)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint from {ckpt_path}")
    return model


def compute_mrr(ranked_ids: List[str], relevant_ids: List[str]) -> float:
    for rank, pid in enumerate(ranked_ids, 1):
        if pid in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_ndcg(ranked_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    dcg = 0.0
    for i, pid in enumerate(ranked_ids[:k]):
        if pid in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)
    idcg = 0.0
    for i in range(min(len(relevant_ids), k)):
        idcg += 1.0 / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


def compute_recall(ranked_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_at_k = set(ranked_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_at_k & relevant_set) / len(relevant_set)


def resolve_question_type(rec: Dict[str, Any]) -> str:
    qt = rec.get("question_type")
    if isinstance(qt, str) and qt.strip():
        return normalize_question_type(qt)
    oh = rec.get("qtype_onehot") or []
    if oh:
        idx = int(np.argmax(np.asarray(oh, dtype=np.float64)))
        if 0 <= idx < len(ORDERED_QUESTION_TYPES):
            return ORDERED_QUESTION_TYPES[idx]
    return "other"


def group_rank_records(
    records: List[Dict[str, Any]],
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
]:
    """Return (baseline_queries, gardian_feature_bundles) keyed by qid."""
    queries: Dict[str, Dict[str, Any]] = {}
    gardian_bundles: Dict[str, Dict[str, Any]] = {}

    for rec in records:
        qid = rec["qid"]
        bm25_score = rec["sparse_feats"][0] if rec.get("sparse_feats") else 0
        dense_score = rec["dense_feats"][0] if rec.get("dense_feats") else 0
        hybrid_score = float(bm25_score) + float(dense_score)

        if qid not in queries:
            qtype = resolve_question_type(rec)
            queries[qid] = {
                "candidates": [],
                "bm25_scores": [],
                "dense_scores": [],
                "hybrid_scores": [],
                "question_type": qtype,
            }
            gardian_bundles[qid] = {
                "sparse_feats": [],
                "dense_feats": [],
                "kg_feats": [],
                "query_emb": None,
                "qtype_onehot": None,
                "kg_coverage": None,
                "candidates": [],
                "question_type": qtype,
            }

        queries[qid]["candidates"].append({"pid": rec["pid"], "label": rec["label"]})
        queries[qid]["bm25_scores"].append(bm25_score)
        queries[qid]["dense_scores"].append(dense_score)
        queries[qid]["hybrid_scores"].append(hybrid_score)

        g = gardian_bundles[qid]
        g["candidates"].append({"pid": rec["pid"], "label": rec["label"]})
        g["sparse_feats"].append(torch.tensor(rec["sparse_feats"], dtype=torch.float32))
        g["dense_feats"].append(torch.tensor(rec["dense_feats"], dtype=torch.float32))
        g["kg_feats"].append(torch.tensor(rec["kg_feats"], dtype=torch.float32))
        if g["query_emb"] is None:
            g["query_emb"] = torch.tensor(rec["query_emb"], dtype=torch.float32)
            g["qtype_onehot"] = torch.tensor(rec["qtype_onehot"], dtype=torch.float32)
            g["kg_coverage"] = rec["kg_coverage"]

    return queries, gardian_bundles


def metrics_for_ranking(
    ranked_ids: List[str],
    relevant_ids: List[str],
    ndcg_k: int,
    recall_k: int,
) -> Dict[str, float]:
    rel = list(dict.fromkeys(relevant_ids))
    return {
        f"ndcg@{ndcg_k}": compute_ndcg(ranked_ids, rel, k=ndcg_k),
        f"recall@{recall_k}": compute_recall(ranked_ids, rel, k=recall_k),
        "mrr": compute_mrr(ranked_ids, rel),
    }


def collect_baseline_per_query(
    queries: Dict[str, Dict[str, Any]],
    score_key: str,
    ndcg_k: int,
    recall_k: int,
) -> List[Tuple[str, str, Dict[str, float]]]:
    """List of (qid, question_type, metrics)."""
    out: List[Tuple[str, str, Dict[str, float]]] = []
    for qid, qdata in queries.items():
        scores = qdata[score_key]
        sorted_indices = np.argsort(scores)[::-1]
        ranked_ids = [qdata["candidates"][i]["pid"] for i in sorted_indices]
        relevant_ids = [c["pid"] for c in qdata["candidates"] if c["label"] == 1]
        m = metrics_for_ranking(ranked_ids, relevant_ids, ndcg_k, recall_k)
        out.append((qid, qdata["question_type"], m))
    return out


def score_gardian_per_query(
    gardian_bundles: Dict[str, Dict[str, Any]],
    model: GARDIAN,
    device: str,
    ndcg_k: int,
    recall_k: int,
) -> List[Tuple[str, str, Dict[str, float]]]:
    out: List[Tuple[str, str, Dict[str, float]]] = []
    with torch.no_grad():
        for qid, qdata in tqdm(gardian_bundles.items(), desc="GARDIAN", leave=False):
            if not qdata["candidates"]:
                continue
            batch_size = len(qdata["candidates"])
            sparse_batch = torch.stack(qdata["sparse_feats"]).to(device)
            dense_batch = torch.stack(qdata["dense_feats"]).to(device)
            kg_batch = torch.stack(qdata["kg_feats"]).to(device)
            query_emb = qdata["query_emb"].unsqueeze(0).expand(batch_size, -1).to(device)
            qtype_onehot = qdata["qtype_onehot"].unsqueeze(0).expand(batch_size, -1).to(device)
            kg_coverage = torch.full((batch_size,), qdata["kg_coverage"], dtype=torch.float32).to(device)
            scores, _ = model(
                sparse_feats=sparse_batch,
                dense_feats=dense_batch,
                kg_feats=kg_batch,
                query_emb=query_emb,
                qtype_onehot=qtype_onehot,
                kg_coverage=kg_coverage,
            )
            sc = scores.cpu().numpy().flatten()
            sorted_indices = np.argsort(sc)[::-1]
            ranked_ids = [qdata["candidates"][i]["pid"] for i in sorted_indices]
            relevant_ids = [c["pid"] for c in qdata["candidates"] if c["label"] == 1]
            m = metrics_for_ranking(ranked_ids, relevant_ids, ndcg_k, recall_k)
            out.append((qid, qdata["question_type"], m))
    return out


def overall_mean(rows: List[Tuple[str, str, Dict[str, float]]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = list(rows[0][2].keys())
    return {k: float(np.mean([r[2][k] for r in rows])) for k in keys}


def print_breakdown_table(
    dataset_name: str,
    system_rows: Dict[str, List[Tuple[str, str, Dict[str, float]]]],
    ndcg_k: int,
    recall_k: int,
) -> None:
    ndcg_key = f"ndcg@{ndcg_k}"
    recall_key = f"recall@{recall_k}"
    systems = ["bm25", "dense", "hybrid", "gardian"]

    qtypes: set[str] = set()
    for sys_name in systems:
        for _, qt, _ in system_rows.get(sys_name, []):
            qtypes.add(qt)
    qtypes_sorted = sorted(qtypes)

    print(f"\n{'=' * 110}")
    print(f"{dataset_name} — metrics by question_type (n = queries per type)")
    print(f"Primary: {ndcg_key} | Recall: {recall_key} | MRR (untruncated rank)")
    print(f"{'=' * 110}")

    for qtype in qtypes_sorted:
        line_parts = [f"{qtype:<22}"]
        n = 0
        vals_ndcg: Dict[str, float] = {}
        for sys in systems:
            rows = system_rows.get(sys, [])
            sub = [m for _, qt, m in rows if qt == qtype]
            if not sub:
                vals_ndcg[sys] = 0.0
                continue
            n = len(sub)
            vals_ndcg[sys] = float(np.mean([m[ndcg_key] for m in sub]))
        line_parts.append(f"n={n:<6}")

        for sys in systems:
            line_parts.append(f"{sys:>8}:{vals_ndcg[sys]:.4f}")

        g, d, h = vals_ndcg.get("gardian", 0), vals_ndcg.get("dense", 0), vals_ndcg.get("hybrid", 0)
        delta_d = (g - d) * 100
        delta_h = (g - h) * 100
        line_parts.append(f"  ΔG-D:{delta_d:+6.2f}bp")
        line_parts.append(f"ΔG-H:{delta_h:+6.2f}bp")
        print(" ".join(line_parts))

    print(f"{'-' * 110}")
    over_bits = ["overall".ljust(22)]
    n_tot = len(system_rows.get("dense", []))
    over_bits.append(f"n={n_tot:<6}")
    for sys in systems:
        om = overall_mean(system_rows.get(sys, []))
        over_bits.append(f"{sys:>8}:{om.get(ndcg_key, 0.0):.4f}")
    om_g = overall_mean(system_rows.get("gardian", []))
    om_d = overall_mean(system_rows.get("dense", []))
    om_h = overall_mean(system_rows.get("hybrid", []))
    over_bits.append(f"  ΔG-D:{(om_g.get(ndcg_key,0)-om_d.get(ndcg_key,0))*100:+6.2f}bp")
    over_bits.append(f"ΔG-H:{(om_g.get(ndcg_key,0)-om_h.get(ndcg_key,0))*100:+6.2f}bp")
    print(" ".join(over_bits))
    print(f"{'=' * 110}\n")


def build_json_payload(
    dataset_name: str,
    system_rows: Dict[str, List[Tuple[str, str, Dict[str, float]]]],
    ndcg_k: int,
    recall_k: int,
) -> Dict[str, Any]:
    ndcg_key = f"ndcg@{ndcg_k}"
    recall_key = f"recall@{recall_k}"
    systems = ["bm25", "dense", "hybrid", "gardian"]
    out: Dict[str, Any] = {"dataset": dataset_name, "by_question_type": {}}

    qtypes: set[str] = set()
    for sys_name in systems:
        for _, qt, _ in system_rows.get(sys_name, []):
            qtypes.add(qt)

    for qtype in sorted(qtypes):
        block: Dict[str, Any] = {"n_queries": 0, "systems": {}}
        for sys in systems:
            rows = [(qid, m) for qid, qt, m in system_rows.get(sys, []) if qt == qtype]
            if not rows:
                continue
            block["n_queries"] = len(rows)
            ms = [m for _, m in rows]
            block["systems"][sys] = {
                ndcg_key: float(np.mean([m[ndcg_key] for m in ms])),
                recall_key: float(np.mean([m[recall_key] for m in ms])),
                "mrr": float(np.mean([m["mrr"] for m in ms])),
            }
        g = block["systems"].get("gardian", {}).get(ndcg_key)
        d = block["systems"].get("dense", {}).get(ndcg_key)
        h = block["systems"].get("hybrid", {}).get(ndcg_key)
        if g is not None and d is not None:
            block["gardian_minus_dense_ndcg_bp"] = round((g - d) * 10000) / 100
        if g is not None and h is not None:
            block["gardian_minus_hybrid_ndcg_bp"] = round((g - h) * 10000) / 100
        out["by_question_type"][qtype] = block

    out["overall"] = {}
    for sys in systems:
        om = overall_mean(system_rows.get(sys, []))
        if om:
            out["overall"][sys] = om
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrieval metrics broken down by question type.")
    p.add_argument(
        "--rank-data",
        type=str,
        default=None,
        help="Path to rank JSONL. If omitted, runs default test splits (same as 05_evaluate).",
    )
    p.add_argument(
        "--ndcg-k",
        type=int,
        default=10,
        help="Cutoff for nDCG (default 10).",
    )
    p.add_argument(
        "--recall-k",
        type=int,
        default=20,
        help="Cutoff for recall (default 20).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write JSON breakdown.",
    )
    p.add_argument(
        "--retriever",
        type=str,
        choices=[
            "hybrid",
            "hybrid_neural",
            "hybrid_bm25_faiss",
            "hybrid_bm25_medcpt",
            "hybrid_spladepp_faiss",
            "hybrid_spladepp_medcpt",
        ],
        default="hybrid_bm25_faiss",
        help="Retriever family to analyze.",
    )
    p.add_argument(
        "--no-gardian",
        action="store_true",
        help="Only compute BM25 / dense / hybrid (no checkpoint).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load("configs/base.yaml")
    assert_cfg_question_types(cfg.model.question_types)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    model = None
    if not args.no_gardian:
        model = build_model(cfg, device, args.retriever)

    default_jobs = [
        ("pubmedqa_labeled", resolve_rank_data_file(args.retriever, "pubmedqa_labeled", "eval")),
        ("pubmedqa_artificial", resolve_rank_data_file(args.retriever, "pubmedqa_artificial", "test")),
        ("medmcqa", resolve_rank_data_file(args.retriever, "medmcqa", "test")),
    ]
    jobs = (
        [("custom", args.rank_data)]
        if args.rank_data
        else default_jobs
    )

    all_payload: Dict[str, Any] = {}

    for dataset_name, rank_path in jobs:
        if not rank_path or not pathlib.Path(rank_path).exists():
            logger.warning(f"Skip missing: {rank_path}")
            continue
        records = load_jsonl(rank_path)
        if not records:
            continue

        queries, gardian_bundles = group_rank_records(records)

        system_rows: Dict[str, List[Tuple[str, str, Dict[str, float]]]] = {
            "bm25": collect_baseline_per_query(queries, "bm25_scores", args.ndcg_k, args.recall_k),
            "dense": collect_baseline_per_query(queries, "dense_scores", args.ndcg_k, args.recall_k),
            "hybrid": collect_baseline_per_query(queries, "hybrid_scores", args.ndcg_k, args.recall_k),
        }
        if model is not None:
            system_rows["gardian"] = score_gardian_per_query(
                gardian_bundles, model, device, args.ndcg_k, args.recall_k
            )

        print_breakdown_table(dataset_name, system_rows, args.ndcg_k, args.recall_k)
        all_payload[dataset_name] = build_json_payload(
            dataset_name, system_rows, args.ndcg_k, args.recall_k
        )
        all_payload[dataset_name]["retriever"] = args.retriever
        # Add nDCG@k confidence intervals per question type for GARDIAN
        ndcg_key = f"ndcg@{args.ndcg_k}"
        for qtype, block in all_payload[dataset_name]["by_question_type"].items():
            g_rows = [m for _, qt, m in system_rows.get("gardian", []) if qt == qtype]
            if not g_rows:
                continue
            vals = [m[ndcg_key] for m in g_rows]
            mean, lo, hi = bootstrap_mean_ci(vals, n_bootstrap=1000, seed=42)
            block["gardian_ndcg_ci95"] = [lo, hi]
            block["gardian_ndcg_mean_query"] = mean

    if args.out and all_payload:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_payload, f, indent=2)
        logger.success(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
