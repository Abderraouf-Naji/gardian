"""
Aggregate GARDIAN controller weights (α_sparse, α_dense, α_kg) by question type.

Reads rank JSONL, runs one forward pass per query (batched over passages),
records the query-level weight vector (same for all passages in a query),
and writes JSON suitable for a bar chart in the paper.

Usage::

    python scripts/07_controller_weight_stats.py --out results/controller_weights_by_qtype.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, ".")

from src.common.question_types import (
    assert_cfg_question_types,
    normalize_question_type,
)
from src.common.rank_data_paths import resolve_rank_data_file
from src.evaluation.rank_jsonl_eval import load_rank_jsonl
from src.evaluation.schemas import validate_controller_weights
from src.model.gardian import GARDIAN, build_gardian_from_model_cfg

torch.set_float32_matmul_precision("high")


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
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rank-data",
        type=str,
        default=None,
        help="Rank JSONL (needs a question_type label for the per-type grouping).",
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
        help="Retriever family checkpoint and default rank-data naming.",
    )
    p.add_argument("--out", type=str, default="results/controller_weights_by_qtype.json")
    p.add_argument("--cfg", type=str, default="configs/base.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)
    assert_cfg_question_types(cfg.evaluation.question_types)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rank_data_path = args.rank_data or resolve_rank_data_file(args.retriever, "medmcqa", "test")
    records = load_rank_jsonl(rank_data_path)
    if not records:
        raise SystemExit(f"No records: {rank_data_path}")

    by_qid: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        qid = rec["qid"]
        if qid not in by_qid:
            qtype = normalize_question_type(rec.get("question_type"))
            by_qid[qid] = {
                "question_type": qtype,
                "query_emb": None,
            }
        g = by_qid[qid]
        if g["query_emb"] is None:
            g["query_emb"] = torch.tensor(rec["query_emb"], dtype=torch.float32)

    model = build_model(cfg, device, args.retriever)
    branch_names = ["alpha_sparse", "alpha_dense"]
    per_type: DefaultDict[str, List[List[float]]] = defaultdict(list)

    # The controller depends only on the query, so one forward per query is
    # enough -- the weights are identical across that query's candidates.
    with torch.no_grad():
        for qid, qdata in tqdm(by_qid.items(), desc="Controller weights"):
            if qdata["query_emb"] is None:
                continue
            qe = qdata["query_emb"].to(device)
            weights = model.controller_weights(qe)
            per_type[qdata["question_type"]].append(weights[0].cpu().tolist())

    summary: Dict[str, Any] = {"by_question_type": {}, "branch_names": branch_names}
    for qtype, rows in sorted(per_type.items()):
        arr = np.asarray(rows, dtype=np.float64)
        summary["by_question_type"][qtype] = {
            "n_queries": int(arr.shape[0]),
            "mean": arr.mean(axis=0).round(6).tolist(),
            "std": arr.std(axis=0, ddof=0).round(6).tolist(),
        }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {"rank_data": rank_data_path, "cfg": args.cfg},
        "retriever": args.retriever,
        "stats": summary,
    }
    validate_controller_weights(payload)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.success(f"Wrote {out}")


if __name__ == "__main__":
    main()
