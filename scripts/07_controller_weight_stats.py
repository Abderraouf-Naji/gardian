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
    ORDERED_QUESTION_TYPES,
    assert_cfg_question_types,
    normalize_question_type,
)
from src.common.rank_data_paths import resolve_rank_data_file
from src.evaluation.rank_jsonl_eval import load_rank_jsonl
from src.evaluation.schemas import validate_controller_weights
from src.model.gardian import GARDIAN

torch.set_float32_matmul_precision("high")


def build_model(cfg, device: str, retriever: str) -> GARDIAN:
    model = GARDIAN(
        sparse_dim=int(cfg.model.sparse_feat_dim),
        dense_dim=int(cfg.model.dense_feat_dim),
        kg_dim=int(cfg.model.kg_feat_dim),
        branch_hidden=int(cfg.model.branch_hidden),
        controller_hidden=int(cfg.model.controller_hidden),
        query_feat_dim=int(cfg.model.query_feat_dim),
        n_qtypes=len(cfg.model.question_types),
        dropout=float(cfg.model.dropout),
    )
    ckpt_path = pathlib.Path(cfg.paths.results_dir) / f"gardian_best_{retriever}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
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
        help="Rank JSONL (needs question_type or qtype_onehot).",
    )
    p.add_argument(
        "--retriever",
        type=str,
        choices=[
            "hybrid",
            "hybrid_neural",
            "hybrid_bm25_biobert",
            "hybrid_doc2query_faiss",
            "doc2query",
        ],
        default="hybrid",
        help="Retriever family checkpoint and default rank-data naming.",
    )
    p.add_argument("--out", type=str, default="results/controller_weights_by_qtype.json")
    p.add_argument("--cfg", type=str, default="configs/base.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)
    assert_cfg_question_types(cfg.model.question_types)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rank_data_path = args.rank_data or resolve_rank_data_file(args.retriever, "medmcqa", "test")
    records = load_rank_jsonl(rank_data_path)
    if not records:
        raise SystemExit(f"No records: {rank_data_path}")

    by_qid: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        qid = rec["qid"]
        if qid not in by_qid:
            qt = rec.get("question_type")
            if isinstance(qt, str) and qt.strip():
                qtype = normalize_question_type(qt)
            else:
                oh = rec.get("qtype_onehot") or []
                if oh:
                    ix = int(np.argmax(np.asarray(oh, dtype=np.float64)))
                    qtype = (
                        ORDERED_QUESTION_TYPES[ix]
                        if 0 <= ix < len(ORDERED_QUESTION_TYPES)
                        else "other"
                    )
                else:
                    qtype = "other"
            by_qid[qid] = {
                "question_type": qtype,
                "sparse_feats": [],
                "dense_feats": [],
                "kg_feats": [],
                "query_emb": None,
                "qtype_onehot": None,
                "kg_coverage": None,
            }
        g = by_qid[qid]
        g["sparse_feats"].append(torch.tensor(rec["sparse_feats"], dtype=torch.float32))
        g["dense_feats"].append(torch.tensor(rec["dense_feats"], dtype=torch.float32))
        g["kg_feats"].append(torch.tensor(rec["kg_feats"], dtype=torch.float32))
        if g["query_emb"] is None:
            g["query_emb"] = torch.tensor(rec["query_emb"], dtype=torch.float32)
            g["qtype_onehot"] = torch.tensor(rec["qtype_onehot"], dtype=torch.float32)
            g["kg_coverage"] = rec["kg_coverage"]

    model = build_model(cfg, device, args.retriever)
    branch_names = ["alpha_sparse", "alpha_dense", "alpha_kg"]
    per_type: DefaultDict[str, List[List[float]]] = defaultdict(list)

    with torch.no_grad():
        for qid, qdata in tqdm(by_qid.items(), desc="Controller weights"):
            n = len(qdata["sparse_feats"])
            if n == 0:
                continue
            sb = torch.stack(qdata["sparse_feats"]).to(device)
            db = torch.stack(qdata["dense_feats"]).to(device)
            kb = torch.stack(qdata["kg_feats"]).to(device)
            qe = qdata["query_emb"].unsqueeze(0).expand(n, -1).to(device)
            qt = qdata["qtype_onehot"].unsqueeze(0).expand(n, -1).to(device)
            kc = torch.full((n,), qdata["kg_coverage"], dtype=torch.float32, device=device)
            _, weights = model(sb, db, kb, qe, qt, kc, ablation=None)
            w0 = weights[0].detach().cpu().tolist()
            per_type[qdata["question_type"]].append(w0)

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
