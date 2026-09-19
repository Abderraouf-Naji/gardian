#!/usr/bin/env python3
"""
Regenerate the Global-$\\alpha$ row of the paper's retrieval tables.

``results/global_alpha_metrics.json`` was an orphan: it is the source of the
Global-$\\alpha$ column in Table~\\ref{tab:main}, but no script in the
repository wrote it, so the strongest non-adaptive baseline -- the one three
reviewers asked for -- could not be reproduced. This script writes that file,
and adds the Hit@k that the evidence table needs and the original artifact
lacked.

Protocol (identical to scripts/run_fusion_baselines.py):
  * one alpha per (back-end, dataset), grid-searched on a DEV split,
  * applied unchanged at test,
  * fusion over within-pool min-max normalised channel scores.

PubMedQA-Labeled has no train/dev split of its own (it is the 1k gold-standard
evaluation set), so its alpha is fitted on PubMedQA-Artificial/dev. That is a
cross-split transfer and is reported as such.

Usage:
  .venv/bin/python scripts/global_alpha_metrics.py
  .venv/bin/python scripts/global_alpha_metrics.py --retrievers hybrid_bm25_faiss
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any, Dict, List, Sequence

os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

from src.baselines.fusion import fuse, global_alpha_fit  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    hit_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from scripts.run_fusion_baselines import EVAL_PLAN, load_pools, rank_path  # noqa: E402

RETRIEVERS = (
    "hybrid_bm25_faiss",
    "hybrid_bm25_medcpt",
    "hybrid_spladepp_faiss",
    "hybrid_spladepp_medcpt",
)


def _ranked_ids(pool: Dict[str, Any], scores: np.ndarray) -> List[str]:
    order = np.argsort(-np.asarray(scores), kind="stable")
    return [pool["pid"][i] for i in order]


def score_pools(
    pools: Dict[str, Dict[str, Any]], alpha: float, *, k: int
) -> Dict[str, float]:
    """
    Mean retrieval metrics of one fixed alpha over a split, both populations.

    A query with no gold passage anywhere in the pool scores 0 for every
    system, the oracle included. Averaging over *all* queries therefore scales
    every row by the same pool_recall factor, and averaging over only the
    queries with gold in the pool isolates ranking from first-stage recall.
    Both are defensible and they differ by ~16% on MedMCQA, so mixing them
    across rows of one table invents wins that are not there.

    The unsuffixed keys use the **all-queries** convention, which is what the
    paper's Table 1 reports for every other system; ``*_gold_in_pool`` keys
    carry the ranking-only view beside them.
    """
    nd: List[float] = []
    rec: List[float] = []
    hit: List[float] = []
    rr: List[float] = []
    n_total = 0
    for pool in pools.values():
        n_total += 1
        relevant = [p for p, lab in zip(pool["pid"], pool["label"]) if int(lab) == 1]
        if not relevant:
            continue
        ranked = _ranked_ids(pool, fuse(pool["sparse"], pool["dense"], alpha))
        nd.append(ndcg_at_k(ranked[:k], relevant, k))
        rec.append(recall_at_k(ranked, relevant, k))
        hit.append(hit_at_k(ranked, relevant, k))
        rr.append(mrr(ranked, relevant))
    if not nd or n_total == 0:
        return {}
    n_gold = len(nd)
    scale = n_gold / n_total  # = pool_recall
    out: Dict[str, float] = {
        "pool_recall": float(scale),
        "n_queries": int(n_total),
        "n_queries_gold_in_pool": int(n_gold),
    }
    for name, vals in ((f"ndcg@{k}", nd), (f"recall@{k}", rec), (f"hit@{k}", hit), ("mrr", rr)):
        gold_only = float(np.mean(vals))
        out[name] = gold_only * scale
        out[f"{name}_gold_in_pool"] = gold_only
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retrievers", nargs="+", default=list(RETRIEVERS))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--out", default="results/global_alpha_metrics.json")
    args = ap.parse_args()

    report: Dict[str, Any] = {}
    for retriever in args.retrievers:
        report[retriever] = {}
        for dataset, split, (fit_ds, fit_split) in EVAL_PLAN:
            test_p = rank_path(retriever, dataset, split)
            dev_p = rank_path(retriever, fit_ds, fit_split)
            if not os.path.exists(test_p) or not os.path.exists(dev_p):
                logger.warning(f"{retriever}/{dataset}: missing rank data, skipping")
                continue

            logger.info(f"{retriever}/{dataset}: fitting alpha on {fit_ds}/{fit_split}")
            dev, _ = load_pools(dev_p, args.max_queries)
            alpha, dev_ndcg = global_alpha_fit(dev, k=args.k)

            test, _ = load_pools(test_p, args.max_queries)
            metrics = score_pools(test, alpha, k=args.k)
            metrics.update(
                {
                    "alpha_sparse": float(alpha),
                    "fit_on": f"{fit_ds}/{fit_split}",
                    "dev_ndcg_at_alpha": float(dev_ndcg),
                    "eval_split": f"{dataset}/{split}",
                }
            )
            report[retriever][dataset] = metrics
            logger.success(
                f"{retriever}/{dataset}: alpha={alpha:.2f} "
                f"nDCG@{args.k}={metrics.get(f'ndcg@{args.k}', 0):.4f} "
                f"Hit@{args.k}={metrics.get(f'hit@{args.k}', 0):.4f} "
                f"Recall@{args.k}={metrics.get(f'recall@{args.k}', 0):.4f} "
                f"(n={metrics.get('n_queries')})"
            )

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.success(f"wrote {out}")


if __name__ == "__main__":
    main()
