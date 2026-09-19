"""
Model-free fusion baselines: the bar GARDIAN has to clear.

None of these needs a trained model or a GPU, so they can be computed as soon as
rank data exists -- before training -- which means that when a GARDIAN
checkpoint lands you immediately know whether it cleared the bar.

Reported per (back-end, dataset):

    Sum            unnormalised sparse + dense (the CoopIS baseline; kept only
                   for continuity -- it is dominated by whichever channel has
                   the larger raw scale)
    RRF            reciprocal rank fusion, k=60
    Global-alpha   one dev-tuned alpha, applied at test  [reviewer 2's baseline]
    Group-alpha    one dev-tuned alpha per question type
    Oracle-alpha   per-query best alpha, post hoc (upper bound on adaptivity)
    pool_recall    fraction of queries with a gold passage anywhere in the pool
    Oracle rerank  perfect ordering of that same pool (ceiling from retrieval)

    python scripts/run_fusion_baselines.py --retriever hybrid_bm25_faiss
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any, Dict

os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

from src.baselines.fusion import (  # noqa: E402
    _query_ndcg, fuse, global_alpha_fit, group_alpha_fit, group_alpha_scores,
    oracle_alpha_per_query, oracle_rerank_ndcg, pool_recall, rrf_scores,
    sum_raw_scores,
)
from src.common.question_types import normalize_question_type  # noqa: E402

# dataset -> (split used for evaluation, split used to fit alpha)
EVAL_PLAN = [
    ("pubmedqa_labeled", "eval", ("pubmedqa_artificial", "dev")),
    ("pubmedqa_artificial", "test", ("pubmedqa_artificial", "dev")),
    ("medmcqa", "test", ("medmcqa", "dev")),
]


def rank_path(retriever: str, dataset: str, split: str) -> str:
    return f"data/{retriever}/rank_data_{retriever}_{dataset}_{split}.jsonl"


def load_pools(path: str, max_queries: int | None = None):
    """Minimal per-query view: pids, labels, both raw scores, question type."""
    pools: Dict[str, Dict[str, Any]] = {}
    groups: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            qid = r["qid"]
            p = pools.get(qid)
            if p is None:
                if max_queries and len(pools) >= max_queries:
                    continue
                p = pools[qid] = {"pid": [], "label": [], "sparse": [], "dense": []}
                groups[qid] = normalize_question_type(r.get("question_type"))
            s = r.get("bm25_score")
            if s is None:
                s = r.get("spladepp_score")
            if s is None:
                s = (r.get("sparse_feats") or [0.0])[0]
            d = r.get("dense_score")
            if d is None:
                d = (r.get("dense_feats") or [0.0])[0]
            p["pid"].append(r["pid"])
            p["label"].append(r["label"])
            p["sparse"].append(float(s))
            p["dense"].append(float(d))
    return pools, groups


def mean_ndcg(pools, score_fn, k=10, *, population: str = "all") -> float:
    """
    Mean nDCG@k over a query population.

    ``population="all"`` scores a query whose gold passage was never retrieved
    as 0.0 and includes it in the mean. This is the convention used by
    ``src/evaluation/rank_jsonl_eval.py`` and therefore by every number in the
    submitted paper, so it is the default: reporting anything else would not be
    comparable with the published table.

    ``population="gold_in_pool"`` averages only over queries a re-ranker could
    possibly win. It is the right denominator for isolating ranking quality
    from first-stage recall, but it inflates every system by roughly
    1/pool_recall and must never be mixed with the other convention in one
    table. On MedMCQA the two differ by 16%.
    """
    vals = []
    for qid, p in pools.items():
        has_gold = any(int(x) == 1 for x in p["label"])
        if has_gold:
            vals.append(_query_ndcg(p["pid"], p["label"], score_fn(qid, p), k))
        elif population == "all":
            vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retriever", default="hybrid_bm25_faiss")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-queries", type=int, default=None,
                    help="Cap queries per split (for a quick look).")
    ap.add_argument("--out", default=None,
                    help="Optional JSON path (default results/baselines_<retriever>.json).")
    args = ap.parse_args()

    out_path = pathlib.Path(
        args.out or f"results/baselines_{args.retriever}.json"
    )
    report: Dict[str, Any] = {"retriever": args.retriever, "k": args.k, "datasets": {}}

    for dataset, split, (fit_ds, fit_split) in EVAL_PLAN:
        test_p = rank_path(args.retriever, dataset, split)
        dev_p = rank_path(args.retriever, fit_ds, fit_split)
        if not os.path.exists(test_p):
            logger.warning(f"missing {test_p}, skipping {dataset}")
            continue
        if not os.path.exists(dev_p):
            logger.warning(f"missing dev {dev_p}, skipping {dataset}")
            continue

        logger.info(f"{dataset}/{split}: loading test pools")
        test, test_groups = load_pools(test_p, args.max_queries)
        logger.info(f"{dataset}/{split}: fitting alpha on {fit_ds}/{fit_split}")
        dev, dev_groups = load_pools(dev_p, args.max_queries)

        a_star, dev_nd = global_alpha_fit(dev, k=args.k)
        g_alphas = group_alpha_fit(dev, dev_groups, k=args.k)

        scorers = {
            "Sum (unnormalised)": lambda q, p: sum_raw_scores(p["sparse"], p["dense"]),
            "RRF": lambda q, p: rrf_scores(p["sparse"], p["dense"]),
            "Global-alpha": lambda q, p: fuse(p["sparse"], p["dense"], a_star),
            "Group-alpha": lambda q, p: group_alpha_scores(
                p, test_groups.get(q, "other"), g_alphas),
        }
        # "all" is the paper's convention and the headline column; the
        # gold-in-pool column is reported beside it for the ranking-only view.
        res = {n: mean_ndcg(test, fn, args.k, population="all") for n, fn in scorers.items()}
        res_gold = {n: mean_ndcg(test, fn, args.k, population="gold_in_pool")
                    for n, fn in scorers.items()}
        scored = {q: p for q, p in test.items() if any(int(x) == 1 for x in p["label"])}
        orc = oracle_alpha_per_query(test, k=args.k)
        pr = pool_recall(test)
        # oracle_alpha_per_query averages over queries that have a gold passage
        res_gold["Oracle-alpha (upper bound)"] = orc["oracle_ndcg"]
        res["Oracle-alpha (upper bound)"] = orc["oracle_ndcg"] * pr

        n_groups_dev = len({g for q, g in dev_groups.items()})
        degenerate = n_groups_dev <= 1
        block = {
            "n_test_queries": len(scored),
            "fit_on": f"{fit_ds}/{fit_split}",
            "global_alpha_sparse": a_star,
            "dev_ndcg_at_global_alpha": dev_nd,
            "group_alphas": g_alphas,
            "group_alpha_is_degenerate": degenerate,
            "pool_recall": pr,
            "results_gold_in_pool": res_gold,
            "oracle_rerank_ndcg": oracle_rerank_ndcg(test, k=args.k),
            "oracle_alpha_tie_fraction": orc["tie_fraction"],
            "oracle_alpha_invariant_fraction": orc["invariant_fraction"],
            "results": res,
        }
        report["datasets"][f"{dataset}/{split}"] = block

        print(f"\n=== {args.retriever} / {dataset}/{split} "
              f"({len(test):,} queries, {len(scored):,} with gold in pool; "
              f"alpha fit on {fit_ds}/{fit_split}) ===")
        print("   nDCG@10 over ALL queries (paper convention; gold-in-pool value in brackets)")
        for name, v in sorted(res.items(), key=lambda kv: -kv[1]):
            mark = ""
            if name == "Global-alpha":
                mark = f"   <- the bar GARDIAN must clear (alpha_s={a_star:.2f})"
            if name == "Group-alpha" and degenerate:
                mark = "   <- degenerate: 1 question type, identical to Global-alpha"
            print(f"   {name:<28}{v:.4f}   [{res_gold[name]:.4f}]{mark}")
        print(f"   {'pool_recall@K':<28}{pr:.4f}")
        print(f"   {'oracle re-rank ceiling':<28}{block['oracle_rerank_ndcg']:.4f}")
        print(f"   (oracle-alpha ties {orc['tie_fraction']:.2f}, "
              f"alpha-invariant queries {orc['invariant_fraction']:.2f})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path}")
    print("GARDIAN is absent by construction: it needs training on this data first.")


if __name__ == "__main__":
    main()
