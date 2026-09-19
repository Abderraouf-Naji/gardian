"""
The go/no-go comparison: GARDIAN against the model-free fusion baselines.

Scores a trained checkpoint on the same splits, the same pools and the same
queries as ``scripts/run_fusion_baselines.py``, then reports paired
significance. This is the number that decides whether the query-adaptive
framing survives.

Reported per dataset:

    GARDIAN vs Global-alpha   the comparison reviewer 2 asked for
    GARDIAN vs Group-alpha    if these tie, the controller is a type detector
    GARDIAN vs Oracle-alpha   NOT an upper bound on GARDIAN: oracle-alpha
                              constrains convex combinations of the two RAW
                              scores, while GARDIAN's branch heads are learned
                              functions of 8 features each. Beating it is
                              possible and would be a strong result.
    oracle re-rank ceiling    the real upper bound (perfect ordering of the pool)

Significance is a one-sided paired bootstrap over per-query nDCG@10, 10,000
resamples, with Holm-Bonferroni correction across the comparisons in the table.

    python scripts/compare_to_baselines.py --retriever hybrid_bm25_faiss --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any, Dict, List

os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.baselines.fusion import (  # noqa: E402
    _query_ndcg, fuse, global_alpha_fit, group_alpha_fit, group_alpha_scores,
    oracle_alpha_per_query, oracle_rerank_ndcg, pool_recall, rrf_scores,
    sum_raw_scores,
)
from src.common.question_types import normalize_question_type  # noqa: E402
from src.common.seeds import seed_path  # noqa: E402
from src.model.gardian import build_gardian_from_model_cfg, load_checkpoint_state  # noqa: E402

EVAL_PLAN = [
    ("pubmedqa_labeled", "eval", ("pubmedqa_artificial", "dev")),
    ("pubmedqa_artificial", "test", ("pubmedqa_artificial", "dev")),
    ("medmcqa", "test", ("medmcqa", "dev")),
]


def rank_path(retriever: str, dataset: str, split: str) -> str:
    return f"data/{retriever}/rank_data_{retriever}_{dataset}_{split}.jsonl"


def load_pools(path: str, *, want_feats: bool):
    """Per-query pools; feature vectors are loaded only when GARDIAN needs them."""
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
                p = pools[qid] = {"pid": [], "label": [], "sparse": [], "dense": []}
                if want_feats:
                    p["sf"], p["df"], p["qemb"] = [], [], None
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
            if want_feats:
                p["sf"].append(r["sparse_feats"])
                p["df"].append(r["dense_feats"])
                if p["qemb"] is None and isinstance(r.get("query_emb"), list):
                    p["qemb"] = r["query_emb"]
    return pools, groups


def attach_query_embeddings(pools, cache_paths: List[str], dim: int) -> int:
    """Fill missing query embeddings from the pickled caches written by script 03."""
    from src.common.query_emb_cache import load_query_emb_cache

    need = {q for q, p in pools.items() if p.get("qemb") is None}
    if not need:
        return 0
    filled = 0
    for cp in cache_paths:
        if not os.path.exists(cp) or not need:
            continue
        cache = load_query_emb_cache(cp, expected_dim=dim) or {}
        for q in list(need):
            v = cache.get(str(q))
            if v is not None:
                pools[q]["qemb"] = v
                need.discard(q)
                filled += 1
    return filled


@torch.no_grad()
def gardian_per_query(pools, model, device: str) -> Dict[str, float]:
    """Per-query nDCG@10 under GARDIAN, plus the predicted alpha_s."""
    model.eval()
    out, alphas = {}, {}
    for qid, p in pools.items():
        if not any(int(x) == 1 for x in p["label"]) or p.get("qemb") is None:
            continue
        sf = torch.tensor(p["sf"], dtype=torch.float32, device=device)
        df = torch.tensor(p["df"], dtype=torch.float32, device=device)
        qe = torch.tensor(p["qemb"], dtype=torch.float32, device=device)
        qe = qe.unsqueeze(0).expand(sf.shape[0], -1)
        scores, w = model(sparse_feats=sf, dense_feats=df, query_emb=qe)
        out[qid] = _query_ndcg(p["pid"], p["label"], scores.cpu().numpy(), 10)
        alphas[qid] = float(w[0, 0].item())
    return out, alphas


def paired_bootstrap(a: List[float], b: List[float], n_boot=10000, seed=42):
    """One-sided paired bootstrap: P(mean(a) <= mean(b))."""
    d = np.asarray(a) - np.asarray(b)
    if d.size == 0:
        return 0.0, 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    return float(d.mean()), float(np.mean(d[idx].mean(axis=1) <= 0.0))


def holm(pvals: Dict[str, float], alpha=0.05) -> Dict[str, bool]:
    """Holm-Bonferroni across the family of comparisons in one table."""
    order = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(order)
    out, rejected_all = {}, True
    for i, (name, p) in enumerate(order):
        thresh = alpha / (m - i)
        rejected_all = rejected_all and (p <= thresh)
        out[name] = rejected_all
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retriever", default="hybrid_bm25_faiss")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cfg", default="configs/base.yaml")
    ap.add_argument("--device", default="cpu",
                    help="cpu by default so this can run while training holds the GPU.")
    ap.add_argument("--datasets", default=None,
                    help="Comma-separated subset, e.g. medmcqa. Default: all three. "
                         "MedMCQA is the smallest and the most decision-relevant, "
                         "since PubMedQA is near-saturated.")
    ap.add_argument("--skip-oracle", action="store_true",
                    help="Skip the oracle-alpha sweep (101 alphas x queries). Those "
                         "numbers are already in results/baselines_<retriever>.json.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.cfg)
    ckpt_path = seed_path(cfg.paths.results_dir, args.seed,
                          f"gardian_best_{args.retriever}.pt")
    if not ckpt_path.exists():
        raise SystemExit(f"No checkpoint at {ckpt_path} -- has training finished?")

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = (ck.get("cfg") or {}).get("model") or cfg.model
    model = build_gardian_from_model_cfg(model_cfg)
    load_checkpoint_state(model, ck["model_state"], strict=True)
    model.to(args.device)
    qdim = int(model.query_feat_dim)
    logger.info(f"loaded {ckpt_path} (dev nDCG@10 at save: {ck.get('best_ndcg10')})")

    report: Dict[str, Any] = {"retriever": args.retriever, "seed": args.seed, "datasets": {}}

    wanted = ({d.strip() for d in args.datasets.split(",") if d.strip()}
              if args.datasets else None)

    for dataset, split, (fit_ds, fit_split) in EVAL_PLAN:
        if wanted and dataset not in wanted:
            continue
        tp, dp = rank_path(args.retriever, dataset, split), rank_path(args.retriever, fit_ds, fit_split)
        if not (os.path.exists(tp) and os.path.exists(dp)):
            logger.warning(f"skipping {dataset}: missing rank data")
            continue

        test, test_groups = load_pools(tp, want_feats=True)
        dev, dev_groups = load_pools(dp, want_feats=False)
        attach_query_embeddings(test, [
            f"data/query_emb_cache_{args.retriever}_{dataset}_{split}.pkl",
            f"data/query_emb_cache_{args.retriever}_all.pkl",
        ], qdim)

        a_star, _ = global_alpha_fit(dev, k=10)
        g_alphas = group_alpha_fit(dev, dev_groups, k=10)

        g_nd, alphas = gardian_per_query(test, model, args.device)
        qids = sorted(g_nd)
        if not qids:
            logger.warning(f"{dataset}: no scorable queries (missing query embeddings?)")
            continue

        gard = [g_nd[q] for q in qids]
        glob = [_query_ndcg(test[q]["pid"], test[q]["label"],
                            fuse(test[q]["sparse"], test[q]["dense"], a_star), 10) for q in qids]
        grp = [_query_ndcg(test[q]["pid"], test[q]["label"],
                           group_alpha_scores(test[q], test_groups.get(q, "other"), g_alphas), 10)
               for q in qids]
        rrf = [_query_ndcg(test[q]["pid"], test[q]["label"],
                           rrf_scores(test[q]["sparse"], test[q]["dense"]), 10) for q in qids]
        summ = [_query_ndcg(test[q]["pid"], test[q]["label"],
                            sum_raw_scores(test[q]["sparse"], test[q]["dense"]), 10) for q in qids]
        sub = {q: test[q] for q in qids}
        orc = ({"oracle_ndcg": float("nan"), "tie_fraction": None,
                "invariant_fraction": None}
               if args.skip_oracle else oracle_alpha_per_query(sub, k=10))

        deltas, pvals = {}, {}
        for name, ref in (("Global-alpha", glob), ("Group-alpha", grp), ("RRF", rrf), ("Sum", summ)):
            deltas[name], pvals[name] = paired_bootstrap(gard, ref)
        sig = holm(pvals)

        a = np.array([alphas[q] for q in qids])
        block = {
            "n_queries": len(qids),
            "oracle_skipped": bool(args.skip_oracle),
            "gardian_ndcg@10": float(np.mean(gard)),
            "global_alpha_ndcg@10": float(np.mean(glob)),
            "group_alpha_ndcg@10": float(np.mean(grp)),
            "oracle_alpha_ndcg@10": orc["oracle_ndcg"],
            "oracle_rerank_ceiling": oracle_rerank_ndcg(test, k=10),
            "population": "queries with a gold passage in the pool "
                          "(both sides scored on the same query list)",
            "pool_recall": pool_recall(sub),
            "global_alpha_sparse": a_star,
            "deltas": deltas, "p_values": pvals, "significant_holm": sig,
            "predicted_alpha_sparse": {
                "mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max()),
                "p05": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
            },
        }
        report["datasets"][f"{dataset}/{split}"] = block

        print(f"\n=== {args.retriever} / {dataset}/{split}  (n={len(qids):,}, seed {args.seed}) ===")
        print(f"   {'GARDIAN':<26}{np.mean(gard):.4f}")
        for name, ref in (("Global-alpha", glob), ("Group-alpha", grp), ("RRF", rrf), ("Sum", summ)):
            d, p = deltas[name], pvals[name]
            mark = "significant" if sig[name] else "NOT significant"
            print(f"   {'vs ' + name:<26}{np.mean(ref):.4f}   delta={d:+.4f}  p={p:.4f}  ({mark}, Holm)")
        if not args.skip_oracle:
            print(f"   {'Oracle-alpha (2 scores)':<26}{orc['oracle_ndcg']:.4f}   "
                  f"{'GARDIAN EXCEEDS IT' if np.mean(gard) > orc['oracle_ndcg'] else 'not exceeded'}")
        print(f"   {'oracle re-rank ceiling':<26}{block['oracle_rerank_ceiling']:.4f}"
              f"   (over all {len(test):,} queries; pool_recall={pool_recall(test):.4f})")
        pa = block["predicted_alpha_sparse"]
        collapsed = pa["std"] < 0.02
        print(f"   predicted alpha_s: mean={pa['mean']:.3f} sd={pa['std']:.3f} "
              f"[{pa['p05']:.3f}, {pa['p95']:.3f}]"
              f"{'   <-- COLLAPSED: the controller is effectively a global alpha' if collapsed else ''}")

    out_path = pathlib.Path(args.out or f"results/compare_{args.retriever}_seed{args.seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
