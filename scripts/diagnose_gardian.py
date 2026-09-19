"""
Where does GARDIAN's score actually come from?

Decomposes a trained checkpoint into its two learned parts by swapping each one
for a non-learned substitute on the same pools:

    A  learned branches + learned alpha      the model as trained
    B  learned branches + best global alpha  does the CONTROLLER add anything?
    C  learned branches + alpha = 0.5        controller replaced by a constant
    D  raw min-max scores + learned alpha    do the BRANCH HEADS add anything?
    E  raw min-max scores + best global alpha  = the Global-alpha baseline

A > B means per-query adaptation is doing work beyond any single global weight.
A ~ B means the controller has collapsed and the gain is entirely in the branch
heads. A ~ D means the branch heads are not adding anything over the raw scores.

Also reports branch-output magnitudes. The training loss is
``softplus(gamma - (r_pos - r_neg))`` over UNBOUNDED branch outputs, so a model
can reduce it by inflating score magnitude rather than improving order --
a failure mode that shows up as training loss falling while dev nDCG falls too.

    python scripts/diagnose_gardian.py --retriever hybrid_bm25_faiss --seed 42 \
        --dataset medmcqa --split test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.baselines.fusion import ALPHA_GRID, _query_ndcg  # noqa: E402
from src.common.query_emb_cache import load_query_emb_store  # noqa: E402
from src.common.seeds import seed_path  # noqa: E402
from src.features.pool_norm import minmax_normalise  # noqa: E402
from src.model.gardian import build_gardian_from_model_cfg, load_checkpoint_state  # noqa: E402


def load_pools(path: str, max_queries: int | None = None):
    pools: Dict[str, Dict[str, Any]] = {}
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
                p = pools[qid] = {"pid": [], "label": [], "sf": [], "df": [],
                                  "sp": [], "de": []}
            s = r.get("bm25_score")
            if s is None:
                s = r.get("spladepp_score")
            if s is None:
                s = r["sparse_feats"][0]
            d = r.get("dense_score")
            if d is None:
                d = r["dense_feats"][0]
            p["pid"].append(r["pid"]); p["label"].append(r["label"])
            p["sf"].append(r["sparse_feats"]); p["df"].append(r["dense_feats"])
            p["sp"].append(float(s)); p["de"].append(float(d))
    return {q: p for q, p in pools.items() if any(int(x) == 1 for x in p["label"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retriever", default="hybrid_bm25_faiss")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default="medmcqa")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-queries", type=int, default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load("configs/base.yaml")
    ck_path = seed_path(cfg.paths.results_dir, args.seed, f"gardian_best_{args.retriever}.pt")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model = build_gardian_from_model_cfg((ck.get("cfg") or {}).get("model") or cfg.model)
    load_checkpoint_state(model, ck["model_state"], strict=True)
    model.to(args.device).eval()
    logger.info(f"{ck_path.name}: epoch={ck.get('epoch')} dev nDCG@10={ck.get('best_ndcg10'):.4f}")

    rp = f"data/{args.retriever}/rank_data_{args.retriever}_{args.dataset}_{args.split}.jsonl"
    pools = load_pools(rp, args.max_queries)
    store = load_query_emb_store(
        f"data/query_emb_cache_{args.retriever}_{args.dataset}_{args.split}.pkl",
        expected_dim=int(model.query_feat_dim))
    logger.info(f"{len(pools):,} scorable queries from {args.dataset}/{args.split}")

    s_sp, s_de, alphas, qids = {}, {}, {}, []
    with torch.no_grad():
        for qid, p in pools.items():
            qe = store.get(qid)
            if qe is None:
                continue
            sf = torch.tensor(np.asarray(p["sf"], dtype=np.float32), device=args.device)
            df = torch.tensor(np.asarray(p["df"], dtype=np.float32), device=args.device)
            q = torch.tensor(np.asarray(qe, dtype=np.float32), device=args.device)
            s_sp[qid] = model.sparse_head(sf).cpu().numpy()
            s_de[qid] = model.dense_head(df).cpu().numpy()
            alphas[qid] = float(model.controller_weights(q)[0, 0].item())
            qids.append(qid)

    def nd(score_fn) -> float:
        return float(np.mean([
            _query_ndcg(pools[q]["pid"], pools[q]["label"], score_fn(q), 10) for q in qids
        ]))

    def best_global(score_fn_of_alpha):
        vals = [(a, nd(lambda q, a=a: score_fn_of_alpha(q, a))) for a in ALPHA_GRID]
        return max(vals, key=lambda t: t[1])

    A = nd(lambda q: alphas[q] * s_sp[q] + (1 - alphas[q]) * s_de[q])
    aB, B = best_global(lambda q, a: a * s_sp[q] + (1 - a) * s_de[q])
    C = nd(lambda q: 0.5 * s_sp[q] + 0.5 * s_de[q])
    D = nd(lambda q: alphas[q] * minmax_normalise(pools[q]["sp"])
           + (1 - alphas[q]) * minmax_normalise(pools[q]["de"]))
    aE, E = best_global(lambda q, a: a * minmax_normalise(pools[q]["sp"])
                        + (1 - a) * minmax_normalise(pools[q]["de"]))

    print(f"\n=== {args.retriever} / {args.dataset}/{args.split}  ({len(qids):,} queries) ===")
    print(f"  A  learned branches + learned alpha      {A:.4f}")
    print(f"  B  learned branches + best global alpha  {B:.4f}  (alpha*={aB:.2f})")
    print(f"  C  learned branches + alpha=0.5          {C:.4f}")
    print(f"  D  raw scores      + learned alpha       {D:.4f}")
    print(f"  E  raw scores      + best global alpha   {E:.4f}  (alpha*={aE:.2f})  [Global-alpha]")
    print()
    print(f"  controller contribution   A - B = {A-B:+.4f}"
          f"   {'the controller beats every global alpha' if A > B else 'NO gain over a global alpha'}")
    print(f"  branch-head contribution  A - D = {A-D:+.4f}")
    print(f"  total over Global-alpha   A - E = {A-E:+.4f}")

    # Ceiling for ANY per-query controller over these same branch heads: pick,
    # per query and post hoc, the alpha that maximises that query's nDCG.
    surf = np.zeros((len(qids), ALPHA_GRID.size))
    for i, q in enumerate(qids):
        for j, a in enumerate(ALPHA_GRID):
            surf[i, j] = _query_ndcg(pools[q]["pid"], pools[q]["label"],
                                     a * s_sp[q] + (1 - a) * s_de[q], 10)
    oracle = float(surf.max(axis=1).mean())
    ties = float((surf >= surf.max(axis=1, keepdims=True) - 1e-12).mean())
    flat = float(np.mean([len(np.unique(r)) == 1 for r in surf]))
    print(f"\n  ORACLE alpha over the SAME learned branches: {oracle:.4f}")
    print(f"    headroom the controller could still win  : {oracle-B:+.4f} (over best global alpha)")
    print(f"    captured by the learned controller       : "
          f"{100*(A-B)/(oracle-B) if oracle > B else float('nan'):.1f}%")
    print(f"    alpha grid tying the per-query max       : {ties:.2f}")
    print(f"    queries where alpha changes nothing      : {flat:.2f}")

    a = np.array([alphas[q] for q in qids])
    allsp = np.concatenate([s_sp[q] for q in qids])
    allde = np.concatenate([s_de[q] for q in qids])
    print(f"\n  predicted alpha_s : mean={a.mean():.3f} sd={a.std():.3f} "
          f"[{np.percentile(a,5):.3f}, {np.percentile(a,95):.3f}]")
    print(f"  branch outputs    : sparse |s| mean={np.abs(allsp).mean():7.2f} max={np.abs(allsp).max():8.2f}")
    print(f"                      dense  |s| mean={np.abs(allde).mean():7.2f} max={np.abs(allde).max():8.2f}")
    print(f"  within-query spread: sparse {np.mean([s_sp[q].std() for q in qids]):.2f}"
          f"   dense {np.mean([s_de[q].std() for q in qids]):.2f}")
    print("  (large magnitudes with the margin at gamma=1 mean the loss can be")
    print("   reduced by inflating scale rather than by improving the ordering)")


if __name__ == "__main__":
    main()
