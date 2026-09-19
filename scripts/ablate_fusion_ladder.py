"""
The fusion isolation ladder: what does each component actually buy?

Every rung changes exactly one thing from the rung above, on the identical
candidate pool and the identical query population:

  1  sum-unnorm            raw scores, unnormalised sum        the floor
  2  RRF (k=60)            raw scores, rank fusion             the usual floor
  3  weighted RRF          rank fusion, weight and k dev-tuned value of tuning ranks
  4  global-alpha          raw scores, alpha dev-tuned         value of tuning at all
  5  branches + global-a   LEARNED branches, alpha dev-tuned   value of learned scoring
  6  GARDIAN               learned branches, per-query alpha   VALUE OF ADAPTIVITY
  7  [ceiling] oracle-a    learned branches, per-query best a  headroom for adaptivity

Row 6 minus row 5 is the paper's central claim expressed as one number, and
row 7 minus row 5 is the most that claim could ever have been worth.

Why not 0.5/0.5 as the fixed-weight control
-------------------------------------------
The ``uniform_alpha`` ablation replaces the controller with 0.5/0.5. That is an
arbitrary constant, so beating it conflates "adaptivity helps" with "0.5 is a
bad constant". Row 5 uses the best *single* alpha over the whole dev split,
applied unchanged to every test query, over the same learned branch outputs.
That is the control the claim actually needs.

Every tuned quantity here is fitted on DEV and applied to TEST. The oracle row
is fitted on the test labels by definition and is reported as a ceiling.

Usage
-----
    .venv/bin/python scripts/ablate_fusion_ladder.py \
        --retriever hybrid_bm25_faiss --dataset medmcqa --seed 42
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.common.query_emb_cache import load_query_emb_store          # noqa: E402
from src.features.pool_features import pool_features                 # noqa: E402
from src.evaluation.qrels import qrels_for_qids                       # noqa: E402
from src.common.rank_data_paths import resolve_rank_data_file        # noqa: E402
from src.common.seeds import DEFAULT_SEEDS                           # noqa: E402
from src.evaluation.metrics import ndcg_at_k                         # noqa: E402
from src.evaluation.rank_jsonl_eval import iter_rank_jsonl_records   # noqa: E402
from src.features.schema import select_active                       # noqa: E402
from src.model.gardian import (                                      # noqa: E402
    build_gardian_from_model_cfg,
    dropped_features_from_cfg,
    load_checkpoint_state,
)

ALPHA_GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)
RRF_K_GRID = (10, 20, 40, 60, 100, 200)


# --------------------------------------------------------------------------
def load_pools(path: str, dropped: List[str]) -> Dict[str, Dict[str, Any]]:
    """qid -> pool of labels, raw channel scores and active feature vectors."""
    pools: Dict[str, Dict[str, Any]] = {}
    for rec in iter_rank_jsonl_records(path):
        p = pools.setdefault(
            rec["qid"],
            {"label": [], "sparse": [], "dense": [], "sf": [], "df": [],
             "pid": [], "s_ok": [], "d_ok": []},
        )
        sf, df = select_active(rec["sparse_feats"], rec["dense_feats"], dropped)
        p["label"].append(int(rec.get("label", 0)))
        p["sparse"].append(float(rec["sparse_feats"][0]))
        p["dense"].append(float(rec["dense_feats"][0]))
        p["sf"].append(sf)
        p["df"].append(df)
        # Kept for scoring against full graded qrels, and for the scale-free
        # pool features the controller reads. Indices 0/7 are the raw score and
        # the channel-retrieval indicator in the *stored* schema, which always
        # has all 8 columns regardless of ``dropped``.
        p["pid"].append(rec["pid"])
        p["s_ok"].append(float(rec["sparse_feats"][7]))
        p["d_ok"].append(float(rec["dense_feats"][7]))

    # Attach the collection's FULL graded judgments to each pool, once. Every
    # nDCG in this script then normalises against every judged document rather
    # than against the positives that happen to be in the pool -- on TREC-COVID
    # that is 493.5 judged per topic against a ~92-candidate pool, so the
    # pool-relative version is not comparable to any published number.
    try:
        qrels = qrels_for_qids(pools.keys())
    except (OSError, ValueError) as exc:
        logger.warning(f"No qrels available ({exc}); falling back to in-pool binary labels")
        qrels = {}
    n_judged = 0
    for qid, pool in pools.items():
        pool["judged"] = qrels.get(str(qid)) or {}
        n_judged += bool(pool["judged"])
    logger.info(
        f"{path}: {len(pools):,} pools | {n_judged:,} with full graded qrels"
        + ("" if n_judged == len(pools) else "  (rest use in-pool binary labels)")
    )
    return pools


def pool_ndcg(p, scores, k: int) -> float:
    """nDCG@k for one pool, graded against full qrels when available."""
    return ndcg(scores, p["label"], k, pids=p.get("pid"), judged=p.get("judged"))


def scorable(p) -> bool:
    """
    Whether a pool contributes to a mean.

    A query with judgments contributes even when its pool holds no positive:
    its nDCG is legitimately (near) zero, and dropping such queries would
    silently raise every system's mean by removing its hardest topics.
    """
    return bool(p.get("judged")) or sum(p["label"]) > 0


def ndcg(scores, labels, k: int, *, pids=None, judged=None) -> float:
    """
    nDCG@k for one pool.

    With ``pids`` and ``judged`` (a pid -> grade map from the collection's full
    qrels) this is graded nDCG normalised against every judged document, which
    is what published numbers report. Without them it falls back to in-pool
    binary labels, whose IDCG counts only the positives retrieval already
    found -- flattering and not comparable across pool sizes.
    """
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    if pids is not None and judged:
        return ndcg_at_k([pids[i] for i in order], judged, k)
    ranked = [str(i) for i in order]
    relevant = {str(i) for i, l in enumerate(labels) if l == 1}
    return ndcg_at_k(ranked, relevant, k)


def minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def rank_positions(v: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(-v, kind="stable"), kind="stable")


def mean_over(pools, score_fn, k: int) -> float:
    vals = [pool_ndcg(p, score_fn(p), k) for p in pools.values() if scorable(p)]
    return float(np.mean(vals)) if vals else 0.0


# --------------------------------------------------------------------------
def _pool_tensors(p, e, device: str):
    """
    One pool as grouped ``(1, N, F)`` model inputs plus its pool features.

    Grouped rather than flat because the model may standardise each branch
    within its pool and the controller may read pool-level evidence; both are
    per-query quantities that a flat batch cannot express.
    """
    sf = torch.tensor(np.asarray(p["sf"], np.float32), device=device).unsqueeze(0)
    df = torch.tensor(np.asarray(p["df"], np.float32), device=device).unsqueeze(0)
    qe = torch.tensor(np.asarray(e, np.float32), device=device).unsqueeze(0)
    pf = torch.from_numpy(
        pool_features(
            np.asarray(p["sparse"], np.float64),
            np.asarray(p["dense"], np.float64),
            np.asarray(p["s_ok"], np.float64),
            np.asarray(p["d_ok"], np.float64),
        )
    ).unsqueeze(0).to(device)
    mk = torch.ones(sf.shape[:2], dtype=torch.float32, device=device)
    return sf, df, qe, pf, mk


@torch.no_grad()
def branch_outputs(model, pools, qemb, device: str, batch: int = 8192):
    """s_sparse / s_dense per candidate -- the model's learned channel scores."""
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    missing = 0
    for qid, p in pools.items():
        e = qemb.get(qid)
        if e is None:
            missing += 1
            continue
        sf, df, qe, pf, mk = _pool_tensors(p, e, device)
        _, _, bd = model(sf, df, qe, pool_feats=pf, mask=mk, return_breakdown=True)
        out[qid] = (
            bd["s_sparse"].squeeze(0).cpu().numpy(),
            bd["s_dense"].squeeze(0).cpu().numpy(),
        )
    if missing:
        logger.warning(f"{missing} queries had no cached query embedding; skipped")
    return out


@torch.no_grad()
def gardian_scores(model, pools, qemb, device: str) -> Dict[str, np.ndarray]:
    out = {}
    for qid, p in pools.items():
        e = qemb.get(qid)
        if e is None:
            continue
        sf, df, qe, pf, mk = _pool_tensors(p, e, device)
        s, _ = model(sf, df, qe, pool_feats=pf, mask=mk)
        out[qid] = s.squeeze(0).cpu().numpy()
    return out


def crossval_alpha_over(pools, per_query_channels, k: int, folds: int = 5, seed: int = 42):
    """
    Per-query alpha by k-fold CV, for collections with no development split.

    Each query is scored with an alpha fitted on the folds that exclude it, so
    no query's weight uses its own labels. Returns ``(per_query_alpha, mean
    held-out nDCG)``. This makes rows 4 and 5 of the ladder available on
    TREC-COVID, which has no dev split and no sibling collection to borrow one
    from.
    """
    import random as _random

    qids = sorted(q for q, p in pools.items()
                  if scorable(p) and q in per_query_channels)
    if not qids:
        return {}, 0.0
    rng = _random.Random(seed)
    rng.shuffle(qids)
    n_folds = max(2, min(int(folds), len(qids)))

    out, scores = {}, []
    for f in range(n_folds):
        held = qids[f::n_folds]
        train = {q: pools[q] for q in qids if q not in set(held)}
        if not train:
            continue
        a, _ = fit_alpha_over(train, per_query_channels, k)
        for q in held:
            out[q] = float(a)
            s_, d_ = per_query_channels[q]
            scores.append(pool_ndcg(pools[q], a * s_ + (1 - a) * d_, k))
    return out, float(np.mean(scores)) if scores else 0.0


def fit_alpha_over(pools, per_query_channels, k: int) -> Tuple[float, float]:
    """Best single alpha over supplied (sparse, dense) score pairs, on dev."""
    qids = [q for q, p in pools.items() if scorable(p) and q in per_query_channels]
    if not qids:
        return 0.5, 0.0
    surf = np.zeros((len(qids), ALPHA_GRID.size))
    for i, q in enumerate(qids):
        s, d = per_query_channels[q]
        for j, a in enumerate(ALPHA_GRID):
            surf[i, j] = pool_ndcg(pools[q], a * s + (1 - a) * d, k)
    means = surf.mean(axis=0)
    j = int(np.argmax(means))
    return float(ALPHA_GRID[j]), float(means[j])


def oracle_alpha_over(pools, per_query_channels, k: int) -> float:
    qids = [q for q, p in pools.items() if scorable(p) and q in per_query_channels]
    best = []
    for q in qids:
        s, d = per_query_channels[q]
        best.append(max(pool_ndcg(pools[q], a * s + (1 - a) * d, k) for a in ALPHA_GRID))
    return float(np.mean(best)) if best else 0.0


def fit_weighted_rrf(pools, k: int) -> Tuple[float, int, float]:
    """Grid-search the RRF channel weight and k on dev."""
    best = (0.5, 60, -1.0)
    for kk in RRF_K_GRID:
        for w in ALPHA_GRID[::5]:
            def fn(p, kk=kk, w=w):
                rs = rank_positions(np.asarray(p["sparse"]))
                rd = rank_positions(np.asarray(p["dense"]))
                return w / (kk + rs + 1) + (1 - w) / (kk + rd + 1)
            v = mean_over(pools, fn, k)
            if v > best[2]:
                best = (float(w), int(kk), v)
    return best


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--retriever", default="hybrid_bm25_faiss")
    ap.add_argument("--dataset", default="medmcqa")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEEDS[2])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5,
                    help="CV folds used when the dataset has no dev split.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="write the ladder as JSON here")
    args = ap.parse_args()

    cfg = OmegaConf.load("configs/base.yaml")
    dropped = dropped_features_from_cfg(cfg.model)

    test_path = resolve_rank_data_file(args.retriever, args.dataset, "test")
    if not pathlib.Path(test_path).is_file():
        raise SystemExit(f"missing rank data: {test_path}")
    dev_path = resolve_rank_data_file(args.retriever, args.dataset, "dev")
    use_cv = not pathlib.Path(dev_path).is_file()
    if use_cv:
        logger.info(
            f"{args.dataset} has no dev split; fitting every tuned quantity by "
            f"{args.folds}-fold cross-validation within the test split "
            "(no query's alpha uses its own labels)"
        )

    ckpt = pathlib.Path(f"results/seeds/seed_{args.seed}/gardian_best_{args.retriever}.pt")
    if not ckpt.is_file():
        raise SystemExit(f"missing checkpoint: {ckpt}")

    logger.info(f"loading test pools: {test_path}")
    test = load_pools(test_path, dropped)
    if use_cv:
        dev = test
    else:
        logger.info(f"loading dev pools: {dev_path}")
        dev = load_pools(dev_path, dropped)
    logger.info(f"dev={len(dev):,} test={len(test):,} queries")

    model = build_gardian_from_model_cfg(cfg.model)
    state = torch.load(ckpt, map_location="cpu")
    load_checkpoint_state(model, state["model_state"], strict=True)
    model.eval().to(args.device)
    logger.info(f"checkpoint epoch={state.get('epoch')} dev nDCG@10={state.get('best_ndcg10'):.4f}")

    qemb = {}
    for name in (
        f"data/query_emb_cache_{args.retriever}_all.pkl",
        f"data/query_emb_cache_{args.retriever}_{args.dataset}_dev.pkl",
        f"data/query_emb_cache_{args.retriever}_{args.dataset}_test.pkl",
    ):
        if pathlib.Path(name).is_file():
            store = load_query_emb_store(name, expected_dim=int(model.query_feat_dim))
            for q in list(dev) + list(test):
                if q not in qemb:
                    v = store.get(q)
                    if v is not None:
                        qemb[q] = v
    logger.info(f"query embeddings available for {len(qemb):,} queries")

    K = args.k
    raw = lambda p: (minmax(np.asarray(p["sparse"])), minmax(np.asarray(p["dense"])))  # noqa: E731
    dev_raw = {q: raw(p) for q, p in dev.items()}
    test_raw = {q: raw(p) for q, p in test.items()}

    rows: List[Tuple[str, float, str]] = []

    rows.append(("1  sum-unnorm",
                 mean_over(test, lambda p: np.asarray(p["sparse"]) + np.asarray(p["dense"]), K),
                 "raw scores, unnormalised"))

    rows.append(("2  RRF (k=60)",
                 mean_over(test, lambda p: 1 / (60 + rank_positions(np.asarray(p["sparse"])) + 1)
                                          + 1 / (60 + rank_positions(np.asarray(p["dense"])) + 1), K),
                 "untuned rank fusion"))

    w, kk, dev_v = fit_weighted_rrf(dev, K)
    rows.append((f"3  weighted RRF (w={w:.2f}, k={kk})",
                 mean_over(test, lambda p: w / (kk + rank_positions(np.asarray(p["sparse"])) + 1)
                                          + (1 - w) / (kk + rank_positions(np.asarray(p["dense"])) + 1), K),
                 f"dev-tuned (dev={dev_v:.4f})"))

    a_br = None
    if use_cv:
        cv_raw, row4 = crossval_alpha_over(test, test_raw, K, args.folds)
        alphas = sorted({round(a, 2) for a in cv_raw.values()})
        a_raw = float(np.mean(list(cv_raw.values()))) if cv_raw else 0.5
        rows.append((f"4  global-alpha (CV, a~{alphas})", row4,
                     f"{args.folds}-fold CV on raw scores"))
    else:
        a_raw, dev_a = fit_alpha_over(dev, dev_raw, K)
        row4 = mean_over(test, lambda p: a_raw * minmax(np.asarray(p["sparse"]))
                                        + (1 - a_raw) * minmax(np.asarray(p["dense"])), K)
        rows.append((f"4  global-alpha (a={a_raw:.2f})", row4,
                     f"dev-tuned on raw scores (dev={dev_a:.4f})"))

    logger.info("scoring learned branches on dev/test ...")
    dev_br = branch_outputs(model, dev, qemb, args.device)
    test_br = branch_outputs(model, test, qemb, args.device)

    if use_cv:
        cv_br, row5 = crossval_alpha_over(test, test_br, K, args.folds)
        br_alphas = sorted({round(a, 2) for a in cv_br.values()})
        a_br = float(np.mean(list(cv_br.values()))) if cv_br else None
        rows.append((f"5  branches + global-alpha (CV, a~{br_alphas})", row5,
                     f"LEARNED branches, {args.folds}-fold CV alpha"))
    else:
        a_br, dev_ab = fit_alpha_over(dev, dev_br, K)
        row5 = float(np.mean([
            pool_ndcg(test[q], a_br * test_br[q][0] + (1 - a_br) * test_br[q][1], K)
            for q in test_br if scorable(test[q])
        ]))
        rows.append((f"5  branches + global-alpha (a={a_br:.2f})", row5,
                     f"LEARNED branches, dev-tuned alpha (dev={dev_ab:.4f})"))

    gs = gardian_scores(model, test, qemb, args.device)
    vals = [pool_ndcg(test[q], gs[q], K) for q in gs if scorable(test[q])]
    row6 = float(np.mean(vals))
    rows.append(("6  GARDIAN (per-query alpha)", row6, "learned branches + controller"))

    row7 = oracle_alpha_over(test, test_br, K)
    rows.append(("7  [ceiling] oracle-alpha", row7, "per-query best alpha, uses TEST labels"))

    print(f"\n{'=' * 92}")
    print(f"FUSION LADDER | {args.retriever} / {args.dataset} test | seed {args.seed} | nDCG@{K}")
    print(f"{'=' * 92}")
    print(f"{'variant':<42}{'nDCG':>9}   {'note'}")
    print("-" * 92)
    for name, v, note in rows:
        print(f"{name:<42}{v:>9.4f}   {note}")
    print("-" * 92)
    print(f"  value of ADAPTIVITY      (6 - 5) = {row6 - row5:+.4f}")
    print(f"  headroom for adaptivity  (7 - 5) = {row7 - row5:+.4f}")
    frac = (row6 - row5) / (row7 - row5) if row7 > row5 else float("nan")
    print(f"  fraction of headroom captured    = {frac * 100:.1f}%")
    print(f"  value of LEARNED BRANCHES(5 - 4) = {row5 - row4:+.4f}")
    print(f"{'=' * 92}\n")

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "retriever": args.retriever, "dataset": args.dataset, "seed": args.seed,
            "k": K, "rows": [{"variant": n, "ndcg": v, "note": c} for n, v, c in rows],
            "adaptivity_gain": row6 - row5, "adaptivity_headroom": row7 - row5,
            "alpha_raw": a_raw, "alpha_branches": a_br,
            "alpha_fitted_by": ("crossval" if use_cv else "dev"),
            "rrf_weight": w, "rrf_k": kk,
        }, indent=2))
        logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
