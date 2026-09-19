"""RQ2 mixes from one GARDIAN forward, scored with the Table 1 metric path.

``scripts/10_paper_run.py`` used to re-run ``evaluate_all_from_rank_data`` once
per ablation. That is metric-identical to Table 1, but it cannot isolate the
paper's claim: ``uniform_alpha`` (0.5/0.5) is a strawman, and there is no
dev-fitted Fixed-α or oracle on the *learned* branch scores.

This module scores the branches and the controller once, then derives every
row from those arrays:

  full              controller mix (GARDIAN)
  uniform_alpha     0.5 / 0.5 on the same branches (diagnostic only)
  fixed_alpha       one alpha fitted on DEV, applied to every test query
  no_sparse_signal  dense branch only
  no_dense_signal   sparse branch only
  oracle_branch     per-query best alpha on the same branches (ceiling)

nDCG uses the same qrels / all-queries convention as
``evaluate_all_from_rank_data`` (missing gold scores 0 and stays in the mean).
Branch mixes are *not* min-maxed: that is what the model itself does at
``fixed_alpha``.
"""

from __future__ import annotations

import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from loguru import logger

from src.baselines.fusion import ALPHA_GRID
from src.evaluation.metrics import ndcg_at_k
from src.evaluation.rank_jsonl_eval import (
    RANK_METRIC_KEYS,
    _aggregate_lists,
    _append_rank_metrics,
    _batch_encode_query_misses,
    _relevance_for_query,
    _resolve_qrels,
    iter_rank_jsonl_records,
)
from src.features.pool_features import pool_features_from_records
from src.model.gardian import GARDIAN

# PubMedQA-Labeled is eval-only. Fit Fixed-α on the sibling collection's
# dev split rather than on the reported split (that would be an oracle).
DEV_FIT_FALLBACK: Dict[str, str] = {
    "pubmedqa_labeled": "pubmedqa_artificial",
}

MIX_NAMES = (
    "full",
    "uniform_alpha",
    "fixed_alpha",
    "no_sparse_signal",
    "no_dense_signal",
    "oracle_branch",
)


def query_emb_cache_for(retriever: str, dataset_name: str, split: str) -> Optional[str]:
    candidates = [
        f"data/query_emb_cache_{retriever}_{dataset_name}_{split}.pkl",
        f"data/query_emb_cache_{retriever}_all.pkl",
        f"data/query_emb_cache_{retriever}_train_all.pkl",
    ]
    for path in candidates:
        if pathlib.Path(path).is_file():
            return path
    return None


def resolve_dev_rank_path(retriever: str, dataset: str) -> Tuple[str, str]:
    """Return ``(rank_jsonl_path, fitted_on_dataset)`` for Fixed-α fitting."""
    from src.common.rank_data_paths import resolve_rank_data_file

    own = resolve_rank_data_file(retriever, dataset, "dev")
    if pathlib.Path(own).is_file():
        return own, dataset
    sibling = DEV_FIT_FALLBACK.get(dataset)
    if sibling:
        cand = resolve_rank_data_file(retriever, sibling, "dev")
        if pathlib.Path(cand).is_file():
            return cand, sibling
    raise FileNotFoundError(
        f"No dev split for {retriever}/{dataset} and no sibling fallback; "
        "refusing to fit Fixed-α on the reported split."
    )


def _metrics_from_score_map(
    pools: Dict[str, Dict[str, Any]],
    scores: Dict[str, np.ndarray],
    qrels,
    *,
    collect_per_query: bool,
) -> Tuple[Dict[str, Any], int]:
    lists: Dict[str, List[float]] = {k: [] for k in RANK_METRIC_KEYS}
    no_positive = 0
    for qid, pool in pools.items():
        sc = scores.get(qid)
        if sc is None or sc.size == 0:
            continue
        # Same (unstable) argsort as evaluate_all_from_rank_data so GARDIAN
        # full matches Table 1 on tied candidates.
        order = np.argsort(np.asarray(sc, dtype=np.float64))[::-1]
        ranked_ids = [pool["candidates"][int(i)]["pid"] for i in order]
        relevant = _relevance_for_query(qid, pool, qrels)
        if not relevant:
            no_positive += 1
        _append_rank_metrics(lists, ranked_ids, relevant)
    block = _aggregate_lists(lists)
    block["mrr@10"] = block["mrr"]
    if collect_per_query:
        block["_per_query"] = lists
    return block, no_positive


def _ndcg10_one(pool: Dict[str, Any], scores: np.ndarray, qrels) -> float:
    order = np.argsort(np.asarray(scores, dtype=np.float64))[::-1]
    ranked = [pool["candidates"][int(i)]["pid"] for i in order]
    relevant = _relevance_for_query(pool["qid"], pool, qrels)
    if not relevant:
        return 0.0
    return float(ndcg_at_k(ranked, relevant, 10))


def _alpha_surface_one(
    s: np.ndarray, d: np.ndarray, pids: List[str], relevant
) -> Tuple[np.ndarray, np.ndarray]:
    """nDCG@10 at every alpha, plus the (n_alpha, N) mix matrix."""
    mixes = ALPHA_GRID[:, None] * s[None, :] + (1.0 - ALPHA_GRID[:, None]) * d[None, :]
    orders = np.argsort(mixes, axis=1)[:, ::-1]
    row = np.empty(ALPHA_GRID.size, dtype=np.float64)
    cache: Dict[bytes, float] = {}
    for j, order in enumerate(orders):
        key = np.asarray(order[:10], dtype=np.int32).tobytes()
        hit = cache.get(key)
        if hit is None:
            ranked = [pids[int(i)] for i in order]
            hit = float(ndcg_at_k(ranked, relevant, 10))
            cache[key] = hit
        row[j] = hit
    return row, mixes


def load_gardian_pools(
    rank_data_path: str,
    *,
    model: GARDIAN,
    query_encoder_name: Optional[str],
    query_encoder_device: str,
    query_emb_cache_path: Optional[str],
    expected_query_feat_dim: Optional[int],
) -> Dict[str, Dict[str, Any]]:
    """Load one rank JSONL into per-query numpy pools with query embeddings."""
    from src.common.query_emb_cache import load_query_emb_store

    grouped: Dict[str, List[dict]] = defaultdict(list)
    pending_q: Dict[str, str] = {}
    prefill: Dict[str, List[float]] = {}
    for rec in iter_rank_jsonl_records(rank_data_path):
        qid = str(rec["qid"])
        grouped[qid].append(rec)
        if qid in prefill:
            continue
        qe = rec.get("query_emb")
        if isinstance(qe, list) and qe:
            prefill[qid] = qe
            continue
        question = rec.get("question")
        if isinstance(question, str) and question.strip():
            pending_q.setdefault(qid, question.strip())

    model_query_dim = int(
        expected_query_feat_dim
        if expected_query_feat_dim is not None
        else model.query_feat_dim
    )
    cache: Dict[str, List[float]] = dict(prefill)
    if query_emb_cache_path:
        store = load_query_emb_store(query_emb_cache_path, expected_dim=model_query_dim)
        hits = 0
        for qid in list(pending_q):
            if qid in cache:
                continue
            vec = store.get(qid)
            if vec is None:
                continue
            cache[qid] = vec.tolist() if hasattr(vec, "tolist") else list(vec)
            hits += 1
        logger.info(
            f"query_emb cache {query_emb_cache_path}: "
            f"{hits:,}/{len(pending_q):,} misses resolved without encoding"
        )
    need_encode = {k: v for k, v in pending_q.items() if k not in cache}
    if need_encode:
        if not query_encoder_name:
            raise ValueError(
                f"{len(need_encode)} queries have no query_emb and no encoder name"
            )
        cache.update(
            _batch_encode_query_misses(
                need_encode,
                query_encoder_name=query_encoder_name,
                query_encoder_device=query_encoder_device,
                model_query_dim=model_query_dim,
            )
        )

    pools: Dict[str, Dict[str, Any]] = {}
    missing_emb = 0
    for qid, recs in grouped.items():
        emb = cache.get(qid)
        if emb is None:
            missing_emb += 1
            continue
        pools[qid] = {
            "qid": qid,
            "candidates": [{"pid": r["pid"], "label": r["label"]} for r in recs],
            "sparse_feats": np.asarray([r["sparse_feats"] for r in recs], dtype=np.float32),
            "dense_feats": np.asarray([r["dense_feats"] for r in recs], dtype=np.float32),
            "query_emb": np.asarray(emb, dtype=np.float32),
            "pool_feats": pool_features_from_records(
                [{"sparse_feats": r["sparse_feats"], "dense_feats": r["dense_feats"]} for r in recs]
            ).astype(np.float32),
        }
    if missing_emb:
        logger.warning(
            f"{missing_emb} queries had no query embedding after cache+encode; skipped"
        )
    logger.info(f"{rank_data_path}: {len(pools):,} queries ready to score")
    return pools


@torch.no_grad()
def score_branches(
    model: GARDIAN,
    pools: Dict[str, Dict[str, Any]],
    device: str,
    *,
    q_per_batch: int = 64,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Return per-query ``(gardian, s_sparse, s_dense)`` arrays."""
    model.eval()
    qids = list(pools.keys())
    gardian: Dict[str, np.ndarray] = {}
    br_s: Dict[str, np.ndarray] = {}
    br_d: Dict[str, np.ndarray] = {}
    for start in range(0, len(qids), q_per_batch):
        chunk = qids[start : start + q_per_batch]
        rows = [pools[q] for q in chunk]
        n_max = max(int(r["sparse_feats"].shape[0]) for r in rows)
        b = len(rows)
        f_s = int(rows[0]["sparse_feats"].shape[1])
        f_d = int(rows[0]["dense_feats"].shape[1])
        sparse = np.zeros((b, n_max, f_s), dtype=np.float32)
        dense = np.zeros((b, n_max, f_d), dtype=np.float32)
        mask = np.zeros((b, n_max), dtype=np.float32)
        qe = np.zeros((b, int(model.query_feat_dim)), dtype=np.float32)
        pf = np.zeros((b, int(model.pool_feat_dim)), dtype=np.float32)
        ns = []
        for i, row in enumerate(rows):
            n = int(row["sparse_feats"].shape[0])
            ns.append(n)
            sparse[i, :n] = row["sparse_feats"]
            dense[i, :n] = row["dense_feats"]
            mask[i, :n] = 1.0
            qe[i] = row["query_emb"]
            pf[i] = row["pool_feats"]
        scores, _, bd = model(
            sparse_feats=torch.from_numpy(sparse).to(device),
            dense_feats=torch.from_numpy(dense).to(device),
            query_emb=torch.from_numpy(qe).to(device),
            pool_feats=torch.from_numpy(pf).to(device),
            mask=torch.from_numpy(mask).to(device),
            return_breakdown=True,
        )
        sc = scores.float().cpu().numpy()
        ss = bd["s_sparse"].float().cpu().numpy()
        sd = bd["s_dense"].float().cpu().numpy()
        for i, qid in enumerate(chunk):
            n = ns[i]
            gardian[qid] = np.asarray(sc[i, :n], dtype=np.float64)
            br_s[qid] = np.asarray(ss[i, :n], dtype=np.float64)
            br_d[qid] = np.asarray(sd[i, :n], dtype=np.float64)
    return gardian, br_s, br_d


def fit_fixed_alpha(
    pools: Dict[str, Dict[str, Any]],
    sparse_sc: Dict[str, np.ndarray],
    dense_sc: Dict[str, np.ndarray],
    qrels,
) -> Tuple[float, float]:
    """Best single alpha on DEV, all-queries nDCG@10 (zeros do not move argmax)."""
    qids = [q for q in pools if q in sparse_sc and q in dense_sc]
    if not qids:
        return 0.5, 0.0
    surf = np.zeros((len(qids), ALPHA_GRID.size), dtype=np.float64)
    for i, qid in enumerate(qids):
        pool = pools[qid]
        relevant = _relevance_for_query(qid, pool, qrels)
        if not relevant:
            continue
        pids = [c["pid"] for c in pool["candidates"]]
        row, _ = _alpha_surface_one(sparse_sc[qid], dense_sc[qid], pids, relevant)
        surf[i] = row
    means = surf.mean(axis=0)
    j = int(np.argmax(means))
    return float(ALPHA_GRID[j]), float(means[j])


def oracle_branch_scores(
    pools: Dict[str, Dict[str, Any]],
    sparse_sc: Dict[str, np.ndarray],
    dense_sc: Dict[str, np.ndarray],
    qrels,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Per-query best mix of learned branches. Ceiling, not a competitor."""
    out: Dict[str, np.ndarray] = {}
    best_ndcg: List[float] = []
    ties: List[float] = []
    invariant: List[bool] = []
    n_gold = 0
    for qid, pool in pools.items():
        if qid not in sparse_sc:
            continue
        s, d = sparse_sc[qid], dense_sc[qid]
        relevant = _relevance_for_query(qid, pool, qrels)
        if not relevant:
            out[qid] = 0.5 * s + 0.5 * d
            best_ndcg.append(0.0)
            continue
        n_gold += 1
        pids = [c["pid"] for c in pool["candidates"]]
        row, mixes = _alpha_surface_one(s, d, pids, relevant)
        j_star = int(np.argmax(row))
        out[qid] = mixes[j_star]
        best_ndcg.append(float(row[j_star]))
        mx = float(row.max())
        ties.append(float(np.mean(row >= mx - 1e-12)))
        invariant.append(bool(len(np.unique(np.round(row, 12))) == 1))
    diag = {
        "oracle_ndcg@10": float(np.mean(best_ndcg)) if best_ndcg else 0.0,
        "tie_fraction": float(np.mean(ties)) if ties else None,
        "invariant_fraction": float(np.mean(invariant)) if invariant else None,
        "n_queries": len(best_ndcg),
        "n_queries_with_gold": n_gold,
    }
    return out, diag


def evaluate_gardian_mixes(
    rank_data_path: str,
    model: GARDIAN,
    device: str,
    *,
    mix_names: Sequence[str],
    fixed_alpha: Optional[float] = None,
    query_encoder_name: Optional[str] = None,
    query_encoder_device: str = "cpu",
    query_emb_cache_path: Optional[str] = None,
    expected_query_feat_dim: Optional[int] = None,
    collect_per_query: bool = True,
    q_per_batch: int = 64,
    precomputed: Optional[
        Tuple[Dict[str, Dict[str, Any]], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]
    ] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    One forward, then every requested mix scored like Table 1.

    Returns ``{mix_name: {gardian: metrics, _meta: ...}}``.
    """
    unknown = [n for n in mix_names if n not in MIX_NAMES]
    if unknown:
        raise ValueError(f"Unknown mix(es) {unknown}; expected one of {MIX_NAMES}")
    if "fixed_alpha" in mix_names and fixed_alpha is None:
        raise ValueError("mix 'fixed_alpha' needs a dev-fitted fixed_alpha value")

    if precomputed is None:
        pools = load_gardian_pools(
            rank_data_path,
            model=model,
            query_encoder_name=query_encoder_name,
            query_encoder_device=query_encoder_device,
            query_emb_cache_path=query_emb_cache_path,
            expected_query_feat_dim=expected_query_feat_dim,
        )
        logger.info(f"  scoring GARDIAN branches on {len(pools):,} queries")
        gardian, br_s, br_d = score_branches(
            model, pools, device, q_per_batch=q_per_batch
        )
    else:
        pools, gardian, br_s, br_d = precomputed

    qrels, qrels_complete = _resolve_qrels(pools.keys())
    meta_base = {
        "rank_data_path": rank_data_path,
        "query_count": len(pools),
        "qrels": {
            "source": "full_graded" if qrels else "pool_labels",
            "queries_judged": len(qrels),
            "complete_coverage": bool(qrels_complete),
        },
        "scored_on_full_pool": True,
        "one_forward_mixes": True,
    }

    mixes: Dict[str, Dict[str, np.ndarray]] = {}
    extra_meta: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for name in mix_names:
        if name == "full":
            mixes[name] = gardian
        elif name == "uniform_alpha":
            mixes[name] = {q: 0.5 * br_s[q] + 0.5 * br_d[q] for q in br_s}
        elif name == "fixed_alpha":
            a = float(fixed_alpha)
            mixes[name] = {q: a * br_s[q] + (1.0 - a) * br_d[q] for q in br_s}
            extra_meta[name]["fixed_alpha"] = a
        elif name == "no_sparse_signal":
            mixes[name] = br_d
        elif name == "no_dense_signal":
            mixes[name] = br_s
        elif name == "oracle_branch":
            scores, diag = oracle_branch_scores(pools, br_s, br_d, qrels)
            mixes[name] = scores
            extra_meta[name]["oracle_branch"] = diag
            extra_meta[name]["oracle_valid_cutoff"] = 10

    out: Dict[str, Dict[str, Any]] = {}
    for name, score_map in mixes.items():
        block, no_pos = _metrics_from_score_map(
            pools, score_map, qrels, collect_per_query=collect_per_query
        )
        meta = dict(meta_base)
        meta["gardian_ablation"] = None if name == "full" else name
        meta["gardian_no_positive_queries"] = no_pos
        meta.update(extra_meta.get(name, {}))
        out[name] = {"gardian": block, "_meta": meta}
    return out


def score_split_for_fit(
    rank_data_path: str,
    model: GARDIAN,
    device: str,
    **load_kw: Any,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load + score one split; used to fit Fixed-α on DEV."""
    pools = load_gardian_pools(rank_data_path, model=model, **load_kw)
    _, br_s, br_d = score_branches(model, pools, device)
    return pools, br_s, br_d
