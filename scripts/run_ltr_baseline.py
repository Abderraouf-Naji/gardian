#!/usr/bin/env python3
"""
LambdaMART control: is a neural re-ranker necessary?

GARDIAN's branches are MLPs over 16 tabular features. A gradient-boosted ranker
consumes exactly those features, so it answers the obvious question directly --
if trees on the same inputs match the neural model, the architecture is not
carrying the result and the paper should say so. ``docs/OBJECTIVE.md`` reports
0.5092 (MedMCQA) / 0.8065 (PubMedQA-artificial) for this baseline against
0.5057 / 0.8015 for GARDIAN, i.e. the trees are *slightly ahead*. This script
is what produces those numbers; the original was never committed.

Fairness conditions, all enforced here rather than assumed:

* **Identical features.** The same 8 sparse + 8 dense columns the model reads,
  through the same ``src.features.schema`` selection, so a ``dropped_features``
  setting applies to both arms.
* **Identical pools.** The same rank JSONL, hence the same candidates from the
  same first-stage retrieval.
* **Identical evaluation.** The same qrels resolution and the same metric
  implementations as ``src.evaluation.rank_jsonl_eval``, so a number here is
  comparable to a number there without re-deriving anything.

It also records what RQ4 needs on the cost axis: training wall clock, model
size on disk, and per-query inference latency on the same pools.

Usage
-----
    python scripts/run_ltr_baseline.py --retriever hybrid_bm25_faiss
    python scripts/run_ltr_baseline.py --retriever all --max-train-queries 50000
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.common.hybrid_retrievers import FOCUS_HYBRID_RETRIEVERS
from src.common.rank_data_paths import (
    normalize_retriever_name,
    rank_data_combined_file,
    resolve_rank_data_file,
)
from src.evaluation.metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k
from src.evaluation.qrels import Qrels, qrels_for_qids
from src.features.schema import active_feature_indices

EVAL_SPLITS: List[Tuple[str, str]] = [
    ("pubmedqa_labeled", "eval"),
    ("medmcqa", "test"),
    ("pubmedqa_artificial", "test"),
]

METRIC_KEYS = ("ndcg@5", "ndcg@10", "ndcg@20", "mrr", "recall@10", "recall@20", "hit@10")

# Rows per preallocated block when streaming a large rank JSONL.
_CHUNK_ROWS = 500_000


def _iter_rows(path: pathlib.Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_feature_matrix(
    path: pathlib.Path,
    *,
    dropped: List[str],
    max_queries: Optional[int] = None,
    collect_ids: bool = True,
    log_every: int = 2_000_000,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str], List[List[str]], List[List[int]]]:
    """
    Stream rank JSONL into a dense float32 matrix.

    Streaming matters: the combined training file is ~16 GB of JSON, but the
    numeric payload is 16 floats per row, so the matrix fits comfortably in RAM
    while the parsed objects would not.

    Returns ``(X, y, group_sizes, qids, pids_per_query, labels_per_query)``.
    """
    # The combined training file is ~16M rows. One small ndarray per row costs
    # more in object overhead than the floats themselves, so rows are filled
    # into preallocated (CHUNK, 16) blocks and concatenated once at the end.
    chunk = np.empty((_CHUNK_ROWS, 16), dtype=np.float32)
    chunk_labels = np.empty(_CHUNK_ROWS, dtype=np.float32)
    chunk_fill = 0
    blocks: List[np.ndarray] = []
    label_blocks: List[np.ndarray] = []

    group_sizes: List[int] = []
    qids: List[str] = []
    pids_per_query: List[List[str]] = []
    labels_per_query: List[List[int]] = []

    current_qid: Optional[str] = None
    current_n = 0
    seen: set[str] = set()
    n_rows = 0

    def _close_group() -> None:
        nonlocal current_n
        if current_qid is not None and current_n > 0:
            group_sizes.append(current_n)
        current_n = 0

    def _flush() -> None:
        nonlocal chunk_fill
        if chunk_fill:
            blocks.append(chunk[:chunk_fill].copy())
            label_blocks.append(chunk_labels[:chunk_fill].copy())
            chunk_fill = 0

    for rec in _iter_rows(path):
        qid = str(rec["qid"])
        if qid != current_qid:
            _close_group()
            if qid in seen:
                raise ValueError(
                    f"Rank JSONL is not grouped by qid: {qid!r} reappears after other "
                    "queries. LambdaMART needs contiguous groups; regenerate the file "
                    "with scripts/03_generate_rank_data.py."
                )
            seen.add(qid)
            if max_queries is not None and len(qids) >= max_queries:
                break
            current_qid = qid
            qids.append(qid)
            if collect_ids:
                pids_per_query.append([])
                labels_per_query.append([])

        if chunk_fill == _CHUNK_ROWS:
            _flush()
        chunk[chunk_fill, :8] = rec["sparse_feats"]
        chunk[chunk_fill, 8:] = rec["dense_feats"]
        label = int(rec["label"])
        chunk_labels[chunk_fill] = float(label)
        chunk_fill += 1
        if collect_ids:
            pids_per_query[-1].append(str(rec["pid"]))
            labels_per_query[-1].append(label)
        current_n += 1
        n_rows += 1
        if n_rows % log_every == 0:
            logger.info(f"    {n_rows:,} rows / {len(qids):,} queries parsed")

    _close_group()
    _flush()

    X = np.vstack(blocks) if blocks else np.zeros((0, 16), dtype=np.float32)
    y = np.concatenate(label_blocks) if label_blocks else np.zeros(0, dtype=np.float32)
    del blocks, label_blocks, chunk, chunk_labels

    keep = _active_columns(dropped)
    if keep is not None:
        X = X[:, keep]

    logger.info(
        f"  Loaded {X.shape[0]:,} rows x {X.shape[1]} features / {len(group_sizes):,} queries"
    )
    return X, y, group_sizes, qids, pids_per_query, labels_per_query


def _active_columns(dropped: List[str]) -> Optional[np.ndarray]:
    """Column indices surviving ``model.dropped_features``, or None for all 16."""
    if not dropped:
        return None
    sparse_idx, dense_idx = active_feature_indices(dropped)
    cols = list(sparse_idx) + [8 + i for i in dense_idx]
    return np.asarray(cols, dtype=np.int64)


def _relevance(qid: str, pids: List[str], labels: List[int], qrels: Qrels):
    """Graded qrels when judged, else in-pool positives -- as rank_jsonl_eval does."""
    judged = qrels.get(str(qid))
    if judged:
        return judged
    return [p for p, lab in zip(pids, labels) if int(lab) >= 1]


def evaluate_scores(
    scores: np.ndarray,
    *,
    group_sizes: List[int],
    qids: List[str],
    pids_per_query: List[List[str]],
    labels_per_query: List[List[int]],
) -> Dict[str, float]:
    """Rank each pool by score and average the same metrics GARDIAN reports."""
    try:
        qrels = qrels_for_qids([str(q) for q in qids])
    except (OSError, ValueError) as exc:
        logger.warning(f"Could not load qrels ({exc}); using pool-relative labels")
        qrels = {}
    missing = sum(1 for q in qids if str(q) not in qrels)
    if missing:
        logger.warning(
            f"  No qrels for {missing}/{len(qids)} queries; those fall back to "
            "pool-relative labels (same fallback as rank_jsonl_eval)"
        )

    acc: Dict[str, List[float]] = defaultdict(list)
    offset = 0
    for i, n in enumerate(group_sizes):
        s = scores[offset : offset + n]
        offset += n
        order = np.argsort(s)[::-1]
        ranked = [pids_per_query[i][j] for j in order]
        rel = _relevance(qids[i], pids_per_query[i], labels_per_query[i], qrels)
        acc["ndcg@5"].append(ndcg_at_k(ranked, rel, 5))
        acc["ndcg@10"].append(ndcg_at_k(ranked, rel, 10))
        acc["ndcg@20"].append(ndcg_at_k(ranked, rel, 20))
        acc["mrr"].append(mrr(ranked, rel))
        acc["recall@10"].append(recall_at_k(ranked, rel, 10))
        acc["recall@20"].append(recall_at_k(ranked, rel, 20))
        acc["hit@10"].append(hit_at_k(ranked, rel, 10))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in acc.items()}


def measure_inference_latency(
    booster: Any,
    X: np.ndarray,
    group_sizes: List[int],
    *,
    n_queries: int,
    warmup: int,
) -> Dict[str, float]:
    """
    Per-query scoring cost, pool by pool.

    Timed the same way as the neural arm (one pool per call) so the two are
    comparable. There is no query encoder here: LambdaMART reads only the
    tabular features, which is exactly why it is cheap -- and exactly what the
    controller buys nothing for.
    """
    samples: List[float] = []
    offset = 0
    bounds: List[Tuple[int, int]] = []
    for n in group_sizes:
        bounds.append((offset, offset + n))
        offset += n
    bounds = bounds[: n_queries + warmup]

    for i, (lo, hi) in enumerate(bounds):
        pool = X[lo:hi]
        t0 = time.perf_counter()
        s = booster.predict(pool)
        np.argsort(np.asarray(s))[::-1]
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            samples.append(dt)
    if not samples:
        return {"p50_ms": 0.0, "mean_ms": 0.0, "n": 0}
    arr = np.asarray(samples)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "mean_ms": float(np.mean(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "n": int(arr.size),
    }


def run_for_retriever(retriever: str, args: argparse.Namespace, cfg: Any) -> Dict[str, Any]:
    import lightgbm as lgb

    dropped = list(cfg.model.get("dropped_features") or [])
    results_dir = pathlib.Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_path = pathlib.Path(rank_data_combined_file(retriever, "train_all"))
    dev_path = pathlib.Path(rank_data_combined_file(retriever, "dev_all"))
    if not train_path.is_file():
        raise FileNotFoundError(f"Training rank data missing: {train_path}")

    if args.load_model:
        booster_path = pathlib.Path(args.load_model)
        if not booster_path.is_file():
            raise FileNotFoundError(f"No booster at {booster_path}")
        logger.info(f"Loading trained booster: {booster_path} (skipping training)")
        booster = lgb.Booster(model_file=str(booster_path))

        class _Wrap:
            """Minimal stand-in exposing the two attributes the eval path uses."""

            def __init__(self, b):
                self.booster_ = b
                self.best_iteration_ = b.best_iteration or b.num_trees()

            def predict(self, X):
                return self.booster_.predict(X)

        ranker = _Wrap(booster)
        train_sec = float("nan")
        gtr: List[int] = []
        model_path = booster_path
        model_bytes = booster_path.stat().st_size
    else:
        logger.info(f"Loading training pools: {train_path}")
        Xtr, ytr, gtr, _qtr, _ptr, _ltr = load_feature_matrix(
            train_path,
            dropped=dropped,
            max_queries=args.max_train_queries,
            collect_ids=False,
        )

        Xdev = ydev = None
        gdev: List[int] = []
        if dev_path.is_file() and not args.no_early_stopping:
            logger.info(f"Loading dev pools: {dev_path}")
            Xdev, ydev, gdev, _qd, _pd, _ld = load_feature_matrix(
                dev_path,
                dropped=dropped,
                max_queries=args.max_dev_queries,
                collect_ids=False,
            )

        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[10],
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_child_samples=args.min_child_samples,
            subsample=args.subsample,
            subsample_freq=1,
            colsample_bytree=args.colsample_bytree,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            verbose=-1,
        )

        fit_kwargs: Dict[str, Any] = {"group": gtr}
        callbacks = [lgb.log_evaluation(period=args.log_period)]
        if Xdev is not None and gdev:
            fit_kwargs["eval_set"] = [(Xdev, ydev)]
            fit_kwargs["eval_group"] = [gdev]
            callbacks.append(lgb.early_stopping(args.early_stopping_rounds, verbose=True))

        logger.info(
            f"Training LambdaMART | {Xtr.shape[0]:,} rows | {len(gtr):,} queries | "
            f"{Xtr.shape[1]} features | {args.n_estimators} trees max"
        )
        t0 = time.perf_counter()
        ranker.fit(Xtr, ytr, callbacks=callbacks, **fit_kwargs)
        train_sec = time.perf_counter() - t0
        logger.success(f"Trained in {train_sec / 60.0:.1f} min")

        # Seed 42 keeps the original filename; other seeds get their own file
        # so a multi-seed run never overwrites the reported model.
        seed_tag = "" if int(args.seed) == 42 else f"_seed{int(args.seed)}"
        model_path = results_dir / f"ltr_lambdamart_{retriever}{seed_tag}.txt"
        ranker.booster_.save_model(str(model_path))
        model_bytes = model_path.stat().st_size

        del Xtr, ytr, Xdev, ydev

    per_dataset: Dict[str, Any] = {}
    for dataset, split in EVAL_SPLITS:
        if args.datasets and dataset not in args.datasets:
            continue
        path = pathlib.Path(resolve_rank_data_file(retriever, dataset, split))
        if args.match_subsample:
            sub = path.with_name(f"{path.stem}_q{int(args.match_subsample)}.jsonl")
            if sub.is_file():
                logger.info(f"Using the same {args.match_subsample}-query subsample as "
                            f"the re-ranker comparison: {sub.name}")
                path = sub
            else:
                logger.warning(
                    f"--match-subsample {args.match_subsample} requested but {sub.name} "
                    "does not exist; falling back to the full split. The resulting "
                    "number is NOT comparable to a subsampled one."
                )
        if not path.is_file():
            logger.warning(f"Missing eval rank data: {path}")
            continue
        logger.info(f"Evaluating {dataset} [{split}]: {path}")
        Xte, _yte, gte, qte, pte, lte = load_feature_matrix(
            path, dropped=dropped, max_queries=args.max_eval_queries
        )
        if Xte.shape[0] == 0:
            continue
        scores = ranker.predict(Xte)
        metrics = evaluate_scores(
            np.asarray(scores),
            group_sizes=gte,
            qids=qte,
            pids_per_query=pte,
            labels_per_query=lte,
        )
        latency = measure_inference_latency(
            ranker.booster_,
            Xte,
            gte,
            n_queries=args.latency_queries,
            warmup=args.latency_warmup,
        )
        per_dataset[dataset] = {
            "split": split,
            "n_queries": len(gte),
            "metrics": {k: metrics[k] for k in METRIC_KEYS if k in metrics},
            "latency_ms": latency,
        }
        logger.success(
            f"  {dataset}: nDCG@10={metrics['ndcg@10']:.4f} "
            f"MRR={metrics['mrr']:.4f} R@10={metrics['recall@10']:.4f} "
            f"| {latency['p50_ms']:.2f} ms/q"
        )
        del Xte

    booster = ranker.booster_
    importances = booster.feature_importance(importance_type="gain")
    total_gain = float(np.sum(importances)) or 1.0

    return {
        "retriever": retriever,
        "n_features": int(booster.num_feature()),
        "n_trees": int(booster.num_trees()),
        "best_iteration": int(getattr(ranker, "best_iteration_", 0) or 0),
        "train_wall_clock_sec": None if args.load_model else train_sec,
        "train_queries": None if args.load_model else len(gtr),
        "loaded_from_saved_model": bool(args.load_model),
        "model_size_bytes": int(model_bytes),
        "model_path": str(model_path),
        "feature_gain_fraction": [float(g) / total_gain for g in importances],
        "results": per_dataset,
        "hyperparameters": {
            "objective": "lambdarank",
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "min_child_samples": args.min_child_samples,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "early_stopping_rounds": (
                None if args.no_early_stopping else args.early_stopping_rounds
            ),
            "seed": args.seed,
            "n_jobs": args.n_jobs,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever", default="hybrid_bm25_faiss")
    parser.add_argument("--cfg", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--log-period", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--max-train-queries",
        type=int,
        default=None,
        help="Cap training queries (default: all). The full combined pool is ~186k.",
    )
    parser.add_argument("--max-dev-queries", type=int, default=20000)
    parser.add_argument("--max-eval-queries", type=int, default=None)
    parser.add_argument("--latency-queries", type=int, default=200)
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument(
        "--datasets",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=None,
        help="Restrict evaluation to these datasets (default: all three).",
    )
    parser.add_argument(
        "--match-subsample",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Evaluate on <split>_qN.jsonl when it exists -- the exact subsample "
            "scripts/14_compare_rerankers.py used -- so LambdaMART and GARDIAN "
            "are scored on the same query population."
        ),
    )
    parser.add_argument(
        "--load-model",
        type=pathlib.Path,
        default=None,
        help="Skip training and score with this saved LightGBM booster.",
    )
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    retrievers = (
        list(FOCUS_HYBRID_RETRIEVERS)
        if args.retriever == "all"
        else [normalize_retriever_name(args.retriever)]
    )

    import lightgbm as lgb

    payload: Dict[str, Any] = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/run_ltr_baseline.py",
            "lightgbm_version": lgb.__version__,
            "platform": platform.platform(),
            "python_version": sys.version,
            "dropped_features": list(cfg.model.get("dropped_features") or []),
            "seed": args.seed,
        },
        "results": {},
    }

    for retriever in retrievers:
        logger.info(f"\n{'#' * 78}\n# LambdaMART baseline: {retriever}\n{'#' * 78}")
        payload["results"][retriever] = run_for_retriever(retriever, args, cfg)

    out = args.out or pathlib.Path(cfg.paths.results_dir) / "ltr_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.success(f"Saved -> {out}")


if __name__ == "__main__":
    main()
