"""
Aggregate raw per-seed results into the mean/std tables the paper reports.

This is a **separate pass** from training and evaluation. It only reads
``results/seeds/seed_<S>/`` and only writes ``results/aggregated/``. It never
mutates a per-seed artifact, so raw results survive every re-aggregation.

Output is the single source for every number in the paper -- no value in a
table should be typed by hand.

Usage
-----
    python scripts/aggregate_seeds.py
    python scripts/aggregate_seeds.py --seeds 13 21 42
    python scripts/aggregate_seeds.py --results-dir results --min-seeds 5

Writes
------
results/aggregated/retrieval_metrics.json
    Nested {retriever: {dataset: {system: {metric: {mean, std, n, seeds, values}}}}}
results/aggregated/retrieval_metrics.csv
    Long format, one row per (retriever, dataset, system, metric).
results/aggregated/training_summary.json / .csv
    Per-(retriever) best dev nDCG@10, wall-clock and parameter counts across seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import statistics
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, ".")

from loguru import logger  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.common.seeds import (  # noqa: E402
    aggregated_root,
    discover_seeds,
    parse_seeds,
    seed_dir,
)

# Metric keys are discovered from the data rather than hardcoded, but a stable
# ordering keeps the CSV diff-friendly across runs.
PREFERRED_METRIC_ORDER = [
    "ndcg@5",
    "ndcg@10",
    "ndcg@20",
    "mrr",
    "mrr@10",
    "recall@5",
    "recall@10",
    "recall@20",
    "hit@5",
    "hit@10",
    "hit@20",
]


def _metric_sort_key(metric: str) -> tuple:
    if metric in PREFERRED_METRIC_ORDER:
        return (0, PREFERRED_METRIC_ORDER.index(metric), metric)
    return (1, 0, metric)


def summarize(values: List[float], seeds: List[int]) -> Dict[str, Any]:
    """
    Mean/std over seeds for one metric cell.

    ``std`` is the *sample* standard deviation (n-1 denominator), which is the
    correct estimator for a small set of seeds and the convention reviewers
    expect in a mean +/- std table. It is None for a single seed, where the
    quantity is undefined -- reported as such rather than silently as 0.0.
    """
    n = len(values)
    return {
        "mean": float(statistics.fmean(values)) if n else None,
        "std": float(statistics.stdev(values)) if n > 1 else None,
        "min": float(min(values)) if n else None,
        "max": float(max(values)) if n else None,
        "n": n,
        "seeds": list(seeds),
        "values": [float(v) for v in values],
    }


def _load_json(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Could not read {path}: {exc}")
        return None


def _record_cell(
    cells: Dict[tuple, Dict[int, float]],
    retriever: str,
    dataset: str,
    system: str,
    metric: str,
    seed: int,
    value: Any,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    cells.setdefault((retriever, dataset, system, metric), {})[seed] = float(value)


def _harvest_fusion_meta(
    cells: Dict[tuple, Dict[int, float]],
    retriever: str,
    dataset: str,
    meta: Dict[str, Any],
    seed: int,
) -> None:
    """
    Promote fusion diagnostics stored under ``_meta`` into aggregatable cells.

    ``normalize_eval_results`` used to drop the ``global_alpha`` / ``oracle_alpha``
    metric blocks before writing the per-seed JSON, so those rows never reached
    this pass. The eval still wrote the fitted weight, pool recall, and the
    oracle nDCG into ``_meta``. Oracle nDCG there is the gold-in-pool mean;
    multiplying by pool recall recovers the all-queries convention used by
    every other row in the paper table. A later eval that persists the metric
    blocks will overwrite these harvested values with the stored ones.
    """
    pool_recall = meta.get("pool_recall")
    if isinstance(pool_recall, (int, float)) and not isinstance(pool_recall, bool):
        _record_cell(cells, retriever, dataset, "pool", "recall", seed, pool_recall)

    ga_weight = meta.get("global_alpha")
    if isinstance(ga_weight, (int, float)) and not isinstance(ga_weight, bool):
        _record_cell(
            cells, retriever, dataset, "global_alpha", "alpha_sparse", seed, ga_weight
        )

    oracle = meta.get("oracle_alpha") if isinstance(meta.get("oracle_alpha"), dict) else {}
    oracle_ndcg = oracle.get("oracle_ndcg")
    if isinstance(oracle_ndcg, (int, float)) and not isinstance(oracle_ndcg, bool):
        _record_cell(
            cells,
            retriever,
            dataset,
            "oracle_alpha",
            "ndcg@10_gold_in_pool",
            seed,
            oracle_ndcg,
        )
        if isinstance(pool_recall, (int, float)) and not isinstance(pool_recall, bool):
            _record_cell(
                cells,
                retriever,
                dataset,
                "oracle_alpha",
                "ndcg@10",
                seed,
                float(oracle_ndcg) * float(pool_recall),
            )
        else:
            _record_cell(
                cells, retriever, dataset, "oracle_alpha", "ndcg@10", seed, oracle_ndcg
            )
    for key in ("tie_fraction", "invariant_fraction", "best_alpha_mean", "best_alpha_std"):
        _record_cell(cells, retriever, dataset, "oracle_alpha", key, seed, oracle.get(key))

    rerank = meta.get("oracle_rerank_ndcg@10")
    if isinstance(rerank, (int, float)) and not isinstance(rerank, bool):
        _record_cell(cells, retriever, dataset, "oracle_rerank", "ndcg@10", seed, rerank)


def collect_retrieval(
    results_dir: pathlib.Path, seeds: List[int]
) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]]:
    """
    Gather every ``evaluation_*.json`` under each seed directory.

    Returns {retriever: {dataset: {system: {metric: summary}}}}.
    """
    # (retriever, dataset, system, metric) -> {seed: value}
    cells: Dict[tuple, Dict[int, float]] = {}

    for seed in seeds:
        sdir = seed_dir(results_dir, seed)
        if not sdir.is_dir():
            logger.warning(f"No directory for seed {seed}: {sdir}")
            continue
        # Per-retriever files are the source of truth. The combined
        # evaluation_results_all_retrievers.json duplicates them and would
        # otherwise overwrite the same (seed, retriever, dataset) cells.
        eval_files = sorted(
            p for p in sdir.glob("evaluation_*.json")
            if "all_retrievers" not in p.name
        )
        if not eval_files:
            eval_files = sorted(sdir.glob("evaluation_*.json"))
        if not eval_files:
            logger.warning(f"Seed {seed}: no evaluation_*.json found in {sdir}")
        for path in eval_files:
            payload = _load_json(path)
            if not isinstance(payload, dict):
                continue
            results = payload.get("results")
            if not isinstance(results, dict):
                continue
            for retriever, ds_block in results.items():
                if not isinstance(ds_block, dict):
                    continue
                for dataset, systems in ds_block.items():
                    if str(dataset).startswith("_") or not isinstance(systems, dict):
                        continue
                    meta = systems.get("_meta")
                    if isinstance(meta, dict):
                        _harvest_fusion_meta(cells, str(retriever), str(dataset), meta, seed)
                    for system, metrics in systems.items():
                        if str(system).startswith("_") or not isinstance(metrics, dict):
                            continue
                        for metric, value in metrics.items():
                            _record_cell(
                                cells,
                                str(retriever),
                                str(dataset),
                                str(system),
                                str(metric),
                                seed,
                                value,
                            )

    out: Dict[str, Any] = {}
    for (retriever, dataset, system, metric), by_seed in cells.items():
        ordered_seeds = sorted(by_seed)
        values = [by_seed[s] for s in ordered_seeds]
        (
            out.setdefault(retriever, {})
            .setdefault(dataset, {})
            .setdefault(system, {})[metric]
        ) = summarize(values, ordered_seeds)
    return out


def collect_training(results_dir: pathlib.Path, seeds: List[int]) -> Dict[str, Any]:
    """
    Gather per-seed training statistics.

    Reads the per-retriever ``gardian_training/<retriever>/run_summary.json``
    files first: they are written once per (seed, retriever) and so always
    cover every family. The combined ``training_summary_all_retrievers.json``
    is then layered on top to fill any gap. (Pre-multi-seed runs left a
    combined summary holding only the last retriever trained -- exactly the
    overwrite hazard this layout removes -- so it cannot be the primary source.)
    """
    fields = ("best_ndcg10", "train_wall_clock_sec", "trainable_parameters")
    cells: Dict[tuple, Dict[int, float]] = {}

    def _absorb(retriever: str, cell: Dict[str, Any], seed: int) -> None:
        for field in fields:
            value = cell.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.setdefault((retriever, field), {}).setdefault(seed, float(value))

    for seed in seeds:
        sdir = seed_dir(results_dir, seed)

        for run_summary in sorted(sdir.glob("gardian_training/*/run_summary.json")):
            payload = _load_json(run_summary)
            if isinstance(payload, dict):
                retriever = payload.get("retriever") or run_summary.parent.name
                _absorb(str(retriever), payload, seed)

        path = sdir / "training_summary_all_retrievers.json"
        payload = _load_json(path) if path.exists() else None
        if not isinstance(payload, dict):
            continue
        for retriever, cell in payload.items():
            if isinstance(cell, dict):
                _absorb(str(retriever), cell, seed)

    out: Dict[str, Any] = {}
    for (retriever, field), by_seed in cells.items():
        ordered_seeds = sorted(by_seed)
        out.setdefault(retriever, {})[field] = summarize(
            [by_seed[s] for s in ordered_seeds], ordered_seeds
        )
    return out


def _fmt(value: Optional[float], digits: int = 6) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def write_retrieval_csv(path: pathlib.Path, agg: Dict[str, Any], seeds: List[int]) -> None:
    header = [
        "retriever",
        "dataset",
        "system",
        "metric",
        "mean",
        "std",
        "min",
        "max",
        "n_seeds",
        "seeds",
    ] + [f"seed_{s}" for s in seeds]

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for retriever in sorted(agg):
            for dataset in sorted(agg[retriever]):
                for system in sorted(agg[retriever][dataset]):
                    metrics = agg[retriever][dataset][system]
                    for metric in sorted(metrics, key=_metric_sort_key):
                        cell = metrics[metric]
                        by_seed = dict(zip(cell["seeds"], cell["values"]))
                        w.writerow(
                            [
                                retriever,
                                dataset,
                                system,
                                metric,
                                _fmt(cell["mean"]),
                                _fmt(cell["std"]),
                                _fmt(cell["min"]),
                                _fmt(cell["max"]),
                                cell["n"],
                                " ".join(str(s) for s in cell["seeds"]),
                            ]
                            + [_fmt(by_seed.get(s)) for s in seeds]
                        )


def write_training_csv(path: pathlib.Path, agg: Dict[str, Any], seeds: List[int]) -> None:
    header = [
        "retriever",
        "field",
        "mean",
        "std",
        "min",
        "max",
        "n_seeds",
        "seeds",
    ] + [f"seed_{s}" for s in seeds]

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for retriever in sorted(agg):
            for field in sorted(agg[retriever]):
                cell = agg[retriever][field]
                by_seed = dict(zip(cell["seeds"], cell["values"]))
                w.writerow(
                    [
                        retriever,
                        field,
                        _fmt(cell["mean"]),
                        _fmt(cell["std"]),
                        _fmt(cell["min"]),
                        _fmt(cell["max"]),
                        cell["n"],
                        " ".join(str(s) for s in cell["seeds"]),
                    ]
                    + [_fmt(by_seed.get(s)) for s in seeds]
                )


def report_incomplete(agg: Dict[str, Any], expected: int) -> List[str]:
    """List cells that were aggregated over fewer than *expected* seeds."""
    warnings: List[str] = []
    for retriever in sorted(agg):
        for dataset in sorted(agg[retriever]):
            for system in sorted(agg[retriever][dataset]):
                for metric, cell in agg[retriever][dataset][system].items():
                    if cell["n"] < expected:
                        warnings.append(
                            f"{retriever}/{dataset}/{system}/{metric}: "
                            f"{cell['n']}/{expected} seeds ({cell['seeds']})"
                        )
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-seed results into mean/std paper tables."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results root (default: cfg.paths.results_dir from configs/base.yaml).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        nargs="+",
        default=None,
        help="Seeds to aggregate (default: every seed directory found).",
    )
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=0,
        help=(
            "Fail if any aggregated cell has fewer than this many seeds. "
            "Set to 5 for the paper run so an incomplete table cannot ship."
        ),
    )
    args = parser.parse_args()

    if args.results_dir:
        results_dir = pathlib.Path(args.results_dir)
    else:
        cfg = OmegaConf.load("configs/base.yaml")
        results_dir = pathlib.Path(cfg.paths.results_dir)

    seeds = parse_seeds(args.seeds) if args.seeds else discover_seeds(results_dir)
    if not seeds:
        raise SystemExit(
            f"No per-seed results found under {results_dir / 'seeds'}. "
            "Run scripts/04_train_gardian.py and scripts/05_evaluate_gardian.py first."
        )
    logger.info(f"Aggregating seeds: {seeds}")

    retrieval = collect_retrieval(results_dir, seeds)
    training = collect_training(results_dir, seeds)

    out_dir = aggregated_root(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/aggregate_seeds.py",
        "seeds": seeds,
        "n_seeds": len(seeds),
        "results_dir": str(results_dir),
        "std_definition": "sample standard deviation (n-1); null when n == 1",
        "source": f"{results_dir / 'seeds'} (raw per-seed artifacts, never modified)",
    }

    retrieval_json = out_dir / "retrieval_metrics.json"
    with open(retrieval_json, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": retrieval}, f, indent=2, ensure_ascii=False)
    write_retrieval_csv(out_dir / "retrieval_metrics.csv", retrieval, seeds)

    training_json = out_dir / "training_summary.json"
    with open(training_json, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": training}, f, indent=2, ensure_ascii=False)
    write_training_csv(out_dir / "training_summary.csv", training, seeds)

    n_cells = sum(
        len(metrics)
        for ds in retrieval.values()
        for systems in ds.values()
        for metrics in systems.values()
    )
    logger.success(
        f"Aggregated {n_cells} metric cells over {len(seeds)} seed(s) -> {out_dir}"
    )

    incomplete = report_incomplete(retrieval, len(seeds))
    if incomplete:
        logger.warning(f"{len(incomplete)} cell(s) missing at least one seed:")
        for line in incomplete[:20]:
            logger.warning(f"  {line}")
        if len(incomplete) > 20:
            logger.warning(f"  ... and {len(incomplete) - 20} more")

    if args.min_seeds:
        short = [
            line
            for line in report_incomplete(retrieval, int(args.min_seeds))
        ]
        if short:
            raise SystemExit(
                f"--min-seeds={args.min_seeds} not satisfied for {len(short)} cell(s). "
                "Refusing to emit paper tables from incomplete seed coverage."
            )


if __name__ == "__main__":
    main()
