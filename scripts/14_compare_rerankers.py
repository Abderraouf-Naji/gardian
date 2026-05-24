#!/usr/bin/env python3
"""
Compare hybrid, RRF, cross-encoders (MonoT5-med, MonoBERT), and GARDIAN on rank JSONL.

Pipeline per dataset split:
  1. Backfill cross_encoder_score for each CE variant (GPU).
  2. Evaluate hybrid + RRF + GARDIAN once on base rank data.
  3. Evaluate cross_encoder per variant on tagged ``_ce_<tag>.jsonl`` files.
  4. Save consolidated JSON (+ CSV summary) under results/.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/14_compare_rerankers.py \\
      --retriever hybrid_bm25_faiss --device cuda

  python scripts/14_compare_rerankers.py \\
      --retriever hybrid_bm25_faiss --datasets pubmedqa_labeled --skip-backfill
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Backfill helpers from script 13 (same repo; not installed as a package).
import importlib.util

_ce13_path = pathlib.Path(__file__).resolve().parent / "13_backfill_cross_encoder_scores.py"
_ce13_spec = importlib.util.spec_from_file_location("backfill_ce", _ce13_path)
ce13 = importlib.util.module_from_spec(_ce13_spec)
assert _ce13_spec.loader is not None
_ce13_spec.loader.exec_module(ce13)

from src.common.question_types import assert_cfg_question_types
from src.common.rank_data_paths import normalize_retriever_name, resolve_rank_data_file
from src.evaluation.rank_jsonl_eval import evaluate_all_from_rank_data
from src.retrieval.cross_encoder import CROSS_ENCODER_PRESETS, CrossEncoderScorer

# Reuse 05 helpers without importing scripts as modules.
build_model = None


def _import_build_model():
    global build_model
    if build_model is not None:
        return build_model
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "evaluate_gardian",
        pathlib.Path(__file__).resolve().parent / "05_evaluate_gardian.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    build_model = mod.build_model
    return build_model


DATASET_SPLITS = ce13.DATASET_SPLITS
METRIC_KEYS = ("ndcg@10", "ndcg@20", "ndcg@50", "mrr", "recall@10", "recall@20", "hit@10")

# Cross-encoder variants requested for paper-style comparison.
DEFAULT_CE_VARIANTS: List[Dict[str, Any]] = [
    {
        "tag": "msmarco_minilm",
        "model": "msmarco_minilm",
        "backend": "st",
        "batch_size": 64,
    },
    {
        "tag": "monot5_med",
        "model": "monot5_med",
        "backend": "monot5",
        "batch_size": 8,
    },
    {
        "tag": "monobert",
        "model": "monobert_large",
        "backend": "monobert",
        "batch_size": 16,
    },
]

# Larger batches for A40-class GPUs (~5h budget: labeled + medmcqa, skip artificial).
FAST_CE_VARIANTS: List[Dict[str, Any]] = [
    {
        "tag": "msmarco_minilm",
        "model": "msmarco_minilm",
        "backend": "st",
        "batch_size": 128,
    },
    {
        "tag": "monot5_med",
        "model": "monot5_med",
        "backend": "monot5",
        "batch_size": 32,
    },
    {
        "tag": "monobert",
        "model": "monobert_large",
        "backend": "monobert",
        "batch_size": 64,
    },
]

FAST_DATASETS = ("pubmedqa_labeled", "medmcqa")
# pubmedqa_artificial test split ≈1.6M rows — typically 12–24h+ for both CE models alone.

HYBRID_FAMILIES = (
    "hybrid_bm25_faiss",
    "hybrid_bm25_medcpt",
    "hybrid_spladepp_faiss",
    "hybrid_spladepp_medcpt",
)


def _parse_retrievers(arg: str) -> List[str]:
    if arg.strip().lower() == "all":
        return list(HYBRID_FAMILIES)
    return [normalize_retriever_name(r.strip()) for r in arg.split(",") if r.strip()]

def _systems_reported(ce_variants: List[Dict[str, Any]]) -> List[str]:
    out = ["hybrid", "rrf"]
    for v in ce_variants:
        out.append(f"cross_encoder_{v['tag']}")
    out.append("gardian")
    return out


def _pick_metrics(block: Any) -> Dict[str, float]:
    if not isinstance(block, dict):
        return {}
    out: Dict[str, float] = {}
    for k in METRIC_KEYS:
        if k in block:
            out[k] = float(block[k])
    return out


def _estimate_rows(paths: List[pathlib.Path]) -> int:
    total = 0
    for p in paths:
        try:
            with p.open("rb") as fh:
                total += sum(1 for _ in fh)
        except OSError:
            pass
    return total


def _ensure_subsampled_rank_path(
    rank_path: pathlib.Path,
    max_queries: int,
    *,
    seed: int,
    rebuild: bool,
) -> pathlib.Path:
    """Keep ``max_queries`` random queries (all pool rows per qid); cache ``*_q{N}.jsonl``."""
    out_path = rank_path.with_name(f"{rank_path.stem}_q{max_queries}.jsonl")
    if out_path.is_file() and not rebuild:
        n = sum(1 for _ in out_path.open("rb"))
        logger.info(f"Using cached subsample ({n:,} rows): {out_path}")
        return out_path

    import random

    records = ce13._load_records(rank_path)
    grouped = ce13._group_by_query(records)
    qids = sorted(grouped.keys())
    if len(qids) > max_queries:
        rng = random.Random(seed)
        qids = sorted(rng.sample(qids, max_queries))
    out_records: List[Dict[str, Any]] = []
    for qid in qids:
        out_records.extend(grouped[qid])
    ce13._write_jsonl(out_path, out_records)
    logger.info(
        f"Subsampled {len(qids):,} queries / {len(out_records):,} rows -> {out_path.name}"
    )
    return out_path


def _dataset_jobs(retriever: str, datasets: List[str]) -> List[Tuple[str, str, pathlib.Path]]:
    retriever = normalize_retriever_name(retriever)
    jobs: List[Tuple[str, str, pathlib.Path]] = []
    for ds in datasets:
        for split in DATASET_SPLITS.get(ds, []):
            p = pathlib.Path(resolve_rank_data_file(retriever, ds, split))
            if p.is_file():
                jobs.append((ds, split, p))
            else:
                logger.warning(f"Missing rank data: {p}")
    return jobs


def _backfill_variant(
    *,
    in_path: pathlib.Path,
    out_path: pathlib.Path,
    dataset_name: str,
    cfg: Any,
    variant: Dict[str, Any],
    device: str,
    overwrite: bool,
    skip_if_exists: bool,
) -> None:
    if skip_if_exists and out_path.is_file() and not overwrite:
        logger.info(f"Skip backfill (exists): {out_path}")
        return

    model_name = str(variant["model"])
    if model_name in CROSS_ENCODER_PRESETS:
        model_name = CROSS_ENCODER_PRESETS[model_name]

    fp16 = bool(getattr(cfg.retrieval, "cross_encoder_fp16", True))
    scorer = CrossEncoderScorer(
        model_name,
        device=device,
        max_length=int(cfg.retrieval.cross_encoder_max_length),
        batch_size=int(variant.get("batch_size") or cfg.retrieval.cross_encoder_batch_size),
        backend=str(variant["backend"]),
        fp16=fp16,
    )
    logger.info(
        f"Backfill {variant['tag']}: {model_name!r} -> {out_path.name} "
        f"(backend={scorer.backend}, device={scorer.device})"
    )

    records = ce13._load_records(in_path)
    corpus_paths = ce13._corpus_paths_for_dataset(dataset_name, cfg)
    if not corpus_paths:
        unified = pathlib.Path(str(getattr(cfg.paths, "corpus_jsonl", "") or ""))
        if unified.is_file():
            corpus_paths = [unified]

    updated, n_scored = ce13.backfill_cross_encoder_scores(
        records,
        scorer=scorer,
        corpus_paths=[p for p in corpus_paths if p.is_file()],
        overwrite=overwrite,
    )
    ce13._write_jsonl(out_path, updated)
    logger.success(f"Wrote {out_path} (scored {n_scored:,} rows)")


def _eval_gardian_stack(
    rank_path: pathlib.Path,
    *,
    cfg: Any,
    model: Any,
    device: str,
    expected_qdim: int,
) -> Dict[str, Dict[str, float]]:
    gardian_adaptive = bool(getattr(cfg.qa, "gardian_adaptive_retrieval", False))
    raw = evaluate_all_from_rank_data(
        str(rank_path),
        model,
        device,
        query_encoder_name=str(cfg.encoder.model_name),
        query_encoder_device="cpu",
        expected_query_feat_dim=expected_qdim,
        include_standalone_spladepp=False,
        gardian_adaptive_retrieval=gardian_adaptive,
        cfg=cfg,
    )
    hybrid_key = "hybrid"
    if "hybrid" not in raw:
        for k in raw:
            if k == "hybrid" or k.startswith("hybrid(") or k.startswith("sum("):
                hybrid_key = k
                break
    out = {
        "hybrid": _pick_metrics(raw.get(hybrid_key, {})),
        "rrf": _pick_metrics(raw.get("rrf", {})),
        "gardian": _pick_metrics(raw.get("gardian", {})),
    }
    return out


def _eval_cross_encoder(rank_path: pathlib.Path) -> Dict[str, float]:
    raw = evaluate_all_from_rank_data(
        str(rank_path),
        model=None,
        device=None,
        include_standalone_spladepp=False,
    )
    return _pick_metrics(raw.get("cross_encoder", {}))


def print_comparison_table(
    dataset_name: str,
    retriever: str,
    systems: Dict[str, Dict[str, float]],
    *,
    system_order: List[str],
) -> None:
    print(f"\n{'=' * 90}")
    print(f"RERANKER COMPARISON — {dataset_name.upper()} [{retriever}]")
    print(f"{'=' * 90}")
    print(f"{'System':<32} {'nDCG@10':>10} {'MRR':>10} {'Recall@10':>11} {'Hit@10':>10}")
    print("-" * 90)
    for name in system_order:
        m = systems.get(name, {})
        hit = m.get("hit@10")
        hit_s = f"{hit:>10.4f}" if hit is not None else f"{'n/a':>10}"
        print(
            f"{name:<32} "
            f"{m.get('ndcg@10', 0.0):>10.4f} "
            f"{m.get('mrr', 0.0):>10.4f} "
            f"{m.get('recall@10', 0.0):>11.4f} "
            f"{hit_s}"
        )


def _write_csv(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    rows: List[Dict[str, str]] = []
    retriever = payload["meta"]["retriever"]
    for dataset_name, systems in payload["results"].items():
        for system, metrics in systems.items():
            row = {
                "retriever": retriever,
                "dataset": dataset_name,
                "system": system,
            }
            for k, v in metrics.items():
                row[k] = f"{v:.6f}"
            rows.append(row)

    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_comparison_for_retriever(
    retriever: str,
    *,
    args: argparse.Namespace,
    cfg: Any,
    device: str,
    datasets: List[str],
    ce_variants: List[Dict[str, Any]],
) -> pathlib.Path:
    jobs = _dataset_jobs(retriever, datasets)
    if not jobs:
        logger.warning(f"No rank JSONL for {retriever!r}; skipping.")
        return pathlib.Path()

    out_dir = pathlib.Path(cfg.paths.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    q_suffix = f"_q{args.max_queries}" if args.max_queries else ""
    json_path = (
        pathlib.Path(args.output)
        if args.output and len(_parse_retrievers(args.retriever)) == 1
        else out_dir / f"reranker_comparison_{retriever}{q_suffix}.json"
    )
    csv_path = json_path.with_suffix(".csv")

    _build = _import_build_model()
    logger.info(f"Loading GARDIAN checkpoint for {retriever!r} on {device!r}")
    model, expected_qdim = _build(cfg, device, retriever)

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    ce_variant_meta: List[Dict[str, str]] = []
    for variant in ce_variants:
        model_id = str(variant["model"])
        if model_id in CROSS_ENCODER_PRESETS:
            model_id = CROSS_ENCODER_PRESETS[model_id]
        ce_variant_meta.append(
            {
                "tag": variant["tag"],
                "model": model_id,
                "backend": variant["backend"],
                "result_key": f"cross_encoder_{variant['tag']}",
            }
        )

    skip_backfill = args.skip_backfill or args.eval_only
    systems_order = _systems_reported(ce_variants)

    seed = int(getattr(cfg, "seed", 42))
    for dataset_name, _split, rank_path in jobs:
        work_path = rank_path
        if args.max_queries:
            work_path = _ensure_subsampled_rank_path(
                rank_path,
                int(args.max_queries),
                seed=seed,
                rebuild=bool(args.overwrite_ce),
            )
        logger.info(f"\n--- {retriever} | {dataset_name} | {work_path} ---")
        if not args.eval_only:
            for variant in ce_variants:
                ce_path = ce13.ce_rank_data_path(work_path, variant["tag"])
                _backfill_variant(
                    in_path=work_path,
                    out_path=ce_path,
                    dataset_name=dataset_name,
                    cfg=cfg,
                    variant=variant,
                    device=device,
                    overwrite=args.overwrite_ce,
                    skip_if_exists=skip_backfill,
                )

        stack = _eval_gardian_stack(
            work_path,
            cfg=cfg,
            model=model,
            device=device,
            expected_qdim=expected_qdim,
        )
        for variant in ce_variants:
            ce_path = ce13.ce_rank_data_path(work_path, variant["tag"])
            if not ce_path.is_file():
                logger.error(f"Missing CE rank file: {ce_path}")
                continue
            stack[f"cross_encoder_{variant['tag']}"] = _eval_cross_encoder(ce_path)

        all_results[dataset_name] = stack
        print_comparison_table(
            dataset_name, retriever, stack, system_order=systems_order
        )

    payload = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/14_compare_rerankers.py",
            "retriever": retriever,
            "device": device,
            "systems": systems_order,
            "cross_encoder_variants": ce_variant_meta,
            "datasets": datasets,
            "max_queries": args.max_queries,
            "args": vars(args),
        },
        "results": all_results,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    _write_csv(csv_path, payload)
    logger.success(f"Saved comparison JSON -> {json_path}")
    logger.success(f"Saved comparison CSV  -> {csv_path}")
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare hybrid, RRF, MonoT5-med, MonoBERT, and GARDIAN; save results."
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default="hybrid_bm25_faiss",
        help=(
            "Hybrid family under data/<retriever>/, comma-separated list, or "
            "'all' for all four: bm25+faiss, bm25+medcpt, spladepp+faiss, spladepp+medcpt."
        ),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated: pubmedqa_labeled, pubmedqa_artificial, medmcqa, or all.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for GARDIAN eval and CE backfill.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/reranker_comparison_<retriever>.json).",
    )
    parser.add_argument(
        "--skip-backfill",
        action="store_true",
        help="Skip cross-encoder scoring if tagged _ce_<tag>.jsonl already exists.",
    )
    parser.add_argument(
        "--overwrite-ce",
        action="store_true",
        help="Recompute cross_encoder_score even when present.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "5h-friendly: pubmedqa_labeled + medmcqa only (skip artificial), "
            "larger CE batches. Does not skip CE backfill."
        ),
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip CE backfill; only run metrics (requires existing _ce_<tag>.jsonl).",
    )
    parser.add_argument(
        "--ce-tags",
        type=str,
        default=None,
        help="Comma-separated CE tags to run (default: monot5_med,monobert). E.g. monobert",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Subsample each rank JSONL to N random queries (seed=cfg.seed). "
            "Writes <stem>_qN.jsonl cache. Use for MedMCQA smoke runs."
        ),
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/base.yaml",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    assert_cfg_question_types(cfg.model.question_types)

    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    retrievers = _parse_retrievers(args.retriever)
    if args.output and len(retrievers) > 1:
        raise SystemExit("--output only allowed with a single --retriever.")

    if args.fast:
        datasets = list(FAST_DATASETS)
        logger.warning(
            "--fast: using pubmedqa_labeled + medmcqa only (skipping pubmedqa_artificial ~1.6M rows)."
        )
    elif args.datasets.strip().lower() == "all":
        datasets = list(DATASET_SPLITS.keys())
    else:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    ce_variants = list(FAST_CE_VARIANTS if args.fast else DEFAULT_CE_VARIANTS)
    if args.ce_tags:
        allowed = {t.strip() for t in args.ce_tags.split(",") if t.strip()}
        ce_variants = [v for v in ce_variants if v["tag"] in allowed]
        if not ce_variants:
            raise SystemExit(f"No CE variants matched --ce-tags {args.ce_tags!r}")

    all_jobs = []
    for r in retrievers:
        all_jobs.extend(_dataset_jobs(r, datasets))
    if not all_jobs:
        raise SystemExit("No rank JSONL matched. Run scripts/03_generate_rank_data.py first.")

    n_rows = _estimate_rows([p for _, _, p in all_jobs])
    n_ce_passes = len(ce_variants) if not args.eval_only else 0
    logger.info(
        f"Plan: retrievers={retrievers} | {len(all_jobs)} rank file(s) | ~{n_rows:,} rows | "
        f"{len(ce_variants)} CE model(s) | ~{n_rows * n_ce_passes:,} CE passes (+ GARDIAN)."
    )

    for retriever in retrievers:
        logger.info(f"\n{'#' * 80}\n# RETRIEVER FAMILY: {retriever}\n{'#' * 80}")
        try:
            run_comparison_for_retriever(
                retriever,
                args=args,
                cfg=cfg,
                device=device,
                datasets=datasets,
                ce_variants=ce_variants,
            )
        except FileNotFoundError as e:
            logger.error(
                f"{e} — train with: python scripts/04_train_gardian.py --retriever {retriever}"
            )


if __name__ == "__main__":
    main()
