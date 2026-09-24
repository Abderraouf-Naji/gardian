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
import os
import pathlib
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

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
from src.common.passage_lookup import build_passage_text_lookup, passage_text_for_record
from src.evaluation.rerank_latency import (
    latency_stats,
    log_summary,
    select_timing_pools,
    time_cross_encoder,
    time_gardian,
    time_rrf,
)
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
        "tag": "bge_v2_m3",
        "model": "bge_v2_m3",
        "backend": "st",
        "batch_size": 32,
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
        "tag": "bge_v2_m3",
        "model": "bge_v2_m3",
        "backend": "st",
        "batch_size": 64,
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
        for system, block in systems.items():
            row = {
                "retriever": retriever,
                "dataset": dataset_name,
                "system": system,
            }
            for k, v in (block.get("metrics") or {}).items():
                row[k] = f"{float(v):.6f}"
            lat = block.get("latency_ms") or {}
            for k in ("p50_ms", "mean_ms", "p95_ms"):
                if k in lat and lat[k] is not None:
                    row[k] = f"{float(lat[k]):.3f}"
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




def _gpu_contention() -> Dict[str, Any]:
    """
    How many other processes hold the GPU while timing.

    A p50 measured under contention reports queueing, not compute: on this
    machine GARDIAN's median went from 24.5 ms to 51.7 ms with two other jobs
    resident, with no code change. Recording the count means a contended
    measurement can be spotted in the artifact instead of being quoted as if
    it were clean.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        pids = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except (OSError, subprocess.SubprocessError):
        return {"available": False}
    mine = str(os.getpid())
    others = [p for p in pids if p != mine]
    return {
        "available": True,
        "total_processes": len(pids),
        "other_processes": len(others),
        "exclusive": len(others) == 0,
    }


def _measure_latency_for_dataset(
    rank_path: pathlib.Path,
    *,
    dataset_name: str,
    cfg: Any,
    model: Any,
    device: str,
    ce_variants: List[Dict[str, Any]],
    n_queries: int,
    warmup: int,
    seed: int,
    ce_batch: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """
    Time every system re-ranking the SAME candidate pools.

    First-stage retrieval is shared by all systems and is excluded, so these
    numbers are the marginal cost of re-ranking. GARDIAN's timed region
    includes the PubMedBERT query-encoder forward whenever the controller is
    enabled, because a deployed system cannot precompute an unseen query's
    embedding.
    """
    from collections import defaultdict

    records_by_qid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in ce13._load_records(rank_path):
        records_by_qid[rec["qid"]].append(rec)

    pools, questions, _qids = select_timing_pools(
        records_by_qid, n_queries=n_queries, seed=seed
    )
    if not pools:
        logger.warning(f"No pools to time for {dataset_name}")
        return {}

    logger.info(
        f"  Latency: {len(pools)} queries, pool size "
        f"{min(len(p) for p in pools)}-{max(len(p) for p in pools)} candidates, "
        f"warmup={warmup}"
    )

    out: Dict[str, Dict[str, Any]] = {}
    out["rrf"] = time_rrf(pools, warmup=warmup)
    log_summary("rrf", out["rrf"])

    # Cross-encoders need the passage text, which rank JSONL does not store.
    corpus_paths = [p for p in ce13._corpus_paths_for_dataset(dataset_name, cfg) if p.is_file()]
    if not corpus_paths:
        unified = pathlib.Path(str(getattr(cfg.paths, "corpus_jsonl", "") or ""))
        if unified.is_file():
            corpus_paths = [unified]
    wanted_pids = {str(r["pid"]) for pool in pools for r in pool}
    logger.info(f"  Resolving {len(wanted_pids):,} passage texts for CE timing...")
    lookup = build_passage_text_lookup(wanted_pids, corpus_paths)
    missing = len(wanted_pids) - len(lookup)
    # A cross-encoder scoring empty strings is fast and meaningless. Silently
    # timing that would hand the paper a latency advantage that does not exist,
    # so an unresolved corpus is a hard failure rather than a warning.
    if missing:
        frac = missing / max(len(wanted_pids), 1)
        msg = (
            f"{missing:,}/{len(wanted_pids):,} passage texts ({frac:.1%}) could not be "
            f"resolved for {dataset_name} from {[str(p) for p in corpus_paths]}. "
            "Cross-encoder timing on empty text is not a measurement."
        )
        if frac > 0.01:
            raise RuntimeError(msg)
        logger.warning(f"  {msg} Continuing: under the 1% tolerance.")

    def _ptext(rec: Dict[str, Any]) -> str:
        return passage_text_for_record(rec, lookup)

    sample = [_ptext(r) for r in pools[0]]
    mean_chars = sum(len(t) for t in sample) / max(len(sample), 1)
    logger.info(f"  Passage text resolved (first pool mean {mean_chars:.0f} chars/passage)")

    fp16 = bool(getattr(cfg.retrieval, "cross_encoder_fp16", True))
    # Batch size for timing. The backfill batch sizes are tuned for bulk
    # throughput over a whole split; timing a single query's pool with them
    # would split ~95 candidates into a dozen forward passes and report a
    # latency no sensible deployment would accept. The default here is one
    # batch per pool, which is the same courtesy GARDIAN gets -- its whole pool
    # goes through in a single forward. Reviewers asked for the batch settings
    # behind the reported latency, so the chosen value is recorded per system.
    max_pool = max(len(p) for p in pools)
    timing_batch = int(ce_batch) if ce_batch and ce_batch > 0 else max_pool
    logger.info(
        f"  CE timing batch size: {timing_batch} "
        f"({'explicit' if ce_batch else 'one batch per pool'}; max pool {max_pool})"
    )
    for variant in ce_variants:
        model_name = str(variant["model"])
        if model_name in CROSS_ENCODER_PRESETS:
            model_name = CROSS_ENCODER_PRESETS[model_name]
        scorer = CrossEncoderScorer(
            model_name,
            device=device,
            max_length=int(cfg.retrieval.cross_encoder_max_length),
            batch_size=timing_batch,
            backend=str(variant["backend"]),
            fp16=fp16,
        )
        key = f"cross_encoder_{variant['tag']}"
        out[key] = time_cross_encoder(
            pools,
            questions=questions,
            passage_text=_ptext,
            scorer=scorer,
            device=device,
            warmup=warmup,
        )
        out[key]["batch_size"] = timing_batch
        out[key]["batch_size_policy"] = (
            "explicit" if ce_batch else "one batch per candidate pool"
        )
        out[key]["fp16"] = fp16
        out[key]["max_length"] = int(cfg.retrieval.cross_encoder_max_length)
        log_summary(key, out[key])
        del scorer
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    # GARDIAN: the controller reads the query embedding, so the encoder pass is
    # on the inference path and is timed. GARDIAN-Lite has no controller and no
    # encoder, and the flag records which arm this is.
    uses_controller = getattr(model, "controller", None) is not None
    encoder = None
    if uses_controller:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(str(cfg.encoder.model_name), device=device)
    out["gardian"] = time_gardian(
        pools,
        questions=questions,
        model=model,
        query_encoder=encoder,
        device=device,
        warmup=warmup,
        encode_query=uses_controller,
    )
    out["gardian"]["query_encoder"] = str(cfg.encoder.model_name) if uses_controller else None
    out["gardian"]["use_controller"] = bool(uses_controller)
    log_summary("gardian", out["gardian"])
    if encoder is not None:
        del encoder
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    # "hybrid" is the unreranked first-stage order: no re-ranking cost at all.
    out["hybrid"] = latency_stats([0.0] * len(pools))
    out["hybrid"]["note"] = "first-stage order, no re-ranking step"
    return out


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

    seed = int(args.seed if args.seed is not None else getattr(cfg, "seed", 42))

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
    ckpt_cfg = cfg
    if args.gardian_results_dir:
        ckpt_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        ckpt_cfg.paths.results_dir = str(args.gardian_results_dir)
        logger.info(f"Resolving GARDIAN checkpoint from {args.gardian_results_dir!r}")
    logger.info(f"Loading GARDIAN checkpoint for {retriever!r} on {device!r}")
    model, expected_qdim = _build(ckpt_cfg, device, retriever, seed)
    logger.info(
        "Loaded model: controller "
        + ("ON" if getattr(model, "controller", None) is not None else "OFF (GARDIAN-Lite)")
    )

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

        print_comparison_table(
            dataset_name, retriever, stack, system_order=systems_order
        )

        lat: Dict[str, Dict[str, Any]] = {}
        if args.measure_latency:
            logger.info(f"  Timing re-rankers on {dataset_name}...")
            lat = _measure_latency_for_dataset(
                work_path,
                dataset_name=dataset_name,
                cfg=cfg,
                model=model,
                device=device,
                ce_variants=ce_variants,
                n_queries=int(args.latency_queries),
                warmup=int(args.latency_warmup),
                seed=seed,
                ce_batch=int(args.latency_ce_batch),
            )

        # Nested schema: metrics and latency are separate blocks per system,
        # which is what scripts/plot_reranker_comparison.py reads.
        all_results[dataset_name] = {
            name: {
                "metrics": metrics,
                "latency_ms": lat.get(name, {}),
            }
            for name, metrics in stack.items()
        }

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
            "seed": seed,
            "latency_measured": bool(args.measure_latency),
            "latency_queries": int(args.latency_queries) if args.measure_latency else None,
            "latency_warmup": int(args.latency_warmup) if args.measure_latency else None,
            "latency_gpu_contention": _gpu_contention() if args.measure_latency else None,
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if str(device).startswith("cuda") and torch.cuda.is_available()
                else None
            ),
            "torch_version": torch.__version__,
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
        "--gardian-results-dir",
        type=str,
        default=None,
        help=(
            "Resolve the GARDIAN checkpoint from this artifact tree instead of "
            "paths.results_dir. Use results/gardian_lite to evaluate or time "
            "the --no-controller arm against the same pools."
        ),
    )
    parser.add_argument(
        "--no-cross-encoders",
        action="store_true",
        help="Evaluate only hybrid / RRF / GARDIAN (skips every cross-encoder arm).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed whose GARDIAN checkpoint is evaluated (default: cfg.seed). "
            "Also seeds --max-queries subsampling."
        ),
    )
    parser.add_argument(
        "--measure-latency",
        action="store_true",
        help=(
            "Time per-query re-ranking for every system on a common candidate "
            "pool. Run this on an otherwise idle GPU: a contended device gives "
            "a latency number that is not reportable."
        ),
    )
    parser.add_argument(
        "--latency-queries",
        type=int,
        default=200,
        metavar="N",
        help="Queries timed per dataset when --measure-latency is set.",
    )
    parser.add_argument(
        "--latency-ce-batch",
        type=int,
        default=0,
        help=(
            "Cross-encoder batch size during timing. 0 (default) scores each "
            "candidate pool in a single batch, matching the single forward "
            "GARDIAN gets, so the baseline is not throttled by a bulk-ingest "
            "batch size."
        ),
    )
    parser.add_argument(
        "--latency-warmup",
        type=int,
        default=10,
        help="Queries discarded before timing starts (allocator/autotuner warmup).",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/base.yaml",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    assert_cfg_question_types(cfg.evaluation.question_types)

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
    if args.no_cross_encoders:
        ce_variants = []
        logger.info("--no-cross-encoders: evaluating hybrid / RRF / GARDIAN only")
    if args.ce_tags:
        allowed = {t.strip() for t in args.ce_tags.split(",") if t.strip()}
        ce_variants = [v for v in ce_variants if v["tag"] in allowed]

    if not ce_variants and not args.no_cross_encoders:
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
