"""Paper experiment driver: GARDIAN component ablations (text-only).

This is the RQ2 source of truth. One GPU forward per (seed, retriever, split)
yields branch scores and the full GARDIAN ranking; every other row is a mix of
those scores, scored with the same qrels / all-queries nDCG as Table 1.

  NoSparse / NoDense   -- one learned branch
  Uniform              -- 0.5/0.5 on the same branches (strawman; not the control)
  Fixed-α              -- learned branches, one alpha fitted on DEV
  GARDIAN              -- learned branches, per-query controller
  Oracle-α             -- learned branches, per-query best alpha (ceiling)

Example (one A40):
  .venv/bin/python scripts/10_paper_run.py \\
    --device cuda --parallel-workers 1 --cuda-devices 0
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, ".")

from src.common.hybrid_retrievers import FOCUS_HYBRID_RETRIEVERS
from src.common.question_types import assert_cfg_question_types
from src.common.seeds import (
    add_seeds_argument,
    parse_seeds,
    resolve_seed_checkpoint,
    seed_path,
    write_seed_json,
)
from src.evaluation.gardian_mixes import MIX_NAMES
from src.evaluation.paper_bundle import run_paper_chunk
from src.evaluation.schemas import validate_paper_bundle

torch.set_float32_matmul_precision("high")

DATASET_SPLITS = [
    ("pubmedqa_labeled", "eval"),
    ("pubmedqa_artificial", "test"),
    ("medmcqa", "test"),
]

RETRIEVER_CHOICES = list(FOCUS_HYBRID_RETRIEVERS) + ["hybrid", "hybrid_neural"]

# Subset of MIX_NAMES. "full" is the unablated model. "oracle_branch" is a
# ceiling computed from the same branch scores, not a GARDIAN.forward ablation.
# "uniform_alpha" is kept as a diagnostic; Fixed-α is the honest control.
ABLATION_CHOICES = [
    "full",
    "fixed_alpha",
    "no_sparse_signal",
    "no_dense_signal",
    "oracle_branch",
    "uniform_alpha",
]


def _git_revision() -> Optional[str]:
    root = pathlib.Path(__file__).resolve().parents[1]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _parse_cuda_devices(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return [0]
    return [int(x) for x in parts]


def _merge_chunk_results(target: Dict[str, Any], partial: Dict[str, Dict[str, Any]]) -> None:
    for ds_name, abl_map in partial.items():
        if ds_name not in target:
            target[ds_name] = {}
        target[ds_name].update(abl_map)


def _ndcg10(block: Dict[str, Any]) -> Optional[float]:
    g = block.get("gardian") if isinstance(block, dict) else None
    if not isinstance(g, dict):
        return None
    v = g.get("ndcg@10")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def aggregate_ablation_bundles(bundles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean ± std of nDCG@10 (and Recall@20) over per-seed paper bundles."""
    cells: Dict[tuple, List[float]] = {}
    r20: Dict[tuple, List[float]] = {}
    alphas: Dict[tuple, List[float]] = {}
    seeds: List[int] = []
    for bundle in bundles:
        seed = int(bundle.get("meta", {}).get("seed", -1))
        seeds.append(seed)
        for retriever, ds_block in bundle.get("results", {}).items():
            for dataset, abl_block in ds_block.items():
                for abl, raw in abl_block.items():
                    key = (retriever, dataset, abl)
                    n = _ndcg10(raw)
                    if n is not None:
                        cells.setdefault(key, []).append(n)
                    g = raw.get("gardian") or {}
                    rec = g.get("recall@20")
                    if isinstance(rec, (int, float)) and not isinstance(rec, bool):
                        r20.setdefault(key, []).append(float(rec))
                    a = (raw.get("_meta") or {}).get("fixed_alpha")
                    if isinstance(a, (int, float)) and not isinstance(a, bool):
                        alphas.setdefault((retriever, dataset), []).append(float(a))
    results: Dict[str, Any] = {}
    for (retriever, dataset, abl), vals in cells.items():
        results.setdefault(retriever, {}).setdefault(dataset, {})[abl] = {
            "ndcg@10": {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "values": vals,
            },
        }
        recs = r20.get((retriever, dataset, abl))
        if recs:
            results[retriever][dataset][abl]["recall@20"] = {
                "mean": float(np.mean(recs)),
                "std": float(np.std(recs, ddof=1)) if len(recs) > 1 else 0.0,
                "values": recs,
            }
    for (retriever, dataset), aval in alphas.items():
        results.setdefault(retriever, {}).setdefault(dataset, {})["fixed_alpha_sparse"] = {
            "mean": float(np.mean(aval)),
            "values": aval,
        }
    return {
        "meta": {"seeds": seeds, "n_seeds": len(seeds), "metric": "all-queries nDCG@10"},
        "results": results,
    }


def _print_summary(agg: Dict[str, Any]) -> None:
    logger.info("RQ2 nDCG@10 mean±std (%)")
    for retriever, ds_block in agg.get("results", {}).items():
        logger.info(f"  {retriever}")
        for dataset, abl_block in ds_block.items():
            parts = []
            for abl in ABLATION_CHOICES:
                cell = abl_block.get(abl)
                if not isinstance(cell, dict) or "ndcg@10" not in cell:
                    continue
                m = 100.0 * cell["ndcg@10"]["mean"]
                s = 100.0 * cell["ndcg@10"]["std"]
                parts.append(f"{abl}={m:.1f}±{s:.1f}")
            if parts:
                logger.info(f"    {dataset}: " + "  ".join(parts))


def load_cell_bundles(results_dir: str) -> List[Dict[str, Any]]:
    """Load every per-retriever ablation JSON under results/seeds/."""
    root = pathlib.Path(results_dir) / "seeds"
    bundles: List[Dict[str, Any]] = []
    if not root.is_dir():
        return bundles
    for path in sorted(root.glob("seed_*/ablation_hybrid_*.json")):
        try:
            bundles.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            logger.warning(f"skip unreadable {path}")
    return bundles


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paper bundle: GARDIAN component ablations (RQ2)."
    )
    p.add_argument("--cfg", type=str, default="configs/base.yaml")
    p.add_argument(
        "--out",
        type=str,
        default="results/aggregated/ablation_rq2.json",
        help="Aggregated JSON path (per-seed files always go under results/seeds/).",
    )
    p.add_argument(
        "--ablations",
        type=str,
        default=",".join(ABLATION_CHOICES),
        help=f"Comma-separated subset of: {','.join(ABLATION_CHOICES)}",
    )
    p.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples (0=skip).")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run a single seed (overrides --seeds). Omit to run the five paper seeds.",
    )
    add_seeds_argument(p)
    p.add_argument(
        "--retrievers",
        type=str,
        default="all",
        help=(
            "Comma list from hybrid_bm25_faiss, hybrid_bm25_medcpt, "
            "hybrid_spladepp_faiss, hybrid_spladepp_medcpt — or 'all'."
        ),
    )
    p.add_argument(
        "--randomization-trials",
        type=int,
        default=10000,
        help="Trials for paired randomization test on nDCG@10 deltas.",
    )
    p.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device.",
    )
    p.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Parallel processes across retrievers (one forward scores every mix). Use 1 on a single GPU.",
    )
    p.add_argument(
        "--cuda-devices",
        type=str,
        default="0",
        help="GPU ids for retriever workers (worker i uses ids[i %% len(ids)]). Ignored on CPU.",
    )
    p.add_argument(
        "--merge-existing",
        action="store_true",
        help="Do not score. Aggregate results/seeds/seed_*/ablation_hybrid_*.json into --out.",
    )
    return p.parse_args()


def _run_one_seed(
    *,
    project_root: str,
    cfg_abs: str,
    retrievers: List[str],
    want: List[str],
    device: str,
    seed: int,
    bootstrap: int,
    randomization_trials: int,
    query_encoder_name: str,
    parallel_workers: int,
    cuda_ids: List[int],
    cfg: Any,
    overwrite: bool,
) -> Dict[str, Any]:
    bundle: Dict[str, Any] = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_revision": _git_revision(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "bootstrap_samples": int(bootstrap),
            "seed": int(seed),
            "platform": platform.platform(),
            "python_version": sys.version,
            "retrievers": retrievers,
            "randomization_trials": int(randomization_trials),
            "parallel_workers": int(parallel_workers),
            "cuda_devices": cuda_ids if device == "cuda" else None,
            "ablations": want,
        },
        "results": {},
    }

    checkpoints = {}
    for retriever in retrievers:
        try:
            checkpoints[retriever] = resolve_seed_checkpoint(
                cfg.paths.results_dir, retriever, int(seed)
            )
        except FileNotFoundError as exc:
            raise SystemExit(
                f"{exc}\nTrain it first: python scripts/04_train_gardian.py "
                f"--retriever {retriever} --seeds {seed}"
            )

    parallel = int(parallel_workers) > 1 and len(retrievers) > 1
    if parallel and device == "cuda":
        logger.info(
            f"Parallel retrievers: {parallel_workers} workers, CUDA devices {cuda_ids}"
        )

    def _save_retriever(retriever: str, partial: Dict[str, Dict[str, Any]]) -> None:
        bundle["results"][retriever] = {}
        _merge_chunk_results(bundle["results"][retriever], partial)
        cell = {
            "meta": {**bundle["meta"], "retrievers": [retriever]},
            "results": {retriever: bundle["results"][retriever]},
        }
        validate_paper_bundle(cell)
        cell_path = seed_path(cfg.paths.results_dir, seed, f"ablation_{retriever}.json")
        write_seed_json(cell_path, cell, overwrite=overwrite)
        logger.success(f"Wrote {cell_path}")
        merged_path = seed_path(cfg.paths.results_dir, seed, "ablation_paper.json")
        merged = {"meta": bundle["meta"], "results": {}}
        if merged_path.exists():
            try:
                prev = json.loads(merged_path.read_text(encoding="utf-8"))
                if isinstance(prev.get("results"), dict):
                    merged["results"].update(prev["results"])
            except json.JSONDecodeError:
                logger.warning(f"Could not read {merged_path}; rewriting from this cell")
        merged["results"][retriever] = bundle["results"][retriever]
        retrievers_done = sorted(merged["results"])
        merged["meta"] = {**bundle["meta"], "retrievers": retrievers_done}
        write_seed_json(merged_path, merged, overwrite=True)
        logger.info(f"Updated {merged_path} ({', '.join(retrievers_done)})")

    def _one_retriever(retriever: str, vis: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        logger.info(f"=== Retriever={retriever} | seed={seed} ===")
        logger.info(f"Checkpoint: {checkpoints[retriever]}")
        return run_paper_chunk(
            project_root,
            retriever,
            cfg_abs,
            list(DATASET_SPLITS),
            want,
            device,
            int(bootstrap),
            int(seed),
            int(randomization_trials),
            query_encoder_name,
            vis,
        )

    if not parallel:
        for retriever in retrievers:
            cell_path = seed_path(cfg.paths.results_dir, seed, f"ablation_{retriever}.json")
            if cell_path.exists() and not overwrite:
                logger.info(f"Skip existing {cell_path}")
                prev = json.loads(cell_path.read_text(encoding="utf-8"))
                bundle["results"][retriever] = prev.get("results", {}).get(retriever, {})
                continue
            partial = _one_retriever(retriever)
            _save_retriever(retriever, partial)
    else:
        ctx = mp.get_context("spawn")
        futures = {}
        with ProcessPoolExecutor(
            max_workers=min(int(parallel_workers), len(retrievers)), mp_context=ctx
        ) as pool:
            for idx, retriever in enumerate(retrievers):
                cell_path = seed_path(cfg.paths.results_dir, seed, f"ablation_{retriever}.json")
                if cell_path.exists() and not overwrite:
                    logger.info(f"Skip existing {cell_path}")
                    prev = json.loads(cell_path.read_text(encoding="utf-8"))
                    bundle["results"][retriever] = prev.get("results", {}).get(retriever, {})
                    continue
                vis = str(cuda_ids[idx % len(cuda_ids)]) if device == "cuda" else None
                futures[
                    pool.submit(
                        run_paper_chunk,
                        project_root,
                        retriever,
                        cfg_abs,
                        list(DATASET_SPLITS),
                        want,
                        device,
                        int(bootstrap),
                        int(seed),
                        int(randomization_trials),
                        query_encoder_name,
                        vis,
                    )
                ] = retriever
            for fut in as_completed(futures):
                retriever = futures[fut]
                _save_retriever(retriever, fut.result())

    return bundle


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)
    assert_cfg_question_types(cfg.evaluation.question_types)

    if args.merge_existing:
        bundles = load_cell_bundles(str(cfg.paths.results_dir))
        if not bundles:
            raise SystemExit("No results/seeds/seed_*/ablation_hybrid_*.json files to merge.")
        agg = aggregate_ablation_bundles(bundles)
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
        _print_summary(agg)
        logger.success(f"Merged {len(bundles)} cells into {out_path}")
        return

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Device (base): {device}")
    if args.parallel_workers > 1 and device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False.")

    want = [x.strip() for x in args.ablations.split(",") if x.strip()]
    allowed = set(ABLATION_CHOICES) | set(MIX_NAMES)
    for name in want:
        if name not in allowed:
            raise SystemExit(f"Unknown ablation {name!r}. Choose from {sorted(allowed)}")

    retrievers = (
        list(FOCUS_HYBRID_RETRIEVERS)
        if args.retrievers == "all"
        else [x.strip() for x in args.retrievers.split(",") if x.strip()]
    )
    for r in retrievers:
        if r not in RETRIEVER_CHOICES:
            raise SystemExit(f"Unknown retriever {r!r}. Choose from {RETRIEVER_CHOICES}")

    seeds = [int(args.seed)] if args.seed is not None else parse_seeds(args.seeds)
    overwrite = bool(getattr(args, "overwrite_seed_artifacts", False))

    project_root = str(pathlib.Path(__file__).resolve().parents[1])
    cfg_abs = str(pathlib.Path(args.cfg).resolve())
    cuda_ids = _parse_cuda_devices(args.cuda_devices)
    query_encoder_name = str(cfg.encoder.model_name)

    for seed in seeds:
        _run_one_seed(
            project_root=project_root,
            cfg_abs=cfg_abs,
            retrievers=retrievers,
            want=want,
            device=device,
            seed=seed,
            bootstrap=int(args.bootstrap),
            randomization_trials=int(args.randomization_trials),
            query_encoder_name=query_encoder_name,
            parallel_workers=int(args.parallel_workers),
            cuda_ids=cuda_ids,
            cfg=cfg,
            overwrite=overwrite,
        )

    cells = load_cell_bundles(str(cfg.paths.results_dir))
    if cells:
        agg = aggregate_ablation_bundles(cells)
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
        _print_summary(agg)
        logger.success(f"Wrote aggregated RQ2 table to {out_path} ({len(cells)} cells on disk)")


if __name__ == "__main__":
    main()
