"""Paper experiment driver for retriever-family ablations + significance stats.
python3 scripts/10_paper_run.py --device cuda --parallel-workers 1 --cuda-devices 0"""

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

import torch
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, ".")

from src.common.question_types import assert_cfg_question_types
from src.evaluation.paper_bundle import run_paper_chunk
from src.evaluation.schemas import validate_paper_bundle

torch.set_float32_matmul_precision("high")

DATASET_SPLITS = [
    ("pubmedqa_labeled", "eval"),
    ("pubmedqa_artificial", "test"),
    ("medmcqa", "test"),
]
RETRIEVER_CHOICES = [
    "hybrid",
    "hybrid_neural",
    "hybrid_bm25_faiss",
    "hybrid_bm25_medcpt",
    "hybrid_spladepp_faiss",
    "hybrid_spladepp_medcpt",
]

ABLATION_CHOICES = [
    "full",
    "uniform_alpha",
    "no_qtype",
    "no_kg_coverage",
    "no_sparse_signal",
    "no_dense_signal",
    "no_kg_signal",
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


def _split_contiguous(items: List[str], max_chunks: int) -> List[List[str]]:
    if not items:
        return []
    n = max(1, min(max_chunks, len(items)))
    k, m = divmod(len(items), n)
    out: List[List[str]] = []
    i = 0
    for j in range(n):
        take = k + (1 if j < m else 0)
        out.append(items[i : i + take])
        i += take
    return out


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper / CIKM evaluation bundle.")
    p.add_argument("--cfg", type=str, default="configs/base.yaml")
    p.add_argument(
        "--out",
        type=str,
        default="results/paper_bundle.json",
        help="Output JSON path.",
    )
    p.add_argument(
        "--ablations",
        type=str,
        default=",".join(ABLATION_CHOICES),
        help=f"Comma-separated subset of: {','.join(ABLATION_CHOICES)}",
    )
    p.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples (0=skip).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--retrievers",
        type=str,
        default="all",
        help=(
            "Comma list from hybrid_bm25_faiss, hybrid_bm25_medcpt, "
            "hybrid_spladepp_faiss, hybrid_spladepp_medcpt — or 'all' "
            "to evaluate every hybrid family."
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
        default=4,
        help="Number of parallel processes per retriever (ablations split across workers). Use 1 to disable.",
    )
    p.add_argument(
        "--cuda-devices",
        type=str,
        default="0,1,2,3",
        help="Comma-separated GPU ids; worker i uses cuda:ids[i %% len(ids)]. Ignored when device is cpu.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)
    assert_cfg_question_types(cfg.model.question_types)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Device (base): {device}")
    if args.parallel_workers > 1 and device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False.")

    want = [x.strip() for x in args.ablations.split(",") if x.strip()]
    for name in want:
        if name not in ABLATION_CHOICES:
            raise SystemExit(f"Unknown ablation {name!r}. Choose from {ABLATION_CHOICES}")

    # ``all`` expands to the four canonical hybrid families used in the
    # paper; legacy aliases ``hybrid`` / ``hybrid_neural`` are still accepted
    # when supplied explicitly via --retrievers.
    canonical_hybrids = [
        "hybrid_bm25_faiss",
        "hybrid_bm25_medcpt",
        "hybrid_spladepp_faiss",
        "hybrid_spladepp_medcpt",
    ]
    retrievers = (
        canonical_hybrids
        if args.retrievers == "all"
        else [x.strip() for x in args.retrievers.split(",") if x.strip()]
    )
    for r in retrievers:
        if r not in RETRIEVER_CHOICES:
            raise SystemExit(f"Unknown retriever {r!r}. Choose from {RETRIEVER_CHOICES}")

    project_root = str(pathlib.Path(__file__).resolve().parents[1])
    cfg_abs = str(pathlib.Path(args.cfg).resolve())
    cuda_ids = _parse_cuda_devices(args.cuda_devices)
    query_encoder_name = str(cfg.encoder.model_name)

    bundle: Dict[str, Any] = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_revision": _git_revision(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "bootstrap_samples": int(args.bootstrap),
            "seed": int(args.seed),
            "platform": platform.platform(),
            "python_version": sys.version,
            "retrievers": retrievers,
            "randomization_trials": int(args.randomization_trials),
            "parallel_workers": int(args.parallel_workers),
            "cuda_devices": cuda_ids if device == "cuda" else None,
        },
        "results": {},
    }

    parallel = int(args.parallel_workers) > 1
    if parallel and device == "cuda":
        logger.info(
            f"Parallel mode: {args.parallel_workers} workers per retriever, CUDA devices {cuda_ids} "
            "(each worker loads a full model; map one GPU id per worker to avoid OOM)."
        )
        if len(cuda_ids) == 1:
            logger.warning(
                "All parallel workers use the same GPU id; this often causes CUDA OOM. "
                "Use e.g. --cuda-devices 0,1,2,3 with four GPUs, or --parallel-workers 1."
            )

    for retriever in retrievers:
        logger.info(f"=== Retriever={retriever} ===")
        ckpt_path = pathlib.Path(cfg.paths.results_dir) / f"gardian_best_{retriever}.pt"
        if not ckpt_path.exists():
            logger.warning(f"Skipping retriever {retriever}: Checkpoint not found: {ckpt_path}")
            continue

        bundle["results"][retriever] = {}
        ablation_chunks = _split_contiguous(want, int(args.parallel_workers))

        if not parallel or len(ablation_chunks) <= 1:
            partial = run_paper_chunk(
                project_root,
                retriever,
                cfg_abs,
                list(DATASET_SPLITS),
                want,
                device,
                int(args.bootstrap),
                int(args.seed),
                int(args.randomization_trials),
                query_encoder_name,
            )
            _merge_chunk_results(bundle["results"][retriever], partial)
            continue

        ctx = mp.get_context("spawn")
        futures = []
        with ProcessPoolExecutor(max_workers=len(ablation_chunks), mp_context=ctx) as pool:
            for idx, ab_chunk in enumerate(ablation_chunks):
                vis = (
                    str(cuda_ids[idx % len(cuda_ids)])
                    if device == "cuda"
                    else None
                )
                futures.append(
                    pool.submit(
                        run_paper_chunk,
                        project_root,
                        retriever,
                        cfg_abs,
                        list(DATASET_SPLITS),
                        ab_chunk,
                        device,
                        int(args.bootstrap),
                        int(args.seed),
                        int(args.randomization_trials),
                        query_encoder_name,
                        vis,
                    )
                )
            for fut in as_completed(futures):
                partial = fut.result()
                _merge_chunk_results(bundle["results"][retriever], partial)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    validate_paper_bundle(bundle)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    logger.success(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
