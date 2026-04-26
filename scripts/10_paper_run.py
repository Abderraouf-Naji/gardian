"""Paper experiment driver for retriever-family ablations + significance stats."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import platform
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, ".")

from src.common.question_types import assert_cfg_question_types
from src.evaluation.rank_jsonl_eval import evaluate_all_from_rank_data
from src.evaluation.schemas import validate_paper_bundle
from src.evaluation.stats import bootstrap_mean_ci, bootstrap_delta_ci, paired_randomization_pvalue
from src.model.gardian import GARDIAN

torch.set_float32_matmul_precision("high")

DATASET_SPLITS = [
    ("pubmedqa_labeled", "eval"),
    ("pubmedqa_artificial", "test"),
    ("medmcqa", "test"),
]
RETRIEVER_CHOICES = ["hybrid", "hybrid_neural", "doc2query"]

ABLATION_CHOICES = ["full", "uniform_alpha", "no_qtype", "no_kg_coverage", "no_kg_signal"]


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


def build_model(cfg, device: str, retriever: str) -> GARDIAN:
    model = GARDIAN(
        sparse_dim=int(cfg.model.sparse_feat_dim),
        dense_dim=int(cfg.model.dense_feat_dim),
        kg_dim=int(cfg.model.kg_feat_dim),
        branch_hidden=int(cfg.model.branch_hidden),
        controller_hidden=int(cfg.model.controller_hidden),
        query_feat_dim=int(cfg.model.query_feat_dim),
        n_qtypes=len(cfg.model.question_types),
        dropout=float(cfg.model.dropout),
    )
    ckpt_path = pathlib.Path(cfg.paths.results_dir) / f"gardian_best_{retriever}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def _abl_to_kw(name: str) -> Optional[str]:
    if name == "full":
        return None
    return name


def _attach_bootstrap(block: Dict[str, Any], n_boot: int, seed: int) -> None:
    pq = block.pop("_per_query", None)
    if not pq or "ndcg@10" not in pq:
        return
    mean, lo, hi = bootstrap_mean_ci(pq["ndcg@10"], n_bootstrap=n_boot, seed=seed)
    block["ndcg@10_query_mean"] = mean
    block["ndcg@10_bootstrap_ci95"] = [lo, hi]
    for metric in ["ndcg@5", "ndcg@10", "ndcg@20", "recall@5", "recall@20", "mrr"]:
        if metric in pq:
            m, ml, mh = bootstrap_mean_ci(pq[metric], n_bootstrap=n_boot, seed=seed)
            block[f"{metric}_bootstrap_ci95"] = [ml, mh]
            block[f"{metric}_query_mean"] = m


def _add_delta_tests(raw: Dict[str, Any], n_boot: int, seed: int, n_trials: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    g = raw.get("gardian", {})
    g_pq = g.get("_per_query", {})
    for baseline in ["dense", "hybrid", "bm25", "doc2query"]:
        b = raw.get(baseline, {})
        b_pq = b.get("_per_query", {})
        if not b_pq or "ndcg@10" not in b_pq or "ndcg@10" not in g_pq:
            continue
        d_mean, d_lo, d_hi = bootstrap_delta_ci(
            g_pq["ndcg@10"], b_pq["ndcg@10"], n_bootstrap=n_boot, seed=seed
        )
        pvalue = paired_randomization_pvalue(
            g_pq["ndcg@10"], b_pq["ndcg@10"], n_trials=n_trials, seed=seed
        )
        out[f"gardian_minus_{baseline}_ndcg10"] = {
            "delta_mean": d_mean,
            "delta_ci95": [d_lo, d_hi],
            "paired_randomization_pvalue": pvalue,
        }
    return out


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
        help="Comma list from hybrid,hybrid_neural,doc2query or 'all'.",
    )
    p.add_argument(
        "--randomization-trials",
        type=int,
        default=10000,
        help="Trials for paired randomization test on nDCG@10 deltas.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)
    assert_cfg_question_types(cfg.model.question_types)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    want = [x.strip() for x in args.ablations.split(",") if x.strip()]
    for name in want:
        if name not in ABLATION_CHOICES:
            raise SystemExit(f"Unknown ablation {name!r}. Choose from {ABLATION_CHOICES}")

    retrievers = RETRIEVER_CHOICES if args.retrievers == "all" else [x.strip() for x in args.retrievers.split(",") if x.strip()]
    for r in retrievers:
        if r not in RETRIEVER_CHOICES:
            raise SystemExit(f"Unknown retriever {r!r}. Choose from {RETRIEVER_CHOICES}")

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
        },
        "results": {},
    }

    for retriever in retrievers:
        logger.info(f"=== Retriever={retriever} ===")
        try:
            model = build_model(cfg, device, retriever)
        except FileNotFoundError as e:
            logger.warning(f"Skipping retriever {retriever}: {e}")
            continue
        bundle["results"][retriever] = {}
        for ds_name, split in DATASET_SPLITS:
            rank_path = f"data/rank_data_{retriever}_{ds_name}_{split}.jsonl"
            if not pathlib.Path(rank_path).exists():
                logger.warning(f"Skip missing dataset file: {rank_path}")
                continue
            bundle["results"][retriever][ds_name] = {}
            for abl_name in want:
                logger.info(f"=== {retriever} | {ds_name} | ablation={abl_name} ===")
                raw = evaluate_all_from_rank_data(
                    rank_path,
                    model,
                    device,
                    gardian_ablation=_abl_to_kw(abl_name),
                    collect_per_query=True,
                    query_encoder_name=str(cfg.encoder.model_name),
                    query_encoder_device="cpu",
                )
                for sys_name, block in raw.items():
                    if isinstance(block, dict) and "_per_query" in block and args.bootstrap > 0:
                        _attach_bootstrap(block, n_boot=int(args.bootstrap), seed=int(args.seed))
                raw["significance"] = _add_delta_tests(
                    raw,
                    n_boot=int(args.bootstrap),
                    seed=int(args.seed),
                    n_trials=int(args.randomization_trials),
                )
                bundle["results"][retriever][ds_name][abl_name] = raw

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    validate_paper_bundle(bundle)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    logger.success(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
