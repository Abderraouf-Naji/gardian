"""
ULTRA-FAST EVALUATION - Uses pre-generated rank data for ALL systems
With CORRECTED MRR calculation
"""

import json
import pathlib
import sys
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict
import argparse

import torch
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, ".")

from src.common.question_types import assert_cfg_question_types
from src.evaluation.rank_jsonl_eval import evaluate_all_from_rank_data
from src.evaluation.schemas import validate_evaluation_results
from src.model.gardian import GARDIAN

torch.set_float32_matmul_precision("high")


def _git_revision() -> str:
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
    return ""


def build_model(cfg, device: str, retriever: str) -> GARDIAN:
    """Load trained GARDIAN model."""
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
    logger.info(f"Loaded checkpoint for {retriever} from {ckpt_path}")
    return model


def print_results_table(dataset_name: str, results: Dict[str, Any]):
    """Print formatted results table (includes Doc2Query when present)."""
    order = [
        "bm25",
        "dense",
        "hybrid",
        "doc2query",
        "gardian",
    ]
    print(f"\n{'='*100}")
    print(f"RESULTS FOR {dataset_name.upper()}")
    print(f"{'='*100}")
    print(
        f"{'System':<12} {'nDCG@5':>10} {'nDCG@10':>10} {'nDCG@20':>10} "
        f"{'Recall@5':>10} {'Recall@20':>10} {'MRR':>10}"
    )
    print(f"{'-'*80}")

    for system in order:
        m = results.get(system)
        if not m or not isinstance(m, dict):
            continue
        if system == "doc2query" and not any(
            k in m for k in ("ndcg@5", "ndcg@10")
        ):
            continue
        print(
            f"{system:<12} "
            f"{m.get('ndcg@5', 0.0):>10.4f} "
            f"{m.get('ndcg@10', 0.0):>10.4f} "
            f"{m.get('ndcg@20', 0.0):>10.4f} "
            f"{m.get('recall@5', 0.0):>10.4f} "
            f"{m.get('recall@20', 0.0):>10.4f} "
            f"{m.get('mrr', 0.0):>10.4f}"
        )

    print(f"{'='*100}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GARDIAN for one or all retriever families")
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["hybrid", "hybrid_neural", "doc2query", "all"],
        default="all",
        help="Retriever family to evaluate",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load("configs/base.yaml")
    assert_cfg_question_types(cfg.model.question_types)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Evaluation on: {device}")
    
    retrievers = ["hybrid", "hybrid_neural", "doc2query"] if args.retriever == "all" else [args.retriever]
    dataset_splits = [
        ("pubmedqa_labeled", "eval"),
        ("pubmedqa_artificial", "test"),
        ("medmcqa", "test"),
    ]

    all_results = {}
    for retriever in retrievers:
        logger.info("\n" + "=" * 72)
        logger.info(f"EVALUATION RUN FOR RETRIEVER: {retriever.upper()}")
        logger.info("=" * 72)

        try:
            model = build_model(cfg, device, retriever)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {retriever}: {e}")
            continue

        all_results[retriever] = {}
        for dataset_name, split in dataset_splits:
            rank_data_path = f"data/rank_data_{retriever}_{dataset_name}_{split}.jsonl"
            if not pathlib.Path(rank_data_path).exists():
                logger.warning(f"Rank data not found: {rank_data_path}, skipping {dataset_name}")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {dataset_name} ({retriever})")
            logger.info(f"Rank data: {rank_data_path}")
            logger.info(f"{'='*60}")

            # Ultra-fast evaluation using pre-computed rank data
            results = evaluate_all_from_rank_data(
                rank_data_path,
                model,
                device,
                query_encoder_name=str(cfg.encoder.model_name),
                query_encoder_device="cpu",
            )
            all_results[retriever][dataset_name] = results

            # Print results table
            print_results_table(f"{dataset_name} [{retriever}]", results)
    
    # Save all results
    out_dir = pathlib.Path(cfg.paths.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "evaluation_results_all_retrievers.json"

    payload = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/05_evaluate_gardian.py",
            "args": vars(args),
            "git_revision": _git_revision(),
            "platform": platform.platform(),
            "python_version": sys.version,
        },
        "results": all_results,
    }
    validate_evaluation_results(payload)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    
    logger.success(f"Results saved to {out_path}")
    
    # Print final summary with analysis
    print("\n" + "="*100)
    print("FINAL SUMMARY - nDCG@10 COMPARISON BY RETRIEVER")
    print("="*100)

    for retriever, retriever_results in all_results.items():
        print(f"\n[{retriever.upper()}]")
        for dataset_name, dataset_results in retriever_results.items():
            print(f"  {dataset_name.upper()}:")
            dense_baseline = dataset_results.get("dense", {}).get("ndcg@10", 0.0)
            gardian_ndcg10 = dataset_results.get("gardian", {}).get("ndcg@10", 0.0)
            gardian_mrr = dataset_results.get("gardian", {}).get("mrr", 0.0)
            if dense_baseline > 0:
                imp = (gardian_ndcg10 - dense_baseline) / dense_baseline * 100
                print(
                    f"    GARDIAN nDCG@10: {gardian_ndcg10:.4f} "
                    f"(Δ {imp:+.1f}% vs Dense) | MRR: {gardian_mrr:.4f}"
                )
            else:
                print(
                    f"    GARDIAN nDCG@10: {gardian_ndcg10:.4f} "
                    f"| MRR: {gardian_mrr:.4f}"
                )

    # Cross-retriever comparison (GARDIAN only) per dataset
    print("\n" + "=" * 100)
    print("CROSS-RETRIEVER COMPARISON (GARDIAN ONLY)")
    print("=" * 100)
    for dataset_name, _ in dataset_splits:
        print(f"\n{dataset_name.upper()}:")
        ranking = []
        for retriever in retrievers:
            r = all_results.get(retriever, {}).get(dataset_name, {})
            if not r:
                continue
            ranking.append((retriever, r.get("gardian", {}).get("ndcg@10", 0.0), r.get("gardian", {}).get("mrr", 0.0)))
        ranking.sort(key=lambda x: x[1], reverse=True)
        for retriever, ndcg10, mrr in ranking:
            print(f"  {retriever:10} nDCG@10: {ndcg10:.4f} | MRR: {mrr:.4f}")
    
    print("="*100)
    
    # Observations
    print("\n" + "="*100)
    print("OBSERVATIONS & ANALYSIS")
    print("="*100)


if __name__ == "__main__":
    main()