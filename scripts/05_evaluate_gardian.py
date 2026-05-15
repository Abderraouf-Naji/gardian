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
from src.common.rank_data_paths import normalize_retriever_name, resolve_rank_data_file
from src.evaluation.rank_jsonl_eval import evaluate_all_from_rank_data
from src.evaluation.schemas import validate_evaluation_results
from src.model.gardian import GARDIAN, build_gardian_from_model_cfg

torch.set_float32_matmul_precision("high")

HYBRID_RETRIEVER_COMBINATIONS = {
    "hybrid_bm25_faiss": "BM25 + FAISS",
    "hybrid_bm25_medcpt": "BM25 + MedCPT",
    "hybrid_spladepp_faiss": "SPLADE++ + FAISS",
    "hybrid_spladepp_medcpt": "SPLADE++ + MedCPT",
}

ALL_EVAL_RETRIEVERS = list(HYBRID_RETRIEVER_COMBINATIONS.keys())

SPARSE_DENSE_COMPONENTS = {
    "hybrid_bm25_faiss": {"sparse": "bm25", "dense": "faiss"},
    "hybrid_bm25_medcpt": {"sparse": "bm25", "dense": "medcpt"},
    "hybrid_spladepp_faiss": {"sparse": "spladepp", "dense": "faiss"},
    "hybrid_spladepp_medcpt": {"sparse": "spladepp", "dense": "medcpt"},
}


def _metric_key_for_component(retriever_name: str, role: str) -> str:
    if role == "sparse":
        if retriever_name == "spladepp":
            return "spladepp"
        return "bm25"
    return "dense"


def _load_component_metric_from_single_rankdata(
    component_retriever: str,
    dataset_name: str,
    split: str,
    role: str,
) -> Dict[str, Any]:
    component_path = resolve_rank_data_file(component_retriever, dataset_name, split)
    if not pathlib.Path(component_path).exists():
        logger.warning(
            f"Component rank data not found for {role}={component_retriever}: {component_path}"
        )
        return {}
    component_results = evaluate_all_from_rank_data(component_path, model=None, device=None)
    metric_key = _metric_key_for_component(component_retriever, role)
    metric = component_results.get(metric_key, {})
    if not isinstance(metric, dict):
        return {}
    return metric


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
    ckpt_path = pathlib.Path(cfg.paths.results_dir) / f"gardian_best_{retriever}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # Load checkpoint weights on CPU first to avoid CUDA OOM spikes during deserialization.
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_model_cfg = ckpt.get("cfg", {}).get("model") if isinstance(ckpt.get("cfg"), dict) else None
    model = build_gardian_from_model_cfg(ckpt_model_cfg or cfg.model)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint for {retriever} from {ckpt_path}")
    return model


def print_results_table(dataset_name: str, results: Dict[str, Any], retriever: str):
    """Print table: sparse, dense, hybrid, RRF, GARDIAN."""
    parts = SPARSE_DENSE_COMPONENTS.get(retriever, {"sparse": "bm25", "dense": "dense"})
    sparse_name = parts["sparse"]
    dense_name = parts["dense"]
    row_specs = [
        ("sparse", "bm25", f"sparse({sparse_name})"),
        ("dense", "dense", f"dense({dense_name})"),
        ("hybrid", "hybrid", f"hybrid({sparse_name}+{dense_name})"),
        ("rrf", "rrf", f"rrf({sparse_name}+{dense_name})"),
        ("gardian", "gardian", "gardian"),
    ]
    print(f"\n{'='*100}")
    print(f"RESULTS FOR {dataset_name.upper()}")
    print(f"{'='*100}")
    print(
        f"{'System':<24} {'nDCG@10':>10} {'nDCG@20':>10} {'nDCG@50':>10} {'nDCG@100':>10} "
        f"{'Recall@20':>11} {'Recall@50':>11} {'Recall@100':>12} {'MRR':>10}"
    )
    print(f"{'-'*120}")

    for _, key, display_name in row_specs:
        m = results.get(key, {})
        if not isinstance(m, dict):
            m = {}
        print(
            f"{display_name:<24} "
            f"{m.get('ndcg@10', 0.0):>10.4f} "
            f"{m.get('ndcg@20', 0.0):>10.4f} "
            f"{m.get('ndcg@50', 0.0):>10.4f} "
            f"{m.get('ndcg@100', 0.0):>10.4f} "
            f"{m.get('recall@20', 0.0):>11.4f} "
            f"{m.get('recall@50', 0.0):>11.4f} "
            f"{m.get('recall@100', 0.0):>12.4f} "
            f"{m.get('mrr', 0.0):>10.4f}"
        )

    print(f"{'='*100}\n")


def _metric(results: Dict[str, Any], system: str, metric_name: str) -> float:
    block = results.get(system, {})
    if not isinstance(block, dict):
        return 0.0
    return float(block.get(metric_name, 0.0) or 0.0)


def _delta_text(value: float, baseline: float) -> str:
    """Return absolute metric delta plus relative change, both signed."""
    delta = float(value) - float(baseline)
    if baseline > 0:
        rel = delta / float(baseline) * 100.0
        return f"{delta:+.4f} ({rel:+.1f}%)"
    return f"{delta:+.4f} (rel n/a)"


def _baseline_label(system: str, retriever: str) -> str:
    parts = SPARSE_DENSE_COMPONENTS.get(retriever, {"sparse": "bm25", "dense": "dense"})
    if system == "bm25":
        return f"sparse({parts['sparse']})"
    if system == "dense":
        return f"dense({parts['dense']})"
    if system == "hybrid":
        return f"hybrid({parts['sparse']}+{parts['dense']})"
    if system == "rrf":
        return f"rrf({parts['sparse']}+{parts['dense']})"
    if system == "spladepp":
        return "spladepp"
    return system


def _best_non_gardian_baseline(results: Dict[str, Any], retriever: str) -> tuple[str, float]:
    candidates = []
    for system in ("bm25", "dense", "hybrid", "rrf", "spladepp"):
        block = results.get(system)
        if isinstance(block, dict) and "ndcg@10" in block:
            candidates.append((_baseline_label(system, retriever), float(block.get("ndcg@10", 0.0) or 0.0)))
    if not candidates:
        return ("baseline", 0.0)
    return max(candidates, key=lambda x: x[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GARDIAN for one or all retriever families")
    parser.add_argument(
        "--retriever",
        type=str,
        choices=[*ALL_EVAL_RETRIEVERS, "hybrid", "hybrid_neural", "all"],
        default="all",
        help=(
            "Retriever family to evaluate. "
            "Hybrid combos used in the paper: "
            "hybrid_bm25_faiss(BM25+FAISS), "
            "hybrid_bm25_medcpt(BM25+MedCPT), "
            "hybrid_spladepp_faiss(SPLADE++ + FAISS), "
            "hybrid_spladepp_medcpt(SPLADE++ + MedCPT). "
            "Aliases: hybrid, hybrid_neural."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Evaluation device selection.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output JSON path. If omitted, defaults to "
            "results/evaluation_results_all_retrievers.json."
        ),
    )
    args = parser.parse_args()

    cfg = OmegaConf.load("configs/base.yaml")
    assert_cfg_question_types(cfg.model.question_types)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Evaluation on: {device}")
    
    retrievers = (
        ALL_EVAL_RETRIEVERS
        if args.retriever == "all"
        else [normalize_retriever_name(args.retriever)]
    )
    logger.info("Hybrid retriever combinations (explicit):")
    for name, combo in HYBRID_RETRIEVER_COMBINATIONS.items():
        logger.info(f"  - {name}: {combo}")
    dataset_splits = [
        ("pubmedqa_labeled", "eval"),
        ("pubmedqa_artificial", "test"),
        ("medmcqa", "test"),
    ]

    all_results = {}
    for retriever in retrievers:
        retriever_desc = HYBRID_RETRIEVER_COMBINATIONS.get(retriever, "single retriever")
        logger.info("\n" + "=" * 72)
        logger.info(f"EVALUATION RUN FOR RETRIEVER: {retriever.upper()} ({retriever_desc})")
        logger.info("=" * 72)

        try:
            model = build_model(cfg, device, retriever)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {retriever}: {e}")
            continue

        all_results[retriever] = {}
        for dataset_name, split in dataset_splits:
            rank_data_path = resolve_rank_data_file(retriever, dataset_name, split)
            if not pathlib.Path(rank_data_path).exists():
                logger.warning(f"Rank data not found: {rank_data_path}, skipping {dataset_name}")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {dataset_name} ({retriever})")
            logger.info(f"Rank data: {rank_data_path}")
            logger.info(f"{'='*60}")

            # Ultra-fast evaluation using pre-computed rank data
            try:
                results = evaluate_all_from_rank_data(
                    rank_data_path,
                    model,
                    device,
                    query_encoder_name=str(cfg.encoder.model_name),
                    query_encoder_device="cpu",
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and device == "cuda":
                    logger.warning(
                        "CUDA OOM during evaluation; retrying this dataset on CPU."
                    )
                    model.to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    results = evaluate_all_from_rank_data(
                        rank_data_path,
                        model,
                        "cpu",
                        query_encoder_name=str(cfg.encoder.model_name),
                        query_encoder_device="cpu",
                    )
                else:
                    raise

            # For hybrid retrievers, report sparse/dense baselines from their own
            # dedicated single-retriever rank files (fair cross-run baseline),
            # while keeping hybrid/gardian on the full hybrid candidate pool.
            if retriever in HYBRID_RETRIEVER_COMBINATIONS:
                parts = SPARSE_DENSE_COMPONENTS[retriever]
                sparse_metric = _load_component_metric_from_single_rankdata(
                    parts["sparse"], dataset_name, split, role="sparse"
                )
                dense_metric = _load_component_metric_from_single_rankdata(
                    parts["dense"], dataset_name, split, role="dense"
                )
                if sparse_metric:
                    results["bm25"] = sparse_metric
                if dense_metric:
                    results["dense"] = dense_metric
                if isinstance(results.get("_meta"), dict):
                    results["_meta"]["sparse_baseline_source"] = parts["sparse"]
                    results["_meta"]["dense_baseline_source"] = parts["dense"]

            all_results[retriever][dataset_name] = results

            # Print results table
            print_results_table(f"{dataset_name} [{retriever}]", results, retriever)
    
    # Save all results
    out_dir = pathlib.Path(cfg.paths.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = pathlib.Path(args.output) if args.output else (out_dir / "evaluation_results_all_retrievers.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
            dense_baseline = _metric(dataset_results, "dense", "ndcg@10")
            gardian_ndcg10 = _metric(dataset_results, "gardian", "ndcg@10")
            gardian_mrr = _metric(dataset_results, "gardian", "mrr")
            best_label, best_baseline = _best_non_gardian_baseline(dataset_results, retriever)
            dense_label = _baseline_label("dense", retriever)
            print(
                f"    GARDIAN nDCG@10: {gardian_ndcg10:.4f} | MRR: {gardian_mrr:.4f} | "
                f"Δ vs {dense_label}: {_delta_text(gardian_ndcg10, dense_baseline)} | "
                f"Δ vs best baseline ({best_label}={best_baseline:.4f}): "
                f"{_delta_text(gardian_ndcg10, best_baseline)}"
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