"""
ULTRA-FAST EVALUATION - Uses pre-generated rank data for ALL systems
With CORRECTED MRR calculation

Also reports Hit@k (success rate) when rank JSONL eval includes hit@ metrics.
"""

import argparse
import json
import pathlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import torch
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, ".")

from src.common.question_types import assert_cfg_question_types
from src.common.rank_data_paths import normalize_retriever_name, resolve_rank_data_file
from src.evaluation.baseline_systems import normalize_eval_results
from src.evaluation.rank_jsonl_eval import evaluate_all_from_rank_data
from src.evaluation.schemas import validate_evaluation_results
from src.model.gardian import GARDIAN, build_gardian_from_model_cfg, load_checkpoint_state

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
    component_results = evaluate_all_from_rank_data(
        component_path,
        model=None,
        device=None,
        include_standalone_spladepp=False,
    )
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


def build_model(cfg, device: str, retriever: str) -> Tuple[GARDIAN, int]:
    """Load trained GARDIAN model."""
    ckpt_path = pathlib.Path(cfg.paths.results_dir) / f"gardian_best_{retriever}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load checkpoint weights on CPU first to avoid CUDA OOM spikes during deserialization.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("cfg", {}) if isinstance(ckpt.get("cfg"), dict) else {}
    ckpt_model_cfg = ckpt_cfg.get("model") if isinstance(ckpt_cfg.get("model"), dict) else None
    model = build_gardian_from_model_cfg(ckpt_model_cfg or cfg.model)
    load_checkpoint_state(model, ckpt["model_state"], strict=False)
    model.to(device)
    model.eval()
    expected_qdim = int(
        (ckpt_model_cfg or {}).get("query_feat_dim", cfg.model.query_feat_dim)
    )
    logger.info(f"Loaded checkpoint for {retriever} from {ckpt_path}")
    return model, expected_qdim


def _has_metric_block(results: Dict[str, Any], key: str) -> bool:
    block = results.get(key)
    return isinstance(block, dict) and "ndcg@10" in block


def _results_row_specs(
    results: Dict[str, Any],
    retriever: str,
    *,
    include_cross_encoder: bool = True,
) -> List[tuple[str, str, str]]:
    parts = SPARSE_DENSE_COMPONENTS.get(retriever, {"sparse": "bm25", "dense": "dense"})
    sparse_name = parts["sparse"]
    dense_name = parts["dense"]
    row_specs = [
        ("sparse", "bm25", f"sparse({sparse_name})"),
        ("dense", "dense", f"dense({dense_name})"),
        ("hybrid", "hybrid", f"hybrid({sparse_name}+{dense_name})"),
        ("rrf", "rrf", f"rrf({sparse_name}+{dense_name})"),
    ]
    if include_cross_encoder and _has_metric_block(results, "cross_encoder"):
        row_specs.append(("cross_encoder", "cross_encoder", "cross_encoder"))
    row_specs.append(("gardian", "gardian", "gardian"))
    return row_specs


def print_results_table(dataset_name: str, results: Dict[str, Any], retriever: str) -> None:
    """Print table: sparse, dense, hybrid, RRF, GARDIAN (+ cross_encoder when scored)."""
    row_specs = _results_row_specs(results, retriever)
    print(f"\n{'=' * 100}")
    print(f"RESULTS FOR {dataset_name.upper()}")
    print(f"{'=' * 100}")
    print(
        f"{'System':<24} {'nDCG@10':>10} {'nDCG@20':>10} {'nDCG@50':>10} {'nDCG@100':>10} "
        f"{'Recall@20':>11} {'Recall@50':>11} {'Recall@100':>12} {'MRR':>10}"
    )
    print(f"{'-' * 120}")

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

    print(f"{'=' * 100}\n")


def print_hit_results_table(dataset_name: str, results: Dict[str, Any], retriever: str) -> None:
    """Additional table: Hit@k (fraction of queries with ≥1 relevant in top-k)."""
    row_specs = _results_row_specs(results, retriever)
    print(f"\n{'=' * 80}")
    print(f"HIT RATE (Success@k) — {dataset_name.upper()}")
    print(f"{'=' * 80}")
    print(f"{'System':<24} {'Hit@5':>10} {'Hit@20':>10} {'Hit@50':>10}")
    print(f"{'-' * 58}")

    for _, key, display_name in row_specs:
        m = results.get(key, {})
        if not isinstance(m, dict):
            m = {}
        print(
            f"{display_name:<24} "
            f"{m.get('hit@5', 0.0):>10.4f} "
            f"{m.get('hit@20', 0.0):>10.4f} "
            f"{m.get('hit@50', 0.0):>10.4f}"
        )
    print(f"{'=' * 80}\n")


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
    if system == "cross_encoder":
        return "cross_encoder"
    return system


def _normalize_dataset_block(
    raw: Dict[str, Any],
    retriever: str,
    *,
    include_cross_encoder: bool,
) -> Dict[str, Any]:
    return normalize_eval_results(
        raw,
        retriever,
        include_cross_encoder=include_cross_encoder,
    )


def _evaluation_output_suffix(use_cross_encoder_rank_data: bool) -> str:
    return "_cross" if use_cross_encoder_rank_data else ""


def _save_retriever_payload(
    out_dir: pathlib.Path,
    retriever: str,
    ds_block: Dict[str, Any],
    *,
    include_cross_encoder: bool,
    output_suffix: str,
    args: argparse.Namespace,
) -> pathlib.Path:
    normalized = {
        ds: _normalize_dataset_block(raw, retriever, include_cross_encoder=include_cross_encoder)
        for ds, raw in ds_block.items()
        if not str(ds).startswith("_")
    }
    per_path = out_dir / f"evaluation_{retriever}{output_suffix}.json"
    per_payload = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/05_evaluate_gardian.py",
            "retriever": retriever,
            "cross_encoder_rank_data": bool(include_cross_encoder),
            "args": vars(args),
            "git_revision": _git_revision(),
        },
        "results": {retriever: normalized},
    }
    validate_evaluation_results(per_payload)
    with open(per_path, "w", encoding="utf-8") as pf:
        json.dump(per_payload, pf, indent=2, default=str)
    return per_path


def _best_non_gardian_baseline(results: Dict[str, Any], retriever: str) -> tuple[str, float]:
    candidates = []
    for system in ("bm25", "dense", "hybrid", "rrf", "spladepp", "cross_encoder"):
        block = results.get(system)
        if isinstance(block, dict) and "ndcg@10" in block:
            candidates.append(
                (_baseline_label(system, retriever), float(block.get("ndcg@10", 0.0) or 0.0))
            )
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
            "Output JSON path. With --use-cross-encoder-rank-data and one retriever, "
            "defaults to results/evaluation_{retriever}_cross.json."
        ),
    )
    parser.add_argument(
        "--per-retriever-json",
        action="store_true",
        help="Also write results/evaluation_{retriever}.json for each hybrid family.",
    )
    parser.add_argument(
        "--print-hit-table",
        action="store_true",
        help="Print an additional Hit@k table after each dataset (requires hit@ in rank eval).",
    )
    parser.add_argument(
        "--use-cross-encoder-rank-data",
        action="store_true",
        help=(
            "Prefer rank JSONL files with a _ce suffix (from "
            "scripts/13_backfill_cross_encoder_scores.py) when they exist."
        ),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help=(
            "Comma-separated datasets to evaluate: pubmedqa_labeled, "
            "pubmedqa_artificial, medmcqa, or all (default all)."
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
    if args.datasets.strip().lower() != "all":
        allowed = {d.strip() for d in args.datasets.split(",") if d.strip()}
        dataset_splits = [ds for ds in dataset_splits if ds[0] in allowed]
        if not dataset_splits:
            raise SystemExit(f"No datasets matched --datasets {args.datasets!r}")

    all_results: Dict[str, Dict[str, Any]] = {}
    out_dir = pathlib.Path(cfg.paths.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_suffix = _evaluation_output_suffix(args.use_cross_encoder_rank_data)
    save_per_retriever = bool(args.per_retriever_json or args.use_cross_encoder_rank_data)

    for retriever in retrievers:
        retriever_desc = HYBRID_RETRIEVER_COMBINATIONS.get(retriever, "single retriever")
        logger.info("\n" + "=" * 72)
        logger.info(f"EVALUATION RUN FOR RETRIEVER: {retriever.upper()} ({retriever_desc})")
        logger.info("=" * 72)

        try:
            model, expected_qdim = build_model(cfg, device, retriever)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device == "cuda":
                logger.warning("CUDA OOM loading GARDIAN; falling back to CPU for this run.")
                device = "cpu"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model, expected_qdim = build_model(cfg, device, retriever)
            else:
                raise
        except FileNotFoundError as e:
            logger.warning(f"Skipping {retriever}: {e}")
            continue

        all_results[retriever] = {}
        for dataset_name, split in dataset_splits:
            rank_data_path = resolve_rank_data_file(retriever, dataset_name, split)
            if args.use_cross_encoder_rank_data:
                ce_path = pathlib.Path(rank_data_path).with_name(
                    pathlib.Path(rank_data_path).stem + "_ce.jsonl"
                )
                if ce_path.exists():
                    rank_data_path = str(ce_path)
                    logger.info(f"Using cross-encoder rank data: {rank_data_path}")
                else:
                    logger.warning(
                        f"No _ce rank file at {ce_path}; using default {rank_data_path}"
                    )
            if not pathlib.Path(rank_data_path).exists():
                logger.warning(f"Rank data not found: {rank_data_path}, skipping {dataset_name}")
                continue

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Evaluating: {dataset_name} ({retriever})")
            logger.info(f"Rank data: {rank_data_path}")
            logger.info(f"{'=' * 60}")

            gardian_adaptive = bool(getattr(cfg.qa, "gardian_adaptive_retrieval", False))
            if gardian_adaptive:
                logger.info(
                    "GARDIAN: adaptive retrieval ON (cfg.qa.gardian_adaptive_retrieval)"
                )

            # Ultra-fast evaluation using pre-computed rank data
            try:
                results = evaluate_all_from_rank_data(
                    rank_data_path,
                    model,
                    device,
                    query_encoder_name=str(cfg.encoder.model_name),
                    query_encoder_device="cpu",
                    expected_query_feat_dim=expected_qdim,
                    include_standalone_spladepp=False,
                    gardian_adaptive_retrieval=gardian_adaptive,
                    cfg=cfg,
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
                        expected_query_feat_dim=expected_qdim,
                        include_standalone_spladepp=False,
                        gardian_adaptive_retrieval=gardian_adaptive,
                        cfg=cfg,
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

            # Print results table (original full metric table)
            print_results_table(f"{dataset_name} [{retriever}]", results, retriever)
            if args.print_hit_table:
                print_hit_results_table(f"{dataset_name} [{retriever}]", results, retriever)

        if save_per_retriever and all_results.get(retriever):
            per_path = _save_retriever_payload(
                out_dir,
                retriever,
                all_results[retriever],
                include_cross_encoder=args.use_cross_encoder_rank_data,
                output_suffix=output_suffix,
                args=args,
            )
            logger.info(f"Per-retriever metrics -> {per_path}")

    # Save combined results (normalized keys for schema validation)
    if args.output:
        out_path = pathlib.Path(args.output)
    elif len(retrievers) == 1:
        out_path = out_dir / f"evaluation_{retrievers[0]}{output_suffix}.json"
    else:
        out_path = out_dir / f"evaluation_results_all_retrievers{output_suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_all: Dict[str, Dict[str, Any]] = {}
    for retriever, ds_block in all_results.items():
        normalized_all[retriever] = {
            ds: _normalize_dataset_block(
                raw,
                retriever,
                include_cross_encoder=args.use_cross_encoder_rank_data,
            )
            for ds, raw in ds_block.items()
            if not str(ds).startswith("_")
        }

    payload = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/05_evaluate_gardian.py",
            "args": vars(args),
            "git_revision": _git_revision(),
            "platform": platform.platform(),
            "python_version": sys.version,
        },
        "results": normalized_all,
    }
    validate_evaluation_results(payload)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    logger.success(f"Results saved to {out_path}")

    # Print final summary with analysis
    print("\n" + "=" * 100)
    print("FINAL SUMMARY - nDCG@10 COMPARISON BY RETRIEVER")
    print("=" * 100)

    for retriever, retriever_results in all_results.items():
        print(f"\n[{retriever.upper()}]")
        for dataset_name, dataset_results in retriever_results.items():
            print(f"  {dataset_name.upper()}:")
            dense_baseline = _metric(dataset_results, "dense", "ndcg@10")
            gardian_ndcg10 = _metric(dataset_results, "gardian", "ndcg@10")
            gardian_mrr = _metric(dataset_results, "gardian", "mrr")
            best_label, best_baseline = _best_non_gardian_baseline(dataset_results, retriever)
            dense_label = _baseline_label("dense", retriever)
            ce_ndcg10 = _metric(dataset_results, "cross_encoder", "ndcg@10")
            ce_line = ""
            if isinstance(dataset_results.get("cross_encoder"), dict):
                ce_line = (
                    f" | cross_encoder nDCG@10: {ce_ndcg10:.4f} | "
                    f"Δ GARDIAN vs CE: {_delta_text(gardian_ndcg10, ce_ndcg10)}"
                )
            print(
                f"    GARDIAN nDCG@10: {gardian_ndcg10:.4f} | MRR: {gardian_mrr:.4f} | "
                f"Δ vs {dense_label}: {_delta_text(gardian_ndcg10, dense_baseline)} | "
                f"Δ vs best baseline ({best_label}={best_baseline:.4f}): "
                f"{_delta_text(gardian_ndcg10, best_baseline)}"
                f"{ce_line}"
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
            ranking.append(
                (
                    retriever,
                    r.get("gardian", {}).get("ndcg@10", 0.0),
                    r.get("gardian", {}).get("mrr", 0.0),
                )
            )
        ranking.sort(key=lambda x: x[1], reverse=True)
        for retriever, ndcg10, mrr in ranking:
            print(f"  {retriever:10} nDCG@10: {ndcg10:.4f} | MRR: {mrr:.4f}")

    print("=" * 100)

    # Observations
    print("\n" + "=" * 100)
    print("OBSERVATIONS & ANALYSIS")
    print("=" * 100)


if __name__ == "__main__":
    main()
