"""
ULTRA-FAST EVALUATION - Uses pre-generated rank data for ALL systems
With CORRECTED MRR calculation

Also reports Hit@k (success rate) when rank JSONL eval includes hit@ metrics.
"""

import argparse
import json
import pathlib
import random
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, ".")

from src.common.question_types import assert_cfg_question_types
from src.common.rank_data_paths import normalize_retriever_name, resolve_rank_data_file
from src.evaluation.baseline_systems import normalize_eval_results
from src.baselines.fusion import global_alpha_fit, group_alpha_fit
from src.common.question_types import normalize_question_type
from src.evaluation.rank_jsonl_eval import (
    evaluate_all_from_rank_data,
    iter_rank_jsonl_records,
)
from src.evaluation.schemas import validate_evaluation_results
from src.common.repro import set_seed
from src.common.seeds import (
    DEFAULT_SEEDS,
    add_seeds_argument,
    guard_seed_artifact,
    parse_seeds,
    resolve_seed_checkpoint,
    seed_dir,
)
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


# The three paper QA collections. `--datasets all` is exactly this list.
EVAL_DATASET_SPLITS: List[Tuple[str, str]] = [
    ("pubmedqa_labeled", "eval"),
    ("pubmedqa_artificial", "test"),
    ("medmcqa", "test"),
]

# Datasets with no dev split of their own, and the sibling whose dev split is
# distributionally closest. Used only to fit fusion baselines; never to fit
# GARDIAN, and never the split being reported.
DEV_FIT_FALLBACK: Dict[str, str] = {
    "pubmedqa_labeled": "pubmedqa_artificial",
}


def fit_fusion_alphas(
    retriever: str,
    dataset_name: str,
    *,
    k: int = 10,
    fit_group_alpha: bool = True,
) -> tuple[Optional[float], Optional[Dict[str, float]]]:
    """
    Grid-search Global-alpha (and Group-alpha) on this dataset's DEV split.

    Fitting on dev and applying at test is what separates a baseline from an
    oracle. PubMedQA-Labeled is eval-only, so it borrows PubMedQA-Artificial
    dev rather than fitting on the split being reported.

    Group-alpha is one alpha per question type. On both PubMedQA splits every
    query is yes/no by construction, so Group-alpha is *identical to
    Global-alpha by definition* there and is only informative on MedMCQA -- it
    is skipped where it would be vacuous.
    """
    dev_path = resolve_rank_data_file(retriever, dataset_name, "dev")
    fitted_on = dataset_name
    if not pathlib.Path(dev_path).exists():
        # PubMedQA-Labeled is eval-only. Fitting alpha on the split we report
        # would make the baseline an oracle; omitting it entirely would leave a
        # ceiling row with no tuned baseline under it, which is worse. Instead
        # fit on a sibling split from the same collection -- PubMedQA-Labeled
        # and PubMedQA-Artificial are both PubMed abstracts with yes/no
        # questions, so the alpha transfers and nothing is fitted on test.
        sibling = DEV_FIT_FALLBACK.get(dataset_name)
        if sibling:
            candidate = resolve_rank_data_file(retriever, sibling, "dev")
            if pathlib.Path(candidate).exists():
                dev_path, fitted_on = candidate, sibling
                logger.info(
                    f"  {dataset_name} has no dev split; fitting fusion alphas on "
                    f"{sibling} dev instead (same collection, never the reported split)"
                )
    if not pathlib.Path(dev_path).exists():
        logger.info(
            f"  no dev split for {dataset_name} and no fallback; skipping "
            "Global-alpha/Group-alpha rather than fitting on the reported split"
        )
        return None, None

    logger.info(f"  fitting fusion alphas on: {dev_path}")
    pools: Dict[str, Dict[str, Any]] = {}
    qtypes: Dict[str, str] = {}
    for rec in iter_rank_jsonl_records(dev_path):
        qid = rec["qid"]
        pool = pools.setdefault(qid, {"pid": [], "label": [], "sparse": [], "dense": []})
        pool["pid"].append(rec["pid"])
        pool["label"].append(int(rec.get("label", 0)))
        pool["sparse"].append(float((rec.get("sparse_feats") or [0.0])[0]))
        pool["dense"].append(float((rec.get("dense_feats") or [0.0])[0]))
        if qid not in qtypes:
            qtypes[qid] = normalize_question_type(rec.get("question_type"))

    if not pools:
        return None, None

    alpha, fit_ndcg = global_alpha_fit(pools, k=k)
    logger.info(
        f"  Global-alpha = {alpha:.2f} (fit nDCG@{k}={fit_ndcg:.4f}, "
        f"{len(pools):,} queries, fitted on {fitted_on})"
    )

    groups = None
    if fit_group_alpha and len({qtypes[q] for q in pools}) > 1:
        groups = group_alpha_fit(pools, qtypes, k=k)
        logger.info(
            "  Group-alpha = "
            + ", ".join(f"{g}:{a:.2f}" for g, a in sorted(groups.items()))
        )
    elif fit_group_alpha:
        logger.info(
            "  Group-alpha skipped: one question type on this dataset, so it is "
            "identical to Global-alpha by construction"
        )
    return float(alpha), groups


def query_emb_cache_for(retriever: str, dataset_name: str, split: str) -> Optional[str]:
    """
    Best available precomputed query-embedding cache for this split.

    Evaluation otherwise re-encodes every query with a BERT forward pass. The
    all-split cache covers every dataset for a back-end, so it is preferred;
    the per-split file is the fallback.
    """
    # Dataset-specific first; the all-split cache is the fallback.
    candidates = [
        f"data/query_emb_cache_{retriever}_{dataset_name}_{split}.pkl",
        f"data/query_emb_cache_{retriever}_all.pkl",
        f"data/query_emb_cache_{retriever}_train_all.pkl",
    ]
    for c in candidates:
        if pathlib.Path(c).is_file():
            return c
    logger.warning(
        f"no query_emb cache for {retriever}/{dataset_name}; queries will be "
        "re-encoded (slow). Build one with scripts/12_precompute_query_cache.py"
    )
    return None


def crossval_global_alpha(
    retriever: str,
    dataset_name: str,
    split: str,
    *,
    k: int = 10,
    folds: int = 5,
    seed: int = 42,
) -> Optional[Dict[str, float]]:
    """
    Per-query Global-alpha by k-fold cross-validation within one split.

    For collections with no development split and no sibling to borrow from
    (TREC-COVID is a different corpus from the PubMedQA family), the choice is
    between omitting the strongest non-adaptive baseline entirely and fitting
    it on the split being reported. Neither is acceptable: the first leaves a
    ceiling with nothing under it, the second is an oracle.

    Cross-validation resolves it. Each query is scored with an alpha fitted on
    the folds that exclude it, so no query's weight is chosen using its own
    labels, while every query still gets a tuned weight. Report it as
    "Global-alpha (k-fold CV)" -- it is a slightly *stronger* baseline than a
    dev-fitted one, since it is tuned on same-collection data.
    """
    path = resolve_rank_data_file(retriever, dataset_name, split)
    if not pathlib.Path(path).is_file():
        return None

    pools: Dict[str, Dict[str, Any]] = {}
    for rec in iter_rank_jsonl_records(path):
        pool = pools.setdefault(
            rec["qid"], {"pid": [], "label": [], "sparse": [], "dense": []}
        )
        pool["pid"].append(rec["pid"])
        pool["label"].append(int(rec.get("label", 0)))
        pool["sparse"].append(float((rec.get("sparse_feats") or [0.0])[0]))
        pool["dense"].append(float((rec.get("dense_feats") or [0.0])[0]))
    if not pools:
        return None

    qids = sorted(pools)
    rng = random.Random(seed)
    rng.shuffle(qids)
    n_folds = max(2, min(int(folds), len(qids)))

    out: Dict[str, float] = {}
    for f in range(n_folds):
        held = set(qids[f::n_folds])
        train = {q: pools[q] for q in qids if q not in held}
        if not train:
            continue
        alpha, _ = global_alpha_fit(train, k=k)
        for q in held:
            out[q] = float(alpha)

    chosen = sorted({round(a, 2) for a in out.values()})
    logger.info(
        f"  Global-alpha by {n_folds}-fold CV on {dataset_name} ({len(out):,} queries); "
        f"per-fold alphas: {chosen}"
    )
    return out


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


def resolve_checkpoint(cfg, retriever: str, seed: int) -> pathlib.Path:
    """Locate the GARDIAN checkpoint for one (retriever, seed) cell."""
    return resolve_seed_checkpoint(cfg.paths.results_dir, retriever, seed)


def build_model(cfg, device: str, retriever: str, seed: int) -> Tuple[GARDIAN, int]:
    """Load trained GARDIAN model for one (retriever, seed) cell."""
    ckpt_path = resolve_checkpoint(cfg, retriever, seed)

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
    logger.info(f"Loaded checkpoint for {retriever} (seed={seed}) from {ckpt_path}")
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
        ("hybrid", "hybrid", f"sum-unnorm({sparse_name}+{dense_name})"),
        ("rrf", "rrf", f"rrf({sparse_name}+{dense_name})"),
    ]
    # Dev-fitted fusion baselines. These are the strongest non-adaptive
    # alternatives and the ones reviewers ask for; they belong above GARDIAN.
    meta = results.get("_meta", {}) if isinstance(results.get("_meta"), dict) else {}
    ga_label = (
        "global-alpha (CV-fit)" if meta.get("global_alpha_cv")
        else "global-alpha (dev-fit)"
    )
    for key, label in (("global_alpha", ga_label),
                       ("group_alpha", "group-alpha (dev-fit)")):
        if _has_metric_block(results, key):
            row_specs.append((key, key, label))
    if include_cross_encoder and _has_metric_block(results, "cross_encoder"):
        row_specs.append(("cross_encoder", "cross_encoder", "cross_encoder"))
    row_specs.append(("gardian", "gardian", "gardian"))
    # Ceiling LAST and clearly marked: it selects each query's alpha using this
    # split's labels, so it is a diagnostic bound on the linear-fusion family,
    # never a system GARDIAN competes with.
    if _has_metric_block(results, "oracle_alpha"):
        row_specs.append(("oracle_alpha", "oracle_alpha", "[ceiling] oracle-alpha"))
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
        def cell(key: str, width: int) -> str:
            """Blank rather than 0.0000 for a metric this row does not define."""
            v = m.get(key)
            return f"{'--':>{width}}" if v is None else f"{float(v):>{width}.4f}"

        print(
            f"{display_name:<24} "
            f"{cell('ndcg@10', 10)} {cell('ndcg@20', 10)} "
            f"{cell('ndcg@50', 10)} {cell('ndcg@100', 10)} "
            f"{cell('recall@20', 11)} {cell('recall@50', 11)} "
            f"{cell('recall@100', 12)} {cell('mrr', 10)}"
        )

    meta = results.get("_meta", {}) if isinstance(results.get("_meta"), dict) else {}
    notes = []
    if meta.get("global_alpha") is not None:
        notes.append(f"global-alpha={meta['global_alpha']:.2f} (fitted on dev)")
    if meta.get("pool_recall") is not None:
        notes.append(f"pool_recall={meta['pool_recall']:.4f}")
    if meta.get("oracle_rerank_ndcg@10") is not None:
        notes.append(f"oracle-rerank nDCG@10={meta['oracle_rerank_ndcg@10']:.4f}")
    oa = meta.get("oracle_alpha") or {}
    if oa.get("tie_fraction") is not None:
        notes.append(
            f"alpha-grid ties={oa['tie_fraction']:.2f}, "
            f"alpha-invariant queries={oa.get('invariant_fraction', float('nan')):.2f}"
        )
    if notes:
        print("  " + " | ".join(notes))
        print(
            "  [ceiling] rows use this split's labels and are upper bounds, "
            "not competing systems."
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
    per_path = guard_seed_artifact(
        out_dir / f"evaluation_{retriever}{output_suffix}.json",
        overwrite=bool(getattr(args, "overwrite_seed_artifacts", False)),
    )
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
            "pubmedqa_artificial, medmcqa, or all (default: those three)."
        ),
    )
    parser.add_argument(
        "--query-encoder-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help=(
            "Device for encoding queries that miss the precomputed cache. "
            "Defaults to CUDA when available; a full cache means nothing is "
            "encoded at all. Use 'cpu' to leave the GPU free."
        ),
    )
    parser.add_argument(
        "--no-oracle-alpha",
        action="store_true",
        help=(
            "Skip the Oracle-alpha ceiling row. It grid-searches every query's "
            "best alpha against this split's labels, which costs a 101-point "
            "sweep per query; the row is a diagnostic bound, never a baseline."
        ),
    )
    add_seeds_argument(
        parser,
        default=DEFAULT_SEEDS,
        help_suffix="Each seed is evaluated from its own checkpoint.",
    )
    args = parser.parse_args()
    args.seeds = parse_seeds(args.seeds)

    cfg = OmegaConf.load("configs/base.yaml")
    assert_cfg_question_types(cfg.evaluation.question_types)

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
    all_known = {name for name, _ in EVAL_DATASET_SPLITS}
    if args.datasets.strip().lower() == "all":
        dataset_splits = list(EVAL_DATASET_SPLITS)
    else:
        allowed = {d.strip() for d in args.datasets.split(",") if d.strip()}
        unknown = allowed - all_known
        if unknown:
            raise SystemExit(
                f"Unknown dataset(s) in --datasets: {sorted(unknown)}. "
                f"Known: {sorted(all_known)}"
            )
        dataset_splits = [ds for ds in EVAL_DATASET_SPLITS if ds[0] in allowed]
        if not dataset_splits:
            raise SystemExit(f"No datasets matched --datasets {args.datasets!r}")

    for seed in args.seeds:
        _evaluate_seed(
            cfg,
            args,
            device=device,
            seed=seed,
            retrievers=retrievers,
            dataset_splits=dataset_splits,
        )

    logger.success(
        f"Evaluation complete for seeds {args.seeds}. Run "
        "scripts/aggregate_seeds.py to produce the mean/std paper tables."
    )


def _evaluate_seed(
    cfg,
    args: argparse.Namespace,
    *,
    device: str,
    seed: int,
    retrievers,
    dataset_splits,
) -> None:
    """
    Run the full evaluation for one seed.

    Reads that seed's GARDIAN checkpoints and writes every artifact beneath
    ``results/seeds/seed_<seed>/``, so no seed can overwrite another.
    ``scripts/aggregate_seeds.py`` turns these into the mean/std tables the
    paper reports.
    """
    set_seed(seed, cudnn_deterministic=True)
    logger.info("#" * 72)
    logger.info(f"# EVALUATING SEED {seed}")
    logger.info("#" * 72)

    all_results: Dict[str, Dict[str, Any]] = {}
    out_dir = seed_dir(cfg.paths.results_dir, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_suffix = _evaluation_output_suffix(args.use_cross_encoder_rank_data)
    save_per_retriever = bool(args.per_retriever_json or args.use_cross_encoder_rank_data)

    for retriever in retrievers:
        retriever_desc = HYBRID_RETRIEVER_COMBINATIONS.get(retriever, "single retriever")
        logger.info("\n" + "=" * 72)
        logger.info(f"EVALUATION RUN FOR RETRIEVER: {retriever.upper()} ({retriever_desc})")
        logger.info("=" * 72)

        try:
            model, expected_qdim = build_model(cfg, device, retriever, seed)
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

            fit_alpha, fit_groups = fit_fusion_alphas(retriever, dataset_name)
            cv_alpha = None
            if fit_alpha is None:
                cv_alpha = crossval_global_alpha(retriever, dataset_name, split)
            qcache = query_emb_cache_for(retriever, dataset_name, split)
            fusion_kw = dict(
                global_alpha=fit_alpha,
                global_alpha_per_query=cv_alpha,
                group_alphas=fit_groups,
                # None means "read the question type from the rank records",
                # which is what makes Group-alpha actually per-group at test time.
                question_types=None,
                include_oracle_alpha=not args.no_oracle_alpha,
            )

            # Ultra-fast evaluation using pre-computed rank data
            try:
                results = evaluate_all_from_rank_data(
                    rank_data_path,
                    model,
                    device,
                    query_encoder_name=str(cfg.encoder.model_name),
                    query_encoder_device=args.query_encoder_device,
                    query_emb_cache_path=qcache,
                    expected_query_feat_dim=expected_qdim,
                    include_standalone_spladepp=False,
                    gardian_adaptive_retrieval=gardian_adaptive,
                    cfg=cfg,
                    **fusion_kw,
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
                        query_emb_cache_path=qcache,
                        expected_query_feat_dim=expected_qdim,
                        include_standalone_spladepp=False,
                        gardian_adaptive_retrieval=gardian_adaptive,
                        cfg=cfg,
                        **fusion_kw,
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
        # Keep an explicit --output per-seed so seeds cannot overwrite each other.
        explicit = pathlib.Path(args.output)
        out_path = out_dir / explicit.name if len(args.seeds) > 1 else explicit
    elif len(retrievers) == 1:
        out_path = out_dir / f"evaluation_{retrievers[0]}{output_suffix}.json"
    else:
        out_path = out_dir / f"evaluation_results_all_retrievers{output_suffix}.json"
    out_path = guard_seed_artifact(
        out_path, overwrite=bool(getattr(args, "overwrite_seed_artifacts", False))
    )

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
            "seed": int(seed),
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
