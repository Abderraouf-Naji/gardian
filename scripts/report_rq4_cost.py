#!/usr/bin/env python3
"""
RQ4 cost table: what adaptive re-ranking costs in latency, training and size.

Reviewer 4 accepted the effectiveness/latency trade-off but objected that the
controller is trained by supervised learning and the training cost never
appears. Reviewer 2 asked for the settings behind the reported latency. This
script assembles both from artifacts that already exist, so nothing here is a
re-derivation or a remembered number:

    results/aggregated/training_summary.json   wall clock + parameters, 5 seeds
    results/seeds/seed_<S>/training_manifest.json   hardware and software stack
    results/gardian_lite/...                   the no-controller arm
    results/ltr_baseline_<retriever>.json      LambdaMART on identical features
    results/reranker_comparison_<retriever>.json    measured per-query latency
    configs/base.yaml                          batch size, precision, objective

One asymmetry is stated rather than hidden: the cross-encoders are used
off-the-shelf, so their *marginal* training cost in this paper is zero. That
does not make them cheaper to obtain -- it makes their training cost external
and amortised, while GARDIAN's is paid here and reported here. The comparison
that is actually like-for-like is GARDIAN against LambdaMART, which is trained
on the same data from the same features.

Writes results/rq4_cost_summary.json and a LaTeX table.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

REPO = pathlib.Path(__file__).resolve().parents[1]

# LaTeX-safe: a bare "SPLADE+++FAISS" is unreadable, and "++" wants a
# fixed-width font to render as the model name rather than as arithmetic.
BACKEND_LABELS = {
    "hybrid_bm25_faiss": "BM25+FAISS",
    "hybrid_bm25_medcpt": "BM25+MedCPT",
    "hybrid_spladepp_faiss": r"SPLADE\texttt{++}+FAISS",
    "hybrid_spladepp_medcpt": r"SPLADE\texttt{++}+MedCPT",
}

# Parameter counts of the re-rankers GARDIAN is compared against. These are
# properties of the published checkpoints, not of anything trained here.
CE_PARAMS = {
    "cross_encoder_msmarco_minilm": ("MiniLM-L6", 22_700_000),
    "cross_encoder_bge_v2_m3": ("BGE-reranker-v2-m3", 568_000_000),
    "cross_encoder_monot5_med": ("MonoT5-base-med", 223_000_000),
    "cross_encoder_monobert": ("MonoBERT-large", 335_000_000),
}


def _load(path: pathlib.Path) -> Optional[Any]:
    if not path.is_file():
        logger.warning(f"Missing: {path}")
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _fmt_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def collect_training_cost(results_dir: pathlib.Path) -> Dict[str, Any]:
    """Per-back-end wall clock and parameter count, aggregated over seeds."""
    agg = _load(results_dir / "aggregated" / "training_summary.json")
    if not agg:
        return {}
    out: Dict[str, Any] = {}
    for retriever, block in agg.get("results", {}).items():
        wall = block.get("train_wall_clock_sec", {})
        params = block.get("trainable_parameters", {})
        ndcg = block.get("best_ndcg10", {})
        out[retriever] = {
            "label": BACKEND_LABELS.get(retriever, retriever),
            "train_wall_clock_sec_mean": wall.get("mean"),
            "train_wall_clock_sec_std": wall.get("std"),
            "train_wall_clock_hms": _fmt_hms(float(wall.get("mean") or 0.0)),
            "n_seeds": wall.get("n"),
            "trainable_parameters": int(params.get("mean") or 0),
            "dev_ndcg10_mean": ndcg.get("mean"),
            "dev_ndcg10_std": ndcg.get("std"),
        }
    out["_meta"] = {"seeds": agg.get("meta", {}).get("seeds", [])}
    return out


def collect_lite(lite_dir: pathlib.Path, seed: int) -> Dict[str, Any]:
    """The --no-controller arm: same protocol, one seed, its own artifact tree."""
    summary = _load(lite_dir / "seeds" / f"seed_{seed}" / "training_summary_all_retrievers.json")
    if not summary:
        return {}
    out: Dict[str, Any] = {}
    for retriever, block in summary.items():
        if not isinstance(block, dict):
            continue
        out[retriever] = {
            "label": BACKEND_LABELS.get(retriever, retriever),
            "train_wall_clock_sec": block.get("train_wall_clock_sec"),
            "train_wall_clock_hms": _fmt_hms(float(block.get("train_wall_clock_sec") or 0.0)),
            "trainable_parameters": block.get("trainable_parameters"),
            "dev_ndcg10": block.get("best_ndcg10"),
            "seed": block.get("seed"),
        }
    return out


def collect_train_step_cost(results_dir: pathlib.Path) -> Dict[str, Any]:
    """
    Controlled per-step training cost of the controller.

    The wall-clock numbers in the two training summaries are NOT comparable:
    the GARDIAN-Lite runs shared the GPU with other jobs and the original runs
    did not, so their difference is contention. scripts/benchmark_train_step.py
    interleaves both arms in one process to get a difference that is about the
    architecture.
    """
    payload = _load(results_dir / "train_step_cost.json")
    if not payload:
        return {}
    over = payload.get("controller_step_overhead", {})
    return {
        "with_controller_ms": payload.get("with_controller", {}).get("p50_ms_per_step_min"),
        "no_controller_ms": payload.get("no_controller", {}).get("p50_ms_per_step_min"),
        "percent_slower_with_controller": over.get("percent_slower_with_controller"),
        "absolute_ms_per_step": over.get("absolute_ms_per_step"),
        "repeats": payload.get("meta", {}).get("repeats"),
        "steps_per_repeat": payload.get("meta", {}).get("steps_per_repeat"),
        "gpu_name": payload.get("meta", {}).get("gpu_name"),
    }


def collect_environment(results_dir: pathlib.Path, seed: int) -> Dict[str, Any]:
    """Hardware and software stack, from the manifest the training run wrote."""
    man = _load(results_dir / "seeds" / f"seed_{seed}" / "training_manifest.json")
    if not man:
        return {}
    return {
        "gpu_name": man.get("gpu_name"),
        "device": man.get("device"),
        "torch_version": man.get("torch_version"),
        "cuda_version": man.get("cuda_version"),
        "python_version": man.get("python_version"),
        "platform": man.get("platform"),
        "git_revision": man.get("git_revision"),
    }


def collect_hyperparameters(cfg: Any) -> Dict[str, Any]:
    """The batch/precision settings reviewers asked to see stated."""
    t = cfg.training
    m = cfg.model
    return {
        "objective": str(t.loss),
        "epochs": int(t.epochs),
        "learning_rate": float(t.lr),
        "weight_decay": float(t.weight_decay),
        "batch_size_queries": int(t.batch_size),
        "listwise_group_size": int(t.listwise_group_size),
        "scored_rows_per_step": int(t.batch_size) * int(t.listwise_group_size),
        "listwise_ndcg_k": int(t.listwise_ndcg_k),
        "warmup_epochs": int(t.warmup_epochs),
        "min_lr_ratio": float(t.min_lr_ratio),
        "num_negatives": int(t.num_negatives),
        "hard_negative_fraction": float(t.hard_negative_fraction),
        "precision": "mixed fp16 (torch.amp.autocast + GradScaler on CUDA)",
        "branch_hidden": int(m.branch_hidden),
        "controller_hidden": int(m.controller_hidden),
        "query_feat_dim": int(m.query_feat_dim),
        "dropout": float(m.dropout),
        "query_encoder": str(cfg.encoder.model_name),
        "query_embeddings_precomputed_in_training": bool(t.precompute_query_emb),
        "candidate_pool_size": int(cfg.retrieval.candidate_pool_size),
    }


def _merged_results(results_dir: pathlib.Path, stem: str) -> Dict[str, Any]:
    """
    Merge a comparison file with its separate PQA-A run.

    PubMedQA-artificial's test split is ~1.8M pool rows, so it is scored on a
    1000-query subsample in its own job and written to <stem>_pqa_a_q1000.json.
    plot_reranker_comparison.py merges the two the same way.
    """
    main = _load(results_dir / f"{stem}.json")
    if not main:
        return {}
    merged = dict(main.get("results") or {})
    extra = _load(results_dir / f"{stem}_pqa_a_q1000.json")
    if extra:
        for dataset, block in (extra.get("results") or {}).items():
            merged[dataset] = block
    return {"meta": main.get("meta", {}), "results": merged}


def collect_latency(results_dir: pathlib.Path, retrievers: List[str]) -> Dict[str, Any]:
    """Measured per-query re-ranking latency, per back-end and dataset."""
    out: Dict[str, Any] = {}
    for retriever in retrievers:
        payload = _merged_results(results_dir, f"reranker_comparison_{retriever}")
        if not payload:
            continue
        meta = payload.get("meta", {})
        if not meta.get("latency_measured"):
            logger.warning(
                f"{retriever}: comparison file has no timing pass "
                "(rerun with --measure-latency on an idle GPU)"
            )
        contention = meta.get("latency_gpu_contention") or {}
        if contention.get("available") and not contention.get("exclusive", True):
            logger.error(
                f"{retriever}: latency was measured with "
                f"{contention.get('other_processes')} other process(es) on the GPU. "
                "These numbers report queueing, not compute -- do NOT publish them. "
                "Re-run scripts/run_rq4_latency.sh on an idle device."
            )
        per_dataset: Dict[str, Any] = {}
        for dataset, systems in payload.get("results", {}).items():
            row: Dict[str, Any] = {}
            for system, block in systems.items():
                lat = block.get("latency_ms") or {}
                if not lat:
                    continue
                entry = {
                    "p50_ms": lat.get("p50_ms"),
                    "mean_ms": lat.get("mean_ms"),
                    "p95_ms": lat.get("p95_ms"),
                    "ndcg@10": (block.get("metrics") or {}).get("ndcg@10"),
                }
                if system == "gardian" and "breakdown" in lat:
                    bd = lat["breakdown"]
                    entry["query_encoder_p50_ms"] = bd.get("query_encoder", {}).get("p50_ms")
                    entry["ranking_head_p50_ms"] = bd.get("ranking_head", {}).get("p50_ms")
                    entry["query_encoder_timed"] = bd.get("query_encoder_timed")
                if system in CE_PARAMS:
                    entry["parameters"] = CE_PARAMS[system][1]
                    entry["model"] = CE_PARAMS[system][0]
                    entry["batch_size"] = lat.get("batch_size")
                    entry["fp16"] = lat.get("fp16")
                row[system] = entry
            if row:
                per_dataset[dataset] = row
        if per_dataset:
            out[retriever] = {
                "gpu_name": meta.get("gpu_name"),
                "latency_queries": meta.get("latency_queries"),
                "latency_warmup": meta.get("latency_warmup"),
                "gpu_contention": contention,
                "latency_trustworthy": bool(contention.get("exclusive", False))
                if contention.get("available")
                else None,
                "datasets": per_dataset,
            }
    return out


def collect_ltr(results_dir: pathlib.Path, retrievers: List[str]) -> Dict[str, Any]:
    """LambdaMART: the like-for-like trained control on identical features."""
    out: Dict[str, Any] = {}
    for retriever in retrievers:
        payload = _load(results_dir / f"ltr_baseline_{retriever}.json")
        if not payload:
            continue
        block = (payload.get("results") or {}).get(retriever)
        if not block:
            continue
        # PubMedQA-artificial must be read from the run that used the SAME
        # 1000-query subsample as the re-ranker comparison. The full-split
        # number is a different population and is not comparable to GARDIAN's.
        sub = _load(results_dir / f"ltr_baseline_{retriever}_pqa_a_q1000.json")
        if sub:
            sub_block = ((sub.get("results") or {}).get(retriever) or {}).get("results") or {}
            if "pubmedqa_artificial" in sub_block:
                block.setdefault("results", {})["pubmedqa_artificial"] = sub_block[
                    "pubmedqa_artificial"
                ]
        else:
            logger.warning(
                f"{retriever}: no matched PQA-A subsample for LambdaMART; its "
                "pubmedqa_artificial number is on the full split and is NOT "
                "comparable to GARDIAN's 1000-query figure."
            )
        out[retriever] = {
            "label": BACKEND_LABELS.get(retriever, retriever),
            "train_wall_clock_sec": block.get("train_wall_clock_sec"),
            "train_wall_clock_hms": _fmt_hms(float(block.get("train_wall_clock_sec") or 0.0)),
            "train_queries": block.get("train_queries"),
            "n_trees": block.get("n_trees"),
            "best_iteration": block.get("best_iteration"),
            "model_size_bytes": block.get("model_size_bytes"),
            "n_features": block.get("n_features"),
            "device": "CPU",
            "results": {
                ds: {
                    "ndcg@10": (b.get("metrics") or {}).get("ndcg@10"),
                    "mrr": (b.get("metrics") or {}).get("mrr"),
                    "recall@10": (b.get("metrics") or {}).get("recall@10"),
                    "p50_ms": (b.get("latency_ms") or {}).get("p50_ms"),
                    "n_queries": b.get("n_queries"),
                }
                for ds, b in (block.get("results") or {}).items()
            },
        }
    return out


DATASET_LABELS = {
    "pubmedqa_labeled": "PQA-L",
    "medmcqa": "MedMCQA",
    "pubmedqa_artificial": "PQA-A",
}


def collect_lite_eval(results_dir: pathlib.Path, retrievers: List[str]) -> Dict[str, Any]:
    """
    Effectiveness and latency of the --no-controller arm on the same pools.

    Produced by scripts/14_compare_rerankers.py with --gardian-results-dir
    pointing at the Lite artifact tree, so the candidates, the splits and the
    metric code are identical to the main arm's.
    """
    out: Dict[str, Any] = {}
    for retriever in retrievers:
        payload = _merged_results(results_dir, f"reranker_comparison_{retriever}_lite")
        if not payload:
            continue
        per_dataset: Dict[str, Any] = {}
        for dataset, systems in payload.get("results", {}).items():
            block = systems.get("gardian")
            if not block:
                continue
            lat = block.get("latency_ms") or {}
            per_dataset[dataset] = {
                "ndcg@10": (block.get("metrics") or {}).get("ndcg@10"),
                "mrr": (block.get("metrics") or {}).get("mrr"),
                "recall@10": (block.get("metrics") or {}).get("recall@10"),
                "p50_ms": lat.get("p50_ms"),
                "query_encoder_timed": (lat.get("breakdown") or {}).get("query_encoder_timed"),
            }
        if per_dataset:
            out[retriever] = per_dataset
    return out


def _gardian_cell(payload: Dict[str, Any], retriever: str, dataset: str) -> Dict[str, Any]:
    """The controller-on GARDIAN row for one (back-end, dataset) cell."""
    lat_block = (payload.get("latency") or {}).get(retriever) or {}
    ds = (lat_block.get("datasets") or {}).get(dataset) or {}
    return ds.get("gardian") or {}


PLAIN_BACKEND_LABELS = {
    "hybrid_bm25_faiss": "BM25+FAISS",
    "hybrid_bm25_medcpt": "BM25+MedCPT",
    "hybrid_spladepp_faiss": "SPLADE+++FAISS",
    "hybrid_spladepp_medcpt": "SPLADE+++MedCPT",
}

# Order of the RQ4 cost comparison, cheapest re-ranking step first.
COST_ROW_ORDER = [
    "rrf",
    "gardian_lite",
    "lambdamart",
    "gardian",
    "cross_encoder_msmarco_minilm",
    "cross_encoder_monot5_med",
    "cross_encoder_monobert",
    "cross_encoder_bge_v2_m3",
]

COST_ROW_LABELS = {
    "rrf": "RRF (no model)",
    "gardian_lite": "GARDIAN-Lite",
    "lambdamart": "LambdaMART",
    "gardian": "GARDIAN",
    "cross_encoder_msmarco_minilm": "CE: MiniLM-L6",
    "cross_encoder_monot5_med": "CE: MonoT5-base-med",
    "cross_encoder_monobert": "CE: MonoBERT-large",
    "cross_encoder_bge_v2_m3": "CE: BGE-v2-m3",
}

COST_DATASETS = ["pubmedqa_labeled", "medmcqa", "pubmedqa_artificial"]


def latex_cost_comparison_table(payload: Dict[str, Any], retriever: str) -> str:
    """
    The table RQ4 asks for directly: latency, training cost and model size for
    every re-ranker, on one back-end, with effectiveness alongside.
    """
    lat_block = (payload.get("latency") or {}).get(retriever) or {}
    per_dataset = lat_block.get("datasets") or {}
    lite_eval = (payload.get("gardian_lite_eval") or {}).get(retriever) or {}
    lite_train = (payload.get("gardian_lite") or {}).get(retriever) or {}
    ltr = (payload.get("lambdamart") or {}).get(retriever) or {}
    train = (payload.get("training") or {}).get(retriever) or {}
    ce_params = payload.get("cross_encoder_parameters") or {}

    def _fmt(x, spec="{:.4f}", dash="--"):
        return spec.format(float(x)) if x is not None else dash

    def _ndcg_cells(getter) -> str:
        return " & ".join(_fmt(getter(ds)) for ds in COST_DATASETS)

    rows: List[str] = []
    for key in COST_ROW_ORDER:
        label = COST_ROW_LABELS[key]

        if key == "lambdamart":
            params = "475 trees"
            size = _fmt(ltr.get("model_size_bytes", 0) / 1e6 if ltr else None, "{:.1f}")
            train_cost = (
                f"{float(ltr['train_wall_clock_sec']) / 60.0:.0f} min (CPU)"
                if ltr.get("train_wall_clock_sec")
                else "--"
            )
            lat = next(
                (
                    b.get("p50_ms")
                    for ds, b in (ltr.get("results") or {}).items()
                    if b.get("p50_ms")
                ),
                None,
            )
            ndcg = _ndcg_cells(
                lambda ds: ((ltr.get("results") or {}).get(ds) or {}).get("ndcg@10")
            )

        elif key == "gardian_lite":
            n = lite_train.get("trainable_parameters")
            params = _fmt(n / 1e6 if n else None, "{:.2f}M")
            size = _fmt(n * 4 / 1e6 if n else None, "{:.1f}")
            train_cost = (
                f"{float(lite_train['train_wall_clock_sec']) / 60.0:.0f} min (GPU)"
                if lite_train.get("train_wall_clock_sec")
                else "--"
            )
            lat = next(
                (b.get("p50_ms") for b in lite_eval.values() if b.get("p50_ms")), None
            )
            ndcg = _ndcg_cells(lambda ds: (lite_eval.get(ds) or {}).get("ndcg@10"))

        elif key == "gardian":
            n = train.get("trainable_parameters")
            params = _fmt(n / 1e6 if n else None, "{:.2f}M")
            size = _fmt(n * 4 / 1e6 if n else None, "{:.1f}")
            mean = train.get("train_wall_clock_sec_mean")
            train_cost = f"{float(mean) / 60.0:.0f} min (GPU)" if mean else "--"
            lat = next(
                (
                    (b.get("gardian") or {}).get("p50_ms")
                    for b in per_dataset.values()
                    if (b.get("gardian") or {}).get("p50_ms")
                ),
                None,
            )
            ndcg = _ndcg_cells(
                lambda ds: ((per_dataset.get(ds) or {}).get("gardian") or {}).get("ndcg@10")
            )

        elif key == "rrf":
            params = "--"
            size = "--"
            train_cost = "none"
            lat = next(
                (
                    (b.get("rrf") or {}).get("p50_ms")
                    for b in per_dataset.values()
                    if (b.get("rrf") or {}).get("p50_ms")
                ),
                None,
            )
            ndcg = _ndcg_cells(
                lambda ds: ((per_dataset.get(ds) or {}).get("rrf") or {}).get("ndcg@10")
            )

        else:
            n = (ce_params.get(key) or {}).get("parameters")
            params = _fmt(n / 1e6 if n else None, "{:.0f}M")
            size = _fmt(n * 4 / 1e6 if n else None, "{:.0f}")
            train_cost = "off-the-shelf"
            lat = next(
                (
                    (b.get(key) or {}).get("p50_ms")
                    for b in per_dataset.values()
                    if (b.get(key) or {}).get("p50_ms")
                ),
                None,
            )
            ndcg = _ndcg_cells(
                lambda ds: ((per_dataset.get(ds) or {}).get(key) or {}).get("ndcg@10")
            )

        lat_s = _fmt(lat, "{:.1f}")
        rows.append(f"{label} & {params} & {size} & {train_cost} & {lat_s} & {ndcg} \\\\")

    body = "\n".join(rows)
    backend = BACKEND_LABELS.get(retriever, retriever)
    return (
        "% Generated by scripts/report_rq4_cost.py -- do not edit by hand.\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{%\n"
        "  \\textbf{Total cost of re-ranking on " + backend + ".} Model size is the "
        "fp32 parameter footprint; LambdaMART's is its serialised booster. "
        "Training cost is one run: GARDIAN and GARDIAN-Lite on one NVIDIA A40, "
        "LambdaMART on CPU, both over the same 186{,}504 training queries. The "
        "cross-encoders are published checkpoints, so their marginal training "
        "cost here is zero and their (substantial) pretraining cost is external. "
        "Latency is the median per-query cost of re-ranking a 100-candidate "
        "pool; first-stage retrieval is shared by every row and excluded. "
        "GARDIAN's latency includes the query-encoder forward pass its "
        "controller requires; GARDIAN-Lite has no such pass.}\n"
        "\\label{tab:rq4-cost}\n"
        "\\small\n"
        "\\begin{tabular}{lrrlrrrr}\n"
        "\\toprule\n"
        " & & Size & Training & Latency & \\multicolumn{3}{c}{nDCG@10} \\\\\n"
        "\\cmidrule(lr){6-8}\n"
        "Re-ranker & Params & (MB) & cost & (ms/q) & PQA-L & MedMCQA & PQA-A \\\\\n"
        "\\midrule\n"
        + body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )


def latex_lite_table(payload: Dict[str, Any]) -> str:
    """What removing the controller saves, and what it costs in nDCG@10."""
    lite_eval = payload.get("gardian_lite_eval", {})
    lite_train = payload.get("gardian_lite", {})
    train = payload.get("training", {})

    rows: List[str] = []
    for retriever, per_dataset in lite_eval.items():
        label = BACKEND_LABELS.get(retriever, retriever)
        for dataset, lite in per_dataset.items():
            full = _gardian_cell(payload, retriever, dataset)
            ds_label = DATASET_LABELS.get(dataset, dataset)

            g_ndcg, l_ndcg = full.get("ndcg@10"), lite.get("ndcg@10")
            g_lat, l_lat = full.get("p50_ms"), lite.get("p50_ms")

            g_ndcg_s = f"{float(g_ndcg):.4f}" if g_ndcg is not None else "--"
            l_ndcg_s = f"{float(l_ndcg):.4f}" if l_ndcg is not None else "--"
            delta_s = (
                f"{float(l_ndcg) - float(g_ndcg):+.4f}"
                if g_ndcg is not None and l_ndcg is not None
                else "--"
            )
            g_lat_s = f"{float(g_lat):.1f}" if g_lat else "--"
            l_lat_s = f"{float(l_lat):.1f}" if l_lat else "--"
            speedup_s = (
                rf"{float(g_lat) / float(l_lat):.1f}$\times$" if g_lat and l_lat else "--"
            )

            rows.append(
                f"{label} & {ds_label} & {g_ndcg_s} & {l_ndcg_s} & {delta_s} & "
                f"{g_lat_s} & {l_lat_s} & {speedup_s} \\\\"
            )

    if not rows:
        rows = [
            r"\multicolumn{8}{c}{\emph{GARDIAN-Lite results not yet available}} \\"
        ]

    lite_params = next(
        (
            b.get("trainable_parameters")
            for b in lite_train.values()
            if b.get("trainable_parameters")
        ),
        None,
    )
    full_params = next(
        (
            b.get("trainable_parameters")
            for k, b in train.items()
            if not k.startswith("_") and b.get("trainable_parameters")
        ),
        None,
    )
    param_note = ""
    if lite_params and full_params:
        pct = 100.0 * (1.0 - float(lite_params) / float(full_params))
        param_note = (
            f" GARDIAN-Lite has {float(lite_params) / 1e6:.2f}M trainable parameters "
            f"against {float(full_params) / 1e6:.2f}M ({pct:.0f}\\% fewer)."
        )

    body = "\n".join(rows)
    return (
        "% Generated by scripts/report_rq4_cost.py -- do not edit by hand.\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{%\n"
        "  \\textbf{Cost and benefit of the query-adaptive controller.} "
        "GARDIAN-Lite is the same model with \\texttt{use\\_controller: false}, "
        "trained under the same protocol at seed~42 on the same candidate pools. "
        "Removing the controller removes the query encoder from the inference "
        "path entirely, so no BERT forward pass is needed per query."
        f"{param_note}"
        " Latency is the median per-query re-ranking time on a single A40.}\n"
        "\\label{tab:rq4-lite}\n"
        "\\small\n"
        "\\begin{tabular}{llrrrrrr}\n"
        "\\toprule\n"
        " & & \\multicolumn{3}{c}{nDCG@10} & \\multicolumn{3}{c}{Latency (ms/q)} \\\\\n"
        "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}\n"
        "Back-end & Split & GARDIAN & Lite & $\\Delta$ & GARDIAN & Lite & speed-up \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def latex_training_table(payload: Dict[str, Any]) -> str:
    """Training cost per back-end -- the table Reviewer 4 asked for."""
    train = payload.get("training", {})
    ltr = payload.get("lambdamart", {})
    env = payload.get("environment", {})
    hp = payload.get("hyperparameters", {})
    step = payload.get("train_step_cost", {})
    seeds = train.get("_meta", {}).get("seeds", [])

    rows: List[str] = []
    for retriever, block in train.items():
        if retriever.startswith("_"):
            continue
        mean = float(block["train_wall_clock_sec_mean"] or 0.0)
        std = float(block["train_wall_clock_sec_std"] or 0.0)
        params = int(block["trainable_parameters"])
        ltr_block = ltr.get(retriever)
        ltr_cell = (
            f"{float(ltr_block['train_wall_clock_sec']) / 60.0:.0f}"
            if ltr_block and ltr_block.get("train_wall_clock_sec")
            else "--"
        )
        rows.append(
            f"{block['label']} & {mean / 60.0:.0f} $\\pm$ {std / 60.0:.0f} & "
            f"{params / 1e6:.2f} & {ltr_cell} \\\\"
        )

    # The controller's training cost is stated from the controlled step
    # benchmark, never from the two runs' wall clocks -- those were measured
    # under different GPU load and their difference is contention.
    if step.get("percent_slower_with_controller") is not None:
        controller_note = (
            f" The query-adaptive controller adds "
            f"{float(step['percent_slower_with_controller']):.0f}\\% to the cost of one "
            f"optimisation step ({float(step['with_controller_ms']):.1f} vs "
            f"{float(step['no_controller_ms']):.1f}~ms, {step.get('repeats')} interleaved "
            f"repeats of {step.get('steps_per_repeat')} steps). We quote the step "
            f"benchmark rather than the two runs' wall clocks, because those runs "
            f"shared the GPU with different neighbours and their difference would "
            f"measure contention rather than architecture."
        )
    else:
        controller_note = ""

    gpu = env.get("gpu_name", "GPU")
    body = "\n".join(rows)
    return (
        "% Generated by scripts/report_rq4_cost.py -- do not edit by hand.\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{%\n"
        "  \\textbf{Training cost of the re-ranker.} Wall-clock minutes for one "
        f"training run per back-end on a single {gpu}, mean $\\pm$ sd over "
        f"{len(seeds)} seeds ({', '.join(str(x) for x in seeds)}). LambdaMART "
        "trains on CPU over the same 186{,}504 queries and the same 16 features. "
        f"GARDIAN: {hp.get('epochs')} epochs, {hp.get('objective')} objective, lr "
        f"{hp.get('learning_rate')}, batches of {hp.get('batch_size_queries')} "
        f"queries $\\times$ {hp.get('listwise_group_size')} candidates "
        f"({hp.get('scored_rows_per_step')} scored rows per step), "
        f"{hp.get('precision')}."
        + controller_note +
        "}\n"
        "\\label{tab:rq4-training-cost}\n"
        "\\small\n"
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        " & \\multicolumn{2}{c}{GARDIAN} & LambdaMART \\\\\n"
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-4}\n"
        "Back-end & min (GPU) & params (M) & min (CPU) \\\\\n"
        "\\midrule\n"
        + body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--retrievers",
        default="hybrid_bm25_faiss,hybrid_spladepp_medcpt",
        help="Back-ends whose latency and LambdaMART artifacts are collected.",
    )
    parser.add_argument("--lite-dir", default="results/gardian_lite")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--latex-out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    results_dir = REPO / str(cfg.paths.results_dir)
    retrievers = [r.strip() for r in args.retrievers.split(",") if r.strip()]

    payload: Dict[str, Any] = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/report_rq4_cost.py",
            "seed": args.seed,
            "retrievers": retrievers,
        },
        "environment": collect_environment(results_dir, args.seed),
        "hyperparameters": collect_hyperparameters(cfg),
        "training": collect_training_cost(results_dir),
        "train_step_cost": collect_train_step_cost(results_dir),
        "gardian_lite": collect_lite(REPO / args.lite_dir, args.seed),
        "gardian_lite_eval": collect_lite_eval(results_dir, retrievers),
        "lambdamart": collect_ltr(results_dir, retrievers),
        "latency": collect_latency(results_dir, retrievers),
        "cross_encoder_parameters": {
            k: {"model": v[0], "parameters": v[1]} for k, v in CE_PARAMS.items()
        },
        "notes": {
            "cross_encoder_training": (
                "Cross-encoders are used off-the-shelf, so their marginal training "
                "cost in this work is zero; their pretraining and MS MARCO "
                "fine-tuning cost is external and not counted here. The "
                "like-for-like trained control is LambdaMART, which sees the same "
                "features and the same pools as GARDIAN."
            ),
            "latency_scope": (
                "Latency is the per-query re-ranking cost over a fixed candidate "
                "pool. First-stage retrieval is shared by every system and is "
                "excluded; it is reported separately by "
                "scripts/benchmark_retrieval_efficiency.py."
            ),
            "query_encoder": (
                "GARDIAN's controller reads a query embedding, so the query-encoder "
                "forward pass is inside the timed region. GARDIAN-Lite has no "
                "controller and therefore no query encoder on the inference path."
            ),
        },
    }

    out = args.out or (results_dir / "rq4_cost_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.success(f"Saved cost summary -> {out}")

    latex_out = args.latex_out or (REPO / "paper" / "table_rq4_training_cost.tex")
    latex_out.parent.mkdir(parents=True, exist_ok=True)
    latex_out.write_text(latex_training_table(payload), encoding="utf-8")
    logger.success(f"Saved LaTeX table  -> {latex_out}")

    cost_out = latex_out.with_name("table_rq4_cost.tex")
    cost_out.write_text(
        latex_cost_comparison_table(payload, retrievers[0]), encoding="utf-8"
    )
    logger.success(f"Saved LaTeX table  -> {cost_out}")

    lite_out = latex_out.with_name("table_rq4_lite.tex")
    lite_out.write_text(latex_lite_table(payload), encoding="utf-8")
    logger.success(f"Saved LaTeX table  -> {lite_out}")

    # Console summary so a run is readable without opening the JSON.
    env = payload["environment"]
    print(f"\nHardware: {env.get('gpu_name')} | torch {env.get('torch_version')} "
          f"| CUDA {env.get('cuda_version')}")
    hp = payload["hyperparameters"]
    print(f"Training: {hp['epochs']} epochs, {hp['objective']}, lr {hp['learning_rate']}, "
          f"{hp['batch_size_queries']}x{hp['listwise_group_size']} "
          f"= {hp['scored_rows_per_step']} rows/step, {hp['precision']}")
    print(f"\n{'Back-end':<20} {'GARDIAN':>14} {'params':>10} {'Lite':>12} {'Lite params':>12}")
    for retriever, block in payload["training"].items():
        if retriever.startswith("_"):
            continue
        lite = payload["gardian_lite"].get(retriever, {})
        lite_s = (
            _fmt_hms(float(lite["train_wall_clock_sec"]))
            if lite.get("train_wall_clock_sec")
            else "not trained"
        )
        lite_p = (
            f"{int(lite['trainable_parameters']) / 1e6:.2f}M"
            if lite.get("trainable_parameters")
            else "--"
        )
        print(
            f"{PLAIN_BACKEND_LABELS.get(retriever, retriever):<20} "
            f"{block['train_wall_clock_hms']:>14} "
            f"{block['trainable_parameters'] / 1e6:>9.2f}M {lite_s:>12} {lite_p:>12}"
        )


if __name__ == "__main__":
    main()
