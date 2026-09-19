"""Shared evaluation helpers for paper bundle JSON (scripts/10_paper_run.py)."""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from omegaconf import OmegaConf

from src.common.question_types import assert_cfg_question_types
from src.common.rank_data_paths import resolve_rank_data_file
from src.common.seeds import resolve_seed_checkpoint
from src.evaluation.gardian_mixes import (
    MIX_NAMES,
    evaluate_gardian_mixes,
    fit_fixed_alpha,
    query_emb_cache_for,
    resolve_dev_rank_path,
    score_split_for_fit,
)
from src.evaluation.rank_jsonl_eval import _resolve_qrels
from src.evaluation.stats import bootstrap_delta_ci, bootstrap_mean_ci, paired_randomization_pvalue
from src.model.gardian import GARDIAN, build_gardian_from_model_cfg, load_checkpoint_state


def build_paper_model(
    cfg: Any,
    device: str,
    retriever: str,
    seed: Optional[int] = None,
) -> tuple[GARDIAN, int]:
    """Load the trained GARDIAN for one ``(retriever, seed)`` cell.

    ``seed`` is required to read the per-seed checkpoint written by
    ``scripts/04_train_gardian.py``; passing ``None`` keeps the legacy flat
    path for callers that predate the multi-seed layout.
    """
    results_dir = cfg.paths.results_dir
    if seed is None:
        ckpt_path = pathlib.Path(results_dir) / f"gardian_best_{retriever}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        ckpt_path = resolve_seed_checkpoint(results_dir, retriever, int(seed))
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
    return model, expected_qdim


def abl_to_kw(name: str) -> Optional[str]:
    if name in (None, "full", "oracle_branch"):
        return None
    return name


def attach_bootstrap(block: Dict[str, Any], n_boot: int, seed: int) -> None:
    pq = block.pop("_per_query", None)
    if not pq or "ndcg@10" not in pq:
        return
    mean, lo, hi = bootstrap_mean_ci(pq["ndcg@10"], n_bootstrap=n_boot, seed=seed)
    block["ndcg@10_query_mean"] = mean
    block["ndcg@10_bootstrap_ci95"] = [lo, hi]
    for metric in [
        "ndcg@5",
        "ndcg@10",
        "ndcg@20",
        "ndcg@50",
        "ndcg@100",
        "recall@5",
        "recall@20",
        "recall@50",
        "recall@100",
        "mrr",
    ]:
        if metric in pq:
            m, ml, mh = bootstrap_mean_ci(pq[metric], n_bootstrap=n_boot, seed=seed)
            block[f"{metric}_bootstrap_ci95"] = [ml, mh]
            block[f"{metric}_query_mean"] = m


def add_delta_tests(raw: Dict[str, Any], n_boot: int, seed: int, n_trials: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    g = raw.get("gardian", {})
    g_pq = g.get("_per_query", {})
    skip = {"_meta", "significance", "gardian", "rrf"}
    for baseline, b in raw.items():
        if baseline in skip or not isinstance(b, dict):
            continue
        b_pq = b.get("_per_query", {})
        if not b_pq or "ndcg@10" not in b_pq or "ndcg@10" not in g_pq:
            continue
        d_mean, d_lo, d_hi = bootstrap_delta_ci(
            g_pq["ndcg@10"], b_pq["ndcg@10"], n_bootstrap=n_boot, seed=seed
        )
        pvalue = paired_randomization_pvalue(
            g_pq["ndcg@10"], b_pq["ndcg@10"], n_trials=n_trials, seed=seed
        )
        slug = baseline.replace(" ", "_").replace("/", "_")
        out[f"gardian_minus_{slug}_ndcg10"] = {
            "delta_mean": d_mean,
            "delta_ci95": [d_lo, d_hi],
            "paired_randomization_pvalue": pvalue,
            "baseline": baseline,
        }
    return out


def run_paper_chunk(
    project_root: str,
    retriever: str,
    cfg_path: str,
    dataset_splits: List[Tuple[str, str]],
    ablation_names: List[str],
    device: str,
    bootstrap: int,
    seed: int,
    randomization_trials: int,
    query_encoder_name: str,
    cuda_visible_devices: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load one GARDIAN checkpoint and evaluate ablation_names on all dataset_splits.

    For multiprocessing on multiple physical GPUs, pass ``cuda_visible_devices`` as a
    single id (e.g. ``\"2\"``). The child process then sees it as ``cuda:0`` only,
    which avoids cross-process CUDA context issues and matches PyTorch's recommended
    pattern.
    """
    sys.path.insert(0, project_root)
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    torch.set_float32_matmul_precision("high")

    if device == "cuda":
        compute_device = "cuda:0"
    else:
        compute_device = device

    cfg = OmegaConf.load(cfg_path)
    assert_cfg_question_types(cfg.evaluation.question_types)

    try:
        model, expected_qdim = build_paper_model(cfg, compute_device, retriever, seed)
        device_for_eval = compute_device
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and str(compute_device).startswith("cuda"):
            logger.warning("CUDA OOM while loading model in worker; retrying on CPU.")
            model, expected_qdim = build_paper_model(cfg, "cpu", retriever, seed)
            device_for_eval = "cpu"
        else:
            raise

    query_encoder_device = device_for_eval if str(device_for_eval).startswith("cuda") else "cpu"

    unknown = [n for n in ablation_names if n not in MIX_NAMES]
    if unknown:
        raise ValueError(f"Unknown ablation(s) {unknown}; expected one of {MIX_NAMES}")

    # One forward per split, then every mix from the same branch scores.
    # Fixed-α is fitted on DEV (PQA-L borrows PQA-A dev) and cached so the
    # sibling split does not re-score 18k queries.
    fit_cache: Dict[str, Tuple[float, float, str]] = {}
    need_fixed = "fixed_alpha" in ablation_names

    out: Dict[str, Dict[str, Any]] = {}
    for ds_name, split in dataset_splits:
        rank_path = resolve_rank_data_file(retriever, ds_name, split)
        if not pathlib.Path(rank_path).exists():
            logger.warning(f"Skip missing dataset file: {rank_path}")
            continue
        qcache = query_emb_cache_for(retriever, ds_name, split)
        load_kw = dict(
            query_encoder_name=query_encoder_name,
            query_encoder_device=query_encoder_device,
            query_emb_cache_path=qcache,
            expected_query_feat_dim=expected_qdim,
        )
        alpha_hat: Optional[float] = None
        if need_fixed:
            try:
                dev_path, fitted_on = resolve_dev_rank_path(retriever, ds_name)
            except FileNotFoundError as exc:
                raise SystemExit(str(exc)) from exc
            cache_key = f"{retriever}:{fitted_on}:dev"
            if cache_key not in fit_cache:
                logger.info(
                    f"=== {retriever} | fit Fixed-α on {fitted_on} dev "
                    f"(for {ds_name}) | device={device_for_eval} ==="
                )
                dev_cache = query_emb_cache_for(retriever, fitted_on, "dev")
                dev_pools, dev_s, dev_d = score_split_for_fit(
                    dev_path,
                    model,
                    device_for_eval,
                    query_encoder_name=query_encoder_name,
                    query_encoder_device=query_encoder_device,
                    query_emb_cache_path=dev_cache,
                    expected_query_feat_dim=expected_qdim,
                )
                qrels, _ = _resolve_qrels(dev_pools.keys())
                a_star, fit_ndcg = fit_fixed_alpha(dev_pools, dev_s, dev_d, qrels)
                fit_cache[cache_key] = (a_star, fit_ndcg, fitted_on)
                logger.info(
                    f"  Fixed-α = {a_star:.2f} (dev nDCG@10={fit_ndcg:.4f}, "
                    f"fitted on {fitted_on})"
                )
                del dev_pools, dev_s, dev_d
                if str(device_for_eval).startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            alpha_hat, fit_ndcg, fitted_on = fit_cache[cache_key]
        else:
            fit_ndcg, fitted_on = None, None

        logger.info(
            f"=== {retriever} | {ds_name}/{split} | mixes={ablation_names} "
            f"| device={device_for_eval} ==="
        )
        mixes = evaluate_gardian_mixes(
            rank_path,
            model,
            device_for_eval,
            mix_names=ablation_names,
            fixed_alpha=alpha_hat,
            collect_per_query=True,
            **load_kw,
        )
        out[ds_name] = {}
        full_pq = (
            mixes.get("full", {})
            .get("gardian", {})
            .get("_per_query", {})
        )
        for abl_name, raw in mixes.items():
            if alpha_hat is not None:
                raw.setdefault("_meta", {})
                raw["_meta"]["fixed_alpha"] = float(alpha_hat)
                raw["_meta"]["fixed_alpha_fitted_on"] = fitted_on
                raw["_meta"]["fixed_alpha_fit_ndcg@10"] = fit_ndcg
            gblock = raw.get("gardian", {})
            abl_pq = (
                gblock.get("_per_query") if isinstance(gblock, dict) else None
            )
            if isinstance(gblock, dict) and "_per_query" in gblock and bootstrap > 0:
                attach_bootstrap(gblock, n_boot=int(bootstrap), seed=int(seed))
            raw["significance"] = {}
            if (
                abl_name != "full"
                and full_pq
                and abl_pq
                and "ndcg@10" in full_pq
                and "ndcg@10" in abl_pq
            ):
                d_mean, d_lo, d_hi = bootstrap_delta_ci(
                    full_pq["ndcg@10"], abl_pq["ndcg@10"], n_bootstrap=int(bootstrap), seed=int(seed)
                )
                pvalue = paired_randomization_pvalue(
                    full_pq["ndcg@10"],
                    abl_pq["ndcg@10"],
                    n_trials=int(randomization_trials),
                    seed=int(seed),
                )
                raw["significance"]["full_minus_ablation_ndcg10"] = {
                    "delta_mean": d_mean,
                    "delta_ci95": [d_lo, d_hi],
                    "paired_randomization_pvalue": pvalue,
                    "ablation": abl_name,
                }
            out[ds_name][abl_name] = raw
        if str(device_for_eval).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out
