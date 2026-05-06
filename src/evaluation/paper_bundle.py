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
from src.evaluation.rank_jsonl_eval import evaluate_all_from_rank_data
from src.evaluation.stats import bootstrap_delta_ci, bootstrap_mean_ci, paired_randomization_pvalue
from src.model.gardian import GARDIAN


def build_paper_model(cfg: Any, device: str, retriever: str) -> GARDIAN:
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
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def abl_to_kw(name: str) -> Optional[str]:
    if name == "full":
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
    assert_cfg_question_types(cfg.model.question_types)

    try:
        model = build_paper_model(cfg, compute_device, retriever)
        device_for_eval = compute_device
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and str(compute_device).startswith("cuda"):
            logger.warning("CUDA OOM while loading model in worker; retrying on CPU.")
            model = build_paper_model(cfg, "cpu", retriever)
            device_for_eval = "cpu"
        else:
            raise

    query_encoder_device = device_for_eval if str(device_for_eval).startswith("cuda") else "cpu"

    out: Dict[str, Dict[str, Any]] = {}
    for ds_name, split in dataset_splits:
        rank_path = resolve_rank_data_file(retriever, ds_name, split)
        if not pathlib.Path(rank_path).exists():
            logger.warning(f"Skip missing dataset file: {rank_path}")
            continue
        out[ds_name] = {}
        for abl_name in ablation_names:
            dev_log = (
                f"{device_for_eval} (CUDA_VISIBLE_DEVICES={cuda_visible_devices})"
                if cuda_visible_devices is not None
                else str(device_for_eval)
            )
            logger.info(f"=== {retriever} | {ds_name} | ablation={abl_name} | device={dev_log} | query_enc={query_encoder_device} ===")
            raw = evaluate_all_from_rank_data(
                rank_path,
                model,
                device_for_eval,
                gardian_ablation=abl_to_kw(abl_name),
                collect_per_query=True,
                query_encoder_name=query_encoder_name,
                query_encoder_device=query_encoder_device,
                canonical_baseline_keys=True,
            )
            for sys_name, block in raw.items():
                if isinstance(block, dict) and "_per_query" in block and bootstrap > 0:
                    attach_bootstrap(block, n_boot=int(bootstrap), seed=int(seed))
            raw["significance"] = add_delta_tests(
                raw,
                n_boot=int(bootstrap),
                seed=int(seed),
                n_trials=int(randomization_trials),
            )
            out[ds_name][abl_name] = raw
    return out
