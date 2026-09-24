#!/usr/bin/env python3
"""
Controlled training-step cost: does removing the controller make training cheaper?

The wall-clock times in ``results/*/training_summary_all_retrievers.json`` cannot
answer this. Those runs happened hours apart under different GPU load -- the
GARDIAN-Lite runs shared an A40 with a cross-encoder scoring job and an 8B
reader, the original runs did not -- so their difference measures contention,
not architecture. Comparing them would be reporting scheduler noise as a result.

This measures the two configurations back to back in one process, on identical
batches, so whatever contention exists applies equally to both arms and cancels
in the ratio. It times the part that actually differs: the forward pass, the
loss and the backward pass. Data loading is excluded deliberately, because it is
shared by both arms and is the dominant cost of a real epoch -- which is itself
the finding, if the step difference turns out to be small.

Usage:
    python scripts/benchmark_train_step.py --steps 200
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.model.gardian import build_gardian_from_model_cfg
from src.training.losses import build_loss, is_listwise


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _make_batch(
    *,
    batch_size: int,
    group_size: int,
    sparse_dim: int,
    dense_dim: int,
    query_dim: int,
    pool_dim: int,
    device: str,
    generator: torch.Generator,
) -> Dict[str, torch.Tensor]:
    """One listwise batch: ``batch_size`` queries, each with ``group_size`` candidates."""
    def r(*shape):
        return torch.rand(*shape, generator=generator).to(device)

    labels = (r(batch_size, group_size) < 0.05).float()
    return {
        "sparse_feats": r(batch_size, group_size, sparse_dim),
        "dense_feats": r(batch_size, group_size, dense_dim),
        "query_emb": r(batch_size, query_dim),
        "pool_feats": r(batch_size, pool_dim),
        "mask": torch.ones(batch_size, group_size, device=device),
        "labels": labels,
    }


def time_config(
    cfg: Any,
    *,
    use_controller: bool,
    steps: int,
    warmup: int,
    device: str,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg["use_controller"] = use_controller
    model = build_gardian_from_model_cfg(model_cfg).to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    loss_name = str(cfg.training.loss)
    loss_fn = build_loss(loss_name)
    loss_kwargs: Dict[str, Any] = {}
    if is_listwise(loss_name):
        loss_kwargs["k"] = int(cfg.training.listwise_ndcg_k)
        if loss_name == "lambdarank":
            loss_kwargs["sigma"] = float(cfg.training.lambdarank_sigma)

    gen = torch.Generator().manual_seed(seed)
    from src.features.pool_features import POOL_FEATURE_DIM

    batch = _make_batch(
        batch_size=int(cfg.training.batch_size),
        group_size=int(cfg.training.listwise_group_size),
        sparse_dim=int(cfg.model.sparse_feat_dim),
        dense_dim=int(cfg.model.dense_feat_dim),
        query_dim=int(cfg.model.query_feat_dim),
        pool_dim=POOL_FEATURE_DIM,
        device=device,
        generator=gen,
    )

    samples: List[float] = []
    for i in range(steps + warmup):
        _sync(device)
        t0 = time.perf_counter()

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            scores, _ = model(
                sparse_feats=batch["sparse_feats"],
                dense_feats=batch["dense_feats"],
                query_emb=batch["query_emb"],
                pool_feats=batch["pool_feats"],
                mask=batch["mask"],
            )
        loss = loss_fn(scores.float(), batch["labels"], mask=batch["mask"], **loss_kwargs)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        _sync(device)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            samples.append(dt)

    arr = np.asarray(samples)
    return {
        "use_controller": use_controller,
        "trainable_parameters": int(n_params),
        "steps": int(arr.size),
        "mean_ms_per_step": float(np.mean(arr)),
        "p50_ms_per_step": float(np.percentile(arr, 50)),
        "std_ms_per_step": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default="configs/base.yaml")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--out", type=pathlib.Path, default=pathlib.Path("results/train_step_cost.json")
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )

    # Interleave the arms across repeats so a drift in GPU load during the run
    # cannot be mistaken for a difference between the configurations.
    runs: Dict[str, List[Dict[str, Any]]] = {"with_controller": [], "no_controller": []}
    for rep in range(args.repeats):
        for flag, key in ((True, "with_controller"), (False, "no_controller")):
            res = time_config(
                cfg,
                use_controller=flag,
                steps=args.steps,
                warmup=args.warmup,
                device=device,
                seed=args.seed + rep,
            )
            runs[key].append(res)
            logger.info(
                f"  repeat {rep + 1}/{args.repeats} "
                f"{'GARDIAN     ' if flag else 'GARDIAN-Lite'}: "
                f"{res['p50_ms_per_step']:.2f} ms/step "
                f"({res['trainable_parameters'] / 1e6:.2f}M params)"
            )

    def _agg(key: str) -> Dict[str, float]:
        vals = [r["p50_ms_per_step"] for r in runs[key]]
        return {
            "p50_ms_per_step_mean": float(np.mean(vals)),
            "p50_ms_per_step_min": float(np.min(vals)),
            "trainable_parameters": runs[key][0]["trainable_parameters"],
        }

    full, lite = _agg("with_controller"), _agg("no_controller")
    # Minimum across repeats is the least contaminated estimate: contention can
    # only ever make a step slower.
    speedup = full["p50_ms_per_step_min"] / lite["p50_ms_per_step_min"]

    contention = None
    if device.startswith("cuda"):
        import subprocess
        import os

        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            pids = [x.strip() for x in out.stdout.splitlines() if x.strip()]
            contention = {"other_processes": len([p for p in pids if p != str(os.getpid())])}
        except (OSError, subprocess.SubprocessError):
            contention = None

    payload = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/benchmark_train_step.py",
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
            "steps_per_repeat": args.steps,
            "repeats": args.repeats,
            "batch_size_queries": int(cfg.training.batch_size),
            "listwise_group_size": int(cfg.training.listwise_group_size),
            "objective": str(cfg.training.loss),
            "precision": "mixed fp16 (autocast + GradScaler)" if device.startswith("cuda") else "fp32",
            "gpu_contention": contention,
            "note": (
                "Both arms are timed in one process on identical batches and "
                "interleaved across repeats, so GPU load applies equally to both "
                "and cancels in the ratio. Data loading is excluded: it is "
                "shared by both arms and dominates a real epoch."
            ),
        },
        "with_controller": full,
        "no_controller": lite,
        "controller_step_overhead": {
            "ratio": float(speedup),
            "percent_slower_with_controller": float((speedup - 1.0) * 100.0),
            "absolute_ms_per_step": float(
                full["p50_ms_per_step_min"] - lite["p50_ms_per_step_min"]
            ),
        },
        "raw": runs,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print()
    print(f"GARDIAN      {full['p50_ms_per_step_min']:7.2f} ms/step  "
          f"({full['trainable_parameters'] / 1e6:.2f}M params)")
    print(f"GARDIAN-Lite {lite['p50_ms_per_step_min']:7.2f} ms/step  "
          f"({lite['trainable_parameters'] / 1e6:.2f}M params)")
    print(f"Controller costs {payload['controller_step_overhead']['percent_slower_with_controller']:+.1f}% "
          f"per step ({payload['controller_step_overhead']['absolute_ms_per_step']:+.2f} ms)")
    logger.success(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
