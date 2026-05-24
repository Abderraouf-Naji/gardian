#!/usr/bin/env python3
"""Add hit@5/20/50 to existing evaluation JSON (baselines offline; optional GARDIAN)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.rank_data_paths import resolve_rank_data_file
from src.evaluation.rank_jsonl_eval import evaluate_all_from_rank_data
from src.model.gardian import build_gardian_from_model_cfg, load_checkpoint_state

DATASET_SPLITS = {
    "pubmedqa_labeled": "eval",
    "pubmedqa_artificial": "test",
    "medmcqa": "test",
}


def _build_gardian(cfg: Any, device: str, retriever: str):
    ckpt_path = Path(cfg.paths.results_dir) / f"gardian_best_{retriever}.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("cfg", {}) if isinstance(ckpt.get("cfg"), dict) else {}
    ckpt_model_cfg = ckpt_cfg.get("model") if isinstance(ckpt_cfg.get("model"), dict) else None
    model = build_gardian_from_model_cfg(ckpt_model_cfg or cfg.model)
    load_checkpoint_state(model, ckpt["model_state"], strict=False)
    model.to(device)
    model.eval()
    return model, int((ckpt_model_cfg or {}).get("query_feat_dim", cfg.model.query_feat_dim))


def backfill_eval_json(
    path: Path,
    *,
    with_gardian: bool = False,
    device: str = "cpu",
    query_encoder_device: Optional[str] = None,
) -> None:
    cfg = OmegaConf.load("configs/base.yaml")
    qenc_device = query_encoder_device or device
    obj = json.loads(path.read_text(encoding="utf-8"))
    results = obj.get("results", {})
    gardian_models: dict[str, Any] = {}
    for retriever, ds_block in results.items():
        if not isinstance(ds_block, dict):
            continue
        for dataset, systems in ds_block.items():
            if dataset.startswith("_") or not isinstance(systems, dict):
                continue
            split = DATASET_SPLITS.get(dataset)
            if not split:
                continue
            rank_path = resolve_rank_data_file(retriever, dataset, split)
            if not Path(rank_path).is_file():
                print(f"skip {retriever}/{dataset}: no rank data")
                continue
            model = qdim = None
            if with_gardian:
                if retriever not in gardian_models:
                    try:
                        gardian_models[retriever] = _build_gardian(cfg, device, retriever)
                    except FileNotFoundError as exc:
                        print(f"  gardian skip {retriever}: {exc}")
                if retriever in gardian_models:
                    model, qdim = gardian_models[retriever]

            if with_gardian and model is not None:
                print(
                    f"  {retriever}/{dataset}: GARDIAN on {device}, "
                    f"query encoder on {qenc_device}"
                )
            raw = evaluate_all_from_rank_data(
                rank_path,
                model=model,
                device=device if model is not None else None,
                expected_query_feat_dim=qdim,
                query_encoder_name=str(cfg.encoder.model_name),
                query_encoder_device=qenc_device,
                include_standalone_spladepp=False,
                gardian_adaptive_retrieval=bool(getattr(cfg.qa, "gardian_adaptive_retrieval", False)),
                cfg=cfg,
            )
            _SYS_RAW_KEYS = {
                "sparse": ("sparse", "bm25", "spladepp"),
                "dense": ("dense", "faiss", "medcpt"),
                "hybrid": ("hybrid",),
                "rrf": ("rrf",),
                "gardian": ("gardian",),
            }
            for sys_name, raw_keys in _SYS_RAW_KEYS.items():
                if sys_name not in systems:
                    continue
                src = None
                for rk in raw_keys:
                    if rk in raw and isinstance(raw[rk], dict):
                        src = raw[rk]
                        break
                if src is None:
                    continue
                for key in ("hit@5", "hit@20", "hit@50"):
                    if key in src:
                        systems[sys_name][key] = src[key]
            print(f"updated {path.name} {retriever}/{dataset}")
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("results/evaluation_hybrid_bm25_faiss.json")],
    )
    parser.add_argument(
        "--with-gardian",
        action="store_true",
        help="Re-score GARDIAN (slow; needs GPU/checkpoint) to fill hit@ for gardian rows.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for GARDIAN scoring (default: cuda if available).",
    )
    parser.add_argument(
        "--query-encoder-device",
        default=None,
        help="Device for PubMedBERT query embeddings (default: same as --device).",
    )
    args = parser.parse_args()
    qenc = args.query_encoder_device or args.device
    if args.with_gardian:
        print(f"GARDIAN device={args.device!r}  query_encoder_device={qenc!r}")
    for p in args.paths:
        if p.is_file():
            backfill_eval_json(
                p,
                with_gardian=args.with_gardian,
                device=args.device,
                query_encoder_device=qenc,
            )
        else:
            print(f"missing {p}")


if __name__ == "__main__":
    main()
