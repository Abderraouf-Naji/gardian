import os
import json
import pathlib
import subprocess
import sys
import hashlib
import platform
from datetime import datetime, timezone
from typing import Optional
import csv

os.environ.setdefault("PYTHONUTF8", "1")

sys.path.insert(0, ".")

import torch
from loguru import logger
from omegaconf import OmegaConf

from src.common.question_types import assert_cfg_question_types
from src.common.rank_data_paths import (
    normalize_retriever_name,
    rank_data_combined_file,
    resolve_rank_data_file,
)
from src.common.repro import set_global_seed
from src.model.gardian import build_gardian_from_model_cfg
from src.training.trainer import GARDIANTrainer

HYBRID_RETRIEVER_COMBINATIONS = {
    "hybrid_bm25_faiss": "BM25 + FAISS",
    "hybrid_bm25_medcpt": "BM25 + MedCPT",
    "hybrid_spladepp_faiss": "SPLADE++ + FAISS",
    "hybrid_spladepp_medcpt": "SPLADE++ + MedCPT",
}

# GARDIAN is trained per hybrid family (one checkpoint per row in the
# 4-cell hybrid table). Single-retriever names are kept here only so that
# legacy CLI calls continue to work for ablation runs over already-existing
# rank-data; they are NOT iterated when ``--retriever all`` is used.
ALL_TRAIN_RETRIEVERS = [
    "hybrid_bm25_faiss",
    "hybrid_bm25_medcpt",
    "hybrid_spladepp_faiss",
    "hybrid_spladepp_medcpt",
    "spladepp",
    "bm25",
    "faiss",
    "medcpt",
]


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


def concat_files(src_paths, dst_path):
    """Concatenate multiple JSONL files into one output file."""
    total_lines = 0
    pathlib.Path(dst_path).parent.mkdir(parents=True, exist_ok=True)

    with open(dst_path, "w", encoding="utf-8") as out_f:
        for src in src_paths:
            if not os.path.exists(src):
                logger.warning(f"Source file not found, skipping: {src}")
                continue
            logger.info(f"Adding {src} -> {dst_path}")
            file_lines = 0
            with open(src, "r", encoding="utf-8") as in_f:  # Explicit UTF-8
                for line in in_f:
                    if line.strip():
                        out_f.write(line)
                        file_lines += 1
                        total_lines += 1
            logger.info(f"  wrote {file_lines:,} lines from {src}")

    logger.info(f"Finished writing {total_lines:,} total lines -> {dst_path}")
    return total_lines


def count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:  # Explicit UTF-8
        return sum(1 for line in f if line.strip())


def _sha256_file(path: str) -> str:
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_feature_dims(path: str, cfg) -> None:
    expected_sparse = int(cfg.model.sparse_feat_dim)
    expected_dense = int(cfg.model.dense_feat_dim)
    expected_kg = int(cfg.model.kg_feat_dim)
    expected_query = int(cfg.model.query_feat_dim)
    expected_qtypes = len(cfg.model.question_types)
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if len(rec.get("sparse_feats", [])) != expected_sparse:
                raise ValueError(f"{path}:{idx} sparse dim mismatch")
            if len(rec.get("dense_feats", [])) != expected_dense:
                raise ValueError(f"{path}:{idx} dense dim mismatch")
            if len(rec.get("kg_feats", [])) != expected_kg:
                raise ValueError(f"{path}:{idx} kg dim mismatch")
            if "query_emb" in rec and len(rec.get("query_emb", [])) != expected_query:
                raise ValueError(f"{path}:{idx} query_emb dim mismatch")
            if len(rec.get("qtype_onehot", [])) != expected_qtypes:
                raise ValueError(f"{path}:{idx} qtype_onehot dim mismatch")
            break


def _existing(paths):
    return [p for p in paths if os.path.exists(p) and count_lines(p) > 0]


def create_combined_rank_files(retriever: str, include_eval_in_dev: bool = False):
    """
    Build train/dev files from real generated rank-data across datasets.
    """
    datasets = ["pubmedqa_artificial", "medmcqa", "pubmedqa_labeled"]
    train_candidates = [resolve_rank_data_file(retriever, ds, "train") for ds in datasets]
    dev_candidates = [resolve_rank_data_file(retriever, ds, "dev") for ds in datasets]
    if include_eval_in_dev:
        dev_candidates += [resolve_rank_data_file(retriever, ds, "eval") for ds in datasets]

    train_sources = _existing(train_candidates)
    dev_sources = _existing(dev_candidates)

    train_path = rank_data_combined_file(retriever, "train_all")
    dev_path = rank_data_combined_file(retriever, "dev_all")

    if train_sources:
        logger.info(f"Combining train files for {retriever}: {train_sources}")
        concat_files(train_sources, train_path)
    else:
        logger.warning(f"No train rank-data sources found for retriever={retriever}")

    if dev_sources:
        logger.info(f"Combining dev/eval/test files for {retriever}: {dev_sources}")
        concat_files(dev_sources, dev_path)
    else:
        logger.warning(f"No dev/eval/test rank-data sources found for retriever={retriever}")

    return train_path, dev_path, train_sources, dev_sources


def _default_query_cache_path(retriever: str) -> str:
    return f"data/query_emb_cache_{retriever}_train_all.pkl"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train GARDIAN for one or all retrievers")
    parser.add_argument(
        "--retriever",
        type=str,
        choices=[*ALL_TRAIN_RETRIEVERS, "all"],
        default="all",
        help=(
            "Retriever rank-data family to train on. "
            "Hybrid combos used in our experiments: "
            "hybrid_bm25_faiss(BM25+FAISS), "
            "hybrid_bm25_medcpt(BM25+MedCPT), "
            "hybrid_spladepp_faiss(SPLADE++ + FAISS), "
            "hybrid_spladepp_medcpt(SPLADE++ + MedCPT). "
            "Use 'all' to train one GARDIAN per hybrid family (4 checkpoints)."
        ),
    )
    parser.add_argument(
        "--include-eval-in-dev",
        action="store_true",
        help=(
            "Also include *_eval rank files in the dev set. "
            "Test splits are never used for dev."
        ),
    )
    parser.add_argument(
        "--no-auto-cache-path",
        action="store_true",
        help=(
            "Disable retriever-specific auto cache path override for "
            "training.query_emb_cache_path."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for training.epochs from configs/base.yaml (e.g., 20).",
    )
    parser.add_argument(
        "--branch-hidden",
        type=int,
        default=None,
        help="Optional override for model.branch_hidden.",
    )
    parser.add_argument(
        "--controller-hidden",
        type=int,
        default=None,
        help="Optional override for model.controller_hidden.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Optional override for model.dropout.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load("configs/base.yaml")
    assert_cfg_question_types(cfg.model.question_types)
    if args.epochs is not None:
        if int(args.epochs) <= 0:
            raise ValueError("--epochs must be > 0")
        cfg.training.epochs = int(args.epochs)
        logger.info(f"Overriding cfg.training.epochs -> {cfg.training.epochs}")
    if args.branch_hidden is not None:
        if int(args.branch_hidden) <= 0:
            raise ValueError("--branch-hidden must be > 0")
        cfg.model.branch_hidden = int(args.branch_hidden)
        logger.info(f"Overriding cfg.model.branch_hidden -> {cfg.model.branch_hidden}")
    if args.controller_hidden is not None:
        if int(args.controller_hidden) <= 0:
            raise ValueError("--controller-hidden must be > 0")
        cfg.model.controller_hidden = int(args.controller_hidden)
        logger.info(f"Overriding cfg.model.controller_hidden -> {cfg.model.controller_hidden}")
    if args.dropout is not None:
        if not (0.0 <= float(args.dropout) < 1.0):
            raise ValueError("--dropout must be in [0, 1)")
        cfg.model.dropout = float(args.dropout)
        logger.info(f"Overriding cfg.model.dropout -> {cfg.model.dropout}")
    cudnn_det = bool(getattr(cfg.training, "cudnn_deterministic", False))
    set_global_seed(int(cfg.seed), cudnn_deterministic=cudnn_det)

    # ── Device ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Training on: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = "cpu"
        logger.warning("Training on: cpu — CUDA not detected.")

    # ``--retriever all`` trains GARDIAN once per hybrid family.
    # The four single retrievers (bm25, faiss, spladepp, medcpt) still work
    # when named explicitly for ablation runs.
    FOCUS_HYBRIDS = [
        "hybrid_bm25_faiss",
        "hybrid_bm25_medcpt",
        "hybrid_spladepp_faiss",
        "hybrid_spladepp_medcpt",
    ]
    retrievers = (
        FOCUS_HYBRIDS
        if args.retriever == "all"
        else [normalize_retriever_name(args.retriever)]
    )
    out_dir = pathlib.Path(cfg.paths.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    logger.info("Hybrid retriever combinations:")
    for name in FOCUS_HYBRIDS:
        logger.info(f"  - {name}: {HYBRID_RETRIEVER_COMBINATIONS[name]}")

    for retriever in retrievers:
        logger.info("=" * 72)
        logger.info(f"Training for retriever family: {retriever}")
        logger.info("=" * 72)

        # ── Data from real generated rank files (all datasets) ─────────────
        train_path, dev_path, train_sources, dev_sources = create_combined_rank_files(
            retriever,
            include_eval_in_dev=bool(args.include_eval_in_dev),
        )
        auto_cache_path = _default_query_cache_path(retriever)
        if not bool(args.no_auto_cache_path):
            cfg.training.query_emb_cache_path = auto_cache_path
            logger.info(f"Using retriever-specific query_emb cache: {auto_cache_path}")
        if not os.path.exists(str(cfg.training.query_emb_cache_path)):
            logger.warning(
                f"query_emb cache not found: {cfg.training.query_emb_cache_path} "
                f"(recommended: precompute before training for retriever={retriever})"
            )
        _validate_feature_dims(train_path, cfg)
        _validate_feature_dims(dev_path, cfg)

        train_lines = count_lines(train_path)
        dev_lines = count_lines(dev_path)
        if train_lines == 0 or dev_lines == 0:
            logger.warning(
                f"Skipping training for {retriever}: empty data "
                f"(train={train_lines}, dev={dev_lines})"
            )
            continue
        logger.info(f"train={train_lines:,} lines | dev={dev_lines:,} lines")

        model = build_gardian_from_model_cfg(cfg.model)

        logger.info(
            "GARDIAN hparams from configs/base.yaml | "
            f"sparse={cfg.model.sparse_feat_dim} dense={cfg.model.dense_feat_dim} "
            f"kg={cfg.model.kg_feat_dim} branch_h={cfg.model.branch_hidden} "
            f"ctrl_h={cfg.model.controller_hidden} dropout={cfg.model.dropout}"
        )

        model.to(device)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"GARDIAN | trainable parameters: {total_params:,}")
        logger.info(
            f"Training | epochs={cfg.training.epochs} lr={cfg.training.lr} "
            f"wd={cfg.training.weight_decay} batch={cfg.training.batch_size} "
            f"num_negatives={cfg.training.num_negatives} margin={cfg.training.margin} "
            f"hard_negative_top_n={getattr(cfg.training, 'hard_negative_top_n', None)} "
            f"hard_negative_fraction={getattr(cfg.training, 'hard_negative_fraction', 0.0)}"
        )

        # ── Train ────────────────────────────────────────────────────────────
        trainer = GARDIANTrainer(model, cfg, device=device)
        best_ndcg10 = trainer.fit(train_path, dev_path)

        # ── Persist epoch-by-epoch training logs (always) ───────────────────
        training_log_dir = out_dir / "gardian_training" / retriever
        training_log_dir.mkdir(parents=True, exist_ok=True)
        epoch_logs_path = training_log_dir / "epoch_logs.jsonl"
        epoch_csv_path = training_log_dir / "epoch_logs.csv"
        run_summary_path = training_log_dir / "run_summary.json"

        with open(epoch_logs_path, "w", encoding="utf-8") as f:
            for row in trainer.epoch_logs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        csv_fields = [
            "epoch",
            "train_loss",
            "did_eval",
            "dev_ndcg@10",
            "is_best",
            "best_ndcg@10_so_far",
            "patience_counter",
            "epoch_elapsed_sec",
        ]
        with open(epoch_csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields)
            w.writeheader()
            for row in trainer.epoch_logs:
                w.writerow(row)

        run_summary = {
            "retriever": retriever,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "best_ndcg10": float(best_ndcg10),
            "epochs_completed": len(trainer.epoch_logs),
            "train_path": train_path,
            "dev_path": dev_path,
            "query_emb_cache_path": str(getattr(cfg.training, "query_emb_cache_path", "")),
            "epoch_logs_jsonl": str(epoch_logs_path),
            "epoch_logs_csv": str(epoch_csv_path),
        }
        with open(run_summary_path, "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved epoch logs -> {epoch_logs_path}")
        logger.info(f"Saved epoch CSV -> {epoch_csv_path}")
        logger.info(f"Saved run summary -> {run_summary_path}")

        # ── Checkpoint ───────────────────────────────────────────────────────
        ckpt_path = out_dir / f"gardian_best_{retriever}.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                "best_ndcg10": best_ndcg10,
                "device": device,
                "dtype": "float32",
                "git_revision": _git_revision(),
                "retriever": retriever,
            },
            ckpt_path,
        )
        summary[retriever] = {
            "best_ndcg10": float(best_ndcg10),
            "checkpoint": str(ckpt_path),
            "train_path": train_path,
            "dev_path": dev_path,
            "train_sources": train_sources,
            "dev_sources": dev_sources,
            "training_log_dir": str(training_log_dir),
            "epoch_logs_jsonl": str(epoch_logs_path),
            "epoch_logs_csv": str(epoch_csv_path),
            "run_summary": str(run_summary_path),
        }
        logger.success(f"Done ({retriever}) | best nDCG@10={best_ndcg10:.4f} | checkpoint -> {ckpt_path}")

    summary_path = out_dir / "training_summary_all_retrievers.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.success(f"Training summary saved -> {summary_path}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/04_train_gardian.py",
        "args": vars(args),
        "seed": int(cfg.seed),
        "git_revision": _git_revision(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "summary_path": str(summary_path),
        "input_files": {},
    }
    for retriever in retrievers:
        p_train = rank_data_combined_file(retriever, "train_all")
        p_dev = rank_data_combined_file(retriever, "dev_all")
        manifest["input_files"][p_train] = {"exists": os.path.exists(p_train), "sha256": _sha256_file(p_train)}
        manifest["input_files"][p_dev] = {"exists": os.path.exists(p_dev), "sha256": _sha256_file(p_dev)}
    manifest_path = out_dir / "training_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote training manifest -> {manifest_path}")


if __name__ == "__main__":
    main()

