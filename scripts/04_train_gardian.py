import os
import json
import pathlib
import random
import subprocess
import sys
import hashlib
import platform
from datetime import datetime, timezone
from typing import Optional

os.environ.setdefault("PYTHONUTF8", "1")

sys.path.insert(0, ".")

import torch
from loguru import logger
from omegaconf import OmegaConf

from src.common.question_types import assert_cfg_question_types
from src.common.repro import set_global_seed
from src.model.gardian import GARDIAN
from src.training.trainer import GARDIANTrainer


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


def _read_rank_by_qid(path: str):
    by_qid = {}
    if not os.path.exists(path):
        return by_qid
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = rec.get("qid")
            by_qid.setdefault(qid, []).append(rec)
    return by_qid


def _write_rank_groups(path: str, groups) -> int:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    n_lines = 0
    with open(path, "w", encoding="utf-8") as f:
        for g in groups:
            for rec in g:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_lines += 1
    return n_lines


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


def create_balanced_rank_files(train_path: str, dev_path: str, retriever: str):
    """
    Create balanced merged train/dev rank-data files.
    Downsample PubMedQA to match MedMCQA for better generalization.
    """
    # Load individual files
    pubmedqa_train = f"data/rank_data_{retriever}_pubmedqa_artificial_train.jsonl"
    medmcqa_train = f"data/rank_data_{retriever}_medmcqa_train.jsonl"
    pubmedqa_dev = f"data/rank_data_{retriever}_pubmedqa_artificial_dev.jsonl"
    medmcqa_dev = f"data/rank_data_{retriever}_medmcqa_dev.jsonl"
    
    # Count lines with explicit encoding
    pubmedqa_train_lines = count_lines(pubmedqa_train)
    medmcqa_train_lines = count_lines(medmcqa_train)
    
    logger.info(f"Original sizes - PubMedQA: {pubmedqa_train_lines:,}, MedMCQA: {medmcqa_train_lines:,}")
    if pubmedqa_train_lines == 0 or medmcqa_train_lines == 0:
        logger.warning(
            "Missing/empty train rank-data files for balancing: "
            f"{pubmedqa_train} ({pubmedqa_train_lines}), "
            f"{medmcqa_train} ({medmcqa_train_lines})"
        )
        return train_path, dev_path
    
    pub_groups = _read_rank_by_qid(pubmedqa_train)
    med_groups = _read_rank_by_qid(medmcqa_train)
    pub_qids = list(pub_groups.keys())
    med_qids = list(med_groups.keys())
    logger.info(f"Original queries - PubMedQA: {len(pub_qids):,}, MedMCQA: {len(med_qids):,}")
    if not pub_qids or not med_qids:
        logger.warning("Could not build query-level balanced train file; missing query groups.")
        return train_path, dev_path
    target_q = len(med_qids)
    sample_q = min(target_q, len(pub_qids))
    logger.info(f"Target query count from PubMedQA: {sample_q:,}")

    balanced_train_path = train_path

    if not os.path.exists(balanced_train_path):
        logger.info("Creating balanced training data...")
        sampled_pub_qids = set(random.sample(pub_qids, sample_q))
        sampled_groups = [pub_groups[qid] for qid in pub_qids if qid in sampled_pub_qids]
        sampled_groups.extend(med_groups[qid] for qid in med_qids)
        random.shuffle(sampled_groups)
        n_lines = _write_rank_groups(balanced_train_path, sampled_groups)
        logger.info(f"Created balanced training with {n_lines:,} lines")
        logger.info(f"  PubMedQA sampled queries: {len(sampled_pub_qids):,}")
        logger.info(f"  MedMCQA queries: {len(med_qids):,}")
    else:
        logger.info(f"Using existing training file: {balanced_train_path}")
    
    # Keep dev as is (or also balance)
    if not os.path.exists(dev_path):
        concat_files([pubmedqa_dev, medmcqa_dev], dev_path)
    
    return balanced_train_path, dev_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train GARDIAN for one or all retrievers")
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["hybrid", "hybrid_neural", "doc2query", "all"],
        default="all",
        help="Retriever rank-data family to train on",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load("configs/base.yaml")
    assert_cfg_question_types(cfg.model.question_types)
    cudnn_det = bool(getattr(cfg.training, "cudnn_deterministic", False))
    set_global_seed(int(cfg.seed), cudnn_deterministic=cudnn_det)

    # ── Device ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Training on: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = "cpu"
        logger.warning("Training on: cpu — CUDA not detected.")

    retrievers = ["hybrid", "hybrid_neural", "doc2query"] if args.retriever == "all" else [args.retriever]
    out_dir = pathlib.Path(cfg.paths.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for retriever in retrievers:
        logger.info("=" * 72)
        logger.info(f"Training for retriever family: {retriever}")
        logger.info("=" * 72)

        # ── Data with balanced sampling ─────────────────────────────────────
        train_path = f"data/rank_data_{retriever}_train_balanced.jsonl"
        dev_path = f"data/rank_data_{retriever}_dev.jsonl"
        train_path, dev_path = create_balanced_rank_files(train_path, dev_path, retriever)
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
            f"num_negatives={cfg.training.num_negatives}"
        )

        # ── Train ────────────────────────────────────────────────────────────
        trainer = GARDIANTrainer(model, cfg, device=device)
        best_ndcg10 = trainer.fit(train_path, dev_path)

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
        p_train = f"data/rank_data_{retriever}_train_balanced.jsonl"
        p_dev = f"data/rank_data_{retriever}_dev.jsonl"
        manifest["input_files"][p_train] = {"exists": os.path.exists(p_train), "sha256": _sha256_file(p_train)}
        manifest["input_files"][p_dev] = {"exists": os.path.exists(p_dev), "sha256": _sha256_file(p_dev)}
    manifest_path = out_dir / "training_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote training manifest -> {manifest_path}")


if __name__ == "__main__":
    main()

