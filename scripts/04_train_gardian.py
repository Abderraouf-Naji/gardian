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
import time

os.environ.setdefault("PYTHONUTF8", "1")

sys.path.insert(0, ".")

import torch
from loguru import logger
from omegaconf import OmegaConf

from src.common.hybrid_retrievers import (
    FOCUS_HYBRID_RETRIEVERS,
    HYBRID_RETRIEVER_COMBINATIONS,
)
from src.common.question_types import assert_cfg_question_types
from src.common.rank_data_paths import (
    normalize_retriever_name,
    rank_data_combined_file,
    resolve_rank_data_file,
)
from src.common.repro import set_seed
from src.common.seeds import (
    DEFAULT_SEEDS,
    add_seeds_argument,
    guard_seed_artifact,
    parse_seeds,
    seed_path,
    seeds_root,
)
from src.model.gardian import build_gardian_from_model_cfg
from src.training.losses import ALL_LOSSES, is_listwise


def _cfg_use_controller(cfg) -> bool:
    return bool(getattr(cfg.model, "use_controller", True))
from src.training.trainer import GARDIANTrainer

# Single-retriever names kept for legacy CLI; not used when ``--retriever all``.
# The three paper QA collections. Training never mixes in other corpora.
DEFAULT_TRAIN_DATASETS = ["pubmedqa_artificial", "medmcqa", "pubmedqa_labeled"]
ALLOWED_TRAIN_DATASETS = frozenset(DEFAULT_TRAIN_DATASETS)

ALL_TRAIN_RETRIEVERS = [
    *FOCUS_HYBRID_RETRIEVERS,
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


def count_queries(path: str) -> int:
    """Number of distinct queries in a query-contiguous rank file."""
    if not os.path.exists(path):
        return 0
    n, last = 0, object()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            qid = json.loads(line).get("qid")
            if qid != last:
                n += 1
                last = qid
    return n


def concat_files(src_paths, dst_path, *, max_queries_per_source=None):
    """
    Concatenate rank JSONL files, optionally capping each source by QUERY count.

    Rank files are query-contiguous, so the cap keeps whole queries: truncating
    mid-query would leave a pool without its positives. ``max_queries_per_source``
    backs both ``--balance-datasets`` (equalise the sources) and
    ``--max-train-queries`` (small runs for configuration search).
    """
    total_lines = 0
    pathlib.Path(dst_path).parent.mkdir(parents=True, exist_ok=True)

    with open(dst_path, "w", encoding="utf-8") as out_f:
        for src in src_paths:
            if not os.path.exists(src):
                logger.warning(f"Source file not found, skipping: {src}")
                continue
            cap = None
            if max_queries_per_source:
                cap = max_queries_per_source.get(src) if isinstance(
                    max_queries_per_source, dict) else int(max_queries_per_source)
            logger.info(f"Adding {src} -> {dst_path}"
                        + (f"  (first {cap:,} queries)" if cap else ""))
            file_lines, seen, last = 0, 0, object()
            with open(src, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    if cap:
                        qid = json.loads(line).get("qid")
                        if qid != last:
                            seen += 1
                            last = qid
                        if seen > cap:
                            break
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
    expected_query = int(cfg.model.query_feat_dim)
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
            if "query_emb" in rec and len(rec.get("query_emb", [])) != expected_query:
                raise ValueError(f"{path}:{idx} query_emb dim mismatch")
            break


def _existing(paths):
    """Keep the rank files that exist and are non-empty, naming the ones that don't."""
    out = []
    for p in paths:
        if os.path.exists(p) and count_lines(p) > 0:
            out.append(p)
        else:
            logger.warning(f"No rank data yet, skipping source: {p}")
    return out


def create_combined_rank_files(
    retriever: str,
    include_eval_in_dev: bool = False,
    *,
    balance_datasets: bool = False,
    max_train_queries: Optional[int] = None,
    train_datasets: Optional[list] = None,
    prefer_denoised: bool = False,
):
    """
    Build train/dev files from real generated rank-data across datasets.

    ``train_datasets`` selects which collections are concatenated; sources with
    no generated rank data are skipped with a warning rather than failing, so a
    collection can be added to the list before its rank data exists.
    """
    datasets = list(train_datasets or DEFAULT_TRAIN_DATASETS)
    unknown = [d for d in datasets if d not in ALLOWED_TRAIN_DATASETS]
    if unknown:
        raise ValueError(
            f"Unsupported training collections {unknown}. "
            f"Allowed: {sorted(ALLOWED_TRAIN_DATASETS)}"
        )

    def _resolve(ds: str, split: str) -> str:
        """
        Rank file for (dataset, split), preferring a denoised variant.

        ``scripts/15_denoise_false_negatives.py`` writes ``*_denoised.jsonl``
        beside the original. With ``prefer_denoised`` the denoised file wins
        where it exists, so a false-negative-cleaned MedMCQA can be trained on
        without renaming or overwriting the generated data.
        """
        base = resolve_rank_data_file(retriever, ds, split)
        if prefer_denoised:
            cand = pathlib.Path(base)
            denoised = cand.with_name(cand.stem + "_denoised.jsonl")
            if denoised.is_file():
                logger.info(f"Using denoised rank data for {ds}/{split}: {denoised}")
                return str(denoised)
        return base

    train_candidates = [_resolve(ds, "train") for ds in datasets]
    # Dev is deliberately NOT denoised: model selection must run against the
    # labels as generated, or the cross-encoder's opinion silently becomes the
    # selection criterion.
    dev_candidates = [resolve_rank_data_file(retriever, ds, "dev") for ds in datasets]
    if include_eval_in_dev:
        dev_candidates += [resolve_rank_data_file(retriever, ds, "eval") for ds in datasets]

    train_sources = _existing(train_candidates)
    dev_sources = _existing(dev_candidates)

    train_path = rank_data_combined_file(retriever, "train_all")
    dev_path = rank_data_combined_file(retriever, "dev_all")

    caps = None
    if balance_datasets and len(train_sources) > 1:
        counts = {p: count_queries(p) for p in train_sources}
        smallest = min(counts.values())
        caps = {p: smallest for p in train_sources}
        logger.info(
            "Balancing datasets: "
            + ", ".join(f"{os.path.basename(p)}={n:,}" for p, n in counts.items())
            + f"  -> capping each to {smallest:,} queries"
        )
    if max_train_queries:
        per = max(1, int(max_train_queries) // max(len(train_sources), 1))
        caps = {p: min(per, caps[p]) if caps else per for p in train_sources}
        logger.info(f"--max-train-queries: {per:,} queries per source")

    if train_sources:
        logger.info(f"Combining train files for {retriever}: {train_sources}")
        concat_files(train_sources, train_path, max_queries_per_source=caps)
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


def _train_one_retriever(
    cfg,
    args,
    *,
    retriever: str,
    device: str,
    seed: int,
    results_dir: pathlib.Path,
) -> Optional[dict]:
    """
    Train GARDIAN for one retriever family under one seed.

    Every artifact is written beneath ``results/seeds/seed_<seed>/`` and is
    guarded against overwriting, so a completed seed can never be clobbered by
    a re-run unless ``--overwrite-seed-artifacts`` is passed.

    Returns the summary dict for this (seed, retriever) cell, or None when the
    rank data is empty and training was skipped.
    """
    overwrite = bool(args.overwrite_seed_artifacts)

    logger.info("=" * 72)
    logger.info(f"Training | seed={seed} | retriever family: {retriever}")
    logger.info("=" * 72)

    # ── Data from real generated rank files (all datasets) ───────────────────
    train_path, dev_path, train_sources, dev_sources = create_combined_rank_files(
        retriever,
        include_eval_in_dev=bool(args.include_eval_in_dev),
        balance_datasets=bool(args.balance_datasets),
        max_train_queries=args.max_train_queries,
        train_datasets=args.train_datasets,
        prefer_denoised=bool(getattr(args, "denoised", False)),
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
            f"Skipping training for {retriever} (seed={seed}): empty data "
            f"(train={train_lines}, dev={dev_lines})"
        )
        return None
    logger.info(f"train={train_lines:,} lines | dev={dev_lines:,} lines")

    # ── Per-seed output locations (checked before the expensive work) ────────
    ckpt_path = guard_seed_artifact(
        seed_path(results_dir, seed, f"gardian_best_{retriever}.pt"),
        overwrite=overwrite,
    )
    training_log_dir = seed_path(results_dir, seed, "gardian_training", retriever)
    epoch_logs_path = guard_seed_artifact(
        training_log_dir / "epoch_logs.jsonl", overwrite=overwrite
    )
    epoch_logs_path.unlink(missing_ok=True)   # appended to from scratch below
    epoch_csv_path = guard_seed_artifact(
        training_log_dir / "epoch_logs.csv", overwrite=overwrite
    )
    run_summary_path = guard_seed_artifact(
        training_log_dir / "run_summary.json", overwrite=overwrite
    )

    model = build_gardian_from_model_cfg(cfg.model)

    logger.info(
        "GARDIAN hparams from configs/base.yaml | "
        f"sparse={cfg.model.sparse_feat_dim} dense={cfg.model.dense_feat_dim} "
        f"branch_h={cfg.model.branch_hidden} "
        f"controller={'on' if _cfg_use_controller(cfg) else 'OFF (GARDIAN-Lite)'} "
        f"ctrl_h={cfg.model.controller_hidden} dropout={cfg.model.dropout} "
        f"ctrl_in={list(cfg.model.get('controller_inputs', ['query_emb']))} "
        f"norm_branches={bool(cfg.model.get('normalize_branches', False))}"
    )

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"GARDIAN | trainable parameters: {total_params:,}")
    logger.info(
        f"Training | seed={seed} epochs={cfg.training.epochs} lr={cfg.training.lr} "
        f"wd={cfg.training.weight_decay} batch={cfg.training.batch_size} "
        f"warmup_epochs={getattr(cfg.training, 'warmup_epochs', 0)} "
        f"min_lr_ratio={getattr(cfg.training, 'min_lr_ratio', 0.0)} "
        f"loss={cfg.training.loss} "
        f"{'group_size=' + str(getattr(cfg.training, 'listwise_group_size', 64)) if is_listwise(str(cfg.training.loss)) else 'num_negatives=' + str(cfg.training.num_negatives) + ' margin=' + str(cfg.training.margin)} "
        f"hard_negative_top_n={getattr(cfg.training, 'hard_negative_top_n', None)} "
        f"hard_negative_fraction={getattr(cfg.training, 'hard_negative_fraction', 0.0)}"
    )

    # ── Train ────────────────────────────────────────────────────────────────
    # Artifacts are written the moment they improve, not after fit() returns, so
    # an interrupted run keeps its best checkpoint and the log can be watched
    # live. Epochs are ~80 minutes; losing one to a crash is not acceptable.
    def _persist_best(state, dev_ndcg: float, epoch: int) -> None:
        torch.save(
            {
                "model_state": state,
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                "best_ndcg10": float(dev_ndcg),
                "epoch": int(epoch),
                "device": device,
                "dtype": "float32",
                "git_revision": _git_revision(),
                "retriever": retriever,
                "seed": int(seed),
                "in_progress": True,
            },
            ckpt_path,
        )
        logger.info(f"  checkpoint written (epoch {epoch}, dev nDCG@10={dev_ndcg:.4f}) -> {ckpt_path}")

    def _persist_epoch(row) -> None:
        with open(epoch_logs_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    wall_start = time.time()
    trainer = GARDIANTrainer(model, cfg, device=device)
    best_ndcg10 = trainer.fit(
        train_path, dev_path, on_best=_persist_best, on_epoch=_persist_epoch
    )
    wall_elapsed = time.time() - wall_start

    # ── Persist epoch-by-epoch training logs (always) ────────────────────────
    # epoch_logs.jsonl was appended incrementally by _persist_epoch above.

    csv_fields = [
        "epoch",
        "train_loss",
        "train_loss_per_pair",
        "did_eval",
        "dev_ndcg@10",
        "is_best",
        "best_ndcg@10_so_far",
        "patience_counter",
        "epoch_elapsed_sec",
        "lr",
    ]
    with open(epoch_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for row in trainer.epoch_logs:
            w.writerow(row)

    run_summary = {
        "retriever": retriever,
        "seed": int(seed),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "best_ndcg10": float(best_ndcg10),
        "epochs_completed": len(trainer.epoch_logs),
        "trainable_parameters": int(total_params),
        "train_wall_clock_sec": float(wall_elapsed),
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

    # ── Checkpoint ───────────────────────────────────────────────────────────
    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "best_ndcg10": best_ndcg10,
            "device": device,
            "dtype": "float32",
            "git_revision": _git_revision(),
            "retriever": retriever,
            "seed": int(seed),
            "in_progress": False,
        },
        ckpt_path,
    )
    logger.success(
        f"Done (seed={seed}, {retriever}) | best nDCG@10={best_ndcg10:.4f} "
        f"| {wall_elapsed / 60.0:.1f} min | checkpoint -> {ckpt_path}"
    )

    return {
        "seed": int(seed),
        "best_ndcg10": float(best_ndcg10),
        "trainable_parameters": int(total_params),
        "train_wall_clock_sec": float(wall_elapsed),
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


def _run_seed(
    cfg,
    args,
    *,
    seed: int,
    device: str,
    retrievers,
    results_dir: pathlib.Path,
) -> dict:
    """Train every requested retriever family under a single seed."""
    cudnn_det = bool(getattr(cfg.training, "cudnn_deterministic", True))
    set_seed(seed, cudnn_deterministic=cudnn_det)
    cfg.seed = int(seed)
    logger.info(f"Seed set to {seed} (cudnn_deterministic={cudnn_det})")

    summary = {}
    for retriever in retrievers:
        cell = _train_one_retriever(
            cfg,
            args,
            retriever=retriever,
            device=device,
            seed=seed,
            results_dir=results_dir,
        )
        if cell is not None:
            summary[retriever] = cell

    summary_path = guard_seed_artifact(
        seed_path(results_dir, seed, "training_summary_all_retrievers.json"),
        overwrite=bool(args.overwrite_seed_artifacts),
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.success(f"Training summary (seed={seed}) saved -> {summary_path}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/04_train_gardian.py",
        "args": vars(args),
        "seed": int(seed),
        "git_revision": _git_revision(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "summary_path": str(summary_path),
        "input_files": {},
    }
    for retriever in retrievers:
        p_train = rank_data_combined_file(retriever, "train_all")
        p_dev = rank_data_combined_file(retriever, "dev_all")
        manifest["input_files"][p_train] = {
            "exists": os.path.exists(p_train),
            "sha256": _sha256_file(p_train),
        }
        manifest["input_files"][p_dev] = {
            "exists": os.path.exists(p_dev),
            "sha256": _sha256_file(p_dev),
        }
    manifest_path = guard_seed_artifact(
        seed_path(results_dir, seed, "training_manifest.json"),
        overwrite=bool(args.overwrite_seed_artifacts),
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote training manifest (seed={seed}) -> {manifest_path}")

    return summary


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
        "--lr",
        type=float,
        default=None,
        help=(
            "Override training.lr. The configured 3e-5 is a transformer "
            "fine-tuning rate; these are from-scratch MLPs over tabular "
            "features, which usually want 1e-4 to 1e-3."
        ),
    )
    parser.add_argument(
        "--max-train-queries",
        type=int,
        default=None,
        help=(
            "Train on the first N queries of the combined file. For fast "
            "configuration search: a 20k-query run finishes in minutes rather "
            "than hours."
        ),
    )
    parser.add_argument(
        "--balance-datasets",
        action="store_true",
        help=(
            "Cap each source dataset to the size of the smallest, so training "
            "is not dominated by PubMedQA-artificial (78%% of the combined "
            "file, and the split where the gold passage is a chunk of the "
            "source abstract and already ranks first)."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override training.num_workers. Data construction is the bottleneck.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Optional override for model.dropout.",
    )
    parser.add_argument(
        "--denoised",
        action="store_true",
        help="Prefer *_denoised.jsonl training files where they exist (written by "
             "scripts/15_denoise_false_negatives.py). Dev files are never "
             "denoised, so model selection stays on the generated labels.",
    )
    parser.add_argument(
        "--train-datasets",
        type=str,
        default=None,
        help="Comma-separated subset of the three paper collections to concatenate "
             f"into the training file (default: {','.join(DEFAULT_TRAIN_DATASETS)}). "
             "Allowed: pubmedqa_artificial, medmcqa, pubmedqa_labeled.",
    )
    parser.add_argument(
        "--controller-inputs",
        type=str,
        default=None,
        help="Override model.controller_inputs (comma-separated subset of "
             "query_emb,pool_feats). 'query_emb' is the CoopIS setting whose "
             "alpha does not transfer zero-shot; 'pool_feats' gives the "
             "controller scale-free evidence about how each channel behaved on "
             "this query's pool. This is the controller-input ablation axis.",
    )
    parser.add_argument(
        "--normalize-branches",
        dest="normalize_branches",
        action="store_true",
        default=None,
        help="Standardise each branch within the candidate pool before mixing, "
             "so alpha is a genuine mixing weight rather than a weight on two "
             "arbitrarily-scaled quantities (listwise only).",
    )
    parser.add_argument(
        "--no-normalize-branches",
        dest="normalize_branches",
        action="store_false",
        default=None,
        help="Mix raw branch outputs, as in the CoopIS submission.",
    )
    parser.add_argument(
        "--no-controller",
        action="store_true",
        help=(
            "Train without the query-adaptive controller (GARDIAN-Lite). The "
            "fusion weight becomes a constant, which the branch heads absorb, "
            "so nothing needs tuning and inference needs no query encoder."
        ),
    )
    parser.add_argument(
        "--loss",
        choices=sorted(ALL_LOSSES),
        default=None,
        help=(
            "Override training.loss. The three listwise objectives consume whole "
            "candidate pools and set --batch-size to count queries, not pairs; "
            "pairwise_softplus_margin is the CoopIS submission, kept for the "
            "loss ablation."
        ),
    )
    parser.add_argument(
        "--listwise-group-size",
        type=int,
        default=None,
        help="Override training.listwise_group_size (candidates per pool, listwise only).",
    )
    add_seeds_argument(
        parser,
        default=DEFAULT_SEEDS,
        help_suffix="One checkpoint per (seed, retriever) is trained.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Deprecated single-seed alias; equivalent to --seeds <SEED>.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load("configs/base.yaml")

    if args.seed is not None:
        seeds = [int(args.seed)]
        logger.warning(
            f"--seed is deprecated; running the single seed {args.seed}. "
            "Use --seeds for the multi-seed protocol."
        )
    else:
        seeds = parse_seeds(args.seeds)
    args.seeds = seeds

    assert_cfg_question_types(cfg.evaluation.question_types)
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
    if args.lr is not None:
        cfg.training.lr = float(args.lr)
        logger.info(f"Overriding cfg.training.lr -> {cfg.training.lr}")
    if args.num_workers is not None:
        if int(args.num_workers) < 0:
            raise ValueError("--num-workers must be >= 0")
        cfg.training.num_workers = int(args.num_workers)
        logger.info(f"Overriding cfg.training.num_workers -> {cfg.training.num_workers}")
    if args.train_datasets is not None:
        args.train_datasets = [t.strip() for t in args.train_datasets.split(",") if t.strip()]
        if not args.train_datasets:
            parser.error("--train-datasets was empty")
        unknown = [d for d in args.train_datasets if d not in ALLOWED_TRAIN_DATASETS]
        if unknown:
            parser.error(
                f"Unknown --train-datasets {unknown}. "
                f"Allowed: {sorted(ALLOWED_TRAIN_DATASETS)}"
            )
        logger.info(f"Training collections -> {args.train_datasets}")

    if args.controller_inputs is not None:
        from src.model.gardian import CONTROLLER_INPUTS

        chosen = [t.strip() for t in args.controller_inputs.split(",") if t.strip()]
        unknown = [t for t in chosen if t not in CONTROLLER_INPUTS]
        if not chosen or unknown:
            parser.error(
                f"--controller-inputs got {args.controller_inputs!r}; expected a "
                f"non-empty comma-separated subset of {','.join(CONTROLLER_INPUTS)}"
            )
        cfg.model.controller_inputs = chosen
        logger.info(f"Overriding cfg.model.controller_inputs -> {chosen}")

    if args.normalize_branches is not None:
        cfg.model.normalize_branches = bool(args.normalize_branches)
        logger.info(
            f"Overriding cfg.model.normalize_branches -> {cfg.model.normalize_branches}"
        )
    if cfg.model.get("normalize_branches") and not is_listwise(cfg.training.loss):
        parser.error(
            f"model.normalize_branches needs a listwise loss (got "
            f"{cfg.training.loss!r}): standardising a branch requires the pool to "
            "standardise it over, and the pairwise path has no pool."
        )

    if args.no_controller:
        cfg.model.use_controller = False
        logger.info("Overriding cfg.model.use_controller -> False (GARDIAN-Lite)")
    if args.loss is not None:
        cfg.training.loss = args.loss
        logger.info(f"Overriding cfg.training.loss -> {cfg.training.loss}")
    if args.listwise_group_size is not None:
        if int(args.listwise_group_size) < 2:
            raise ValueError("--listwise-group-size must be >= 2")
        cfg.training.listwise_group_size = int(args.listwise_group_size)
        logger.info(
            f"Overriding cfg.training.listwise_group_size -> {cfg.training.listwise_group_size}"
        )
    if args.dropout is not None:
        if not (0.0 <= float(args.dropout) < 1.0):
            raise ValueError("--dropout must be in [0, 1)")
        cfg.model.dropout = float(args.dropout)
        logger.info(f"Overriding cfg.model.dropout -> {cfg.model.dropout}")

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
    retrievers = (
        FOCUS_HYBRID_RETRIEVERS
        if args.retriever == "all"
        else [normalize_retriever_name(args.retriever)]
    )
    results_dir = pathlib.Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Hybrid retriever combinations:")
    for name in FOCUS_HYBRID_RETRIEVERS:
        logger.info(f"  - {name}: {HYBRID_RETRIEVER_COMBINATIONS[name]}")
    logger.info(
        f"Multi-seed protocol | seeds={seeds} | retrievers={list(retrievers)} | "
        f"{len(seeds) * len(list(retrievers))} training run(s)"
    )

    all_summaries = {}
    for seed in seeds:
        all_summaries[str(seed)] = _run_seed(
            cfg,
            args,
            seed=seed,
            device=device,
            retrievers=retrievers,
            results_dir=results_dir,
        )

    logger.success(
        f"All seeds complete: {seeds}. Raw per-seed artifacts under "
        f"{seeds_root(results_dir)}. Run scripts/aggregate_seeds.py to produce "
        "the mean/std tables consumed by the paper."
    )


if __name__ == "__main__":
    main()
