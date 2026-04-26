"""Controlled end-to-end QA evaluation (RQ4) over precomputed rank data."""

import argparse
import json
import pathlib
import platform
import subprocess
import sys
from datetime import datetime, timezone

import torch
from loguru import logger
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, ".")

from src.common.question_types import assert_cfg_question_types
from src.evaluation.qa_eval import evaluate_qa_from_rank_records
from src.model.gardian import GARDIAN


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


def _load_questions(path: str, dataset_name: str, max_questions: int):
    if not pathlib.Path(path).exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rec["dataset"] = dataset_name
            if "answer" not in rec:
                rec["answer"] = rec.get("final_decision", rec.get("label", ""))
            rows.append(rec)
            if max_questions is not None and len(rows) >= max_questions:
                break
    return rows


def _build_model(cfg, device: str, retriever: str):
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
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Controlled end-to-end QA evaluation")
    p.add_argument("--cfg", type=str, default="configs/base.yaml")
    p.add_argument("--retriever", type=str, choices=["hybrid", "hybrid_neural", "doc2query"], default="hybrid")
    p.add_argument("--max-questions", type=int, default=200)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None)
    p.add_argument(
        "--systems",
        type=str,
        default="bm25,dense,hybrid,gardian",
        help="Comma-separated systems from bm25,dense,hybrid,doc2query,gardian",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)
    assert_cfg_question_types(cfg.model.question_types)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading reader: {cfg.qa.reader_model}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.qa.reader_model)
    reader = AutoModelForCausalLM.from_pretrained(
        cfg.qa.reader_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    systems = [x.strip() for x in args.systems.split(",") if x.strip()]

    gardian_model = None
    if "gardian" in systems:
        gardian_model = _build_model(cfg, device, args.retriever)

    dataset_jobs = [
        ("pubmedqa_labeled", "data/pubmedqa_labeled_eval.jsonl", "eval"),
        ("pubmedqa_artificial", "data/pubmedqa_artificial_test.jsonl", "test"),
        ("medmcqa", "data/medmcqa_test.jsonl", "test"),
    ]

    payload = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/06_end_to_end_qa.py",
            "args": vars(args),
            "git_revision": _git_revision(),
            "platform": platform.platform(),
            "python_version": sys.version,
            "reader_model": cfg.qa.reader_model,
        },
        "datasets": {},
    }

    for dataset_name, q_path, split in dataset_jobs:
        rank_path = f"data/rank_data_{args.retriever}_{dataset_name}_{split}.jsonl"
        if not pathlib.Path(rank_path).exists():
            logger.warning(f"Missing rank data: {rank_path}, skipping")
            continue
        questions = _load_questions(q_path, dataset_name, args.max_questions)
        if not questions:
            logger.warning(f"No questions from {q_path}, skipping")
            continue
        with open(rank_path, "r", encoding="utf-8") as f:
            rank_records = [json.loads(line) for line in f if line.strip()]
        agg, per_q = evaluate_qa_from_rank_records(
            questions,
            rank_records,
            systems=systems,
            gardian_model=gardian_model,
            tokenizer=tokenizer,
            reader_model=reader,
            cfg=cfg,
            device=device,
            bootstrap_samples=int(args.bootstrap),
            bootstrap_seed=int(args.seed),
        )
        payload["datasets"][dataset_name] = {
            "aggregate": agg,
            "per_question": per_q,
        }
        logger.info(f"Completed QA eval for {dataset_name} | systems={list(agg.keys())}")

    out_path = pathlib.Path(args.out) if args.out else pathlib.Path(cfg.paths.results_dir) / f"qa_results_{args.retriever}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.success(f"QA results -> {out_path}")


if __name__ == "__main__":
    main()