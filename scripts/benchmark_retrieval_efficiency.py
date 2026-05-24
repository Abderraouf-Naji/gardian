#!/usr/bin/env python3
"""
Live retrieval latency benchmark (ms/query) + merge index-size report.

Times per query (per-dataset indices):
  - sparse: first channel only (BM25 or SPLADE++)
  - dense:  second channel only (FAISS or MedCPT)
  - hybrid: union retrieve (both channels + RRF merge)
  - gardian: adaptive retrieve + GARDIAN rerank (end-to-end retrieval stack)

Writes ``results/retrieval_efficiency.json`` for plotting.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.hybrid_retrievers import FOCUS_HYBRID_RETRIEVERS
from src.common.question_types import normalize_question_type, qtype_onehot
from src.evaluation.index_stats import report_all_index_sizes
from src.model.gardian import build_gardian_from_model_cfg, load_checkpoint_state
from src.pipeline.gardian_adaptive import retrieve_adaptive_candidates_live
from src.pipeline.rag_reader import build_retriever_for_qa

DATASET_QUESTION_FILES = {
    "pubmedqa_labeled": "data/pubmedqa_labeled_eval.jsonl",
    "pubmedqa_artificial": "data/pubmedqa_artificial_test.jsonl",
    "medmcqa": "data/medmcqa_test.jsonl",
}


def _load_questions(path: Path, n: int, seed: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    if n > 0 and len(rows) > n:
        rng = random.Random(seed)
        rows = rng.sample(rows, n)
    return rows


def _timed_ms(fn: Callable[[], Any]) -> float:
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000.0


def _latency_stats(samples_ms: List[float]) -> Dict[str, float]:
    if not samples_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "n": 0}
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "n": int(len(arr)),
    }


def _load_gardian(cfg: Any, retriever: str, device: str):
    ckpt_path = Path(cfg.paths.results_dir) / f"gardian_best_{retriever}.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"GARDIAN checkpoint missing: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("cfg", {}) if isinstance(ckpt.get("cfg"), dict) else {}
    ckpt_model_cfg = ckpt_cfg.get("model") if isinstance(ckpt_cfg.get("model"), dict) else None
    model = build_gardian_from_model_cfg(ckpt_model_cfg or cfg.model)
    load_checkpoint_state(model, ckpt["model_state"], strict=False)
    model.to(device)
    model.eval()
    return model


def benchmark_one_setting(
    cfg: Any,
    retriever: str,
    dataset: str,
    *,
    n_queries: int,
    warmup: int,
    seed: int,
    device: str,
    use_faiss_gpu: bool,
    skip_gardian: bool,
) -> Dict[str, Any]:
    qpath = Path(DATASET_QUESTION_FILES[dataset])
    if not qpath.is_file():
        raise FileNotFoundError(f"Questions file missing: {qpath}")

    questions = _load_questions(qpath, n_queries, seed)
    hybrid = build_retriever_for_qa(
        cfg,
        retriever,
        device=device,
        use_faiss_gpu=use_faiss_gpu,
        dataset_name=dataset,
        use_per_dataset_indices=True,
    )
    first = getattr(hybrid, "first", getattr(hybrid, "bm25", None))
    second = getattr(hybrid, "second", getattr(hybrid, "dense", None))
    if first is None or second is None:
        raise RuntimeError(f"Cannot access sparse/dense channels on {type(hybrid)}")

    gardian = None
    encoder = None
    if not skip_gardian:
        try:
            gardian = _load_gardian(cfg, retriever, device)
            encoder = SentenceTransformer(str(cfg.encoder.model_name), device=device)
        except FileNotFoundError as exc:
            logger.warning(f"Skipping GARDIAN timing for {retriever}: {exc}")

    sparse_ms: List[float] = []
    dense_ms: List[float] = []
    hybrid_ms: List[float] = []
    gardian_ms: List[float] = []

    def _run_all(qtext: str, q_emb: Optional[List[float]], q_oh: Optional[List[float]]) -> None:
        sparse_ms.append(_timed_ms(lambda: first.retrieve(qtext)))
        dense_ms.append(_timed_ms(lambda: second.retrieve(qtext)))
        hybrid_ms.append(_timed_ms(lambda: hybrid.retrieve(qtext)))
        if gardian is not None and encoder is not None and q_emb is not None and q_oh is not None:

            def _gardian_retrieve_path() -> None:
                # Live path: adaptive retrieve (rerank needs offline sparse_feats in candidates).
                retrieve_adaptive_candidates_live(
                    qtext,
                    hybrid,
                    gardian,
                    query_emb=q_emb,
                    qtype_onehot=q_oh,
                    cfg=cfg,
                    device=device,
                )

            gardian_ms.append(_timed_ms(_gardian_retrieve_path))

    for item in questions[:warmup]:
        q = (item.get("question") or "").strip()
        if not q:
            continue
        q_emb = q_oh = None
        if encoder is not None:
            q_emb = encoder.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()
            q_oh = qtype_onehot(normalize_question_type(item.get("question_type") or "other"))
        _run_all(q, q_emb, q_oh)

    sparse_ms.clear()
    dense_ms.clear()
    hybrid_ms.clear()
    gardian_ms.clear()

    for item in questions:
        q = (item.get("question") or "").strip()
        if not q:
            continue
        q_emb = q_oh = None
        if encoder is not None:
            q_emb = encoder.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()
            q_oh = qtype_onehot(normalize_question_type(item.get("question_type") or "other"))
        _run_all(q, q_emb, q_oh)

    out: Dict[str, Any] = {
        "dataset": dataset,
        "n_queries": len(questions),
        "warmup": warmup,
        "sparse": _latency_stats(sparse_ms),
        "dense": _latency_stats(dense_ms),
        "hybrid": _latency_stats(hybrid_ms),
    }
    if gardian_ms:
        out["gardian"] = _latency_stats(gardian_ms)
        out["gardian"]["note"] = "adaptive retrieve only (rerank uses offline rank features)"
    else:
        out["gardian"] = {"mean_ms": None, "p50_ms": None, "p95_ms": None, "n": 0, "skipped": True}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retriever",
        default="hybrid_bm25_faiss",
        choices=[*FOCUS_HYBRID_RETRIEVERS, "all"],
    )
    parser.add_argument(
        "--datasets",
        default="pubmedqa_labeled",
        help="Comma-separated datasets (default: pubmedqa_labeled smoke).",
    )
    parser.add_argument("--n-queries", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--faiss-gpu", action="store_true")
    parser.add_argument("--skip-gardian", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/retrieval_efficiency.json"),
    )
    parser.add_argument("--index-sizes-only", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load("configs/base.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device != "auto":
        device = args.device

    index_report = report_all_index_sizes()
    if args.index_sizes_only:
        payload = {
            "meta": {"created_utc": datetime.now(timezone.utc).isoformat()},
            "index_sizes": index_report,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.success(f"Index sizes -> {args.out}")
        return

    retrievers = (
        list(FOCUS_HYBRID_RETRIEVERS) if args.retriever == "all" else [args.retriever]
    )
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    latency: Dict[str, Any] = {}
    for retriever in retrievers:
        latency[retriever] = {}
        for dataset in datasets:
            logger.info(f"Benchmark {retriever} / {dataset} ({args.n_queries} queries)")
            try:
                latency[retriever][dataset] = benchmark_one_setting(
                    cfg,
                    retriever,
                    dataset,
                    n_queries=args.n_queries,
                    warmup=args.warmup,
                    seed=args.seed,
                    device=device,
                    use_faiss_gpu=args.faiss_gpu,
                    skip_gardian=args.skip_gardian,
                )
            except FileNotFoundError as exc:
                logger.warning(f"Skip {retriever}/{dataset}: {exc}")
                latency[retriever][dataset] = {"error": str(exc)}

    payload = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/benchmark_retrieval_efficiency.py",
            "device": device,
            "n_queries": args.n_queries,
            "warmup": args.warmup,
            "per_dataset_indices": True,
        },
        "index_sizes": index_report,
        "latency_ms": latency,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.success(f"Efficiency report -> {args.out}")


if __name__ == "__main__":
    main()
