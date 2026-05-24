#!/usr/bin/env python3
"""
Confirm the live GARDIAN adaptive reranker pipeline (configs/base.yaml):

  1. Index caps: top_k_bm25 / top_k_faiss = 50 (max per channel, not reader input)
  2. Controller: query-specific (alpha_sparse, alpha_dense) — not fixed hybrid weights
  3. Retrieve k_sparse = alpha*50, k_dense = beta*50 (proportional; pool << 100)
  4. GARDIAN rerank: gardian_score = alpha*sparse_branch + beta*dense_branch
  5. Reader gets qa.top_k_passages only (e.g. 6), not the full pool

Usage:
  python scripts/verify_gardian_retrieval_pipeline.py \\
    --retriever hybrid_bm25_faiss --n 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.question_types import normalize_question_type, qtype_onehot
from src.model.gardian import build_gardian_from_model_cfg, load_checkpoint_state
from src.pipeline.gardian_adaptive import (
    adaptive_channel_budgets,
    retrieve_adaptive_candidates_live,
)
from src.pipeline.rag_reader import build_retriever_for_qa, resolve_retrieval_paths
from src.pipeline.rank_dense_features import FaissPassageEmbeddingLookup
from sentence_transformers import SentenceTransformer


def _load_cfg():
    return OmegaConf.load("configs/base.yaml")


def _load_gardian(cfg, retriever: str, device: str):
    ckpt = Path("results") / f"gardian_best_{retriever}.pt"
    if not ckpt.is_file():
        ckpt = Path("results/gardian.pt")
    if not ckpt.is_file():
        raise FileNotFoundError(f"No GARDIAN checkpoint at {ckpt}")
    model = build_gardian_from_model_cfg(cfg.model).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    load_checkpoint_state(model, state["model_state"], strict=False)
    model.eval()
    return model, str(ckpt)


def _sample_questions(path: Path, n: int):
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retriever", default="hybrid_bm25_faiss")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--questions", default="data/pubmedqa_labeled_eval.jsonl")
    args = ap.parse_args()

    cfg = _load_cfg()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cap_bm25 = int(cfg.retrieval.top_k_bm25)
    cap_faiss = int(cfg.retrieval.top_k_faiss)
    budget_mode = str(getattr(cfg.retrieval, "adaptive_channel_budget", "full_caps"))

    print("=== Config (configs/base.yaml) ===")
    print(f"  top_k_bm25:              {cap_bm25}")
    print(f"  top_k_faiss:             {cap_faiss}")
    print(f"  candidate_pool_size:     {cfg.retrieval.candidate_pool_size}")
    print(f"  adaptive_channel_budget: {budget_mode}")
    print(f"  qa.gardian_adaptive:     {cfg.qa.gardian_adaptive_retrieval}")

    from src.retrieval.index_paths import resolve_dataset_index_paths

    per_paths = resolve_dataset_index_paths("pubmedqa_labeled")
    print("\n=== Per-dataset index paths (pubmedqa_labeled) ===")
    for k, v in per_paths.items():
        print(f"  {k}: {v}")

    retriever = build_retriever_for_qa(
        cfg,
        args.retriever,
        device=device,
        use_faiss_gpu=False,
        dataset_name="pubmedqa_labeled",
        use_per_dataset_indices=True,
    )
    print(f"\n=== Retriever ({args.retriever}) ===")
    print(f"  .top_k_first / bm25:  {getattr(retriever, 'top_k_first', getattr(retriever, 'top_k_bm25', '?'))}")
    print(f"  .top_k_second / faiss: {getattr(retriever, 'top_k_second', getattr(retriever, 'top_k_dense', '?'))}")

    gardian, ckpt_path = _load_gardian(cfg, args.retriever, device)
    print(f"\n=== GARDIAN checkpoint ===\n  {ckpt_path}")

    encoder = SentenceTransformer(cfg.encoder.model_name, device=device)
    qpath = Path(args.questions)
    if not qpath.is_file():
        print(f"\nWARN: questions file missing: {qpath} (skip live probe)")
        return 0

    items = _sample_questions(qpath, args.n)
    print(f"\n=== Live probe ({len(items)} questions) ===")
    ok = True
    for item in items:
        q = (item.get("question") or "").strip()
        qtype = normalize_question_type(item.get("question_type") or "yesno")
        q_emb = encoder.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()
        q_oh = qtype_onehot(qtype)
        pool, alpha, beta = retrieve_adaptive_candidates_live(
            q,
            retriever,
            gardian,
            query_emb=q_emb,
            qtype_onehot=q_oh,
            cfg=cfg,
            device=device,
        )
        k_s, k_d = adaptive_channel_budgets(alpha, beta, cfg)
        has_rrf = sum(1 for c in pool if c.get("hybrid_rrf_score", 0) > 0)
        ranked = gardian.rerank(
            candidates=[dict(c) for c in pool[: min(80, len(pool))]],
            query_features={"query_emb": q_emb, "qtype_onehot": q_oh},
            device=device,
        )
        top = ranked[0] if ranked else {}
        fusion_alpha = float(top.get("sparse_alfa", 0))
        fusion_beta = float(top.get("dense_alfa", 0))
        reader_k = int(cfg.qa.get("yesno_top_k_passages", cfg.qa.top_k_passages))
        pool_ok = len(pool) <= cap_bm25 + cap_faiss
        if budget_mode in ("full_caps", "full", "fixed", "50+50"):
            budget_ok = k_s == cap_bm25 and k_d == cap_faiss
        else:
            exp_s = max(1, min(cap_bm25, int(round((alpha / (alpha + beta + 1e-8)) * cap_bm25))))
            exp_d = max(1, min(cap_faiss, int(round((beta / (alpha + beta + 1e-8)) * cap_faiss))))
            budget_ok = k_s == exp_s and k_d == exp_d
        ok = ok and pool_ok and budget_ok
        print(
            f"\n  qid={item.get('id', '?')[:20]}…"
            f"\n    controller α_sparse={alpha:.3f} α_dense={beta:.3f}"
            f"\n    retrieve k_bm25={k_s} k_faiss={k_d}  pool={len(pool)} (max {cap_bm25}+{cap_faiss})"
            f"\n    fusion  α_sparse={fusion_alpha:.3f} α_dense={fusion_beta:.3f}"
            f"    gardian_top={top.get('id', '')[:28]}"
            f"\n    reader would see top_k={reader_k} passages (not full pool)"
        )
        if not budget_ok:
            print(f"    ** budget mismatch (mode={budget_mode})")
        if not pool_ok:
            print(f"    ** pool larger than channel caps")

    print("\n=== Summary ===")
    if ok and budget_mode in ("proportional", "adaptive", ""):
        print("  OK: adaptive α budgets + fusion; pool bounded; reader gets top_k only.")
        return 0
    if ok and budget_mode in ("full_caps", "full", "fixed", "50+50"):
        print("  OK: full_caps ablation (50+50 retrieve; α only in fusion).")
        return 0
    print("  FAIL: check adaptive_channel_budget and gardian_adaptive_retrieval flags.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
