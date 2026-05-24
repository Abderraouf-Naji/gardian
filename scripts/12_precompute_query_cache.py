"""
Precompute qid -> query_emb pickle caches from rank JSONL (no KG required).

Example:
  python scripts/12_precompute_query_cache.py \\
    --rank-jsonl data/hybrid_bm25_faiss/rank_data_train_all.jsonl \\
    --out data/query_emb_cache_hybrid_bm25_faiss_train_all.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from loguru import logger
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.common.query_emb_cache import save_query_emb_cache


def _collect_qid_questions(rank_path: Path) -> dict[str, str]:
    qid_to_q: dict[str, str] = {}
    with rank_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = rec.get("qid")
            if qid is None:
                continue
            sqid = str(qid)
            if sqid in qid_to_q:
                continue
            emb = rec.get("query_emb")
            if isinstance(emb, list) and emb:
                continue
            question = rec.get("question")
            if isinstance(question, str) and question.strip():
                qid_to_q[sqid] = question
    return qid_to_q


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute query embedding cache from rank JSONL")
    p.add_argument("--cfg", default="configs/base.yaml")
    p.add_argument("--rank-jsonl", required=True, help="Rank JSONL path (train/dev/all)")
    p.add_argument("--out", required=True, help="Output .pkl path")
    p.add_argument(
        "--merge",
        nargs="*",
        default=[],
        help="Optional existing cache paths to merge before writing",
    )
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    cfg = OmegaConf.load(args.cfg)
    rank_path = Path(args.rank_jsonl)
    out_path = Path(args.out)
    feat_dim = int(cfg.model.query_feat_dim)

    merged: dict[str, list[float]] = {}
    for mp in args.merge:
        from src.common.query_emb_cache import load_query_emb_cache

        merged.update(load_query_emb_cache(mp, expected_dim=feat_dim))

    qid_to_q = _collect_qid_questions(rank_path)
    if not qid_to_q and not merged:
        logger.warning("No questions to encode and no merge inputs; nothing to write.")
        return

    if qid_to_q:
        encoder = SentenceTransformer(cfg.encoder.model_name, device=args.device)
        qids = list(qid_to_q.keys())
        questions = [qid_to_q[q] for q in qids]
        bs = max(1, int(args.batch_size))
        for i in tqdm(range(0, len(questions), bs), desc="Encode queries"):
            batch_q = questions[i : i + bs]
            batch_ids = qids[i : i + bs]
            embs = encoder.encode(
                batch_q,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=bs,
            )
            for qid, emb in zip(batch_ids, embs):
                merged[qid] = emb.tolist()

    save_query_emb_cache(out_path, merged, expected_dim=feat_dim)
    logger.success(f"Wrote {len(merged):,} query embeddings -> {out_path}")


if __name__ == "__main__":
    main()
