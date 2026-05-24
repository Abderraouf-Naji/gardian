#!/usr/bin/env python3
"""Quick validator for text-only rank JSONL + query_emb cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from omegaconf import OmegaConf
from src.common.query_emb_cache import load_query_emb_cache


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--query-cache", default=None)
    p.add_argument("--cfg", default="configs/base.yaml")
    p.add_argument("--max-lines", type=int, default=5000)
    args = p.parse_args()

    cfg = OmegaConf.load(args.cfg)
    sparse_d = int(cfg.model.sparse_feat_dim)
    dense_d = int(cfg.model.dense_feat_dim)
    kg_d = int(cfg.model.kg_feat_dim)
    q_d = int(cfg.model.query_feat_dim)
    qtype_d = len(cfg.model.question_types)
    text_only = bool(cfg.model.text_only)

    n = 0
    has_kg = 0
    with Path(args.jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            n += 1
            if "kg_feats" in rec:
                has_kg += 1
            assert len(rec["sparse_feats"]) == sparse_d, rec.get("qid")
            assert len(rec["dense_feats"]) == dense_d, rec.get("qid")
            assert len(rec["qtype_onehot"]) == qtype_d, rec.get("qid")
            if "kg_coverage" in rec:
                assert float(rec["kg_coverage"]) == 0.0
            if args.max_lines and n >= args.max_lines:
                break

    print(f"lines_checked={n} has_kg_feats={has_kg} text_only_cfg={text_only}")
    if text_only and has_kg:
        raise SystemExit("FAIL: kg_feats found in records (expected text-only)")
    if text_only and has_kg == 0:
        print("OK: no kg_feats in sampled records")

    if args.query_cache:
        cache = load_query_emb_cache(args.query_cache, expected_dim=q_d)
        print(f"query_cache entries={len(cache)} dim={q_d}")


if __name__ == "__main__":
    main()
