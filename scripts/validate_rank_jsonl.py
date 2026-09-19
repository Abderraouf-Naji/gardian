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
    q_d = int(cfg.model.query_feat_dim)

    n = 0
    with Path(args.jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            n += 1
            assert len(rec["sparse_feats"]) == sparse_d, rec.get("qid")
            assert len(rec["dense_feats"]) == dense_d, rec.get("qid")
            if args.max_lines and n >= args.max_lines:
                break

    print(f"lines_checked={n}")

    if args.query_cache:
        cache = load_query_emb_cache(args.query_cache, expected_dim=q_d)
        print(f"query_cache entries={len(cache)} dim={q_d}")


if __name__ == "__main__":
    main()
