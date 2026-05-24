#!/usr/bin/env python3
"""Export Hit@5/20/50 from evaluation JSON into a paper-ready summary."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

FOCUS = [
    "hybrid_bm25_faiss",
    "hybrid_bm25_medcpt",
    "hybrid_spladepp_faiss",
    "hybrid_spladepp_medcpt",
]
DATASETS = ["pubmedqa_labeled", "pubmedqa_artificial", "medmcqa"]
SYSTEMS = ["sparse", "dense", "hybrid", "rrf", "gardian"]
HIT_KEYS = ("hit@5", "hit@20", "hit@50")


def _load_eval(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    results = obj.get("results", {})
    if len(results) == 1:
        return next(iter(results.values()))
    return results


def export_hit_rates(eval_dir: Path) -> dict:
    out: dict = {"retrievers": {}}
    for ht in FOCUS:
        path = eval_dir / f"evaluation_{ht}.json"
        if not path.is_file():
            continue
        ds_block = _load_eval(path)
        out["retrievers"][ht] = {}
        for ds in DATASETS:
            systems = ds_block.get(ds, {})
            if not isinstance(systems, dict):
                continue
            out["retrievers"][ht][ds] = {}
            for sys in SYSTEMS:
                block = systems.get(sys, {})
                if not isinstance(block, dict) or not block:
                    # Fair sparse baseline may be stored under bm25/spladepp
                    if sys == "sparse":
                        block = systems.get("bm25") or systems.get("spladepp") or {}
                    elif sys == "dense":
                        block = systems.get("faiss") or systems.get("medcpt") or {}
                if not isinstance(block, dict):
                    continue
                hits = {k: float(block[k]) for k in HIT_KEYS if k in block}
                if hits:
                    out["retrievers"][ht][ds][sys] = hits
    return out


def print_table(summary: dict) -> None:
    print(f"\n{'Retriever':<28} {'Dataset':<22} {'System':<8} {'Hit@5':>8} {'Hit@20':>8} {'Hit@50':>8}")
    print("-" * 90)
    for ht, ds_block in summary.get("retrievers", {}).items():
        for ds, sys_block in ds_block.items():
            for sys, metrics in sys_block.items():
                print(
                    f"{ht:<28} {ds:<22} {sys:<8} "
                    f"{metrics.get('hit@5', 0):>8.4f} "
                    f"{metrics.get('hit@20', 0):>8.4f} "
                    f"{metrics.get('hit@50', 0):>8.4f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/retrieval_hit_rates.json"))
    args = parser.parse_args()

    summary = export_hit_rates(args.eval_dir)
    payload = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/export_retrieval_hit_rates.py",
            "metrics": list(HIT_KEYS),
        },
        "hit_rates": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_table(summary)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
