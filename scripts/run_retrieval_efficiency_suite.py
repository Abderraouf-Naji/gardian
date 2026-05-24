#!/usr/bin/env python3
"""
Run retrieval efficiency suite end-to-end:

  1. Hit@5/20/50  — backfill evaluation JSON + export summary
  2. Index size   — disk footprint per hybrid × dataset
  3. Latency      — live ms/query benchmark
  4. Figure       — retrieval_hit_latency_mrr (PNG + PDF)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(cmd: list[str], desc: str) -> None:
    print(f"\n{'=' * 72}\n{desc}\n{'=' * 72}")
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-hit-backfill",
        action="store_true",
        help="Skip hit@ backfill on evaluation JSON (use existing hit@ fields).",
    )
    parser.add_argument(
        "--with-gardian-hit",
        action="store_true",
        help="Re-score GARDIAN for hit@ in one process (slow; GPU). Do not run backfill in parallel.",
    )
    parser.add_argument("--n-queries", type=int, default=50, help="Latency benchmark queries per dataset.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Skip live latency benchmark.",
    )
    parser.add_argument(
        "--retriever",
        default="all",
        help="Latency benchmark retriever (default: all four hybrids).",
    )
    args = parser.parse_args()

    eval_files = sorted((ROOT / "results").glob("evaluation_hybrid_*.json"))
    if not eval_files:
        print("No results/evaluation_hybrid_*.json found. Run 05_evaluate_gardian.py --per-retriever-json first.")

    if not args.skip_hit_backfill and eval_files:
        backfill_cmd = [PY, "scripts/backfill_hit_metrics.py", *[str(p) for p in eval_files]]
        if args.with_gardian_hit:
            backfill_cmd.extend(["--with-gardian", "--device", "cuda"])
        _run(
            backfill_cmd,
            "1/4 Hit@ backfill (single job; pass --with-gardian-hit only when GARDIAN hit@ is missing)",
        )

    _run([PY, "scripts/export_retrieval_hit_rates.py"], "1/4 Export Hit@ summary")

    _run([PY, "scripts/report_retrieval_index_sizes.py"], "2/4 Index disk sizes")

    if not args.skip_latency:
        _run(
            [
                PY,
                "scripts/benchmark_retrieval_efficiency.py",
                "--retriever",
                args.retriever,
                "--datasets",
                "pubmedqa_labeled,pubmedqa_artificial,medmcqa",
                "--n-queries",
                str(args.n_queries),
                "--warmup",
                str(args.warmup),
                "--out",
                "results/retrieval_efficiency.json",
            ],
            f"3/4 Live latency ({args.n_queries} queries/dataset)",
        )

    _run([PY, "scripts/plot_retrieval_efficiency.py"], "4/4 Combined Hit@ / latency / MRR figure")

    print("\nDone. Outputs:")
    print("  results/retrieval_hit_rates.json")
    print("  results/retrieval_index_sizes.json")
    print("  results/retrieval_efficiency.json")
    print("  results/figures/retrieval_hit_latency_mrr.png")
    print("  results/figures/retrieval_hit_latency_mrr.pdf")


if __name__ == "__main__":
    main()
