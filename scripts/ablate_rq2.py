"""Deprecated. Paper RQ2 numbers come from ``scripts/10_paper_run.py``.

This file used pool-label nDCG and skipped queries missing from the embedding
cache, so its numbers are not comparable to Table 1. The paper driver scores
one forward per split, then derives NoSparse / NoDense / Uniform / Fixed-α /
GARDIAN / Oracle-α with the same qrels path as Table 1.

    .venv/bin/python scripts/10_paper_run.py --device cuda --parallel-workers 1 --cuda-devices 0
"""

from __future__ import annotations

import sys


def main() -> None:
    raise SystemExit(
        "scripts/ablate_rq2.py is not the paper ablation.\n"
        "It used a different nDCG path than Table 1 (pool labels, skipped "
        "missing embeddings) and would not survive review.\n\n"
        "Run the Table-1-identical driver instead:\n\n"
        "  .venv/bin/python scripts/10_paper_run.py "
        "--device cuda --parallel-workers 1 --cuda-devices 0\n\n"
        "That writes results/seeds/seed_*/ablation_paper.json and "
        "results/aggregated/ablation_rq2.json (5 seeds, Fixed-α control, "
        "oracle on learned branches)."
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
