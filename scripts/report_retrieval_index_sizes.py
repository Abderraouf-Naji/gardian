#!/usr/bin/env python3
"""Report on-disk index sizes per hybrid family and benchmark corpus."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.index_stats import format_index_size_table, report_all_index_sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/retrieval_index_sizes.json"),
    )
    args = parser.parse_args()

    report = report_all_index_sizes()
    payload = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/report_retrieval_index_sizes.py",
            "note": "Disk footprint of per-dataset sparse + dense index directories.",
        },
        "index_sizes": report,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(format_index_size_table(report))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
