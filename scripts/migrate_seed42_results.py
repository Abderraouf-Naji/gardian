"""
One-shot migration of the pre-multi-seed artifacts into the per-seed layout.

Everything currently at the top level of ``results/`` was produced by the
original single-seed run with ``seed: 42``. This script *copies* those files to
``results/seeds/seed_42/`` so they become the seed-42 row of the multi-seed
tables, while leaving the originals in place so nothing that still reads the
flat paths breaks.

Idempotent: an already-migrated file is skipped, never overwritten.

    python scripts/migrate_seed42_results.py --dry-run
    python scripts/migrate_seed42_results.py
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys
from typing import List, Tuple

os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, ".")

from loguru import logger  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.common.seeds import seed_dir  # noqa: E402

LEGACY_SEED = 42

# Top-level result files produced by the single-seed run, as glob patterns
# relative to the results root.
FILE_PATTERNS = [
    "gardian_best_*.pt",
    "evaluation_*.json",
    "ablation_*.json",
    "training_summary_all_retrievers.json",
    "training_manifest.json",
    "rank_data_manifest.json",
]

# Whole directories to copy across.
DIR_NAMES = ["gardian_training"]


def plan(results_dir: pathlib.Path) -> Tuple[List[Tuple[pathlib.Path, pathlib.Path]], List[pathlib.Path]]:
    """Return (copies, skipped) where copies are (src, dst) pairs."""
    dest_root = seed_dir(results_dir, LEGACY_SEED)
    copies: List[Tuple[pathlib.Path, pathlib.Path]] = []
    skipped: List[pathlib.Path] = []

    for pattern in FILE_PATTERNS:
        for src in sorted(results_dir.glob(pattern)):
            if not src.is_file():
                continue
            dst = dest_root / src.name
            (skipped if dst.exists() else copies).append(dst if dst.exists() else (src, dst))

    for name in DIR_NAMES:
        src_dir = results_dir / name
        if not src_dir.is_dir():
            continue
        for src in sorted(src_dir.rglob("*")):
            if not src.is_file():
                continue
            dst = dest_root / src.relative_to(results_dir)
            (skipped if dst.exists() else copies).append(dst if dst.exists() else (src, dst))

    return copies, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without writing anything.",
    )
    args = parser.parse_args()

    if args.results_dir:
        results_dir = pathlib.Path(args.results_dir)
    else:
        cfg = OmegaConf.load("configs/base.yaml")
        results_dir = pathlib.Path(cfg.paths.results_dir)

    copies, skipped = plan(results_dir)

    if not copies and not skipped:
        logger.warning(f"Nothing to migrate under {results_dir}")
        return

    total_bytes = sum(src.stat().st_size for src, _ in copies)
    logger.info(
        f"Migrating {len(copies)} file(s) ({total_bytes / 1e6:.1f} MB) "
        f"-> {seed_dir(results_dir, LEGACY_SEED)}"
    )
    for src, dst in copies:
        logger.info(f"  {src}  ->  {dst}")
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    if skipped:
        logger.info(f"Skipped {len(skipped)} already-migrated file(s):")
        for dst in skipped[:10]:
            logger.info(f"  exists: {dst}")
        if len(skipped) > 10:
            logger.info(f"  ... and {len(skipped) - 10} more")

    if args.dry_run:
        logger.warning("--dry-run: no files were written.")
    else:
        logger.success(
            f"Seed-{LEGACY_SEED} artifacts are now the seed-{LEGACY_SEED} row of the "
            "multi-seed tables. Originals were left in place."
        )


if __name__ == "__main__":
    main()
