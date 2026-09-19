"""
Multi-seed run layout: seed lists, per-seed artifact paths, write guards.

Every trained component (GARDIAN, its ablations, and the learned baselines)
is run once per seed. Raw per-seed artifacts live under::

    results/seeds/seed_<SEED>/<relative path>

and are **never overwritten** by a later run unless the caller explicitly
opts in. Aggregation (``scripts/aggregate_seeds.py``) is a separate pass that
reads only that tree and writes mean/std summaries to ``results/aggregated/``,
which is the single source for every number in the paper.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from typing import Any, Iterable, List, Sequence

from loguru import logger

# Fixed seed set for all reported runs. Five seeds is the number the revision
# checklist asks for and gives a usable std on margins of ~0.1 pp. Seed 42 is
# retained deliberately: the original single-seed artifacts were migrated to
# results/seeds/seed_42 and remain a valid row of the aggregated table.
DEFAULT_SEEDS: tuple[int, ...] = (13, 21, 42, 87, 100)

# Sub-tree holding raw per-seed artifacts (never aggregated in place).
SEEDS_SUBDIR = "seeds"

# Sub-tree holding aggregated mean/std artifacts consumed by paper tables.
AGGREGATED_SUBDIR = "aggregated"


class SeedArtifactExistsError(FileExistsError):
    """Raised when a per-seed artifact would be overwritten."""


def parse_seeds(values: Iterable[Any]) -> List[int]:
    """
    Normalise a CLI seed specification into a sorted, de-duplicated int list.

    Accepts already-split tokens (``--seeds 13 21``) as well as comma-joined
    forms (``--seeds 13,21``), so both shell habits work.

    Raises
    ------
    ValueError
        If any token is not a base-10 integer, or the result is empty.
    """
    out: List[int] = []
    for value in values:
        for token in str(value).replace(",", " ").split():
            try:
                seed = int(token)
            except ValueError as exc:
                raise ValueError(f"Invalid seed {token!r}: seeds must be integers") from exc
            if seed not in out:
                out.append(seed)
    if not out:
        raise ValueError("No seeds provided")
    return sorted(out)


def add_seeds_argument(
    parser: argparse.ArgumentParser,
    *,
    default: Sequence[int] = DEFAULT_SEEDS,
    help_suffix: str = "",
) -> argparse.ArgumentParser:
    """Attach the standard ``--seeds`` flag to *parser*."""
    parser.add_argument(
        "--seeds",
        type=str,
        nargs="+",
        default=[str(s) for s in default],
        help=(
            "Seeds to run, space- or comma-separated "
            f"(default: {' '.join(str(s) for s in default)}). "
            "Each seed produces its own artifacts under results/seeds/seed_<S>/. "
            + help_suffix
        ).strip(),
    )
    parser.add_argument(
        "--overwrite-seed-artifacts",
        action="store_true",
        help=(
            "Allow re-running a seed to overwrite its existing artifacts. "
            "Off by default so completed per-seed results cannot be silently lost."
        ),
    )
    return parser


def seeds_root(results_dir: os.PathLike[str] | str) -> pathlib.Path:
    """Root of the raw per-seed tree: ``<results_dir>/seeds``."""
    return pathlib.Path(results_dir) / SEEDS_SUBDIR


def aggregated_root(results_dir: os.PathLike[str] | str) -> pathlib.Path:
    """Root of the aggregated tree: ``<results_dir>/aggregated``."""
    return pathlib.Path(results_dir) / AGGREGATED_SUBDIR


def seed_dir(results_dir: os.PathLike[str] | str, seed: int) -> pathlib.Path:
    """Directory holding every artifact for one seed."""
    return seeds_root(results_dir) / f"seed_{int(seed)}"


def seed_path(
    results_dir: os.PathLike[str] | str,
    seed: int,
    *parts: str,
) -> pathlib.Path:
    """
    Path to one per-seed artifact.

    ``seed_path("results", 13, "gardian_best_hybrid_bm25_faiss.pt")`` returns
    ``results/seeds/seed_13/gardian_best_hybrid_bm25_faiss.pt``.
    """
    return seed_dir(results_dir, seed).joinpath(*parts)


def resolve_seed_checkpoint(
    results_dir: os.PathLike[str] | str,
    retriever: str,
    seed: int,
) -> pathlib.Path:
    """
    Locate the GARDIAN checkpoint for one ``(retriever, seed)`` cell.

    Prefers the per-seed artifact written by ``scripts/04_train_gardian.py``
    (``results/seeds/seed_<S>/gardian_best_<retriever>.pt``) and falls back to
    the legacy flat path for checkpoints trained before the multi-seed
    refactor, warning loudly so a stale single-seed artifact is never silently
    reported as a multi-seed result.
    """
    per_seed = seed_path(results_dir, seed, f"gardian_best_{retriever}.pt")
    if per_seed.exists():
        return per_seed

    legacy = pathlib.Path(results_dir) / f"gardian_best_{retriever}.pt"
    if legacy.exists():
        logger.warning(
            f"No per-seed checkpoint at {per_seed}; falling back to legacy "
            f"{legacy}. This checkpoint is NOT seed-{seed} specific -- retrain "
            "with scripts/04_train_gardian.py --seeds before reporting."
        )
        return legacy

    raise FileNotFoundError(
        f"Checkpoint not found for retriever={retriever} seed={seed}. "
        f"Looked in {per_seed} and {legacy}."
    )


def guard_seed_artifact(path: os.PathLike[str] | str, *, overwrite: bool = False) -> pathlib.Path:
    """
    Ensure *path* can be written, creating parents.

    Raises
    ------
    SeedArtifactExistsError
        If the file already exists and ``overwrite`` is False. This is the
        mechanism that makes "never overwrite per-seed results" enforceable
        rather than a convention.
    """
    p = pathlib.Path(path)
    if p.exists() and not overwrite:
        raise SeedArtifactExistsError(
            f"Per-seed artifact already exists: {p}\n"
            "Refusing to overwrite completed seed results. Pass "
            "--overwrite-seed-artifacts to replace it, or delete the file first."
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_seed_json(
    path: os.PathLike[str] | str,
    payload: Any,
    *,
    overwrite: bool = False,
    indent: int = 2,
) -> pathlib.Path:
    """Write one per-seed JSON artifact under the no-overwrite guard."""
    p = guard_seed_artifact(path, overwrite=overwrite)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False, default=str)
    return p


def discover_seeds(results_dir: os.PathLike[str] | str) -> List[int]:
    """Return the seeds that currently have a directory under ``results/seeds``."""
    root = seeds_root(results_dir)
    if not root.is_dir():
        return []
    found: List[int] = []
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("seed_"):
            try:
                found.append(int(child.name[len("seed_") :]))
            except ValueError:
                continue
    return sorted(found)
