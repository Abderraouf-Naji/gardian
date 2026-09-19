"""
Rewrite query-embedding caches with float32 arrays instead of Python float lists.

A cache pickled as ``{qid: [float, ...]}`` costs ~24 bytes per dimension in
memory, because every element is a separate Python float object. ``pickle.load``
must materialise all of them before anything can convert them, so loading a
186k-query cache peaks at 6.2 GB even though the data is 0.53 GB as float32.
With several caches open at once that peak is what gets a training run
OOM-killed.

Storing ``{qid: np.ndarray(dtype=float32)}`` removes the transient entirely:
the same file then loads at roughly its on-disk size.

Existing readers need no change -- both ``load_query_emb_cache`` and
``load_query_emb_store`` already pass values through ``np.asarray``.

    python scripts/compact_query_emb_caches.py --dry-run
    python scripts/compact_query_emb_caches.py --glob 'data/query_emb_cache_hybrid_bm25_faiss_*.pkl'
"""

from __future__ import annotations

import argparse
import glob as globmod
import os
import pickle
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402


def already_compact(data: dict) -> bool:
    if not data:
        return True
    v = next(iter(data.values()))
    return isinstance(v, np.ndarray) and v.dtype == np.float32


def compact_one(path: Path, *, dry_run: bool = False) -> tuple[bool, str]:
    try:
        with path.open("rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        return False, f"unreadable ({exc})"
    if not isinstance(data, dict):
        return False, "not a dict"
    if already_compact(data):
        return False, "already float32"

    n = len(data)
    dim = int(np.asarray(next(iter(data.values()))).reshape(-1).size)
    out = {str(k): np.asarray(v, dtype=np.float32).reshape(-1) for k, v in data.items()}
    del data
    if dry_run:
        return True, f"would compact {n:,} x {dim}"

    fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return True, f"compacted {n:,} x {dim}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="data/query_emb_cache_*.pkl")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-recent-seconds", type=int, default=120,
                    help="Skip files modified this recently: another process may "
                         "still be writing them.")
    args = ap.parse_args()

    import time
    now = time.time()
    paths = sorted(Path(p) for p in globmod.glob(args.glob))
    if not paths:
        raise SystemExit(f"no files matched {args.glob!r}")

    changed = 0
    for p in paths:
        age = now - p.stat().st_mtime
        if age < args.skip_recent_seconds:
            logger.warning(f"  skip (modified {age:.0f}s ago, may be in use): {p.name}")
            continue
        before = p.stat().st_size
        did, msg = compact_one(p, dry_run=args.dry_run)
        after = p.stat().st_size
        if did:
            changed += 1
            logger.info(f"  {p.name}: {msg}  {before/1e6:.0f} MB -> {after/1e6:.0f} MB")
        else:
            logger.info(f"  {p.name}: {msg}")
    logger.success(f"{changed} file(s) {'would be ' if args.dry_run else ''}compacted")


if __name__ == "__main__":
    main()
