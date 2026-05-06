"""
Script 01 – Build BM25 and FAISS indices from MULTIPLE corpora.

Supports two modes:
1) Per-corpus indexing:
   - Builds separate BM25 and FAISS indices for each corpus.
   - Useful for routed / source-specific retrieval experiments.

2) Unified indexing:
   - Reuses an existing unified corpus if available.
   - Otherwise merges all corpora into one unified corpus file.
   - Builds a single BM25 and FAISS index over the combined corpus.
   - Useful for GARDIAN's default unified retrieval baseline.

Expected inputs
---------------
data/corpus_pubmedqa_artificial.jsonl
data/corpus_pubmedqa_labeled.jsonl
data/corpus_medmcqa.jsonl
data/corpus_medrag_pubmed.jsonl   

Expected outputs
----------------
Per corpus:
data/indices/<corpus_name>/bm25/
data/indices/<corpus_name>/faiss.index
data/indices/<corpus_name>/faiss_meta.jsonl

Unified:
data/indices/bm25/unified/
data/indices/faiss/unified/faiss.index
data/indices/faiss/unified/faiss_meta.jsonl
data/indices/unified/corpus_unified.jsonl

If unified BM25 exists but FAISS is missing, run:
  python scripts/01_Build_bm25_faiss_indices.py --unified-faiss-only
"""

import argparse
import json
import pathlib
import shutil
import sys
from typing import Dict, Iterable, Tuple

sys.path.insert(0, ".")

from loguru import logger
from omegaconf import OmegaConf

from src.retrieval.bm25 import build_bm25_index
from src.retrieval.dense import build_faiss_index


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CORPORA: Dict[str, str] = {
    "pubmedqa_artificial": "data/corpus_pubmedqa_artificial.jsonl",
    "pubmedqa_labeled": "data/corpus_pubmedqa_labeled.jsonl",
    "medmcqa": "data/corpus_medmcqa.jsonl",
    "medrag_pubmed": "data/corpus_medrag_pubmed.jsonl",  
}

UNIFIED_NAME = "unified"
UNIFIED_CORPUS_PATH = pathlib.Path("data/indices/unified/corpus_unified.jsonl")


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate_corpus(corpus_path: pathlib.Path, sample_lines: int = 5) -> int:
    """
    Validate a corpus JSONL file and return the number of passages.
    Checks required fields on the first few lines and ensures file is non-empty.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    n = 0
    with corpus_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            n += 1
            if n <= sample_lines:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {n} in {corpus_path}: {e}") from e

                required = {"id", "text", "title", "source"}
                missing = required - set(obj.keys())
                if missing:
                    raise ValueError(
                        f"Invalid corpus record at line {n} in {corpus_path}: "
                        f"missing fields {sorted(missing)}"
                    )

                if not isinstance(obj["id"], str) or not obj["id"].strip():
                    raise ValueError(f"Invalid 'id' at line {n} in {corpus_path}")

                if not isinstance(obj["text"], str) or not obj["text"].strip():
                    raise ValueError(f"Invalid 'text' at line {n} in {corpus_path}")

                if not isinstance(obj["title"], str):
                    raise ValueError(f"Invalid 'title' at line {n} in {corpus_path}")

                if not isinstance(obj["source"], str) or not obj["source"].strip():
                    raise ValueError(f"Invalid 'source' at line {n} in {corpus_path}")

    if n == 0:
        raise ValueError(f"Corpus file is empty: {corpus_path}")

    logger.info(f"Validated corpus: {corpus_path} ({n:,} passages)")
    return n


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

def get_index_paths(base_dir: pathlib.Path, corpus_name: str) -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """
    Return (bm25_dir, faiss_index_path, faiss_meta_path) for a corpus name.
    """
    bm25_dir = base_dir / "bm25" / corpus_name
    faiss_dir = base_dir / "faiss" / corpus_name
    faiss_index_path = faiss_dir / "faiss.index"
    faiss_meta_path = faiss_dir / "faiss_meta.jsonl"
    return bm25_dir, faiss_index_path, faiss_meta_path


def ensure_output_dirs(indices_dir: pathlib.Path, corpus_name: str) -> None:
    """
    Ensure output directories exist for a given corpus.
    """
    bm25_dir, faiss_index_path, faiss_meta_path = get_index_paths(indices_dir, corpus_name)
    bm25_dir.mkdir(parents=True, exist_ok=True)
    faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss_meta_path.parent.mkdir(parents=True, exist_ok=True)


def warn_existing_outputs(indices_dir: pathlib.Path, corpus_name: str) -> None:
    """
    Warn if index artifacts already exist.
    """
    bm25_dir, faiss_index_path, faiss_meta_path = get_index_paths(indices_dir, corpus_name)

    already_exists = (
        (bm25_dir.exists() and any(bm25_dir.iterdir()))
        or faiss_index_path.exists()
        or faiss_meta_path.exists()
    )

    if already_exists:
        logger.warning(
            f"Existing index artifacts detected for '{corpus_name}'. "
            "If corpus/model settings changed, delete old outputs before rebuilding."
        )


def _move_if_missing(src: pathlib.Path, dst: pathlib.Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        logger.warning(f"[organize] target exists, skipping move: {dst}")
        return
    shutil.move(src.as_posix(), dst.as_posix())
    logger.info(f"[organize] moved {src} -> {dst}")


def organize_existing_index_artifacts(indices_dir: pathlib.Path, corpus_names: Iterable[str]) -> None:
    """
    Move legacy per-corpus layout into normalized typed layout.
    Legacy:
      data/indices/<corpus>/bm25
      data/indices/<corpus>/faiss.index
      data/indices/<corpus>/faiss_meta.jsonl
    New:
      data/indices/bm25/<corpus>/
      data/indices/faiss/<corpus>/faiss.index
      data/indices/faiss/<corpus>/faiss_meta.jsonl
    """
    indices_dir.mkdir(parents=True, exist_ok=True)
    (indices_dir / "bm25").mkdir(parents=True, exist_ok=True)
    (indices_dir / "faiss").mkdir(parents=True, exist_ok=True)

    for corpus_name in corpus_names:
        legacy_dir = indices_dir / corpus_name
        _move_if_missing(legacy_dir / "bm25", indices_dir / "bm25" / corpus_name)
        _move_if_missing(legacy_dir / "faiss.index", indices_dir / "faiss" / corpus_name / "faiss.index")
        _move_if_missing(legacy_dir / "faiss_meta.jsonl", indices_dir / "faiss" / corpus_name / "faiss_meta.jsonl")

        if legacy_dir.exists():
            try:
                next(legacy_dir.iterdir())
            except StopIteration:
                legacy_dir.rmdir()
                logger.info(f"[organize] removed empty legacy dir: {legacy_dir}")


# -----------------------------------------------------------------------------
# Unified corpus builder
# -----------------------------------------------------------------------------

def iter_jsonl(path: pathlib.Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_unified_corpus(corpora: Dict[str, pathlib.Path], output_path: pathlib.Path) -> int:
    """
    Merge multiple corpora into one unified JSONL.
    Deduplicates by passage id.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = set()
    written = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for corpus_name, corpus_path in corpora.items():
            logger.info(f"Merging corpus '{corpus_name}' from {corpus_path}")

            for obj in iter_jsonl(corpus_path):
                pid = obj["id"]
                if pid in seen_ids:
                    continue

                seen_ids.add(pid)
                out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1

    logger.success(f"Unified corpus written to {output_path} ({written:,} unique passages)")
    return written


# -----------------------------------------------------------------------------
# Index builders
# -----------------------------------------------------------------------------

def build_indices_for_corpus(
    corpus_name: str,
    corpus_path: pathlib.Path,
    indices_dir: pathlib.Path,
    encoder_name: str,
    batch_size: int,
) -> None:
    """
    Build BM25 and FAISS indices for one corpus.
    """
    bm25_dir, faiss_index_path, faiss_meta_path = get_index_paths(indices_dir, corpus_name)

    logger.info("=" * 72)
    logger.info(f"Building indices for corpus: {corpus_name}")
    logger.info("=" * 72)
    logger.info(f"Corpus JSONL : {corpus_path}")
    logger.info(f"BM25 dir     : {bm25_dir}")
    logger.info(f"FAISS path   : {faiss_index_path}")
    logger.info(f"Meta path    : {faiss_meta_path}")
    logger.info(f"Encoder      : {encoder_name}")
    logger.info(f"Batch size   : {batch_size}")

    logger.info("-" * 72)
    logger.info("Building BM25 index")
    build_bm25_index(
        corpus_jsonl=str(corpus_path),
        index_dir=str(bm25_dir),
    )

    logger.info("-" * 72)
    logger.info("Building FAISS dense index")
    build_faiss_index(
        corpus_jsonl=str(corpus_path),
        faiss_path=str(faiss_index_path),
        meta_path=str(faiss_meta_path),
        encoder_name=encoder_name,
        batch_size=batch_size,
    )

    logger.success(f"Finished building indices for corpus: {corpus_name}")


def build_unified_faiss_only(cfg) -> None:
    """
    Encode ``data/indices/unified/corpus_unified.jsonl`` and write only
    ``data/indices/faiss/unified/faiss.index`` (+ meta). Skips BM25 and
    per-corpus indices. Use when unified BM25 exists but FAISS is missing.
    """
    indices_dir = pathlib.Path("data/indices")
    if not UNIFIED_CORPUS_PATH.is_file():
        raise FileNotFoundError(
            f"Unified corpus not found: {UNIFIED_CORPUS_PATH}\n"
            "Run this script without --unified-faiss-only first, or merge corpora into that path."
        )
    validate_corpus(UNIFIED_CORPUS_PATH)
    ensure_output_dirs(indices_dir, UNIFIED_NAME)
    _, faiss_index_path, faiss_meta_path = get_index_paths(indices_dir, UNIFIED_NAME)
    encoder_name = str(cfg.encoder.model_name)
    batch_size = int(cfg.encoder.batch_size)
    logger.info("=" * 72)
    logger.info("Unified FAISS only (skipping BM25 and per-corpus builds)")
    logger.info("=" * 72)
    logger.info(f"Corpus JSONL : {UNIFIED_CORPUS_PATH}")
    logger.info(f"FAISS path   : {faiss_index_path}")
    logger.info(f"Meta path    : {faiss_meta_path}")
    build_faiss_index(
        corpus_jsonl=str(UNIFIED_CORPUS_PATH),
        faiss_path=str(faiss_index_path),
        meta_path=str(faiss_meta_path),
        encoder_name=encoder_name,
        batch_size=batch_size,
    )
    logger.success(f"Unified FAISS index ready: {faiss_index_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 + FAISS indices (per corpus and unified).")
    parser.add_argument(
        "--unified-faiss-only",
        action="store_true",
        help="Only build FAISS for data/indices/unified/corpus_unified.jsonl (matches hybrid in 08_ask_gardian).",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load("configs/base.yaml")
    indices_dir = pathlib.Path("data/indices")

    if args.unified_faiss_only:
        build_unified_faiss_only(cfg)
        return

    organize_existing_index_artifacts(indices_dir, list(CORPORA.keys()) + [UNIFIED_NAME])

    encoder_name = str(cfg.encoder.model_name)
    batch_size = int(cfg.encoder.batch_size)

    corpora_paths: Dict[str, pathlib.Path] = {
        name: pathlib.Path(path_str) for name, path_str in CORPORA.items()
    }

    # Keep only corpora that actually exist
    available_corpora: Dict[str, pathlib.Path] = {}
    missing_corpora = []

    for name, path in corpora_paths.items():
        if path.exists():
            available_corpora[name] = path
        else:
            missing_corpora.append((name, path))

    if missing_corpora:
        for name, path in missing_corpora:
            logger.warning(f"Skipping missing corpus '{name}': {path}")

    if not available_corpora:
        raise FileNotFoundError("No corpus files found. Run Script 00 first.")

    logger.info("=" * 72)
    logger.info("Validating input corpora")
    logger.info("=" * 72)

    counts: Dict[str, int] = {}
    for corpus_name, corpus_path in available_corpora.items():
        counts[corpus_name] = validate_corpus(corpus_path)

    logger.info("=" * 72)
    logger.info("Input corpus summary")
    logger.info("=" * 72)
    for corpus_name, count in counts.items():
        logger.info(f"{corpus_name:<22} {count:>12,} passages")

    # Build per-corpus indices
    logger.info("=" * 72)
    logger.info("Building per-corpus indices")
    logger.info("=" * 72)

    for corpus_name, corpus_path in available_corpora.items():
        ensure_output_dirs(indices_dir, corpus_name)
        warn_existing_outputs(indices_dir, corpus_name)
        build_indices_for_corpus(
            corpus_name=corpus_name,
            corpus_path=corpus_path,
            indices_dir=indices_dir,
            encoder_name=encoder_name,
            batch_size=batch_size,
        )

    # Unified corpus
    logger.info("=" * 72)
    logger.info("Building unified corpus and unified indices")
    logger.info("=" * 72)

    ensure_output_dirs(indices_dir, UNIFIED_NAME)
    warn_existing_outputs(indices_dir, UNIFIED_NAME)

    if UNIFIED_CORPUS_PATH.exists():
        logger.info(f"Reusing existing unified corpus: {UNIFIED_CORPUS_PATH}")
    else:
        build_unified_corpus(
            corpora=available_corpora,
            output_path=UNIFIED_CORPUS_PATH,
        )

    validate_corpus(UNIFIED_CORPUS_PATH)

    build_indices_for_corpus(
        corpus_name=UNIFIED_NAME,
        corpus_path=UNIFIED_CORPUS_PATH,
        indices_dir=indices_dir,
        encoder_name=encoder_name,
        batch_size=batch_size,
    )

    logger.success("=" * 72)
    logger.success("All per-corpus and unified indices built successfully.")
    logger.success("=" * 72)


if __name__ == "__main__":
    main()