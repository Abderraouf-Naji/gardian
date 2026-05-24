#!/usr/bin/env python3
"""
Backfill cross_encoder_score on existing rank JSONL for baseline comparison.

Scores every (query, passage) pair in the hybrid candidate pool with a
pretrained cross-encoder, then writes an updated JSONL that
``scripts/05_evaluate_gardian.py`` can rank alongside sparse /comment/RRF/GARDIAN.

Usage:
  python scripts/13_backfill_cross_encoder_scores.py --retriever hybrid_bm25_faiss --dataset all
  python scripts/13_backfill_cross_encoder_scores.py --rank-jsonl data/hybrid_bm25_faiss/rank_data_....jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.passage_lookup import build_passage_text_lookup, passage_text_for_record
from src.common.rank_data_paths import normalize_retriever_name, resolve_rank_data_file
from src.retrieval.cross_encoder import CROSS_ENCODER_PRESETS, CrossEncoderScorer

DATASET_SPLITS = {
    "pubmedqa_labeled": ["eval"],
    "pubmedqa_artificial": ["test"],
    "medmcqa": ["test"],
}

DATASET_CORPUS_PATHS: Dict[str, List[Path]] = {
    "pubmedqa_labeled": [Path("data/corpus_pubmedqa_labeled.jsonl")],
    "pubmedqa_artificial": [Path("data/corpus_pubmedqa_artificial.jsonl")],
    "medmcqa": [Path("data/corpus_medmcqa.jsonl")],
}


def subsample_rank_path(
    rank_path: Path,
    max_queries: int,
    *,
    seed: int,
    rebuild: bool = False,
) -> Path:
    """Write ``<stem>_q{N}.jsonl`` with *max_queries* random queries (all pool rows each)."""
    import random

    out_path = rank_path.with_name(f"{rank_path.stem}_q{max_queries}.jsonl")
    if out_path.is_file() and not rebuild:
        logger.info(f"Using cached subsample: {out_path}")
        return out_path

    records = _load_records(rank_path)
    grouped = _group_by_query(records)
    qids = sorted(grouped.keys())
    if len(qids) > max_queries:
        rng = random.Random(seed)
        qids = sorted(rng.sample(qids, max_queries))
    out_records: List[Dict[str, Any]] = []
    for qid in qids:
        out_records.extend(grouped[qid])
    _write_jsonl(out_path, out_records)
    logger.success(
        f"Subsampled {len(qids):,} queries / {len(out_records):,} rows -> {out_path}"
    )
    return out_path


def _resolve_input_rank_path(
    in_path: Path,
    max_queries: int | None,
    *,
    seed: int,
    rebuild: bool,
) -> Path:
    """Resolve rank JSONL; build ``_qN`` subsample when requested or when path is missing."""
    import re

    if max_queries is None:
        if not in_path.is_file():
            raise FileNotFoundError(
                f"Rank JSONL not found: {in_path}\n"
                "Create it with scripts/03_generate_rank_data.py, or pass --max-queries N "
                "to subsample from the full rank file."
            )
        return in_path

    base = in_path
    m = re.match(r"^(.*)_q(\d+)$", in_path.stem)
    if m and not in_path.is_file():
        base = in_path.with_name(m.group(1) + ".jsonl")
        if int(m.group(2)) != max_queries:
            logger.warning(
                f"Path asks for q{m.group(2)} but --max-queries={max_queries}; using N={max_queries}"
            )
    elif m and in_path.is_file():
        return in_path

    if not base.is_file():
        raise FileNotFoundError(
            f"Base rank JSONL not found: {base}\n"
            f"Run: python scripts/03_generate_rank_data.py --retriever <family>"
        )
    return subsample_rank_path(base, max_queries, seed=seed, rebuild=rebuild)


def ce_rank_data_path(rank_path: Path, tag: str | None = None) -> Path:
    """Path for cross-encoder-scored rank JSONL (``_ce`` or ``_ce_{tag}``)."""
    stem = rank_path.stem
    suffix = f"_ce_{tag}" if tag else "_ce"
    return rank_path.with_name(stem + suffix + ".jsonl")


def _resolve_rank_paths(
    *,
    rank_jsonl: str | None,
    retriever: str | None,
    dataset: str,
    tag: str | None = None,
) -> List[Tuple[str, Path, Path]]:
    """Return (dataset_name, input_path, output_path) tuples to process."""
    if rank_jsonl:
        in_path = Path(rank_jsonl)
        out_path = ce_rank_data_path(in_path, tag)
        return [("", in_path, out_path)]

    if not retriever:
        raise ValueError("Provide --rank-jsonl or --retriever.")

    retriever = normalize_retriever_name(retriever)
    datasets = list(DATASET_SPLITS.keys()) if dataset == "all" else [dataset]
    out: List[Tuple[str, Path, Path]] = []
    for ds in datasets:
        for split in DATASET_SPLITS.get(ds, []):
            in_path = Path(resolve_rank_data_file(retriever, ds, split))
            if not in_path.is_file():
                logger.warning(f"Skipping missing rank file: {in_path}")
                continue
            out_path = ce_rank_data_path(in_path, tag)
            out.append((ds, in_path, out_path))
    return out


def _corpus_paths_for_dataset(dataset_name: str, cfg: Any) -> List[Path]:
    paths = list(DATASET_CORPUS_PATHS.get(dataset_name, []))
    unified = Path(str(getattr(cfg.paths, "corpus_jsonl", "") or ""))
    if unified.is_file():
        paths.append(unified)
    return paths


def _load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def _group_by_query(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[str(rec["qid"])].append(rec)
    return grouped


def backfill_cross_encoder_scores(
    records: List[Dict[str, Any]],
    *,
    scorer: CrossEncoderScorer,
    corpus_paths: Sequence[Path],
    overwrite: bool,
) -> Tuple[List[Dict[str, Any]], int]:
    """Return updated records and count of newly scored rows."""
    if not records:
        return records, 0

    lookup = build_passage_text_lookup(
        (rec.get("pid") for rec in records),
        corpus_paths,
    )
    missing_pids = {
        str(rec["pid"])
        for rec in records
        if not passage_text_for_record(rec, lookup).strip()
    }
    if missing_pids:
        sample = sorted(missing_pids)[:5]
        raise ValueError(
            f"Missing passage text for {len(missing_pids)} ids (e.g. {sample}). "
            "Ensure corpus JSONL paths are available."
        )

    grouped = _group_by_query(records)
    scored_rows = 0
    out_records: List[Dict[str, Any]] = []
    qids = sorted(grouped.keys())
    total_q = len(qids)
    log_every = max(1, total_q // 50)

    for qi, qid in enumerate(qids):
        if qi == 0 or (qi + 1) % log_every == 0 or qi + 1 == total_q:
            logger.info(
                f"  CE backfill query {qi + 1}/{total_q} "
                f"({100 * (qi + 1) / total_q:.1f}%, scored_rows={scored_rows:,})"
            )
        rows = grouped[qid]
        question = str(rows[0].get("question", "")).strip()
        if not question:
            raise ValueError(f"Query {qid!r} missing non-empty question text.")

        need_score_idx: List[int] = []
        for i, rec in enumerate(rows):
            if overwrite or rec.get("cross_encoder_score") is None:
                need_score_idx.append(i)

        if need_score_idx:
            queries = [question] * len(need_score_idx)
            passages = [
                passage_text_for_record(rows[i], lookup) for i in need_score_idx
            ]
            scores = scorer.score_pairs(queries, passages)
            for i, score in zip(need_score_idx, scores):
                rows[i] = dict(rows[i])
                rows[i]["cross_encoder_score"] = float(score)
                scored_rows += 1

        out_records.extend(rows)

    return out_records, scored_rows


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill cross_encoder_score on rank JSONL for baseline comparison."
    )
    parser.add_argument(
        "--rank-jsonl",
        type=str,
        default=None,
        help="Single rank JSONL to score (writes <stem>_ce.jsonl unless --output set).",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default=None,
        help="Hybrid retriever key (e.g. hybrid_bm25_faiss). Used with --dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", *DATASET_SPLITS.keys()],
        help="Dataset to backfill when using --retriever.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path (only valid with a single --rank-jsonl input).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input rank JSONL in place.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute cross_encoder_score even when the field already exists.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for cross-encoder inference (default: cuda when available).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override cfg.retrieval.cross_encoder_model (HuggingFace id or preset key).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "st", "monobert", "monot5"],
        default=None,
        help="Reranker backend (default: cfg.retrieval.cross_encoder_backend or auto).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override cfg.retrieval.cross_encoder_batch_size.",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 on GPU (use full FP32).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Suffix for output files: <stem>_ce_<tag>.jsonl (used by compare script).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        metavar="N",
        help="Subsample to N random queries first (writes <stem>_qN.jsonl).",
    )
    parser.add_argument(
        "--subsample-only",
        action="store_true",
        help="Only write the _qN.jsonl subsample; do not run cross-encoder scoring.",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/base.yaml",
        help="Config path (cross_encoder_model / batch_size / max_length).",
    )
    args = parser.parse_args()

    if args.output and not args.rank_jsonl:
        raise SystemExit("--output requires --rank-jsonl.")
    if args.in_place and args.output:
        raise SystemExit("Use either --in-place or --output, not both.")

    cfg = OmegaConf.load(args.cfg)
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    model_name = str(args.model or cfg.retrieval.cross_encoder_model)
    if model_name in CROSS_ENCODER_PRESETS:
        model_name = CROSS_ENCODER_PRESETS[model_name]
    backend = str(
        args.backend
        if args.backend is not None
        else getattr(cfg.retrieval, "cross_encoder_backend", "auto")
    )
    batch_size = int(
        args.batch_size if args.batch_size is not None else cfg.retrieval.cross_encoder_batch_size
    )
    max_length = int(cfg.retrieval.cross_encoder_max_length)
    fp16 = not args.no_fp16 and bool(getattr(cfg.retrieval, "cross_encoder_fp16", True))
    seed = int(getattr(cfg, "seed", 42))

    jobs = _resolve_rank_paths(
        rank_jsonl=args.rank_jsonl,
        retriever=args.retriever,
        dataset=args.dataset,
        tag=args.tag,
    )
    if not jobs:
        raise SystemExit("No rank JSONL files matched the requested filters.")

    if args.subsample_only and args.max_queries is None:
        raise SystemExit("--subsample-only requires --max-queries N.")

    scorer = None
    if not args.subsample_only:
        scorer = CrossEncoderScorer(
            model_name,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            backend=backend,
            fp16=fp16,
        )
        logger.info(
            f"Cross-encoder baseline scorer: model={model_name!r} backend={scorer.backend!r} "
            f"device={scorer.device!r} batch_size={batch_size} max_length={max_length} fp16={fp16}"
        )

    for dataset_name, in_path, default_out in jobs:
        in_path = _resolve_input_rank_path(
            in_path,
            args.max_queries,
            seed=seed,
            rebuild=bool(args.overwrite),
        )
        if args.subsample_only:
            continue

        if args.in_place:
            out_path = in_path
        elif args.output:
            out_path = Path(args.output)
        else:
            out_path = ce_rank_data_path(in_path, args.tag)

        logger.info(f"Scoring {in_path} -> {out_path}")
        records = _load_records(in_path)
        corpus_paths = _corpus_paths_for_dataset(dataset_name, cfg) if dataset_name else []
        if not corpus_paths:
            corpus_paths = [
                Path(str(getattr(cfg.paths, "corpus_jsonl", "") or "")),
            ]
        updated, n_scored = backfill_cross_encoder_scores(
            records,
            scorer=scorer,  # type: ignore[arg-type]
            corpus_paths=[p for p in corpus_paths if p.is_file()],
            overwrite=bool(args.overwrite),
        )
        _write_jsonl(out_path, updated)
        logger.success(
            f"Wrote {out_path} | records={len(updated):,} newly_scored={n_scored:,}"
        )


if __name__ == "__main__":
    main()
