"""
Full relevance judgments (qrels), loaded independently of the candidate pool.

Why this module exists
----------------------
Evaluation used to build its relevant set from the *pool*::

    relevant_ids = [c["pid"] for c in qdata["candidates"] if c["label"] == 1]

That makes every denominator pool-relative: recall@100 is 1.0 by construction
once the pool holds <=100 candidates, and nDCG normalises against only the
positives that retrieval already found. On TREC-COVID the gap is enormous --
493.5 judged-relevant passages per topic against a ~92-candidate pool -- so a
pool-relative score cannot be compared to any published number.

Qrels here are the *full* judgment set for a query, read from the source
evaluation JSONL (``data/trec_covid_test.jsonl`` and friends), which stores
``gold_passage_ids`` plus a ``relevance_grades`` map of pid -> grade.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, Optional

from loguru import logger

# qid -> {pid: grade}; grades are ints >= 1 (unjudged/irrelevant are absent).
Qrels = Dict[str, Dict[str, int]]


def load_qrels(path: str | Path) -> Qrels:
    """
    Read full graded qrels from an evaluation JSONL.

    Each record contributes one query. ``relevance_grades`` is authoritative
    when present; otherwise every id in ``gold_passage_ids`` is given grade 1,
    which reproduces binary judgments for the QA-style collections that have
    no graded labels.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Qrels source not found: {path}")

    qrels: Qrels = {}
    for line in path.open("r", encoding="utf-8", errors="ignore"):
        if not line.strip():
            continue
        rec = json.loads(line)
        qid = str(rec.get("id") or rec.get("qid") or "")
        if not qid:
            continue

        grades = rec.get("relevance_grades")
        if isinstance(grades, Mapping) and grades:
            judged = {str(pid): int(g) for pid, g in grades.items() if int(g) > 0}
        else:
            judged = {str(pid): 1 for pid in (rec.get("gold_passage_ids") or [])}

        if judged:
            qrels[qid] = judged

    if not qrels:
        raise ValueError(f"No judgments found in {path}")

    n_rel = sum(len(v) for v in qrels.values())
    n_graded = sum(1 for v in qrels.values() if any(g > 1 for g in v.values()))
    logger.info(
        f"Loaded qrels from {path.name}: {len(qrels):,} queries | "
        f"{n_rel / len(qrels):.1f} relevant/query | {n_graded} queries with graded labels"
    )
    return qrels


@lru_cache(maxsize=8)
def _load_qrels_cached(path: str) -> Qrels:
    return load_qrels(path)


# Rank JSONL qids are prefixed with the collection name ("trec_covid_1"), which
# is also the ``id`` of the source record, so no id translation is needed.
_EVAL_SOURCES = {
    "trec_covid": "data/trec_covid_test.jsonl",
    "medmcqa": "data/medmcqa_test.jsonl",
    "pubmedqa": "data/pubmedqa_labeled_eval.jsonl",
    # Densely-judged BEIR collections. Both carry graded (NFCorpus) or
    # multi-positive (SciFact) judgments, which the QA splits do not, so a
    # controller can actually be trained on them. Entries are inert until the
    # files exist -- see qrels_for_collection.
    "nfcorpus": ("data/nfcorpus_train.jsonl", "data/nfcorpus_dev.jsonl",
                 "data/nfcorpus_test.jsonl"),
    "scifact": ("data/scifact_train.jsonl", "data/scifact_test.jsonl"),
}


def qrels_for_eval_file(eval_path: str | Path) -> Qrels:
    """Load qrels for an evaluation JSONL, memoised by path."""
    return _load_qrels_cached(str(Path(eval_path)))


def qrels_for_collection(collection: str) -> Optional[Qrels]:
    """
    Load qrels by collection name ("trec_covid"), or ``None`` if no source file
    for that collection is present on disk.

    A collection may declare several sources -- one per split -- because a
    training collection's dev queries need judgments too, and qids are unique
    across splits. Missing files are skipped, so a collection can be declared
    before all of its splits have been downloaded.
    """
    src = _EVAL_SOURCES.get(collection)
    if not src:
        return None
    paths = (src,) if isinstance(src, str) else tuple(src)
    merged: Qrels = {}
    for path in paths:
        if Path(path).exists():
            merged.update(_load_qrels_cached(path))
    return merged or None


def infer_collection(qid: str) -> Optional[str]:
    """Collection a rank-JSONL qid belongs to, from its name prefix."""
    for name in _EVAL_SOURCES:
        if qid.startswith(name):
            return name
    return None


def qrels_for_qids(qids) -> Qrels:
    """
    Union of the qrels covering ``qids``, resolved per collection from the
    qid prefixes. Collections with no source file on disk are skipped, so a
    caller can detect partial coverage by comparing key counts.
    """
    out: Qrels = {}
    wanted = set(map(str, qids))
    for collection in {c for c in map(infer_collection, wanted) if c}:
        loaded = qrels_for_collection(collection)
        if not loaded:
            logger.warning(f"No qrels source on disk for collection '{collection}'")
            continue
        out.update({q: v for q, v in loaded.items() if q in wanted})
    return out
