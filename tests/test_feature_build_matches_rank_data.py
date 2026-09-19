"""
The shared feature builder must reproduce the offline rank data exactly.

``src/features/build.py`` was extracted from ``scripts/03_generate_rank_data.py``
so the live QA path and the training data stop drifting apart. This test pins
that extraction against the real rank JSONL on disk: for every query it
rebuilds the within-pool dimensions from the stored raw channel scores and
compares them to the feature vectors the offline builder wrote.

Only the pool-normalised dimensions are checked. The lexical dims
(``idf_overlap``, ``jaccard``) and the embedding dims (``mean_abs_diff``,
``max_abs_diff``) need passage text and passage embeddings, which compact rank
JSONL does not store -- and they were never the drifting half: the live path
already computed those, and omitted every dimension checked here.
"""

from __future__ import annotations

import collections
import json
import pathlib

import pytest

from src.features.build import build_branch_features
from src.features.schema import DENSE_FEATURE_NAMES, SPARSE_FEATURE_NAMES

RANK_FILE = pathlib.Path(
    "data/hybrid_bm25_faiss/rank_data_hybrid_bm25_faiss_pubmedqa_labeled_eval.jsonl"
)
RETRIEVER = "hybrid_bm25_faiss"
MAX_QUERIES = 25

# Dimensions reconstructible from the raw channel scores alone.
SPARSE_POOL_DIMS = [
    SPARSE_FEATURE_NAMES.index(n)
    for n in (
        "sparse_score",
        "sparse_minmax",
        "sparse_z",
        "sparse_rank",
        "sparse_dispersion",
        "retrieved_by_sparse",
    )
]
DENSE_POOL_DIMS = [
    DENSE_FEATURE_NAMES.index(n)
    for n in (
        "dense_score",
        "dense_z",
        "dense_minmax",
        "dense_rank",
        "dense_dispersion",
        "retrieved_by_dense",
    )
]


def _load_pools(path: pathlib.Path, max_queries: int):
    """Group rank records into per-query pools, preserving file order."""
    pools: "collections.OrderedDict[str, list]" = collections.OrderedDict()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = rec["qid"]
            if qid not in pools and len(pools) >= max_queries:
                break
            pools.setdefault(qid, []).append(rec)
    return pools


@pytest.mark.skipif(not RANK_FILE.exists(), reason=f"{RANK_FILE} not generated")
def test_builder_reproduces_offline_pool_features():
    pools = _load_pools(RANK_FILE, MAX_QUERIES)
    assert pools, "no queries loaded from rank data"

    for qid, recs in pools.items():
        candidates = [
            {
                "id": r["pid"],
                "text": "",
                "bm25_score": float(r.get("bm25_score", 0.0)),
                "dense_score": float(r.get("dense_score", 0.0)),
            }
            for r in recs
        ]
        sparse_feats, dense_feats, pool_stats = build_branch_features(
            question=recs[0]["question"],
            candidates=candidates,
            retriever_type=RETRIEVER,
        )

        for i, rec in enumerate(recs):
            for d in SPARSE_POOL_DIMS:
                assert sparse_feats[i][d] == pytest.approx(
                    rec["sparse_feats"][d], rel=1e-5, abs=1e-6
                ), f"{qid} cand {i} sparse dim {d} ({SPARSE_FEATURE_NAMES[d]})"
            for d in DENSE_POOL_DIMS:
                assert dense_feats[i][d] == pytest.approx(
                    rec["dense_feats"][d], rel=1e-5, abs=1e-6
                ), f"{qid} cand {i} dense dim {d} ({DENSE_FEATURE_NAMES[d]})"

        stored = recs[0].get("pool_stats") or {}
        for key, value in stored.items():
            if key in pool_stats:
                assert pool_stats[key] == pytest.approx(
                    value, rel=1e-5, abs=1e-6
                ), f"{qid} pool_stats[{key}]"


@pytest.mark.skipif(not RANK_FILE.exists(), reason=f"{RANK_FILE} not generated")
def test_builder_emits_full_schema_width():
    pools = _load_pools(RANK_FILE, 1)
    recs = next(iter(pools.values()))
    candidates = [
        {
            "id": r["pid"],
            "text": "",
            "bm25_score": float(r.get("bm25_score", 0.0)),
            "dense_score": float(r.get("dense_score", 0.0)),
        }
        for r in recs
    ]
    sparse_feats, dense_feats, _ = build_branch_features(
        question=recs[0]["question"],
        candidates=candidates,
        retriever_type=RETRIEVER,
    )
    assert all(len(f) == len(SPARSE_FEATURE_NAMES) for f in sparse_feats)
    assert all(len(f) == len(DENSE_FEATURE_NAMES) for f in dense_feats)


def test_empty_pool_is_handled():
    sparse_feats, dense_feats, pool_stats = build_branch_features(
        question="q", candidates=[], retriever_type=RETRIEVER
    )
    assert sparse_feats == [] and dense_feats == [] and pool_stats == {}
