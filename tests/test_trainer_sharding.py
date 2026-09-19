"""
Multi-worker sharding must keep each query's candidate pool intact.

Rank JSONL rows are query-contiguous. Sharding by line (the previous
behaviour) splits one query across workers, so positives get paired against a
fraction of their true negatives and whole groups vanish. These tests pin the
group-level sharding contract.
"""

from __future__ import annotations

import json

import pytest

from src.training.trainer import StreamingRankDataset

QUERY_FEAT_DIM = 4


def _record(qid: str, pid: str, label: int) -> dict:
    return {
        "qid": qid,
        "pid": pid,
        "label": label,
        # The full stored schema (8 + 8); the model consumes a subset of it.
        "sparse_feats": [1.0, 2.0, 3.0, 0.4, 0.5, 0.6, 0.7, 1.0],
        "dense_feats": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0],
        "query_emb": [0.5] * QUERY_FEAT_DIM,
    }


@pytest.fixture
def rank_jsonl(tmp_path):
    """Four queries, each with one positive followed by three negatives."""
    path = tmp_path / "rank.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for q in range(4):
            f.write(json.dumps(_record(f"q{q}", f"q{q}_pos", 1)) + "\n")
            for n in range(3):
                f.write(json.dumps(_record(f"q{q}", f"q{q}_neg{n}", 0)) + "\n")
    return path


def _dataset(path, **kw):
    return StreamingRankDataset(
        str(path),
        num_negatives=3,
        query_feat_dim=QUERY_FEAT_DIM,
        precompute_query_emb=False,
        **kw,
    )


def test_single_worker_emits_every_query(rank_jsonl):
    pairs = list(_dataset(rank_jsonl))
    # 4 queries x 1 positive x 3 negatives
    assert len(pairs) == 12


def test_group_sharding_partitions_without_loss(rank_jsonl, monkeypatch):
    """Union over workers == single-worker output; no query is split or dropped."""
    import src.training.trainer as trainer_mod

    expected = len(list(_dataset(rank_jsonl)))

    class _FakeWorkerInfo:
        def __init__(self, wid, n):
            self.id = wid
            self.num_workers = n

    total = 0
    for wid in range(2):
        monkeypatch.setattr(
            trainer_mod, "get_worker_info", lambda w=wid: _FakeWorkerInfo(w, 2)
        )
        total += len(list(_dataset(rank_jsonl)))

    assert total == expected, (
        "sharded workers must together emit exactly the single-worker pair set; "
        "line-level sharding loses pairs by splitting query groups"
    )


def test_each_worker_gets_whole_queries_only(rank_jsonl, monkeypatch):
    """A worker either sees all of a query's pairs or none of them."""
    import src.training.trainer as trainer_mod

    class _FakeWorkerInfo:
        def __init__(self, wid, n):
            self.id = wid
            self.num_workers = n

    for wid in range(2):
        monkeypatch.setattr(
            trainer_mod, "get_worker_info", lambda w=wid: _FakeWorkerInfo(w, 2)
        )
        n_pairs = len(list(_dataset(rank_jsonl)))
        # Each whole query contributes exactly 3 pairs (1 positive x 3 negatives).
        assert n_pairs % 3 == 0, f"worker {wid} received a partial query group"


def test_negatives_are_sampled_from_the_full_pool(rank_jsonl):
    """Every negative of a query must be reachable, not just a per-worker slice."""
    ds = _dataset(rank_jsonl)
    groups = []
    with open(rank_jsonl, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    groups = [r for r in recs if r["qid"] == "q0"]
    negatives = [r for r in groups if r["label"] == 0]
    sampled = ds._sample_negatives(negatives, 3)
    assert len({r["pid"] for r in sampled}) == 3
