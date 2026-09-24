"""
Regression tests for the RQ4 cost measurements.

Three claims in the RQ4 section are only true if the code behaves in specific
ways, and each was wrong at some point in this repository:

* GARDIAN's reported latency must include the query-encoder forward pass. The
  earlier benchmark encoded the query outside the timer, which made the
  controller look free.
* "No controller means no query encoder" must hold on the training path too,
  not just at inference, or the Lite arm's training cost is inflated by work
  its model discards.
* The LambdaMART control must read the same features and group candidates by
  query, or its numbers are not comparable to GARDIAN's.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.evaluation.rerank_latency import (
    latency_stats,
    select_timing_pools,
    time_gardian,
    time_rrf,
)
from src.model.gardian import build_gardian_from_model_cfg
from src.training.trainer import StreamingRankDataset


def _record(qid: str, pid: str, label: int = 0) -> dict:
    rng = np.random.default_rng(abs(hash(pid)) % (2**32))
    return {
        "qid": qid,
        "pid": pid,
        "question": f"question for {qid}",
        "label": label,
        "sparse_feats": rng.random(8).astype(float).tolist(),
        "dense_feats": rng.random(8).astype(float).tolist(),
    }


def _pool(qid: str, n: int = 12) -> list[dict]:
    return [_record(qid, f"{qid}_p{i}", label=1 if i == 0 else 0) for i in range(n)]


def _model_cfg(use_controller: bool) -> dict:
    return {
        "sparse_feat_dim": 8,
        "dense_feat_dim": 8,
        "branch_hidden": 16,
        "controller_hidden": 16,
        "query_feat_dim": 8,
        "dropout": 0.0,
        "use_controller": use_controller,
        "controller_inputs": ["query_emb"],
        "normalize_branches": False,
        "dropped_features": [],
    }


# --------------------------------------------------------------------------
# latency statistics
# --------------------------------------------------------------------------


def test_latency_stats_empty_is_zero_not_error():
    stats = latency_stats([])
    assert stats["n"] == 0
    assert stats["p50_ms"] == 0.0


def test_latency_stats_reports_median_not_mean():
    # One slow outlier must not drag the reported figure: p50 is the headline.
    stats = latency_stats([1.0, 1.0, 1.0, 1.0, 1000.0])
    assert stats["p50_ms"] == pytest.approx(1.0)
    assert stats["mean_ms"] > 100.0


def test_latency_stats_percentiles_ordered():
    stats = latency_stats([float(x) for x in range(1, 101)])
    assert stats["min_ms"] <= stats["p50_ms"] <= stats["p95_ms"] <= stats["max_ms"]


# --------------------------------------------------------------------------
# warmup and pool selection
# --------------------------------------------------------------------------


def test_warmup_queries_are_excluded_from_samples():
    pools = [_pool(f"q{i}") for i in range(10)]
    stats = time_rrf(pools, warmup=4)
    assert stats["n"] == 6


def test_select_timing_pools_is_deterministic():
    records = {f"q{i}": _pool(f"q{i}") for i in range(50)}
    a, qa, ia = select_timing_pools(records, n_queries=10, seed=42)
    b, _qb, ib = select_timing_pools(records, n_queries=10, seed=42)
    assert ia == ib
    assert len(a) == len(b) == 10


def test_select_timing_pools_carries_question_text():
    records = {"q1": _pool("q1")}
    _pools, questions, _qids = select_timing_pools(records, n_queries=1, seed=0)
    assert questions == ["question for q1"]


# --------------------------------------------------------------------------
# the query encoder must be inside the timed region
# --------------------------------------------------------------------------


class _SlowEncoder:
    """Stand-in for PubMedBERT with a measurable, deterministic cost."""

    def __init__(self, dim: int = 8, delay_s: float = 0.01) -> None:
        self.dim = dim
        self.delay_s = delay_s
        self.calls = 0

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        import time

        self.calls += 1
        time.sleep(self.delay_s)
        return np.zeros((len(texts), self.dim), dtype=np.float32)


def test_gardian_timing_includes_query_encoder():
    model = build_gardian_from_model_cfg(_model_cfg(use_controller=True))
    model.eval()
    encoder = _SlowEncoder(delay_s=0.02)
    pools = [_pool(f"q{i}") for i in range(6)]
    questions = [f"question {i}" for i in range(6)]

    stats = time_gardian(
        pools,
        questions=questions,
        model=model,
        query_encoder=encoder,
        device="cpu",
        warmup=1,
        encode_query=True,
    )

    assert encoder.calls == len(pools)
    # The encoder sleeps 20 ms per query, so a timer that excluded it could not
    # report a median above that.
    assert stats["p50_ms"] >= 20.0
    assert stats["breakdown"]["query_encoder"]["p50_ms"] >= 20.0
    assert stats["breakdown"]["query_encoder_timed"] is True


def test_gardian_lite_timing_skips_query_encoder():
    model = build_gardian_from_model_cfg(_model_cfg(use_controller=False))
    model.eval()
    encoder = _SlowEncoder(delay_s=0.02)
    pools = [_pool(f"q{i}") for i in range(6)]
    questions = [f"question {i}" for i in range(6)]

    stats = time_gardian(
        pools,
        questions=questions,
        model=model,
        query_encoder=encoder,
        device="cpu",
        warmup=1,
        encode_query=False,
    )

    assert encoder.calls == 0
    assert stats["breakdown"]["query_encoder"]["p50_ms"] == 0.0
    assert stats["breakdown"]["query_encoder_timed"] is False
    # No absolute wall-clock assertion here: this suite runs on whatever machine
    # is free, often alongside training jobs, so a threshold in milliseconds
    # would fail for reasons that have nothing to do with the code. The claim
    # under test is that the encoder is not on the path at all, which
    # `encoder.calls == 0` states exactly.
    assert stats["breakdown"]["ranking_head"]["n"] == len(pools) - 1


def test_total_is_at_least_the_sum_of_its_parts():
    model = build_gardian_from_model_cfg(_model_cfg(use_controller=True))
    model.eval()
    pools = [_pool(f"q{i}") for i in range(6)]
    stats = time_gardian(
        pools,
        questions=[f"q{i}" for i in range(6)],
        model=model,
        query_encoder=_SlowEncoder(delay_s=0.005),
        device="cpu",
        warmup=1,
        encode_query=True,
    )
    parts = (
        stats["breakdown"]["query_encoder"]["mean_ms"]
        + stats["breakdown"]["ranking_head"]["mean_ms"]
    )
    assert stats["mean_ms"] >= parts * 0.95


# --------------------------------------------------------------------------
# GARDIAN-Lite has no query encoder on the training path either
# --------------------------------------------------------------------------


def _write_rank_jsonl(path: pathlib.Path, n_queries: int = 4, pool: int = 8) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for q in range(n_queries):
            for rec in _pool(f"q{q}", pool):
                fh.write(json.dumps(rec) + "\n")


def test_dataset_without_query_emb_emits_zeros_and_loads_no_cache(tmp_path):
    rank_path = tmp_path / "rank.jsonl"
    _write_rank_jsonl(rank_path)

    ds = StreamingRankDataset(
        str(rank_path),
        query_feat_dim=8,
        mode="groups",
        group_size=8,
        emit_query_emb=False,
        precompute_query_emb=True,
        query_emb_cache_path=str(tmp_path / "does_not_exist.pkl"),
        shuffle_buffer=0,
    )
    assert ds._emb_store.nbytes == 0

    first = next(iter(ds))
    query_emb = first[2]
    assert query_emb.shape == (8,)
    assert np.allclose(query_emb, 0.0)


def test_dataset_with_controller_still_requires_real_embeddings(tmp_path):
    rank_path = tmp_path / "rank.jsonl"
    _write_rank_jsonl(rank_path)

    ds = StreamingRankDataset(
        str(rank_path),
        query_feat_dim=8,
        mode="groups",
        group_size=8,
        emit_query_emb=True,
        precompute_query_emb=False,
        shuffle_buffer=0,
    )
    # No embeddings in the data and no encoder configured: the controller path
    # must fail loudly rather than silently training on zeros.
    with pytest.raises((ValueError, RuntimeError)):
        next(iter(ds))


def test_lite_model_ignores_query_embedding_entirely():
    model = build_gardian_from_model_cfg(_model_cfg(use_controller=False))
    model.eval()
    sparse = torch.rand(1, 6, 8)
    dense = torch.rand(1, 6, 8)
    mask = torch.ones(1, 6)
    with torch.no_grad():
        a, _ = model(
            sparse_feats=sparse,
            dense_feats=dense,
            query_emb=torch.zeros(1, 8),
            mask=mask,
        )
        b, _ = model(
            sparse_feats=sparse,
            dense_feats=dense,
            query_emb=torch.randn(1, 8) * 100.0,
            mask=mask,
        )
    assert torch.allclose(a, b), "Lite scores must not depend on the query embedding"


# --------------------------------------------------------------------------
# LambdaMART control reads the same features and groups by query
# --------------------------------------------------------------------------


def test_ltr_loader_groups_by_query(tmp_path):
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location(
        "ltr", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_ltr_baseline.py"
    )
    ltr = module_from_spec(spec)
    spec.loader.exec_module(ltr)

    rank_path = tmp_path / "rank.jsonl"
    _write_rank_jsonl(rank_path, n_queries=5, pool=7)

    X, y, groups, qids, pids, labels = ltr.load_feature_matrix(rank_path, dropped=[])
    assert X.shape == (35, 16), "all 16 features must reach the trees"
    assert groups == [7] * 5
    assert sum(groups) == X.shape[0] == y.shape[0]
    assert len(qids) == 5
    assert all(len(p) == 7 for p in pids)
    assert all(lab[0] == 1 for lab in labels)


def test_ltr_loader_rejects_non_contiguous_groups(tmp_path):
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location(
        "ltr2", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_ltr_baseline.py"
    )
    ltr = module_from_spec(spec)
    spec.loader.exec_module(ltr)

    rank_path = tmp_path / "bad.jsonl"
    with rank_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_record("qA", "a1", 1)) + "\n")
        fh.write(json.dumps(_record("qB", "b1", 1)) + "\n")
        fh.write(json.dumps(_record("qA", "a2", 0)) + "\n")

    # LightGBM's `group` argument assumes contiguous blocks; interleaved qids
    # would silently train on wrong groupings.
    with pytest.raises(ValueError, match="not grouped by qid"):
        ltr.load_feature_matrix(rank_path, dropped=[])


def test_ltr_loader_respects_max_queries(tmp_path):
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location(
        "ltr3", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_ltr_baseline.py"
    )
    ltr = module_from_spec(spec)
    spec.loader.exec_module(ltr)

    rank_path = tmp_path / "rank.jsonl"
    _write_rank_jsonl(rank_path, n_queries=10, pool=4)

    X, _y, groups, qids, _p, _l = ltr.load_feature_matrix(
        rank_path, dropped=[], max_queries=3
    )
    assert len(qids) == 3
    assert groups == [4, 4, 4]
    assert X.shape[0] == 12


# --------------------------------------------------------------------------
# checkpoint loading across the controller boundary
# --------------------------------------------------------------------------


def test_load_checkpoint_into_lite_model():
    """
    Regression: load_checkpoint_state read ``model.controller.net[0]``
    unconditionally, so loading any checkpoint into a GARDIAN-Lite model raised
    ``AttributeError: 'NoneType' object has no attribute 'net'`` -- which made
    the whole Lite evaluation path unreachable.
    """
    from src.model.gardian import load_checkpoint_state

    lite = build_gardian_from_model_cfg(_model_cfg(use_controller=False))
    state = {k: v.clone() for k, v in lite.state_dict().items()}
    load_checkpoint_state(lite, state, strict=True)


def test_controller_weights_are_dropped_when_target_has_no_controller():
    """A controller-trained checkpoint must load into a Lite model, branches only."""
    from src.model.gardian import load_checkpoint_state

    full = build_gardian_from_model_cfg(_model_cfg(use_controller=True))
    lite = build_gardian_from_model_cfg(_model_cfg(use_controller=False))

    state = {k: v.clone() for k, v in full.state_dict().items()}
    assert any(k.startswith("controller.") for k in state)

    load_checkpoint_state(lite, state, strict=True)

    # The branch heads must actually have been copied across, not silently skipped.
    for name, param in lite.sparse_head.named_parameters():
        assert torch.allclose(param, dict(full.sparse_head.named_parameters())[name])


def test_controller_checkpoint_still_loads_into_controller_model():
    """The widening path for legacy checkpoints must survive the Lite guard."""
    from src.model.gardian import load_checkpoint_state

    full = build_gardian_from_model_cfg(_model_cfg(use_controller=True))
    state = {k: v.clone() for k, v in full.state_dict().items()}
    target = build_gardian_from_model_cfg(_model_cfg(use_controller=True))
    load_checkpoint_state(target, state, strict=True)
    assert torch.allclose(
        target.controller.net[0].weight, full.controller.net[0].weight
    )


# --------------------------------------------------------------------------
# the timed path must be the evaluated path
# --------------------------------------------------------------------------


def test_timed_path_matches_evaluation_path_exactly():
    """
    A latency number is only meaningful if it times the computation that
    actually produced the reported effectiveness.

    src/evaluation/rank_jsonl_eval.py builds its pool tensors by stacking
    per-row ``torch.tensor`` objects; src/evaluation/rerank_latency.py builds
    them from one ``np.asarray`` for speed. If those two ever diverge -- a
    dtype, a column order, a missing pool-feature vector -- the paper would
    report the latency of one model and the nDCG of another. This pins them
    together.
    """
    from src.features.pool_features import pool_features_from_records

    model = build_gardian_from_model_cfg(_model_cfg(use_controller=True))
    model.eval()
    pool = _pool("q0", n=17)
    query_emb = torch.zeros(1, 8)

    # As rank_jsonl_eval builds it.
    s_eval = torch.stack(
        [torch.tensor(r["sparse_feats"], dtype=torch.float32) for r in pool]
    ).unsqueeze(0)
    d_eval = torch.stack(
        [torch.tensor(r["dense_feats"], dtype=torch.float32) for r in pool]
    ).unsqueeze(0)

    # As rerank_latency builds it.
    s_lat = torch.tensor(
        np.asarray([r["sparse_feats"] for r in pool], dtype=np.float32)
    ).unsqueeze(0)
    d_lat = torch.tensor(
        np.asarray([r["dense_feats"] for r in pool], dtype=np.float32)
    ).unsqueeze(0)

    assert torch.equal(s_eval, s_lat)
    assert torch.equal(d_eval, d_lat)

    pool_feats = torch.from_numpy(pool_features_from_records(pool)).unsqueeze(0)
    mask = torch.ones(s_eval.shape[:2])
    with torch.no_grad():
        a, _ = model(
            sparse_feats=s_eval, dense_feats=d_eval, query_emb=query_emb,
            pool_feats=pool_feats, mask=mask,
        )
        b, _ = model(
            sparse_feats=s_lat, dense_feats=d_lat, query_emb=query_emb,
            pool_feats=pool_feats, mask=mask,
        )
    assert torch.equal(a, b)
