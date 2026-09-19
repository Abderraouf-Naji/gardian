"""
The listwise path: pools in, masked (B, N) scores out, one optimiser step.

The pairwise path emits (positive, negative) tuples; a listwise objective needs
whole candidate pools instead, so the dataset, the collate function and the
trainer step all change together. These tests pin that contract end to end on
CPU.
"""

from __future__ import annotations

import json

import pytest
import torch
from omegaconf import OmegaConf

from src.model.gardian import build_gardian_from_model_cfg
from src.training.trainer import (
    GARDIANTrainer,
    StreamingRankDataset,
    collate_groups,
)

QUERY_FEAT_DIM = 8
POOL = 6


def _record(qid: str, pid: str, label: int) -> dict:
    return {
        "qid": qid,
        "pid": pid,
        "label": label,
        "sparse_feats": [float(label)] * 8,
        "dense_feats": [float(label) * 0.5] * 8,
        "query_emb": [0.25] * QUERY_FEAT_DIM,
    }


@pytest.fixture
def rank_jsonl(tmp_path):
    """Five queries, each one positive followed by five negatives."""
    path = tmp_path / "rank.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for q in range(5):
            f.write(json.dumps(_record(f"q{q}", f"q{q}_pos", 1)) + "\n")
            for n in range(5):
                f.write(json.dumps(_record(f"q{q}", f"q{q}_neg{n}", 0)) + "\n")
    return path


def _pools(path, group_size=POOL):
    return list(
        StreamingRankDataset(
            str(path),
            num_negatives=5,
            query_feat_dim=QUERY_FEAT_DIM,
            precompute_query_emb=False,
            mode="groups",
            group_size=group_size,
            shuffle_buffer=0,
        )
    )


def test_group_mode_emits_one_item_per_query(rank_jsonl):
    """By default every stored feature reaches the model: 8 + 8."""
    pools = _pools(rank_jsonl)
    assert len(pools) == 5
    for sparse, dense, qemb, labels in pools:
        assert sparse.shape == (POOL, 8)
        assert dense.shape == (POOL, 8)
        assert qemb.shape == (QUERY_FEAT_DIM,)
        assert labels.sum() == 1.0


def test_dropping_features_removes_the_named_columns(rank_jsonl):
    """
    A drop list must remove exactly the named columns, keeping the rest in order.

    ``retrieved_by_*`` is dimension 7 of each branch, so dropping both must keep
    dims 0-6. Pinned because silently selecting the wrong columns still trains
    and still produces a plausible nDCG -- it fails silently, not loudly.
    """
    dataset = StreamingRankDataset(
        str(rank_jsonl), num_negatives=5, query_feat_dim=QUERY_FEAT_DIM,
        precompute_query_emb=False, mode="groups", group_size=POOL,
        shuffle_buffer=0,
        dropped_features=["retrieved_by_sparse", "retrieved_by_dense"],
    )
    assert dataset.sparse_cols == [0, 1, 2, 3, 4, 5, 6]
    assert dataset.dense_cols == [0, 1, 2, 3, 4, 5, 6]
    pools = list(dataset)
    assert pools[0][0].shape == (POOL, 7)
    assert pools[0][1].shape == (POOL, 7)


def test_data_is_unchanged_by_a_drop_list(rank_jsonl):
    """
    Dropping features is a column selection, never an edit to the rank JSONL.

    This is what lets a feature set change without re-running
    scripts/03_generate_rank_data.py.
    """
    before = rank_jsonl.read_text()
    list(
        StreamingRankDataset(
            str(rank_jsonl), num_negatives=5, query_feat_dim=QUERY_FEAT_DIM,
            precompute_query_emb=False, mode="groups", group_size=POOL,
            shuffle_buffer=0, dropped_features=["retrieved_by_dense"],
        )
    )
    assert rank_jsonl.read_text() == before


def test_group_size_caps_the_pool(rank_jsonl):
    """A smaller group_size truncates negatives but always keeps a positive."""
    for sparse, _, _, labels in _pools(rank_jsonl, group_size=3):
        assert sparse.shape[0] == 3
        assert labels.sum() == 1.0


def test_invalid_mode_is_rejected(rank_jsonl):
    with pytest.raises(ValueError, match="mode must be"):
        StreamingRankDataset(
            str(rank_jsonl), query_feat_dim=QUERY_FEAT_DIM,
            precompute_query_emb=False, mode="listwise",
        )


def test_collate_pads_and_masks():
    """Ragged pools pad to the batch maximum; padding is masked, never labelled."""
    import numpy as np

    small = (np.ones((2, 8), np.float32), np.ones((2, 8), np.float32),
             np.zeros(QUERY_FEAT_DIM, np.float32), np.array([1.0, 0.0], np.float32))
    big = (np.ones((5, 8), np.float32), np.ones((5, 8), np.float32),
           np.zeros(QUERY_FEAT_DIM, np.float32), np.array([1.0, 0, 0, 0, 0], np.float32))

    b = collate_groups([small, big])
    assert b["sparse_feats"].shape == (2, 5, 8)
    assert b["mask"][0].tolist() == [1, 1, 0, 0, 0]
    assert b["mask"][1].tolist() == [1, 1, 1, 1, 1]
    assert float((b["labels"] * (1 - b["mask"])).sum()) == 0.0


def _cfg(loss: str):
    cfg = OmegaConf.load("configs/base.yaml")
    cfg.model.sparse_feat_dim = 8
    cfg.model.dense_feat_dim = 8
    cfg.model.query_feat_dim = QUERY_FEAT_DIM
    cfg.model.branch_hidden = 16
    cfg.model.controller_hidden = 16
    cfg.training.loss = loss
    return cfg


@pytest.mark.parametrize("loss", ["lambdarank", "approxndcg", "listnet"])
def test_trainer_takes_a_listwise_step(rank_jsonl, loss):
    """Forward, loss and backward run, and the weights actually move."""
    cfg = _cfg(loss)
    model = build_gardian_from_model_cfg(cfg.model)
    trainer = GARDIANTrainer(model, cfg, device="cpu")
    assert trainer.listwise

    batch = collate_groups(_pools(rank_jsonl))
    before = [p.detach().clone() for p in model.parameters()]

    loss_value = trainer._listwise_step(batch)
    assert torch.isfinite(loss_value)
    loss_value.backward()
    trainer.opt.step()

    assert any(
        not torch.equal(b, a) for b, a in zip(before, model.parameters())
    ), "a listwise step must update the model"


def test_forward_groups_shape_and_query_broadcast(rank_jsonl):
    """Scores come back (B, N), and one query's pool shares its embedding."""
    cfg = _cfg("lambdarank")
    model = build_gardian_from_model_cfg(cfg.model)
    trainer = GARDIANTrainer(model, cfg, device="cpu")
    batch = collate_groups(_pools(rank_jsonl))

    assert (model.sparse_dim, model.dense_dim) == (8, 8)
    with torch.no_grad():
        scores = trainer.forward_groups(batch)
    assert scores.shape == (5, POOL)
    assert torch.isfinite(scores).all()

    with torch.no_grad():
        weights = model.controller_weights(batch["query_emb"])
    assert weights.shape == (5, 2)
    assert torch.allclose(weights.sum(dim=1), torch.ones(5), atol=1e-5)


def test_pairwise_mode_still_selected_by_config(rank_jsonl):
    cfg = _cfg("pairwise_softplus_margin")
    model = build_gardian_from_model_cfg(cfg.model)
    trainer = GARDIANTrainer(model, cfg, device="cpu")
    assert not trainer.listwise
