"""Behavioural tests for the ranking objectives."""

import pytest
import torch

from src.training.losses import (
    ALL_LOSSES,
    LISTWISE_LOSSES,
    approxndcg_loss,
    build_loss,
    is_listwise,
    lambdarank_loss,
    listnet_loss,
    pairwise_softplus_margin,
)

LISTWISE = list(LISTWISE_LOSSES.items())


def _perfect():
    scores = torch.tensor([[5.0, 4.0, 3.0, 2.0]])
    labels = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mask = torch.ones(1, 4)
    return scores, labels, mask


def _reversed():
    scores = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
    labels = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mask = torch.ones(1, 4)
    return scores, labels, mask


@pytest.mark.parametrize("name,fn", LISTWISE)
def test_worse_ordering_costs_more(name, fn):
    """The gold at rank 1 must score strictly lower loss than gold at rank 4."""
    assert float(fn(*_perfect())) < float(fn(*_reversed()))


@pytest.mark.parametrize("name,fn", LISTWISE)
def test_non_negative(name, fn):
    for args in (_perfect(), _reversed()):
        assert float(fn(*args)) >= -1e-6


@pytest.mark.parametrize("name,fn", LISTWISE)
def test_query_without_positive_is_skipped(name, fn):
    """A pool with no gold contributes zero, not NaN."""
    scores = torch.randn(1, 6)
    labels = torch.zeros(1, 6)
    mask = torch.ones(1, 6)
    out = fn(scores, labels, mask)
    assert torch.isfinite(out) and abs(float(out)) < 1e-6


@pytest.mark.parametrize("name,fn", LISTWISE)
def test_padding_is_ignored(name, fn):
    """Padding columns must not change the loss, whatever score they carry."""
    scores = torch.tensor([[5.0, 4.0, 3.0]])
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.ones(1, 3)
    base = float(fn(scores, labels, mask))

    padded_scores = torch.cat([scores, torch.tensor([[99.0, -99.0]])], dim=1)
    padded_labels = torch.cat([labels, torch.zeros(1, 2)], dim=1)
    padded_mask = torch.cat([mask, torch.zeros(1, 2)], dim=1)
    assert float(fn(padded_scores, padded_labels, padded_mask)) == pytest.approx(base, abs=1e-5)


@pytest.mark.parametrize("name,fn", LISTWISE)
def test_gradient_flows(name, fn):
    scores = torch.randn(3, 8, requires_grad=True)
    labels = torch.zeros(3, 8)
    labels[:, 0] = 1.0
    loss = fn(scores, labels, torch.ones(3, 8))
    loss.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert float(scores.grad.abs().sum()) > 0.0


@pytest.mark.parametrize("name,fn", LISTWISE)
def test_batch_mean_is_order_invariant(name, fn):
    """Two queries in one batch = mean of the two evaluated separately."""
    s1, l1, m1 = _perfect()
    s2, l2, m2 = _reversed()
    both = fn(torch.cat([s1, s2]), torch.cat([l1, l2]), torch.cat([m1, m2]))
    apart = 0.5 * (float(fn(s1, l1, m1)) + float(fn(s2, l2, m2)))
    assert float(both) == pytest.approx(apart, rel=1e-5)


def test_lambdarank_weights_the_cutoff_more_than_the_tail():
    """
    The discount curvature must dominate: moving the gold between ranks 1 and 2
    changes the loss far more than moving it between ranks 6 and 7, even though
    both are one-position moves.

    Scores are a near-flat ramp so every pair sits at essentially the same
    softplus value; what is left is the |delta nDCG| weight alone. This is the
    property the margin loss does not have -- there, a one-position move costs
    the same wherever it happens.
    """
    n = 50
    scores = (-1e-4 * torch.arange(n, dtype=torch.float32)).unsqueeze(0)
    mask = torch.ones(1, n)

    def loss_with_gold_at(pos):
        lab = torch.zeros(1, n)
        lab[0, pos] = 1.0
        return float(lambdarank_loss(scores, lab, mask, k=10))

    top_move = abs(loss_with_gold_at(0) - loss_with_gold_at(1))
    tail_move = abs(loss_with_gold_at(5) - loss_with_gold_at(6))
    assert top_move > 5.0 * tail_move


def test_lambdarank_ignores_pairs_of_equal_label():
    """Two irrelevant candidates in any order contribute exactly zero."""
    scores = torch.tensor([[3.0, 1.0, 2.0]])
    labels = torch.zeros(1, 3)
    mask = torch.ones(1, 3)
    assert float(lambdarank_loss(scores, labels, mask)) == pytest.approx(0.0, abs=1e-7)


def test_pairwise_margin_saturates():
    """Documents the failure mode that motivated the listwise objectives."""
    pos = torch.tensor([10.0], requires_grad=True)
    neg = torch.tensor([0.0])
    pairwise_softplus_margin(pos, neg, margin=1.0).backward()
    assert float(pos.grad.abs()) < 1e-3          # separated pair -> no gradient

    pos2 = torch.tensor([0.5], requires_grad=True)
    pairwise_softplus_margin(pos2, neg, margin=1.0).backward()
    assert float(pos2.grad.abs()) > 0.3          # pair at the margin -> gradient


def test_registry():
    assert is_listwise("lambdarank")
    assert not is_listwise("pairwise_softplus_margin")
    assert build_loss("approxndcg") is approxndcg_loss
    assert build_loss("listnet") is listnet_loss
    with pytest.raises(ValueError, match="Unknown training.loss"):
        build_loss("nope")
    assert set(ALL_LOSSES) == {
        "pairwise_softplus_margin", "lambdarank", "approxndcg", "listnet"
    }
