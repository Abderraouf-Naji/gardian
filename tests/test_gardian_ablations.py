"""Ablation semantics of the GARDIAN forward pass."""

import pytest
import torch

from src.model.gardian import ABLATIONS, GARDIAN


def _tiny_model():
    return GARDIAN(
        sparse_dim=3,
        dense_dim=4,
        branch_hidden=8,
        controller_hidden=8,
        query_feat_dim=12,
        dropout=0.0,
    )


def _inputs(b: int):
    return torch.randn(b, 3), torch.randn(b, 4), torch.randn(b, 12)


def test_controller_takes_the_query_embedding_alone():
    m = _tiny_model()
    assert m.controller.net[0].in_features == 12, (
        "the controller must be driven by the query embedding only; "
        "question-type conditioning was removed (see docs/QUESTION_TYPE.md)"
    )


def test_weights_lie_on_the_simplex():
    m = _tiny_model()
    sparse, dense, qe = _inputs(6)
    _, w = m(sparse, dense, query_emb=qe)
    assert w.shape == (6, 2)
    assert torch.allclose(w.sum(dim=1), torch.ones(6), atol=1e-5)
    assert (w >= 0).all()


def test_uniform_alpha_overrides_controller():
    m = _tiny_model()
    b = 5
    sparse, dense, qe = _inputs(b)
    scores, w = m(sparse, dense, query_emb=qe, ablation="uniform_alpha")
    assert w.shape == (b, 2)
    assert torch.allclose(w, torch.full_like(w, 0.5))
    assert scores.shape == (b,)


def test_no_sparse_signal_first_weight_zero():
    m = _tiny_model()
    b = 4
    sparse, dense, qe = _inputs(b)
    _, w = m(sparse, dense, query_emb=qe, ablation="no_sparse_signal")
    assert torch.allclose(w[:, 0], torch.zeros(b), atol=1e-5)
    assert torch.allclose(w[:, 1], torch.ones(b), atol=1e-4)


def test_no_dense_signal_second_weight_zero():
    m = _tiny_model()
    b = 4
    sparse, dense, qe = _inputs(b)
    _, w = m(sparse, dense, query_emb=qe, ablation="no_dense_signal")
    assert torch.allclose(w[:, 1], torch.zeros(b), atol=1e-5)
    assert torch.allclose(w[:, 0], torch.ones(b), atol=1e-4)


def test_reported_weights_match_the_scores_that_were_produced():
    """The returned weights must be the ones actually used to fuse."""
    m = _tiny_model()
    sparse, dense, qe = _inputs(4)
    scores, w = m(sparse, dense, query_emb=qe, ablation="no_dense_signal")
    s_sparse = m.sparse_head(sparse)
    expected = w[:, 0] * s_sparse
    assert torch.allclose(scores, expected, atol=1e-5)


def test_one_forward_mixes_match_named_ablations():
    """RQ2 rows must be mixes of one forward, identical to GARDIAN.forward."""
    m = _tiny_model()
    m.eval()
    b, n = 2, 5
    sf = torch.randn(b, n, 3)
    df = torch.randn(b, n, 4)
    qe = torch.randn(b, 12)
    mask = torch.ones(b, n)
    scores, _, bd = m(sf, df, qe, mask=mask, return_breakdown=True)
    uni, _ = m(sf, df, qe, mask=mask, ablation="uniform_alpha")
    ns, _ = m(sf, df, qe, mask=mask, ablation="no_sparse_signal")
    nd, _ = m(sf, df, qe, mask=mask, ablation="no_dense_signal")
    fx, _ = m(sf, df, qe, mask=mask, ablation="fixed_alpha", fixed_alpha=0.3)
    assert torch.allclose(uni, 0.5 * bd["s_sparse"] + 0.5 * bd["s_dense"], atol=1e-5)
    assert torch.allclose(ns, bd["s_dense"], atol=1e-5)
    assert torch.allclose(nd, bd["s_sparse"], atol=1e-5)
    assert torch.allclose(fx, 0.3 * bd["s_sparse"] + 0.7 * bd["s_dense"], atol=1e-5)
    assert torch.allclose(scores, m(sf, df, qe, mask=mask)[0], atol=1e-5)


def test_unknown_ablation_is_rejected():
    m = _tiny_model()
    sparse, dense, qe = _inputs(2)
    with pytest.raises(ValueError, match="Unknown ablation"):
        m(sparse, dense, query_emb=qe, ablation="no_qtype")


def test_ablation_registry_has_no_question_type_entry():
    assert "no_qtype" not in ABLATIONS
    assert set(ABLATIONS) == {
        "uniform_alpha",
        "fixed_alpha",
        "no_sparse_signal",
        "no_dense_signal",
    }


def test_fixed_alpha_applies_the_supplied_weight():
    """
    ``fixed_alpha`` is the honest fixed-weight control for the adaptive claim.

    ``uniform_alpha`` pins 0.5/0.5, which is an arbitrary constant -- beating it
    conflates "adaptivity helps" with "0.5 is a bad constant". ``fixed_alpha``
    takes the single best alpha fitted on dev, so GARDIAN minus this ablation
    is the value of per-query adaptation and nothing else.
    """
    import torch

    from src.model.gardian import GARDIAN

    model = GARDIAN(
        sparse_dim=8, dense_dim=8, branch_hidden=16,
        controller_hidden=16, query_feat_dim=8,
    )
    sf, df, qe = torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8)

    _, w = model(sf, df, qe, ablation="fixed_alpha", fixed_alpha=0.66)
    assert torch.allclose(w[:, 0], torch.full((4,), 0.66), atol=1e-6)
    assert torch.allclose(w[:, 1], torch.full((4,), 0.34), atol=1e-6)

    # identical for every query in the batch -- that is the whole point
    assert torch.allclose(w, w[0].expand_as(w))


def test_fixed_alpha_refuses_an_untuned_default():
    """Silently defaulting the alpha would turn the control into a strawman."""
    import pytest
    import torch

    from src.model.gardian import GARDIAN

    model = GARDIAN(
        sparse_dim=8, dense_dim=8, branch_hidden=16,
        controller_hidden=16, query_feat_dim=8,
    )
    with pytest.raises(ValueError, match="tuned on dev"):
        model(
            torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 8),
            ablation="fixed_alpha",
        )


def test_fixed_alpha_and_uniform_alpha_differ():
    """They are different controls; conflating them is the reviewer's objection."""
    import torch

    from src.model.gardian import GARDIAN

    model = GARDIAN(
        sparse_dim=8, dense_dim=8, branch_hidden=16,
        controller_hidden=16, query_feat_dim=8,
    )
    sf, df, qe = torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8)
    s_uniform, _ = model(sf, df, qe, ablation="uniform_alpha")
    s_fixed, _ = model(sf, df, qe, ablation="fixed_alpha", fixed_alpha=0.9)
    assert not torch.allclose(s_uniform, s_fixed)


def test_paper_run_ablation_choices_are_all_real():
    """
    The driver's ablation list must stay a subset of the model's.

    ``no_qtype`` outlived the question-type input it named and would raise at
    evaluation time -- after a full checkpoint load and rank-data pass.
    """
    import importlib.util
    import pathlib

    from src.model.gardian import ABLATIONS

    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "10_paper_run.py"
    src = path.read_text()
    start = src.index("ABLATION_CHOICES = [")
    block = src[start:src.index("]", start) + 1]
    names = {n.strip().strip('",') for n in block.splitlines()[1:-1] if n.strip()}

    assert "full" in names, "the unablated model must be runnable"
    assert "fixed_alpha" in names, (
        "RQ2 needs the dev-fitted Fixed-α control, not only uniform 0.5/0.5"
    )
    extra_ok = {"full", "oracle_branch"}
    assert names - extra_ok <= set(ABLATIONS), (
        f"unknown ablation(s) in 10_paper_run.py: {sorted(names - extra_ok - set(ABLATIONS))}"
    )
