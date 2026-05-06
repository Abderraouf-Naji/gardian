import torch

from src.model.gardian import GARDIAN


def _tiny_model():
    return GARDIAN(
        sparse_dim=3,
        dense_dim=4,
        kg_dim=6,
        branch_hidden=8,
        controller_hidden=8,
        query_feat_dim=12,
        n_qtypes=7,
        dropout=0.0,
    )


def test_uniform_alpha_overrides_controller():
    m = _tiny_model()
    b = 5
    z = torch.zeros
    sparse = z((b, 3))
    dense = z((b, 4))
    kg = z((b, 6))
    qe = torch.randn(b, 12)
    qt = torch.zeros(b, 7)
    qt[:, 0] = 1.0
    kc = torch.ones(b)
    scores, w = m(sparse, dense, kg, qe, qt, kc, ablation="uniform_alpha")
    assert w.shape == (b, 3)
    assert torch.allclose(w, torch.full_like(w, 1.0 / 3.0))
    assert scores.shape == (b,)


def test_no_kg_signal_third_weight_zero():
    m = _tiny_model()
    b = 4
    sparse = torch.randn(b, 3)
    dense = torch.randn(b, 4)
    kg = torch.randn(b, 6)
    qe = torch.randn(b, 12)
    qt = torch.zeros(b, 7)
    qt[:, 2] = 1.0
    kc = torch.ones(b) * 0.5
    _, w = m(sparse, dense, kg, qe, qt, kc, ablation="no_kg_signal")
    assert torch.allclose(w[:, 2], torch.zeros(b), atol=1e-5)
    assert torch.allclose(w[:, :2].sum(dim=1), torch.ones(b), atol=1e-4)


def test_no_sparse_signal_first_weight_zero():
    m = _tiny_model()
    b = 4
    sparse = torch.randn(b, 3)
    dense = torch.randn(b, 4)
    kg = torch.randn(b, 6)
    qe = torch.randn(b, 12)
    qt = torch.zeros(b, 7)
    qt[:, 1] = 1.0
    kc = torch.ones(b) * 0.5
    _, w = m(sparse, dense, kg, qe, qt, kc, ablation="no_sparse_signal")
    assert torch.allclose(w[:, 0], torch.zeros(b), atol=1e-5)
    assert torch.allclose(w[:, 1:].sum(dim=1), torch.ones(b), atol=1e-4)


def test_no_dense_signal_second_weight_zero():
    m = _tiny_model()
    b = 4
    sparse = torch.randn(b, 3)
    dense = torch.randn(b, 4)
    kg = torch.randn(b, 6)
    qe = torch.randn(b, 12)
    qt = torch.zeros(b, 7)
    qt[:, 3] = 1.0
    kc = torch.ones(b) * 0.5
    _, w = m(sparse, dense, kg, qe, qt, kc, ablation="no_dense_signal")
    assert torch.allclose(w[:, 1], torch.zeros(b), atol=1e-5)
    assert torch.allclose(torch.stack([w[:, 0], w[:, 2]], dim=1).sum(dim=1), torch.ones(b), atol=1e-4)
