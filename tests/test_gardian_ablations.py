import torch

from src.model.gardian import GARDIAN


def _tiny_model():
    return GARDIAN(
        sparse_dim=3,
        dense_dim=4,
        branch_hidden=8,
        controller_hidden=8,
        query_feat_dim=12,
        n_qtypes=7,
        dropout=0.0,
        text_only=True,
    )


def test_uniform_alpha_overrides_controller():
    m = _tiny_model()
    b = 5
    z = torch.zeros
    sparse = z((b, 3))
    dense = z((b, 4))
    qe = torch.randn(b, 12)
    qt = torch.zeros(b, 7)
    qt[:, 0] = 1.0
    scores, w = m(sparse, dense, query_emb=qe, qtype_onehot=qt, ablation="uniform_alpha")
    assert w.shape == (b, 2)
    assert torch.allclose(w, torch.full_like(w, 0.5))
    assert scores.shape == (b,)


def test_no_sparse_signal_first_weight_zero():
    m = _tiny_model()
    b = 4
    sparse = torch.randn(b, 3)
    dense = torch.randn(b, 4)
    qe = torch.randn(b, 12)
    qt = torch.zeros(b, 7)
    qt[:, 1] = 1.0
    _, w = m(sparse, dense, query_emb=qe, qtype_onehot=qt, ablation="no_sparse_signal")
    assert torch.allclose(w[:, 0], torch.zeros(b), atol=1e-5)
    assert torch.allclose(w[:, 1], torch.ones(b), atol=1e-4)


def test_no_dense_signal_second_weight_zero():
    m = _tiny_model()
    b = 4
    sparse = torch.randn(b, 3)
    dense = torch.randn(b, 4)
    qe = torch.randn(b, 12)
    qt = torch.zeros(b, 7)
    qt[:, 3] = 1.0
    _, w = m(sparse, dense, query_emb=qe, qtype_onehot=qt, ablation="no_dense_signal")
    assert torch.allclose(w[:, 1], torch.zeros(b), atol=1e-5)
    assert torch.allclose(w[:, 0], torch.ones(b), atol=1e-4)
