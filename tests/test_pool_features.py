"""
The pool-feature vector must be scale-free, or it reintroduces the bug it exists
to fix: a controller that reads anything proportional to a raw channel score is
miscalibrated the moment that channel's score distribution shifts, which is
exactly what happens between PubMed abstracts and CORD-19.
"""

import numpy as np
import pytest

from src.features.pool_features import (
    POOL_FEATURE_DIM,
    POOL_FEATURE_NAMES,
    pool_features,
    pool_features_from_records,
)


def _pool(seed=0, n=92, cov=0.7):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(8.0, 3.0, n),          # BM25-like
        rng.normal(0.8, 0.05, n),         # cosine-like
        (rng.random(n) < cov).astype(float),
        (rng.random(n) < cov).astype(float),
    )


def test_dim_and_names_agree():
    assert POOL_FEATURE_DIM == len(POOL_FEATURE_NAMES)
    assert pool_features(*_pool()).shape == (POOL_FEATURE_DIM,)


@pytest.mark.parametrize("a_s,c_s,a_d,c_d", [
    (137.0, 50.0, 0.001, -9.0),
    (1e4, -1e3, 2.0, 0.0),
    (0.5, 0.0, 1e3, 7.5),
])
def test_invariant_to_independent_affine_rescaling(a_s, c_s, a_d, c_d):
    """
    The property the controller depends on: rescaling either channel
    independently must not move a single feature.
    """
    s, d, s_ok, d_ok = _pool()
    base = pool_features(s, d, s_ok, d_ok)
    moved = pool_features(a_s * s + c_s, a_d * d + c_d, s_ok, d_ok)
    assert np.allclose(base, moved, atol=1e-6), dict(
        zip(POOL_FEATURE_NAMES, np.abs(base - moved))
    )


def test_agreement_features_detect_agreement():
    """When the channels rank alike, alpha cannot matter -- so say so."""
    n = 50
    s = np.linspace(0, 10, n)[::-1].copy()
    ok = np.ones(n)
    agree = pool_features(s, s * 0.01 + 3.0, ok, ok)
    disagree = pool_features(s, s[::-1].copy(), ok, ok)
    i_sp = POOL_FEATURE_NAMES.index("rank_spearman")
    i_ov = POOL_FEATURE_NAMES.index("overlap_at_10")
    assert agree[i_sp] > 0.99 and agree[i_ov] == 1.0
    assert disagree[i_sp] < -0.99 and disagree[i_ov] == 0.0


def test_coverage_reflects_retrieval_indicators():
    n = 100
    s, d = np.random.default_rng(1).normal(size=(2, n))
    s_ok = np.zeros(n); s_ok[:30] = 1.0
    d_ok = np.zeros(n); d_ok[:80] = 1.0
    f = pool_features(s, d, s_ok, d_ok)
    assert f[POOL_FEATURE_NAMES.index("sparse_coverage")] == pytest.approx(0.30)
    assert f[POOL_FEATURE_NAMES.index("dense_coverage")] == pytest.approx(0.80)


def test_unscored_candidates_excluded_from_channel_shape():
    """
    A floor-imputed score is a marker, not a measurement. Changing the value
    stored for candidates a channel never retrieved must not move that
    channel's shape statistics.
    """
    s, d, s_ok, d_ok = _pool(seed=3)
    s2 = s.copy()
    s2[s_ok == 0] = -1e6
    a, b = pool_features(s, d, s_ok, d_ok), pool_features(s2, d, s_ok, d_ok)
    shape = [i for i, nm in enumerate(POOL_FEATURE_NAMES) if nm.startswith("sparse_")
             and nm != "sparse_coverage"]
    assert np.allclose(a[shape], b[shape], atol=1e-6)


def test_degenerate_pools_are_finite():
    for args in [
        (np.array([]), np.array([]), np.array([]), np.array([])),
        (np.array([1.0]), np.array([2.0]), np.array([1.0]), np.array([1.0])),
        (np.ones(10), np.ones(10), np.ones(10), np.ones(10)),          # constant
        (np.ones(10), np.ones(10), np.zeros(10), np.zeros(10)),        # nothing scored
    ]:
        f = pool_features(*args)
        assert f.shape == (POOL_FEATURE_DIM,)
        assert np.isfinite(f).all()


def test_from_records_reads_schema_columns():
    """Raw score is stored dim 0 and the retrieval indicator dim 7."""
    recs = [
        {"sparse_feats": [5.0, 0, 0, 0, 0, 0, 0, 1.0],
         "dense_feats":  [0.9, 0, 0, 0, 0, 0, 0, 1.0]},
        {"sparse_feats": [1.0, 0, 0, 0, 0, 0, 0, 1.0],
         "dense_feats":  [0.1, 0, 0, 0, 0, 0, 0, 0.0]},
    ]
    f = pool_features_from_records(recs)
    assert f[POOL_FEATURE_NAMES.index("sparse_coverage")] == pytest.approx(1.0)
    assert f[POOL_FEATURE_NAMES.index("dense_coverage")] == pytest.approx(0.5)
    assert f[POOL_FEATURE_NAMES.index("pool_size_norm")] == pytest.approx(0.02)
