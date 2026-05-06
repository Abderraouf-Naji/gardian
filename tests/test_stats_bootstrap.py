from src.evaluation.stats import bootstrap_mean_ci


def test_bootstrap_deterministic():
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    m1, lo1, hi1 = bootstrap_mean_ci(x, n_bootstrap=500, seed=123)
    m2, lo2, hi2 = bootstrap_mean_ci(x, n_bootstrap=500, seed=123)
    assert m1 == m2 == sum(x) / len(x)
    assert lo1 == lo2 and hi1 == hi2


def test_bootstrap_single_sample():
    m, lo, hi = bootstrap_mean_ci([0.42], n_bootstrap=100, seed=0)
    assert m == lo == hi == 0.42
