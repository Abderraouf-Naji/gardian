"""Sanity checks for retrieval metrics used in evaluation scripts."""

import math

from src.evaluation.metrics import ndcg_at_k


def test_ndcg_perfect_ranking():
    ranked = ["a", "b", "c"]
    relevant = {"a", "b"}
    assert math.isclose(ndcg_at_k(ranked, relevant, 10), 1.0, rel_tol=1e-9)


def test_ndcg_partial():
    ranked = ["x", "a", "b"]
    relevant = {"a"}
    # Only "a" at rank 2 contributes: 1/log2(3)
    dcg = 1.0 / math.log2(3)
    idcg = 1.0  # single relevant
    expected = dcg / idcg
    assert math.isclose(ndcg_at_k(ranked, relevant, 10), expected, rel_tol=1e-9)


def test_ndcg_empty_relevant():
    ranked = ["a", "b"]
    relevant = set()
    idcg = 0.0
    assert ndcg_at_k(ranked, relevant, 5) == 0.0
