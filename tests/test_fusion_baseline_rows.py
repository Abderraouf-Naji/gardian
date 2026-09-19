"""
The fusion baselines must be evaluated on the same pools as GARDIAN.

Global-alpha and Group-alpha are fitted on dev by the caller and applied here;
Oracle-alpha selects each query's best alpha from the evaluated split's own
labels and is therefore a ceiling. These tests pin that distinction, because a
ceiling accidentally reported as a baseline would silently invert the paper's
central comparison.
"""

from __future__ import annotations

import json

import pytest

from src.evaluation.rank_jsonl_eval import evaluate_all_from_rank_data


def _rec(qid, pid, label, sparse, dense, qtype="factoid"):
    return {
        "qid": qid,
        "pid": pid,
        "label": label,
        "question_type": qtype,
        "retriever_type": "hybrid_bm25_faiss",
        "sparse_feats": [sparse, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "dense_feats": [dense, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "bm25_score": sparse,
        "dense_score": dense,
    }


@pytest.fixture
def rank_file(tmp_path):
    """
    Two queries with opposite channel preferences.

    q0's gold is found by the sparse channel only, q1's by the dense channel
    only -- so no single alpha is right for both, which is exactly the setting
    where Oracle-alpha must exceed Global-alpha.
    """
    path = tmp_path / "rank.jsonl"
    rows = [
        _rec("q0", "q0_gold", 1, 9.0, 0.1),
        _rec("q0", "q0_a", 0, 1.0, 0.9),
        _rec("q0", "q0_b", 0, 0.5, 0.8),
        _rec("q1", "q1_gold", 1, 0.1, 9.0, qtype="yesno"),
        _rec("q1", "q1_a", 0, 0.9, 1.0, qtype="yesno"),
        _rec("q1", "q1_b", 0, 0.8, 0.5, qtype="yesno"),
    ]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


def test_oracle_alpha_is_at_least_global_alpha(rank_file):
    """A per-query alpha can never be beaten by one fixed alpha."""
    res = evaluate_all_from_rank_data(rank_file, global_alpha=0.5)
    assert res["oracle_alpha"]["ndcg@10"] >= res["global_alpha"]["ndcg@10"] - 1e-9


def test_oracle_alpha_finds_what_no_fixed_alpha_can(rank_file):
    """
    With opposite per-query preferences, the ceiling must be strictly higher.

    This is the measurement the paper's negative result rests on: the gap is
    real, it is just not predictable from the query.
    """
    res = evaluate_all_from_rank_data(rank_file, global_alpha=0.5)
    assert res["oracle_alpha"]["ndcg@10"] > res["global_alpha"]["ndcg@10"]
    assert res["oracle_alpha"]["ndcg@10"] == pytest.approx(1.0)


def test_global_alpha_row_absent_unless_fitted(rank_file):
    """No dev fit means no row -- never silently fitted on the reported split."""
    res = evaluate_all_from_rank_data(rank_file)
    assert "global_alpha" not in res
    assert "group_alpha" not in res
    assert "oracle_alpha" in res          # the ceiling needs no fit


def test_oracle_alpha_can_be_skipped(rank_file):
    res = evaluate_all_from_rank_data(rank_file, include_oracle_alpha=False)
    assert "oracle_alpha" not in res


def test_group_alpha_applies_per_question_type(rank_file):
    """Each group's alpha is routed by that query's question type."""
    res = evaluate_all_from_rank_data(
        rank_file,
        group_alphas={"__global__": 0.5, "factoid": 1.0, "yesno": 0.0},
        question_types={"q0": "factoid", "q1": "yesno"},
    )
    # factoid -> all sparse (finds q0's gold), yesno -> all dense (finds q1's)
    assert res["group_alpha"]["ndcg@10"] == pytest.approx(1.0)


def test_meta_reports_pool_recall_and_ceilings(rank_file):
    res = evaluate_all_from_rank_data(rank_file, global_alpha=0.5)
    meta = res["_meta"]
    assert meta["pool_recall"] == pytest.approx(1.0)
    assert meta["global_alpha"] == pytest.approx(0.5)
    assert meta["oracle_rerank_ndcg@10"] == pytest.approx(1.0)
    for key in ("tie_fraction", "invariant_fraction", "n_queries"):
        assert key in meta["oracle_alpha"]


def test_baselines_share_gardian_population(rank_file):
    """
    Every row must be averaged over the same queries.

    Mixing populations is how a fusion baseline gets accidentally compared
    against a different query set; ``query_count`` pins that they agree.
    """
    res = evaluate_all_from_rank_data(rank_file, global_alpha=0.5)
    assert res["_meta"]["query_count"] == 2
    for key in ("rrf", "global_alpha", "oracle_alpha"):
        assert res[key]["ndcg@10"] >= 0.0


def test_group_alpha_reads_question_types_from_records(rank_file):
    """
    Group-alpha must route by each query's own type without being handed a map.

    Regression test: passing ``question_types=None`` previously sent every
    query to ``__global__``, so the Group-alpha row silently reported
    Global-alpha while looking like a distinct baseline.
    """
    res = evaluate_all_from_rank_data(
        rank_file,
        global_alpha=0.5,
        group_alphas={"__global__": 0.5, "factoid": 1.0, "yesno": 0.0},
        question_types=None,          # types come from the records themselves
    )
    assert res["group_alpha"]["ndcg@10"] == pytest.approx(1.0)
    assert res["group_alpha"]["ndcg@10"] > res["global_alpha"]["ndcg@10"]


def test_group_alpha_equals_global_when_no_group_matches(rank_file):
    """An unmatched group falls back to the global alpha -- and says so."""
    res = evaluate_all_from_rank_data(
        rank_file,
        global_alpha=0.5,
        group_alphas={"__global__": 0.5},      # no per-type entries at all
    )
    assert res["group_alpha"]["ndcg@10"] == pytest.approx(
        res["global_alpha"]["ndcg@10"]
    )
