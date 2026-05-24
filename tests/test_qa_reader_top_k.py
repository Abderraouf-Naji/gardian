"""Reader passage selection aligns Hybrid (RRF) vs GARDIAN (rerank) under RQ4."""

from types import SimpleNamespace

from src.evaluation.qa_eval import (
    _gold_evidence_in_reader_context,
    _hybrid_use_balanced_top_k,
    _select_reader_passages,
)


def _cfg(*, balanced: bool = False, align: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        qa=SimpleNamespace(
            top_k_passages=10,
            yesno_top_k_passages=10,
            mcq_top_k_passages=10,
            hybrid_balanced_top_k=balanced,
            rq4_align_retrieval_top_k=align,
        )
    )


def test_rq4_align_disables_hybrid_balanced():
    assert _hybrid_use_balanced_top_k(_cfg(balanced=True, align=True)) is False
    assert _hybrid_use_balanced_top_k(_cfg(balanced=True, align=False)) is True


def test_gardian_top_k_by_gardian_score():
    cfg = _cfg()
    candidates = [
        {"id": "a", "text": "A", "bm25_score": 9.0, "dense_score": 0.1, "hybrid_rrf_score": 0.5},
        {"id": "b", "text": "B", "bm25_score": 0.1, "dense_score": 9.0, "hybrid_rrf_score": 0.6},
        {"id": "c", "text": "C", "bm25_score": 5.0, "dense_score": 5.0, "hybrid_rrf_score": 0.9},
    ]
    gardian_ranked = [
        {**candidates[0], "gardian_score": 0.2},
        {**candidates[1], "gardian_score": 0.3},
        {**candidates[2], "gardian_score": 0.95},
    ]
    scored = [(0.9, candidates[2]), (0.6, candidates[1]), (0.5, candidates[0])]
    top = _select_reader_passages(
        "gardian",
        candidates=candidates,
        scored=scored,
        k=2,
        cfg=cfg,
        gardian_ranked=gardian_ranked,
    )
    assert [p["id"] for p in top] == ["c", "b"]


def test_hybrid_top_k_by_rrf_when_aligned():
    cfg = _cfg()
    candidates = [
        {"id": "a", "text": "A", "bm25_score": 9.0, "dense_score": 0.1, "hybrid_rrf_score": 0.1},
        {"id": "b", "text": "B", "bm25_score": 0.1, "dense_score": 9.0, "hybrid_rrf_score": 0.95},
    ]
    scored = [(0.95, candidates[1]), (0.1, candidates[0])]
    top = _select_reader_passages(
        "hybrid",
        candidates=candidates,
        scored=scored,
        k=1,
        cfg=cfg,
    )
    assert top[0]["id"] == "b"


def test_gold_evidence_in_reader_context():
    passages = [{"id": "g1", "text": "a"}, {"id": "x", "text": "b"}]
    assert _gold_evidence_in_reader_context(passages, ["g1", "g2"]) == 0.5
    assert _gold_evidence_in_reader_context(passages, []) is None
