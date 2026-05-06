"""Doc2Query metrics appear when rank JSONL carries nonzero scores."""

import json
import tempfile

from src.evaluation.rank_jsonl_eval import evaluate_all_from_rank_data


def _pair(qid, pid, label, doc2query, bm25=0.1, dense=0.2):
    sparse = [bm25, 0.0, 0.0]
    dense_f = [dense, 0.0, 0.0, 0.0]
    kg = [0.0] * 6
    return {
        "qid": qid,
        "pid": pid,
        "question": "test?",
        "question_type": "factoid",
        "label": label,
        "gold_passage_ids": ["a"] if label == 1 else [],
        "bm25_score": bm25,
        "dense_score": dense,
        "doc2query_score": doc2query,
        "sparse_feats": sparse,
        "dense_feats": dense_f,
        "kg_feats": kg,
        "query_emb": [0.0] * 384,
        "qtype_onehot": [0.0] * 6 + [1.0],
        "kg_coverage": 0.0,
    }


def test_doc2query_evaluated_when_nonzero():
    # Query 1: gold is pid "a", higher Doc2Query score on "a"
    rows = [
        _pair("q1", "a", 1, doc2query=10.0),
        _pair("q1", "b", 0, doc2query=1.0),
        _pair("q2", "x", 1, doc2query=2.0),
        _pair("q2", "y", 0, doc2query=8.0),
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        path = f.name
    out = evaluate_all_from_rank_data(path, model=None, device=None)
    assert "doc2query" in out
    assert out["doc2query"]["ndcg@10"] > 0.0


def _neural_row(qid, pid, label, doc2query, biobert_dense):
    sparse = [doc2query, 0.0, 0.0]
    dense_f = [biobert_dense, 0.0, 0.0, 0.0]
    kg = [0.0] * 6
    return {
        "qid": qid,
        "pid": pid,
        "question": "test?",
        "question_type": "factoid",
        "label": label,
        "gold_passage_ids": ["a"] if label == 1 else [],
        "retriever_type": "hybrid_neural",
        "dense_score": biobert_dense,
        "doc2query_score": doc2query,
        "sparse_feats": sparse,
        "dense_feats": dense_f,
        "kg_feats": kg,
        "query_emb": [0.0] * 384,
        "qtype_onehot": [0.0] * 6 + [1.0],
        "kg_coverage": 0.0,
    }


def test_canonical_baseline_keys_hybrid_neural():
    rows = [
        _neural_row("q1", "a", 1, doc2query=10.0, biobert_dense=0.5),
        _neural_row("q1", "b", 0, doc2query=1.0, biobert_dense=0.4),
        _neural_row("q2", "x", 1, doc2query=2.0, biobert_dense=0.3),
        _neural_row("q2", "y", 0, doc2query=8.0, biobert_dense=0.2),
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        path = f.name
    out = evaluate_all_from_rank_data(path, model=None, device=None, canonical_baseline_keys=True)
    assert "sparse(doc2query)" in out
    assert "dense(biobert)" in out
    assert "sum(doc2query,biobert)" in out
    assert "bm25" not in out
    assert "hybrid" not in out
    assert out["_meta"]["retriever_type"] == "hybrid_neural"


def test_rrf_is_reported():
    rows = [
        _pair("q1", "a", 1, doc2query=0.0, bm25=10.0, dense=0.1),
        _pair("q1", "b", 0, doc2query=0.0, bm25=0.1, dense=10.0),
        _pair("q2", "x", 1, doc2query=0.0, bm25=9.0, dense=9.0),
        _pair("q2", "y", 0, doc2query=0.0, bm25=1.0, dense=1.0),
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        path = f.name
    out = evaluate_all_from_rank_data(path, model=None, device=None)
    assert "rrf" in out
    assert out["rrf"]["ndcg@10"] > 0.0
