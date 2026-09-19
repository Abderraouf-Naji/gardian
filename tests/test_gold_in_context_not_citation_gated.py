"""
Gold-in-context must be reported wherever gold passages exist.

It is a retrieval diagnostic -- did the retriever put the annotated evidence in
front of the reader -- and is independent of whether the dataset supports
citation scoring. It used to share the early return that nulls citation metrics
on MedMCQA, so Hit@k came back null on exactly the dataset whose weak results
the paper blamed on first-stage retrieval.
"""

from src.pipeline.rag.metrics import compute_citation_metrics

PASSAGES = [{"id": "a"}, {"id": "b"}, {"id": "gold"}]


def test_medmcqa_reports_gold_in_context_without_citation_metrics():
    m = compute_citation_metrics(
        [], PASSAGES, ["gold"], reader_task="mcq", dataset="medmcqa"
    )
    assert m["gold_in_context_any"] == 1.0
    assert m["gold_in_context_recall"] == 1.0
    # Citation metrics stay undefined: MedMCQA has no passage-level annotation.
    assert m["citation_precision"] is None
    assert m["citation_recall"] is None
    assert m["unsupported_citation_rate"] is None


def test_medmcqa_gold_miss_is_zero_not_none():
    m = compute_citation_metrics(
        [], PASSAGES, ["absent"], reader_task="mcq", dataset="medmcqa"
    )
    assert m["gold_in_context_any"] == 0.0
    assert m["gold_in_context_recall"] == 0.0


def test_no_gold_ids_stays_none():
    m = compute_citation_metrics([], PASSAGES, [], reader_task="mcq", dataset="medmcqa")
    assert m["gold_in_context_any"] is None
    assert m["gold_in_context_recall"] is None


def test_pubmedqa_still_reports_both_families():
    m = compute_citation_metrics(
        ["3"], PASSAGES, ["gold"], reader_task="yesno", dataset="pubmedqa_labeled"
    )
    assert m["gold_in_context_any"] == 1.0
    assert m["citation_precision"] == 1.0
