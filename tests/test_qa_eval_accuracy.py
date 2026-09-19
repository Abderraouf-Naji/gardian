"""Unit tests for QA answer scoring dataset tags."""

from src.evaluation.qa_eval import _check_accuracy, _unsupported_claim_rate, format_qa_question_for_reader


def test_pubmedqa_labeled_tag():
    assert _check_accuracy("The answer is yes.", "yes", "pubmedqa_labeled") == 1.0
    assert _check_accuracy("no evidence", "yes", "pubmedqa_labeled") == 0.0
    # "no" must not match as substring of "know" (common in "I don't know").
    assert _check_accuracy("I don't know.", "no", "pubmedqa_labeled") == 0.0
    assert _check_accuracy("Label: no", "no", "pubmedqa_labeled") == 1.0
    assert _check_accuracy("Long rationale without label words.\nyes", "yes", "pubmedqa_labeled") == 1.0
    assert _check_accuracy("Long rationale.\nno", "yes", "pubmedqa_labeled") == 0.0
    assert _check_accuracy("I don't know. The answer: no.", "no", "pubmedqa_labeled") == 1.0
    assert _check_accuracy("I don't know. The answer: no.", "yes", "pubmedqa_labeled") == 0.0


def test_pubmedqa_artificial_tag():
    assert _check_accuracy("maybe.", "maybe", "pubmedqa_artificial") == 1.0


def test_medmcqa_tag():
    gold = "aspirin"
    assert _check_accuracy("The correct treatment is aspirin.", gold, "medmcqa") == 1.0
    assert _check_accuracy("Wrong", gold, "medmcqa") == 0.0
    assert (
        _check_accuracy("Brief rationale.\nAnswer: B — aspirin", gold, "medmcqa", gold_letter="B")
        == 1.0
    )
    assert _check_accuracy("Answer: B", gold, "medmcqa", gold_letter="B") == 1.0


def test_medmcqa_reader_question_includes_options():
    item = {
        "question": "What is the best drug?",
        "dataset": "medmcqa",
        "options": {"A": "aspirin", "B": "ibuprofen"},
    }
    text = format_qa_question_for_reader(item)
    assert "Options:" in text
    assert "A. aspirin" in text


def test_unsupported_claim_rate_is_none_when_nothing_is_cited():
    """
    An answer that cites nothing has no citations to audit, so the rate is
    undefined -- not 0.0. Scoring it 0.0 would reward a reader for citing
    nothing at all, which inverts the metric. Callers aggregate over answers
    that actually cited, and report answer_without_citation_rate beside it.
    """
    assert _unsupported_claim_rate([], [{"id": "x"}], ["x"]) is None


def test_legacy_pubmedqa_medqa_tags():
    assert _check_accuracy("yes", "yes", "pubmedqa") == 1.0
    assert _check_accuracy("x", "y", "medqa") == 0.0


# ---------------------------------------------------------------------------
# Grounding metrics (src/pipeline/rag/metrics.py)
# ---------------------------------------------------------------------------

import pytest

from src.pipeline.rag.metrics import (
    CitationAnnotationError,
    citation_precision,
    citation_recall,
    compute_citation_metrics,
    gold_in_context_any,
    gold_in_context_recall,
    require_citation_support,
    unsupported_citation_rate,
)

PASSAGES = [{"id": "g1"}, {"id": "x2"}, {"id": "g3"}, {"id": "x4"}]
GOLD = ["g1", "g3", "g5"]  # g5 was never retrieved


def test_citation_precision_hand_computed():
    # cites [P1]=g1 (gold) and [P2]=x2 (not gold) -> 1/2
    assert citation_precision(["1", "2"], PASSAGES, GOLD) == pytest.approx(0.5)


def test_citation_recall_hand_computed():
    # cites g1 and g3 of the three gold passages -> 2/3
    assert citation_recall(["1", "3"], PASSAGES, GOLD) == pytest.approx(2 / 3)


def test_unsupported_is_one_minus_precision():
    p = citation_precision(["1", "2"], PASSAGES, GOLD)
    assert unsupported_citation_rate(["1", "2"], PASSAGES, GOLD) == pytest.approx(1 - p)


def test_out_of_range_and_duplicate_citations_are_ignored():
    # [P9] does not exist; [P1] repeated must count once
    assert citation_precision(["1", "1", "9"], PASSAGES, GOLD) == pytest.approx(1.0)


def test_gold_in_context_recall_is_fractional():
    # g1 and g3 present of three gold -> 2/3, NOT 1.0
    assert gold_in_context_recall(PASSAGES, GOLD) == pytest.approx(2 / 3)


def test_gold_in_context_any_is_binary():
    assert gold_in_context_any(PASSAGES, GOLD) == 1.0
    assert gold_in_context_any([{"id": "x2"}], GOLD) == 0.0


def test_fractional_and_binary_context_metrics_differ():
    """The distinction that makes Ctx=0.87 readable: it is not '13% had no evidence'."""
    assert gold_in_context_recall(PASSAGES, GOLD) < gold_in_context_any(PASSAGES, GOLD)


def test_compute_returns_dict_with_all_fields():
    out = compute_citation_metrics(
        ["1", "2"], PASSAGES, GOLD,
        reader_task="yesno", dataset="pubmedqa_labeled",
    )
    for key in ("citation_precision", "citation_recall", "unsupported_citation_rate",
                "gold_in_context_recall", "gold_in_context_any",
                "answer_without_citation", "n_citations"):
        assert key in out
    assert out["answer_without_citation"] == 0.0
    assert out["n_citations"] == 2


def test_uncited_answer_is_flagged_and_unsupported_is_none():
    out = compute_citation_metrics(
        [], PASSAGES, GOLD, reader_task="yesno", dataset="pubmedqa_labeled",
    )
    assert out["answer_without_citation"] == 1.0
    assert out["unsupported_citation_rate"] is None
    assert out["citation_recall"] == 0.0


def test_medmcqa_raises_in_strict_mode_instead_of_silently_skipping():
    """Task 8: a missing column must be deliberate, never an unnoticed gap."""
    with pytest.raises(CitationAnnotationError, match="no passage-level evidence"):
        require_citation_support("mcq", "medmcqa")
    with pytest.raises(CitationAnnotationError):
        compute_citation_metrics(
            ["1"], PASSAGES, GOLD, reader_task="mcq", dataset="medmcqa", strict=True,
        )


def test_medmcqa_returns_nulls_when_not_strict():
    out = compute_citation_metrics(
        ["1"], PASSAGES, GOLD, reader_task="mcq", dataset="medmcqa",
    )
    assert out["citation_precision"] is None
    assert out["citation_recall"] is None
