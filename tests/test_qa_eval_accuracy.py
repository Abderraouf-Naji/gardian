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


def test_unsupported_claim_rate_no_citations_is_zero():
    assert _unsupported_claim_rate([], [{"id": "x"}], ["x"]) == 0.0


def test_legacy_pubmedqa_medqa_tags():
    assert _check_accuracy("yes", "yes", "pubmedqa") == 1.0
    assert _check_accuracy("x", "y", "medqa") == 0.0
