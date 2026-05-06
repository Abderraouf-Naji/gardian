from src.common.question_types import (
    N_QTYPES,
    ORDERED_QUESTION_TYPES,
    assert_cfg_question_types,
    normalize_question_type,
    qtype_onehot,
)


def test_onehot_length_and_axis():
    assert len(qtype_onehot("diagnosis")) == N_QTYPES == len(ORDERED_QUESTION_TYPES)
    assert sum(qtype_onehot("mechanism")) == 1.0


def test_normalize_heuristic():
    assert normalize_question_type("Diagnosis question") == "diagnosis"
    assert normalize_question_type("") == "other"


def test_assert_cfg_ok():
    assert_cfg_question_types(list(ORDERED_QUESTION_TYPES))


def test_assert_cfg_rejects_wrong_order():
    bad = ["diagnosis", "treatment", "mechanism", "contraindication", "factoid", "other", "yesno"]
    try:
        assert_cfg_question_types(bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError for swapped yesno/other")
