"""
Question-type labels: an **analysis dimension**, never a model input.

Question type is used only for reporting -- the per-type nDCG breakdown and the
alpha-by-type distribution. It is derived from the question text at report time,
is not stored in rank data, and is not seen by the model.

The one-hot that used to condition the controller was removed: it is constant on
both PubMedQA splits (100% ``yesno``) and separates PubMedQA from MedMCQA
perfectly in the combined training set, so it functioned as a dataset indicator
rather than a query signal. Measurements: ``docs/QUESTION_TYPE.md``; regenerate
them with ``scripts/analyze_question_types.py``.

Categories are configured under ``evaluation.question_types`` in
``configs/base.yaml``.
"""

from __future__ import annotations

from typing import Sequence

# Reporting order for per-type tables and plots.
ORDERED_QUESTION_TYPES = (
    "diagnosis",
    "treatment",
    "mechanism",
    "contraindication",
    "factoid",
    "yesno",
    "other",
)

N_QTYPES = len(ORDERED_QUESTION_TYPES)

QTYPE_TO_IDX = {name: i for i, name in enumerate(ORDERED_QUESTION_TYPES)}

def normalize_question_type(qtype: str | None) -> str:
    """
    Map a free-text or coarse label onto the closed reporting label set.

    Substring matching is deliberate: the source datasets carry labels such as
    "Diagnosis question" or "management", which must land on the same category
    as the bare keyword. Anything unrecognised becomes ``"other"`` rather than
    raising, so a new dataset never breaks a reporting run.
    """
    if not qtype:
        return "other"
    q = qtype.strip().lower()
    if q in QTYPE_TO_IDX:
        return q
    if "diagnos" in q:
        return "diagnosis"
    if "treat" in q or "therapy" in q or "management" in q:
        return "treatment"
    if "mechan" in q or "cause" in q or "pathoph" in q:
        return "mechanism"
    if "contra" in q or "interaction" in q:
        return "contraindication"
    if "fact" in q or "definition" in q:
        return "factoid"
    if "yes" in q or "no" in q:
        return "yesno"
    return "other"


def assert_cfg_question_types(cfg_types: Sequence[str]) -> None:
    """
    Check that the configured reporting categories match this module's order.

    Per-type tables are indexed positionally, so a mismatch would silently
    relabel rows.
    """
    configured = tuple(str(t) for t in cfg_types)
    if configured != ORDERED_QUESTION_TYPES:
        raise ValueError(
            "evaluation.question_types must match src.common.question_types."
            "ORDERED_QUESTION_TYPES exactly (order included).\n"
            f"  configured: {configured}\n"
            f"  expected:   {ORDERED_QUESTION_TYPES}"
        )
