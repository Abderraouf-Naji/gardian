"""
Canonical question-type ordering for GARDIAN.

``qtype_onehot`` in rank JSONL **must** use the same axis order as
``cfg.model.question_types`` and ``ControllerMLP`` input width. This module is
the single source of truth (matches ``scripts/03_generate_rank_data.py``).
"""

from __future__ import annotations

from typing import List, Sequence

# Index i corresponds to dimension i of qtype_onehot and row i of the
# question-type slice concatenated to the query embedding in the controller.
ORDERED_QUESTION_TYPES: tuple[str, ...] = (
    "diagnosis",
    "treatment",
    "mechanism",
    "contraindication",
    "factoid",
    "yesno",
    "other",
)

QTYPE_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(ORDERED_QUESTION_TYPES)}
N_QTYPES: int = len(ORDERED_QUESTION_TYPES)


def normalize_question_type(qtype: str | None) -> str:
    """Map free-text or coarse labels onto the closed label set."""
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


def qtype_onehot(normalized_qtype: str) -> List[float]:
    """One-hot vector aligned with ``ORDERED_QUESTION_TYPES``."""
    key = normalize_question_type(normalized_qtype)
    v = [0.0] * N_QTYPES
    v[QTYPE_TO_IDX.get(key, QTYPE_TO_IDX["other"])] = 1.0
    return v


def assert_cfg_question_types(cfg_types: Sequence[str]) -> None:
    """Fail fast if config order drifts from serialized training data."""
    got = tuple(str(x).lower() for x in cfg_types)
    if got != ORDERED_QUESTION_TYPES:
        raise ValueError(
            "cfg.model.question_types must exactly match rank-data one-hot layout.\n"
            f"  expected: {list(ORDERED_QUESTION_TYPES)}\n"
            f"  got:      {list(got)}\n"
            "Regenerate rank JSONL after any change, or restore the canonical order "
            "in configs/base.yaml."
        )
