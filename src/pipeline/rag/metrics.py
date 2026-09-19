"""
Grounding metrics for end-to-end RAG answers.

Every metric is defined here once, with its exact formula, because the paper's
Section 4.4 promises these numbers and reviewers asked what they mean.

Notation for one question:
  P            passages shown to the reader, in rank order (P[0] is [P1])
  G            set of gold passage ids annotated as evidence for the question
  C            set of DISTINCT passage indices the answer cites via [P#] tags,
               restricted to indices that actually exist in P
  C_gold       {p in C : p in G}

  citation_precision       = |C_gold| / |C|            (None when C is empty)
  citation_recall          = |C_gold| / |G|
  unsupported_citation_rate= 1 - |C_gold| / |C|        (None when C is empty)
  gold_in_context_recall   = |P ∩ G| / |G|             ("Ctx" in the paper)
  gold_in_context_any      = 1 if P ∩ G else 0
  answer_without_citation  = 1 if C is empty else 0

Two definitions deserve care.

``unsupported_citation_rate`` is **None**, not 0.0, when the answer cites
nothing. Scoring an uncited answer as 0.0 unsupported rewards a reader for
citing nothing at all, which inverts the metric. Callers must aggregate over
answers that actually made a citation, and report
``answer_without_citation_rate`` alongside so abstention stays visible.

``gold_in_context_recall`` is a fraction, so a value of 0.87 means "87% of the
gold passages were retrieved on average", NOT "13% of questions had no
evidence". ``gold_in_context_any`` is the binary companion that answers the
second question, and the two differ sharply: on PubMedQA-labeled the fractional
value is ~0.87 while at least one gold passage is present ~99% of the time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.pipeline.rag.parser import extract_citations


class CitationAnnotationError(ValueError):
    """Raised when citation metrics are requested but the data cannot support them."""


# Datasets whose annotation identifies which passages are evidence.
CITATION_ANNOTATED_DATASETS = frozenset(
    {"pubmedqa", "pubmedqa_labeled", "pubmedqa_artificial"}
)


def citation_metrics_applicable(reader_task: str, dataset: str) -> bool:
    """
    Whether *dataset* annotates which passages are evidence for an answer.

    MedMCQA ships explanations rather than passage-level evidence annotations,
    so citation precision/recall are undefined there and are not reported.
    """
    rt = (reader_task or "").strip().lower()
    ds = (dataset or "").strip().lower()
    if rt == "mcq" or ds in ("medmcqa", "medqa"):
        return False
    return ds in CITATION_ANNOTATED_DATASETS or rt == "yesno"


def require_citation_support(reader_task: str, dataset: str) -> None:
    """
    Fail loudly when citation metrics are requested for an unsupported dataset.

    Reviewers asked why grounding metrics appear only for PubMedQA. The answer
    is that MedMCQA has no passage-level evidence annotation; this raises rather
    than silently emitting nulls, so a missing column is always a deliberate
    choice and never an unnoticed gap.
    """
    if not citation_metrics_applicable(reader_task, dataset):
        raise CitationAnnotationError(
            f"Citation metrics are not defined for dataset={dataset!r} "
            f"(reader_task={reader_task!r}): it has no passage-level evidence "
            "annotation. Report accuracy and gold-context recall instead, or "
            "pass a citation-annotated dataset "
            f"({sorted(CITATION_ANNOTATED_DATASETS)})."
        )


def _resolve_cited_indices(
    cited_idxs: Sequence[str], passages: Sequence[dict]
) -> List[int]:
    """Distinct, in-range, 0-based passage indices referenced by [P#] tags."""
    out: List[int] = []
    for raw in cited_idxs:
        try:
            idx = int(raw) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(passages) and idx not in out:
            out.append(idx)
    return out


def citation_precision(
    cited_idxs: Sequence[str], passages: Sequence[dict], gold_ids: Sequence[str]
) -> Optional[float]:
    """|C_gold| / |C|. None when the answer cites nothing."""
    resolved = _resolve_cited_indices(cited_idxs, passages)
    if not resolved:
        return None
    gold = set(gold_ids)
    return sum(1 for i in resolved if passages[i]["id"] in gold) / len(resolved)


def citation_recall(
    cited_idxs: Sequence[str], passages: Sequence[dict], gold_ids: Sequence[str]
) -> float:
    """|C_gold| / |G|. Zero when the answer cites no gold passage."""
    gold = set(gold_ids)
    if not gold:
        return 0.0
    resolved = _resolve_cited_indices(cited_idxs, passages)
    cited_gold = {passages[i]["id"] for i in resolved if passages[i]["id"] in gold}
    return len(cited_gold) / len(gold)


def unsupported_citation_rate(
    cited_idxs: Sequence[str], passages: Sequence[dict], gold_ids: Sequence[str]
) -> Optional[float]:
    """
    1 - precision: the share of citations that do not point at gold evidence.

    None when the answer cites nothing -- see the module docstring for why this
    is not 0.0.
    """
    precision = citation_precision(cited_idxs, passages, gold_ids)
    return None if precision is None else 1.0 - precision


# Retained for backwards compatibility with existing result files.
unsupported_claim_rate = unsupported_citation_rate


def gold_in_context_recall(
    passages: Sequence[dict], gold_ids: Sequence[str]
) -> Optional[float]:
    """|P ∩ G| / |G| -- the "Ctx" column. None when the question has no gold set."""
    gold = set(gold_ids)
    if not gold:
        return None
    return len({p["id"] for p in passages if p["id"] in gold}) / len(gold)


def gold_in_context_any(
    passages: Sequence[dict], gold_ids: Sequence[str]
) -> Optional[float]:
    """1.0 if at least one gold passage is in context. None when no gold set."""
    gold = set(gold_ids)
    if not gold:
        return None
    return 1.0 if any(p["id"] in gold for p in passages) else 0.0


def compute_citation_metrics(
    cited_idxs: Sequence[str],
    passages: Sequence[dict],
    gold_ids: Sequence[str],
    *,
    reader_task: str,
    dataset: str,
    answer_text: str = "",
    strict: bool = False,
) -> Dict[str, Any]:
    """
    All grounding metrics for one answer.

    Returns a dict rather than a tuple so new metrics cannot silently shift
    positional call sites. Values are None where the metric is undefined for
    this question, and callers must skip Nones when averaging.

    ``strict=True`` raises :class:`CitationAnnotationError` instead of returning
    an all-None block when the dataset carries no evidence annotation.
    """
    # Gold-in-context is a RETRIEVAL diagnostic: it asks whether the passages
    # the retriever put in front of the reader contain the annotated evidence,
    # and says nothing about what the answer cited. It is therefore defined
    # wherever gold_passage_ids exist -- MedMCQA included, whose single gold
    # explanation is a perfectly good retrieval target even though it carries
    # no passage-level *citation* annotation. Bundling it behind the citation
    # gate reported it as null on MedMCQA, which is the Hit@k reviewers asked
    # for on the one dataset where first-stage retrieval was blamed.
    grounding = {
        "gold_in_context_recall": gold_in_context_recall(passages, gold_ids),
        "gold_in_context_any": gold_in_context_any(passages, gold_ids),
    }
    empty = {
        "citation_precision": None,
        "citation_recall": None,
        "unsupported_citation_rate": None,
        "answer_without_citation": None,
        "n_citations": 0,
        **grounding,
    }

    if not citation_metrics_applicable(reader_task, dataset):
        if strict:
            require_citation_support(reader_task, dataset)
        return dict(empty)

    if not gold_ids:
        if strict:
            raise CitationAnnotationError(
                f"dataset={dataset!r} supports citation metrics but this question "
                "carries no gold_passage_ids; cannot score grounding."
            )
        return dict(empty)

    unique = extract_citations(answer_text, unique=True) if answer_text else list(
        dict.fromkeys(cited_idxs)
    )
    resolved = _resolve_cited_indices(unique, passages)

    return {
        "citation_precision": citation_precision(unique, passages, gold_ids),
        "citation_recall": citation_recall(unique, passages, gold_ids),
        "unsupported_citation_rate": unsupported_citation_rate(
            unique, passages, gold_ids
        ),
        **grounding,
        "answer_without_citation": 0.0 if resolved else 1.0,
        "n_citations": len(resolved),
    }
