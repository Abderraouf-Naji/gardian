"""
RAG reader prompts -- strict, task-specific contracts.

Two yes/no variants are available, selected by ``qa.yesno_prompt`` in
``configs/base.yaml`` and measurable head-to-head with
``scripts/ablate_reader_prompt.py`` (which freezes the retrieved passages so the
prompt is the only variable):

``v1_permissive`` (default)
    The prompt used for the CoopIS submission, and still the default because it
    is the only variant with a completed 300-question measurement: 0.5333
    against the "decisive" rewrite's 0.5300. Its "maybe" clause -- "the
    evidence is insufficient, mixed, or the key question is not addressed" --
    licenses hedging almost anywhere. Measured effect on PubMedQA-labeled:
    25% of gold-yes/no questions answered "maybe" with Llama-3-8B and 44% with
    Qwen2.5-14B, against a gold "maybe" rate of 11%. Both readers land at or
    below the 55.2% majority-class baseline.

``v2_calibrated``
    Narrows "maybe" to genuine conflict or an explicitly inconclusive study,
    states the label prior, and names the specific failure ("a null result is
    'no', not 'maybe'"). Also asks for full citation coverage, because readers
    emit ~1.75 citations against ~3.4 gold passages, which is what caps
    citation recall at ~0.48.

Keep both: the ablation is part of the paper, so the submitted prompt must stay
reproducible.
"""

from __future__ import annotations
from typing import Tuple
from src.pipeline.rag.reader_types import ReaderTask

SYSTEM_YESNO_V1_PERMISSIVE = """You are answering a biomedical research question using ONLY the passages below.
The passages are labeled [P1] through [P10] in the order they were ranked.

Instructions:
1. Read the question carefully.
2. Use all passages that are relevant to the question; ignore only those that are
   completely off-topic.
3. Write 2–3 sentences summarising the evidence. Cite every passage that supports
   each sentence using [P#] tags — a single sentence may carry multiple citations
   (e.g. "X has been shown [P2][P5].").
4. Last line MUST be exactly one of:
     Answer: yes
     Answer: no
     Answer: maybe

When to choose each answer:
- yes  : the passages collectively support the claim in the question.
- no   : the passages contradict the claim or consistently show no effect.
- maybe: the evidence is insufficient, mixed, or the key question is not
         addressed by the passages.

Do not say "I don't know". Do not cite a passage you did not use.
Do not fabricate [P#] tags."""

SYSTEM_YESNO_V2_CALIBRATED = """You are answering a biomedical research question using ONLY the passages below.
The passages are labeled [P1] onward in the order they were ranked.

Step 1 - Evidence. Write 2-4 sentences summarising what the passages show.
Cite EVERY passage that supports a sentence with [P#] tags; one sentence may carry
several (e.g. "X was observed [P2][P5][P7]."). Every passage you relied on must
appear at least once.

Step 2 - Direction. In one short sentence, state which way the evidence leans:
supporting the claim, contradicting it, or genuinely split.

Step 3 - Verdict. Last line MUST be exactly one of:
     Answer: yes
     Answer: no
     Answer: maybe

How to decide:
- yes  : the evidence leans toward supporting the claim.
- no   : the evidence leans against the claim, or shows no significant effect.
         A null or negative result is "no". It is NOT "maybe".
- maybe: ONLY when the passages directly conflict with each other, or the study
         itself reports an inconclusive result. Roughly 1 question in 10.

Incomplete or indirect evidence is not a reason for "maybe". If the passages lean
one way at all, commit to that direction. Reserve "maybe" for true conflict.

Do not say "I don't know". Do not cite a passage you did not use.
Do not fabricate [P#] tags."""

YESNO_PROMPTS = {
    "v1_permissive": SYSTEM_YESNO_V1_PERMISSIVE,
    "v2_calibrated": SYSTEM_YESNO_V2_CALIBRATED,
}

DEFAULT_YESNO_PROMPT = "v1_permissive"


def yesno_system_prompt(variant: str | None = None) -> str:
    """Resolve the configured yes/no system prompt by name."""
    key = (variant or DEFAULT_YESNO_PROMPT).strip()
    if key not in YESNO_PROMPTS:
        raise ValueError(
            f"Unknown qa.yesno_prompt={variant!r}; choose from {sorted(YESNO_PROMPTS)}"
        )
    return YESNO_PROMPTS[key]


# Backwards-compatible alias for callers that import the symbol directly.
SYSTEM_YESNO = SYSTEM_YESNO_V1_PERMISSIVE


SYSTEM_MCQ = """You are answering a multiple-choice biomedical question using ONLY the passages below.
The passages are labeled [P1] through [P10] in the order they were ranked.

Instructions:
1. Read the question and all options (A, B, C, D).
2. Identify the option best supported by the passages.
3. Write 2–4 sentences of reasoning. Cite every passage you draw on with [P#] tags;
   a single sentence may carry multiple citations.
4. Last line MUST be exactly:
     Answer: <LETTER> — <copy the full option text verbatim>

If no option is supported by any passage, use:
     Answer: UNSURE

Do not discuss every passage. Do not fabricate [P#] tags."""

SYSTEM_OPEN = """You are answering a biomedical question using ONLY the passages below.
The passages are labeled [P1] through [P10] in the order they were ranked.

Instructions:
1. Use all passages relevant to the question; ignore only those that are completely
   off-topic.
2. Write a concise answer. Cite every claim with [P#] tags; a single sentence may
   carry multiple citations.
3. Do not make any claim that is not supported by at least one passage.
4. Do not fabricate [P#] tags."""


def build_prompt(
    *,
    task: ReaderTask,
    question: str,
    context: str,
    routing_note: str = "",
    yesno_prompt: str | None = None,
) -> Tuple[str, str]:
    route = (
        f"\n\nRetrieval note (for your reference only): {routing_note}"
        if routing_note
        else ""
    )
    if task == ReaderTask.MCQ:
        system = f"{SYSTEM_MCQ}{route}"
        user = (
            f"Passages:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Give brief reasoning with [P#] tags, then a final line "
            "Answer: <LETTER> — <full option text>.\n"
            "Answer:"
        )
        return system, user
    if task == ReaderTask.YESNO:
        system = f"{yesno_system_prompt(yesno_prompt)}{route}"
        user = (
            f"Passages:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )
        return system, user
    system = f"{SYSTEM_OPEN}{route}"
    user = f"Passages:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    return system, user


def build_retry_prompt(
    *,
    task: ReaderTask,
    question: str,
    context: str,
) -> Tuple[str, str]:
    """Minimal second pass when the first answer violated the output contract.
    Fixes only the format violation; does not re-impose structural constraints
    that were already satisfied in the first pass."""
    if task == ReaderTask.YESNO:
        system = (
            "Use only the passages. Provide brief reasoning with [P#] citations. "
            "End with exactly one line: Answer: yes, Answer: no, or Answer: maybe."
        )
        user = f"Passages:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        return system, user
    if task == ReaderTask.MCQ:
        system = (
            "Use only the passages. Provide brief reasoning with [P#] citations. "
            "Last line must be: Answer: <LETTER> — <verbatim option text>, "
            "or Answer: UNSURE if no option is supported."
        )
        user = f"Passages:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        return system, user
    return build_prompt(task=task, question=question, context=context)