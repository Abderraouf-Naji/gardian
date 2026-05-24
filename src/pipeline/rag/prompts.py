"""RAG reader prompts — strict, task-specific contracts."""

from __future__ import annotations
from typing import Tuple
from src.pipeline.rag.reader_types import ReaderTask

SYSTEM_YESNO = """You are answering a biomedical research question using ONLY the passages below.
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
        system = f"{SYSTEM_YESNO}{route}"
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