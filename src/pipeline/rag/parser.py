"""Parse and validate reader outputs."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from src.pipeline.rag.reader_types import ParsedAnswer, ReaderTask

_CITE_RE = re.compile(r"\[P(\d+)\]", re.IGNORECASE)
_YESNO_VERDICT_RE = re.compile(
    r"(?i)\banswer\s*:\s*(yes|no|maybe)\b"
)
_MCQ_VERDICT_RE = re.compile(
    r"(?i)^answer\s*:\s*([A-D])\s*[—\-–]\s*(.+?)\s*$",
    re.MULTILINE,
)
_MCQ_LETTER_LINE_RE = re.compile(r"(?i)\banswer\s*:\s*([A-D])\b")
_MCQ_UNSURE_RE = re.compile(r"(?i)\banswer\s*:\s*unsure\b")
_MCQ_OPTION_LETTER_RE = re.compile(
    r"(?i)\b(?:option|choice)\s+([A-D])\b|(?:choose|select|pick)\s+(?:option\s+)?([A-D])\b"
)
_IDK_RE = re.compile(r"(?i)\bi\s+don'?t\s+know\b")


def extract_citations(text: str, *, unique: bool = False) -> List[str]:
    """Return passage indices cited as [P#]. With ``unique=True``, first occurrence order."""
    found = _CITE_RE.findall(text or "")
    if not unique:
        return found
    seen: set[str] = set()
    out: List[str] = []
    for idx in found:
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def extract_pubmedqa_label(text: str) -> Optional[str]:
    spans = list(_YESNO_VERDICT_RE.finditer(text or ""))
    if spans:
        return spans[-1].group(1).lower()
    last = (text or "").strip().splitlines()[-1].strip().lower()
    m = re.match(r"^(yes|no|maybe)[\s.!?,;:]*$", last)
    return m.group(1) if m else None


def is_mcq_unsure(text: str) -> bool:
    return bool(_MCQ_UNSURE_RE.search(text or ""))


def extract_mcq_letter(text: str) -> Optional[str]:
    spans = list(_MCQ_VERDICT_RE.finditer(text or ""))
    if spans:
        return spans[-1].group(1).upper()
    lines = list(_MCQ_LETTER_LINE_RE.finditer(text or ""))
    if lines:
        return lines[-1].group(1).upper()
    last: Optional[str] = None
    for m in _MCQ_OPTION_LETTER_RE.finditer(text or ""):
        letter = m.group(1) or m.group(2)
        if letter:
            last = letter.upper()
    return last


def normalize_mcq_raw(text: str) -> str:
    """Add a scorable final line when the model states a letter without ``Answer:``."""
    t = (text or "").strip()
    if not t:
        return t
    if is_mcq_unsure(t):
        return t if _MCQ_UNSURE_RE.search(t) else f"{t}\n\nAnswer: UNSURE"
    letter = extract_mcq_letter(t)
    if letter and not _MCQ_LETTER_LINE_RE.search(t):
        return f"{t}\n\nAnswer: {letter}"
    return t


def truncate_at_final_answer(text: str, task: ReaderTask) -> str:
    """Keep content through the first well-formed final answer line (drops rambling tail)."""
    raw = (text or "").strip()
    if not raw:
        return raw
    if task == ReaderTask.MCQ:
        for m in _MCQ_VERDICT_RE.finditer(raw):
            return raw[: m.end()].strip()
        m = re.search(r"(?i)\banswer\s*:\s*[A-D]\b.*", raw)
        if m:
            return raw[: m.end()].strip()
    if task == ReaderTask.YESNO:
        spans = list(_YESNO_VERDICT_RE.finditer(raw))
        if spans:
            return raw[: spans[-1].end()].strip()
    return raw


def validate_answer(
    text: str,
    task: ReaderTask,
    *,
    require_citations: bool,
    n_passages: int,
    max_citations: int = 0,
) -> Tuple[bool, List[str]]:
    """Return (ok, reasons)."""
    reasons: List[str] = []
    t = (text or "").strip()
    if not t:
        return False, ["empty"]
    if n_passages > 0 and _IDK_RE.search(t) and task != ReaderTask.OPEN:
        if task == ReaderTask.MCQ:
            if not is_mcq_unsure(t) and extract_mcq_letter(t) is None:
                reasons.append("i_dont_know")
        else:
            reasons.append("i_dont_know")
    cites = extract_citations(t)
    unique_cites = extract_citations(t, unique=True)
    if require_citations and not cites:
        if not (task == ReaderTask.MCQ and is_mcq_unsure(t)):
            reasons.append("missing_citations")
    if max_citations > 0 and len(unique_cites) > max_citations:
        reasons.append("too_many_citations")
    if task == ReaderTask.YESNO:
        if extract_pubmedqa_label(t) is None:
            reasons.append("missing_yesno_verdict")
    elif task == ReaderTask.MCQ:
        if not is_mcq_unsure(t) and extract_mcq_letter(t) is None:
            reasons.append("missing_mcq_verdict")
    return (len(reasons) == 0), reasons


def parse_answer(
    text: str,
    task: ReaderTask,
    *,
    require_citations: bool,
    n_passages: int,
    max_citations: int = 0,
    used_retry: bool = False,
) -> ParsedAnswer:
    raw_in = normalize_mcq_raw(text) if task == ReaderTask.MCQ else (text or "")
    trimmed = truncate_at_final_answer(raw_in, task)
    ok, _ = validate_answer(
        trimmed,
        task,
        require_citations=require_citations,
        n_passages=n_passages,
        max_citations=max_citations,
    )
    return ParsedAnswer(
        raw=trimmed,
        citations=extract_citations(trimmed, unique=True),
        pubmedqa_label=extract_pubmedqa_label(trimmed) if task == ReaderTask.YESNO else None,
        mcq_letter=extract_mcq_letter(trimmed) if task == ReaderTask.MCQ else None,
        valid_format=ok,
        used_retry=used_retry,
    )


def yesno_fallback_with_citation(passage_index: int = 1) -> str:
    """Last resort when the model fails twice — still scorable."""
    p = max(1, min(passage_index, 10))
    return (
        f"The passages do not give conclusive evidence for a firm yes or no [{p}].\n\n"
        "Answer: maybe"
    )
