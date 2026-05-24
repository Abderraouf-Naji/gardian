"""Types for the GARDIAN RAG reader stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ReaderTask(str, Enum):
    YESNO = "yesno"
    MCQ = "mcq"
    OPEN = "open"


# Paper systems: sparse | dense | hybrid | gardian (hybrid pool + GARDIAN rerank + reader).
EVAL_SYSTEMS = ("sparse", "dense", "hybrid", "gardian")

SYSTEM_ALIASES = {
    "bm25": "sparse",
    "spladepp": "sparse",
    "hybrid_rag": "gardian",
    "rag_gardian": "gardian",
}


def normalize_system_name(name: str) -> str:
    key = (name or "").strip().lower()
    return SYSTEM_ALIASES.get(key, key)


@dataclass
class ReaderConfig:
    """Generation and formatting limits — tuned for publishable E2E QA."""

    top_k_passages: int = 5
    max_chars_per_passage: int = 500
    max_input_length: int = 8192
    max_new_tokens_yesno: int = 384
    max_new_tokens_mcq: int = 320
    max_new_tokens_open: int = 512
    require_citations_yesno: bool = True
    require_citations_mcq: bool = True
    repetition_penalty: float = 1.12
    no_repeat_ngram_size: int = 4
    include_signal_features: bool = False
    max_citations_yesno: int = 3
    max_citations_mcq: int = 4
    allow_retry: bool = True

    @classmethod
    def from_cfg(cls, cfg: Any) -> "ReaderConfig":
        q = cfg.qa
        return cls(
            top_k_passages=int(q.get("top_k_passages", 5)),
            max_chars_per_passage=int(q.get("max_chars_per_passage", 500)),
            max_input_length=int(q.get("reader_max_input_length", 8192)),
            max_new_tokens_yesno=int(q.get("max_new_tokens_yesno", 384)),
            max_new_tokens_mcq=int(q.get("max_new_tokens_mcq", 320)),
            max_new_tokens_open=int(q.get("max_new_tokens", 512)),
            require_citations_yesno=bool(q.get("require_citations_yesno", True)),
            require_citations_mcq=bool(q.get("require_citations_mcq", True)),
            repetition_penalty=float(q.get("reader_repetition_penalty", 1.12)),
            no_repeat_ngram_size=int(q.get("reader_no_repeat_ngram_size", 4)),
            include_signal_features=bool(q.get("reader_include_signal_features", False)),
            max_citations_yesno=int(q.get("reader_max_citations_yesno", 3)),
            max_citations_mcq=int(q.get("reader_max_citations_mcq", 4)),
            allow_retry=bool(q.get("reader_allow_retry", True)),
        )


@dataclass
class ParsedAnswer:
    raw: str
    citations: List[str] = field(default_factory=list)
    pubmedqa_label: Optional[str] = None
    mcq_letter: Optional[str] = None
    valid_format: bool = False
    used_retry: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw": self.raw,
            "citations": list(self.citations),
            "pubmedqa_label": self.pubmedqa_label,
            "mcq_letter": self.mcq_letter,
            "valid_format": self.valid_format,
            "used_retry": self.used_retry,
        }
