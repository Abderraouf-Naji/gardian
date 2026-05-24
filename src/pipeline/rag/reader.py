"""GARDIAN RAG reader — single entry point for live / offline QA."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from src.pipeline.rag.context import format_passages
from src.pipeline.rag.generation import format_chat_prompt, generate_text
from src.pipeline.rag.parser import parse_answer, validate_answer, yesno_fallback_with_citation
from src.pipeline.rag.prompts import build_prompt, build_retry_prompt
from src.pipeline.rag.reader_types import ParsedAnswer, ReaderConfig, ReaderTask


def reader_task_from_name(name: str) -> ReaderTask:
    key = (name or "open").strip().lower()
    if key == "yesno":
        return ReaderTask.YESNO
    if key == "mcq":
        return ReaderTask.MCQ
    return ReaderTask.OPEN


def _task_limits(task: ReaderTask, cfg: ReaderConfig) -> Dict[str, Any]:
    if task == ReaderTask.YESNO:
        return {
            "top_k": cfg.top_k_passages,
            "max_new_tokens": cfg.max_new_tokens_yesno,
            "require_citations": cfg.require_citations_yesno,
            "max_citations": cfg.max_citations_yesno,
        }
    if task == ReaderTask.MCQ:
        return {
            "top_k": cfg.top_k_passages,
            "max_new_tokens": cfg.max_new_tokens_mcq,
            "require_citations": cfg.require_citations_mcq,
            "max_citations": cfg.max_citations_mcq,
        }
    return {
        "top_k": cfg.top_k_passages,
        "max_new_tokens": cfg.max_new_tokens_open,
        "require_citations": False,
        "max_citations": 0,
    }


class RAGReader:
    """Run retrieval-conditioned generation with contract enforcement + one retry."""

    def __init__(
        self,
        tokenizer,
        reader_model,
        device: str,
        config: Optional[ReaderConfig] = None,
    ):
        self.tokenizer = tokenizer
        self.reader_model = reader_model
        self.device = device
        self.config = config or ReaderConfig()

    def run(
        self,
        *,
        question: str,
        passages: List[Dict[str, Any]],
        task: ReaderTask,
        routing_note: str = "",
    ) -> ParsedAnswer:
        limits = _task_limits(task, self.config)
        top_k = min(int(limits["top_k"]), len(passages))
        if top_k < 1:
            if task == ReaderTask.YESNO:
                return parse_answer(
                    yesno_fallback_with_citation(1),
                    task,
                    require_citations=limits["require_citations"],
                    n_passages=0,
                    max_citations=int(limits.get("max_citations") or 0),
                )
            return ParsedAnswer(raw="Answer: UNSURE — I don't know", valid_format=False)

        context = format_passages(
            passages,
            top_k=top_k,
            max_chars_per_passage=self.config.max_chars_per_passage,
            include_signal_features=self.config.include_signal_features,
        )
        sys_p, usr_p = build_prompt(
            task=task,
            question=question,
            context=context,
            routing_note=routing_note,
        )
        prompt = format_chat_prompt(self.tokenizer, self.reader_model, sys_p, usr_p)
        raw = generate_text(
            self.tokenizer,
            self.reader_model,
            prompt,
            max_new_tokens=limits["max_new_tokens"],
            max_input_length=self.config.max_input_length,
            repetition_penalty=self.config.repetition_penalty,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
        )
        max_cites = int(limits.get("max_citations") or 0)
        parsed = parse_answer(
            raw,
            task,
            require_citations=limits["require_citations"],
            n_passages=top_k,
            max_citations=max_cites,
        )
        if parsed.valid_format:
            return parsed

        if not self.config.allow_retry:
            return parsed

        sys_r, usr_r = build_retry_prompt(task=task, question=question, context=context)
        prompt_r = format_chat_prompt(self.tokenizer, self.reader_model, sys_r, usr_r)
        raw_r = generate_text(
            self.tokenizer,
            self.reader_model,
            prompt_r,
            max_new_tokens=limits["max_new_tokens"],
            max_input_length=self.config.max_input_length,
            repetition_penalty=self.config.repetition_penalty,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
        )
        parsed_r = parse_answer(
            raw_r,
            task,
            require_citations=limits["require_citations"],
            n_passages=top_k,
            max_citations=max_cites,
            used_retry=True,
        )
        if parsed_r.valid_format:
            return parsed_r

        if task == ReaderTask.YESNO:
            return parse_answer(
                yesno_fallback_with_citation(1),
                task,
                require_citations=limits["require_citations"],
                n_passages=top_k,
                max_citations=max_cites,
                used_retry=True,
            )
        if task == ReaderTask.MCQ:
            return ParsedAnswer(
                raw="The passages do not support any listed option.\n\nAnswer: UNSURE — I don't know",
                valid_format=False,
                used_retry=True,
            )
        return parsed_r


def run_rag_reader(
    *,
    question: str,
    passages_top_k: List[Dict[str, Any]],
    tokenizer,
    reader_model,
    device: str,
    reader_task: str = "open",
    config: Optional[ReaderConfig] = None,
    routing_note: str = "",
    **kwargs: Any,
) -> str:
    """Backward-compatible: return answer string only."""
    _ = kwargs
    task = reader_task_from_name(reader_task)
    reader = RAGReader(tokenizer, reader_model, device, config=config)
    return reader.run(
        question=question,
        passages=passages_top_k,
        task=task,
        routing_note=routing_note,
    ).raw
