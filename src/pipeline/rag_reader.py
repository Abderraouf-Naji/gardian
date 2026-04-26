"""
Reader (RAG) — final block in the GARDIAN architecture.

Upstream (already in the repo):
  * Hybrid retrieval + KG features → candidate pool (~300–400 passages).
  * ``GARDIAN.forward`` fuses branch scores with controller weights
    :math:`(\\alpha_s, \\alpha_d, \\alpha_g)` → relevance score per passage;
    sort by score → **top-k passages**.

This module is the **last block**: it takes the **query** and **GARDIAN-ordered
top-k passages**, builds the LLM context, runs the **medical reader** (causal
LM), and returns the **answer string** (your eval code may then parse citations
like ``[P1]``, ``[P2]`` from that text).

See also: ``src.model.gardian.GARDIAN`` (fusion + weights) and
``configs/base.yaml`` keys ``qa.reader_model``, ``qa.top_k_passages``,
``qa.max_new_tokens``.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import torch

SYSTEM_PROMPT = """You are a helpful medical assistant. Answer the question using ONLY
the provided passages. After your answer, cite the passage IDs you used in brackets like [P1],[P2].
If you cannot answer from the passages, say "I don't know".

When passage metadata includes dense/sparse/kg evidence scores, use them as reliability signals:
- sparse: lexical overlap / exact terminology
- dense: semantic similarity / paraphrase match
- kg: medical entity-graph support
Prefer passages whose evidence profile matches the query intent."""

REACT_TOOL_PROTOCOL = """
You are an **agentic ReAct** medical reader. You may only use information that appears in **tool observations** below (passages are revealed via tools).

## Loop
Each turn you MUST write:
1) A short **Thought:** (one or two sentences)
2) Exactly one line **Action:** chosen from:
   - `Action: READ_PASSAGE[n]` — read full text of passage *n* (1-based, e.g. READ_PASSAGE[2] for [P2])
   - `Action: LIST_SIGNALS` — see score metadata for all indexed passages
   - `Action: FINAL` — you are ready to answer

After you choose READ_PASSAGE or LIST_SIGNALS, stop and wait; the user message will append an **Observation:** with the tool result.

When you choose FINAL, on the **next lines** give your **final answer** only (no more actions). End with citations like [P1],[P2] using only passages you actually read. If evidence is insufficient, say "I don't know".

Rules:
- Do not invent facts not supported by observations.
- Prefer passages whose evidence profile matches the routing note (if any).
"""


def build_query_routing_note(
    *,
    question_type: Optional[str],
    alpha_sparse: Optional[float],
    alpha_dense: Optional[float],
    alpha_kg: Optional[float],
) -> str:
    """
    Build a query-adaptive routing note for the reader.
    """
    if alpha_sparse is None or alpha_dense is None or alpha_kg is None:
        return "Routing note: no explicit branch weights provided."
    triples = [
        ("sparse", float(alpha_sparse)),
        ("dense", float(alpha_dense)),
        ("kg", float(alpha_kg)),
    ]
    primary = sorted(triples, key=lambda x: x[1], reverse=True)[0][0]
    qtype = (question_type or "other").strip().lower()
    return (
        f"Routing note: question_type={qtype}; controller_weights="
        f"(sparse={alpha_sparse:.3f}, dense={alpha_dense:.3f}, kg={alpha_kg:.3f}); "
        f"primary_signal={primary}. Prefer passages with stronger {primary} evidence."
    )


def format_reader_context(
    passages: List[Dict],
    top_k: int = 5,
    max_chars_per_passage: int = 600,
    include_signal_features: bool = True,
) -> str:
    """
    Turn top-k retrieved passages into a single context block for the reader.

    Each passage dict must include ``id`` and ``text`` (same shape as hybrid
    retriever output, optionally after GARDIAN adds ``gardian_score`` /
    ``ctrl_weights`` — those fields are ignored here).
    """
    lines: List[str] = []
    for i, p in enumerate(passages[:top_k]):
        pid = p.get("id", f"doc_{i}")
        text = (p.get("text") or "")[:max_chars_per_passage]
        if include_signal_features:
            signal_bits = []
            for key in (
                "gardian_score",
                "sparse_contribution",
                "dense_contribution",
                "kg_contribution",
                "bm25_score",
                "dense_score",
                "biobert_score",
                "doc2query_score",
            ):
                if key in p:
                    try:
                        signal_bits.append(f"{key}={float(p.get(key, 0.0)):.4f}")
                    except (TypeError, ValueError):
                        continue
            if signal_bits:
                lines.append(f"[P{i + 1}] (id={pid}; {'; '.join(signal_bits)}): {text}")
            else:
                lines.append(f"[P{i + 1}] (id={pid}): {text}")
        else:
            lines.append(f"[P{i + 1}] (id={pid}): {text}")
    return "\n\n".join(lines)


def _compact_passage_index(passages: List[Dict], top_k: int, preview_chars: int = 220) -> str:
    """Short [P1]..[Pk] lines for ReAct indexing (full text via READ_PASSAGE)."""
    lines: List[str] = []
    for i, p in enumerate(passages[:top_k]):
        pid = p.get("id", f"doc_{i}")
        text = (p.get("text") or "")[:preview_chars].replace("\n", " ")
        lines.append(f"[P{i + 1}] id={pid} preview: {text}")
    return "\n".join(lines)


def _signals_one_liner(passages: List[Dict], top_k: int) -> str:
    parts: List[str] = []
    for i, p in enumerate(passages[:top_k]):
        bits = []
        for key in (
            "gardian_score",
            "sparse_contribution",
            "dense_contribution",
            "kg_contribution",
            "bm25_score",
            "dense_score",
            "biobert_score",
            "doc2query_score",
        ):
            if key in p:
                try:
                    bits.append(f"{key}={float(p.get(key, 0.0)):.4f}")
                except (TypeError, ValueError):
                    continue
        pid = p.get("id", f"doc_{i}")
        if bits:
            parts.append(f"[P{i + 1}] id={pid}: " + "; ".join(bits))
        else:
            parts.append(f"[P{i + 1}] id={pid}: (no scores)")
    return "\n".join(parts)


def _tool_read_passage(passages: List[Dict], top_k: int, n: int, max_chars: int) -> str:
    if n < 1 or n > top_k or n > len(passages):
        return f"Observation error: invalid passage index {n}. Use 1..{min(top_k, len(passages))}."
    p = passages[n - 1]
    pid = p.get("id", "")
    text = (p.get("text") or "")[:max_chars]
    return f"Full text of [P{n}] (id={pid}):\n{text}"


def _parse_react_action(text: str) -> Optional[Tuple[str, Optional[int]]]:
    """Parse last Action: line from model output."""
    for line in reversed(text.strip().splitlines()):
        raw = line.strip()
        m = re.match(
            r"(?i)^Action:\s*(READ_PASSAGE|LIST_SIGNALS|FINAL)\b",
            raw,
        )
        if not m:
            continue
        name = m.group(1).upper()
        if name == "FINAL":
            return ("FINAL", None)
        if name == "LIST_SIGNALS":
            return ("LIST_SIGNALS", None)
        # READ_PASSAGE[n] or READ_PASSAGE (n)
        m2 = re.search(r"(?i)READ_PASSAGE\s*\[?\s*(\d+)\s*\]?", raw)
        if m2:
            return ("READ_PASSAGE", int(m2.group(1)))
        m3 = re.search(r"(?i)READ_PASSAGE\D+(\d+)", raw)
        if m3:
            return ("READ_PASSAGE", int(m3.group(1)))
        return ("READ_PASSAGE", None)
    return None


def _extract_final_answer(generation: str) -> str:
    """Text after last Action: FINAL, else whole generation stripped of boilerplate."""
    lines = generation.strip().splitlines()
    last_final = -1
    for i, line in enumerate(lines):
        if re.match(r"(?i)^\s*Action:\s*FINAL\s*$", line.strip()):
            last_final = i
    if last_final >= 0:
        tail = "\n".join(lines[last_final + 1 :]).strip()
        if tail:
            return tail
    # Sometimes model puts answer on same line
    m = re.search(r"(?i)Action:\s*FINAL\s*(.+)$", generation, re.MULTILINE)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return generation.strip()


def run_reader_react_rag_block(
    *,
    question: str,
    passages_top_k: List[Dict],
    tokenizer,
    reader_model,
    device: str,
    top_k_passages: int,
    max_new_tokens: int,
    max_input_length: int = 2048,
    max_chars_per_passage: int = 600,
    question_type: Optional[str] = None,
    alpha_sparse: Optional[float] = None,
    alpha_dense: Optional[float] = None,
    alpha_kg: Optional[float] = None,
    include_signal_features: bool = True,
    react_max_steps: int = 6,
    react_tokens_per_step: Optional[int] = None,
) -> str:
    """
    Agentic ReAct reader: multi-turn Thought / Action / Observation over passages.

    Tools (simulated, deterministic):
      READ_PASSAGE[n], LIST_SIGNALS, FINAL
    """
    react_max_steps = max(1, int(react_max_steps))
    per_step = react_tokens_per_step or max(128, min(512, max_new_tokens // max(2, react_max_steps // 2)))
    per_step = int(per_step)

    routing_note = build_query_routing_note(
        question_type=question_type,
        alpha_sparse=alpha_sparse,
        alpha_dense=alpha_dense,
        alpha_kg=alpha_kg,
    )
    routing_block = f"\n\n{routing_note}" if include_signal_features else ""

    index_block = _compact_passage_index(passages_top_k, top_k_passages)
    header = (
        f"{SYSTEM_PROMPT}\n{REACT_TOOL_PROTOCOL}{routing_block}\n\n"
        f"## Passage index (use READ_PASSAGE to load full text)\n{index_block}\n\n"
        f"## Question\n{question}\n\n"
        "Begin. Write Thought: then Action: on the following lines."
    )

    transcript = header
    final_answer: Optional[str] = None

    for step in range(react_max_steps):
        prompt = transcript + "\n\nAssistant:"
        gen = reader_generate(
            tokenizer,
            reader_model,
            prompt,
            device,
            max_new_tokens=per_step,
            max_input_length=max_input_length,
        )
        transcript = transcript + "\n\nAssistant:\n" + gen.strip()
        action = _parse_react_action(gen)

        if action and action[0] == "FINAL":
            final_answer = _extract_final_answer(gen)
            break

        if action and action[0] == "LIST_SIGNALS":
            obs = _signals_one_liner(passages_top_k, top_k_passages)
            transcript = transcript + f"\n\nUser:\nObservation:\n{obs}\n"
            continue

        if action and action[0] == "READ_PASSAGE":
            n = action[1]
            if n is None:
                obs = "Observation error: READ_PASSAGE requires an index, e.g. Action: READ_PASSAGE[2]."
            else:
                obs = _tool_read_passage(
                    passages_top_k, top_k_passages, n, max_chars_per_passage
                )
            transcript = transcript + f"\n\nUser:\nObservation:\n{obs}\n"
            continue

        # No parseable action — nudge model
        transcript = (
            transcript
            + "\n\nUser:\nObservation:\n"
            "No valid Action line found. End your next reply with exactly one line among:\n"
            "  Action: READ_PASSAGE[n]\n  Action: LIST_SIGNALS\n  Action: FINAL\n"
        )

    if final_answer:
        return final_answer.strip()

    # Force one closing generation
    transcript = (
        transcript
        + "\n\nUser:\nObservation:\nStep budget exhausted. "
        "You MUST now output Action: FINAL on its own line, then your answer with [Pn] citations.\n"
    )
    prompt = transcript + "\n\nAssistant:"
    gen = reader_generate(
        tokenizer,
        reader_model,
        prompt,
        device,
        max_new_tokens=min(512, max_new_tokens),
        max_input_length=max_input_length,
    )
    last_act = _parse_react_action(gen)
    if last_act and last_act[0] == "FINAL":
        return _extract_final_answer(gen).strip()
    return gen.strip()


def build_rag_prompt(question: str, context: str, routing_note: Optional[str] = None) -> str:
    """Concatenate system instructions, passages, and the user question."""
    routing = f"\n\n{routing_note}" if routing_note else ""
    return (
        f"{SYSTEM_PROMPT}{routing}\n\nPassages:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def reader_generate(
    tokenizer,
    reader_model,
    prompt: str,
    device: str,
    *,
    max_new_tokens: int,
    max_input_length: int = 2048,
) -> str:
    """
    Run the causal reader LM on ``prompt`` and return **only** the generated
    continuation (decoded text after the prompt tokens).
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    ).to(device)

    with torch.no_grad():
        out = reader_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)


def run_reader_rag_block(
    *,
    question: str,
    passages_top_k: List[Dict],
    tokenizer,
    reader_model,
    device: str,
    top_k_passages: int,
    max_new_tokens: int,
    max_input_length: int = 2048,
    max_chars_per_passage: int = 600,
    question_type: Optional[str] = None,
    alpha_sparse: Optional[float] = None,
    alpha_dense: Optional[float] = None,
    alpha_kg: Optional[float] = None,
    include_signal_features: bool = True,
    use_react: bool = False,
    react_max_steps: int = 6,
    react_tokens_per_step: Optional[int] = None,
) -> str:
    """
    One-call **last block**: format context → build prompt → ``generate``.

    ``passages_top_k`` should already be sorted by GARDIAN (or baseline) score,
    most relevant first.

    When ``use_react`` is True, runs a bounded Thought / Action / Observation
    loop with simulated tools (READ_PASSAGE, LIST_SIGNALS, FINAL).
    """
    if use_react:
        return run_reader_react_rag_block(
            question=question,
            passages_top_k=passages_top_k,
            tokenizer=tokenizer,
            reader_model=reader_model,
            device=device,
            top_k_passages=top_k_passages,
            max_new_tokens=max_new_tokens,
            max_input_length=max_input_length,
            max_chars_per_passage=max_chars_per_passage,
            question_type=question_type,
            alpha_sparse=alpha_sparse,
            alpha_dense=alpha_dense,
            alpha_kg=alpha_kg,
            include_signal_features=include_signal_features,
            react_max_steps=react_max_steps,
            react_tokens_per_step=react_tokens_per_step,
        )
    context = format_reader_context(
        passages_top_k,
        top_k=top_k_passages,
        max_chars_per_passage=max_chars_per_passage,
        include_signal_features=include_signal_features,
    )
    routing_note = build_query_routing_note(
        question_type=question_type,
        alpha_sparse=alpha_sparse,
        alpha_dense=alpha_dense,
        alpha_kg=alpha_kg,
    )
    prompt = build_rag_prompt(question, context, routing_note=routing_note)
    return reader_generate(
        tokenizer,
        reader_model,
        prompt,
        device,
        max_new_tokens=max_new_tokens,
        max_input_length=max_input_length,
    )
