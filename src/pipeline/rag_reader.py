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

Causal readers (e.g. BioMistral-7B) use ``tokenizer.chat_template`` when present,
otherwise a Mistral-style ``[INST] … [/INST]`` envelope for single-turn prompts.
Encoder–decoder readers (Flan-T5) keep the legacy flat ``system + user`` string.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

# ``config.model_type`` values that map to encoder–decoder / seq2seq ``AutoModelForSeq2SeqLM``.
_SEQ2SEQ_MODEL_TYPES = frozenset(
    {
        "t5",
        "mt5",
        "umt5",
        "bart",
        "mbart",
        "pegasus",
        "marian",
        "blenderbot",
        "blenderbot-small",
        "m2m_100",
        "nllb",
        "nllb_moe",
        "led",
        "longt5",
        "bigbird_pegasus",
        "mvp",
        "prophetnet",
        "xlm_prophetnet",
        "fsmt",
        "seamless_m4t",
        "seamless_m4t_v2",
        "switch_transformers",
        "gptsan_japanese",
        "encoder_decoder",
    }
)

SYSTEM_PROMPT_LLM_ONLY = """You are a helpful medical assistant. Answer the medical question using
your general training knowledge. No reference passages are provided.
Do not invent document citations such as [P1] or [P2]. Give a direct, concise answer.
If you are unsure, say "I don't know"."""

# Yes/no questions: do not mix with "I don't know" first — Flan-T5 then loops on "I don't know" + filler.
SYSTEM_PROMPT_LLM_ONLY_YESNO = """You are a biomedical assistant. The user asks a yes/no (or maybe) question.
Use your training knowledge briefly. Give at most two short sentences of reasoning, then a new line
with EXACTLY one of these words: yes, no, maybe (lowercase). You may also write "Answer: yes" (or no/maybe) on the last line.
Do not repeat the same sentence. Do not paste the question back. Do not output more than 80 words total."""

SYSTEM_PROMPT = """You are a helpful medical assistant. Answer the question using ONLY
the provided passages.

Citation protocol (required whenever you use passage content):
- Use ONLY the labels shown in the passage list, in square brackets: [P1], [P2], … up to [P10].
- Put citations on the same line as the fact they support (at least one [P#] per supported sentence).
- After your answer, add a final line: Sources: [P1],[P2],… listing every passage you cited (no extras).

If you cannot answer from the passages, reply with a single line: I don't know

When passage metadata includes dense/sparse/kg evidence scores, use them as reliability signals:
- sparse: lexical overlap / exact terminology
- dense: semantic similarity / paraphrase match
- kg: medical entity-graph support
Prefer passages whose evidence profile matches the query intent."""

# PubMedQA-style yes/no RAG: the generic SYSTEM_PROMPT encourages "I don't know" + long summaries
# without a parseable verdict — our scorer needs Answer: yes|no|maybe (or bare last line).
SYSTEM_PROMPT_YESNO_RAG = """You are a biomedical assistant. Use ONLY the provided passages.
The question is yes/no (or maybe): you must commit to a single verdict supported by those passages.

Rules:
- Every sentence that states a fact from a passage MUST include at least one bracket citation like [P1] or [P2] pointing to that passage (use the [P#] labels from the passage list).
- At most 3 short sentences of reasoning before your verdict line.
- The LAST line of your reply must be exactly one of: Answer: yes — Answer: no — Answer: maybe (lowercase after the colon).
- If evidence is weak or conflicting, still end with Answer: maybe (do not reply with only "I don't know").
- Do not repeat the same sentence. Do not paste the question back as the answer.

When passage metadata includes dense/sparse/kg evidence scores, use them as reliability signals:
- sparse: lexical overlap / exact terminology
- dense: semantic similarity / paraphrase match
- kg: medical entity-graph support
Prefer passages whose evidence profile matches the query intent."""

SYSTEM_PROMPT_MCQ_RAG = """You are a biomedical assistant. You are given a multiple-choice question (options A, B, C, D, etc.) and several reference passages.

Rules:
- Use ONLY information supported by the passages. Prefer passages whose [P#] label matches the evidence you need.
- Show brief reasoning (2–4 sentences). Each factual claim drawn from a passage MUST include that passage's citation tag, e.g. [P1] or [P2], in the same sentence or immediately after it.
- On the LAST line, write exactly: Answer: <LETTER> — <full option text copied verbatim>
  Example: Answer: B — Distal ischaemia affecting the skin of the toes
- If the passages do not support any option, last line: Answer: UNSURE — I don't know

When passage metadata includes dense/sparse/kg evidence scores, treat them as weak reliability hints (sparse=lexical, dense=semantic, kg=graph)."""

SYSTEM_PROMPT_LLM_ONLY_MCQ = """You are a biomedical assistant answering a multiple-choice exam question.
Use your training knowledge. The question lists all options with letters.
Reply with at most 3 short sentences of reasoning, then a final line EXACTLY in this form:
Answer: <LETTER> — <full correct option text>
Use a single uppercase letter (A–D). Do not cite [P#] (no passages are provided). If you cannot decide, end with: Answer: UNSURE — I don't know"""

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


def _is_weak_answer(text: str) -> bool:
    """Detect near-empty or citation-only answers that need fallback regeneration."""
    if not text or not text.strip():
        return True
    t = text.strip().lower()
    # PubMedQA-style verdicts are intentionally short; do not discard as "weak".
    if re.match(r"^(yes|no|maybe)[\s.!?,;:]*$", t):
        return False
    if re.match(r"^(?:answer|label)\s*:\s*(yes|no|maybe)[\s.!?,;:]*$", t):
        return False
    t_raw = text.strip()
    # MedMCQA-style "Answer: B — option text" (last line often short but valid).
    if re.match(r"(?i)^answer\s*:\s*[a-d]\s*[—\-–]\s*\S", t_raw):
        return False
    if re.match(r"(?i)^answer\s*:\s*unsure\b", t_raw):
        return False
    if len(t_raw) < 24:
        return True
    # Citation-only patterns like "[P1]." or "[P1],[P2]"
    if re.fullmatch(r"[\[\]Pp0-9,\.\s\-;:()]+", t_raw):
        return True
    alpha_chars = sum(ch.isalpha() for ch in t_raw)
    return alpha_chars < 12


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
    reader_task: str = "open",
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
    per_step = react_tokens_per_step or max(
        128, min(max(512, max_new_tokens // 2), max_new_tokens, 2048)
    )
    per_step = int(per_step)

    routing_note = build_query_routing_note(
        question_type=question_type,
        alpha_sparse=alpha_sparse,
        alpha_dense=alpha_dense,
        alpha_kg=alpha_kg,
    )
    routing_block = f"\n\n{routing_note}" if include_signal_features else ""

    index_block = _compact_passage_index(passages_top_k, top_k_passages)
    rt = (reader_task or "open").strip().lower()
    if rt == "mcq":
        sys_rag = SYSTEM_PROMPT_MCQ_RAG
    elif (question_type or "").strip().lower() == "yesno":
        sys_rag = SYSTEM_PROMPT_YESNO_RAG
    else:
        sys_rag = SYSTEM_PROMPT
    yesno_tail = ""
    if (question_type or "").strip().lower() == "yesno":
        yesno_tail = (
            "\n\nWhen you output Action: FINAL and your answer, end with a new line exactly: "
            "Answer: yes OR Answer: no OR Answer: maybe (from the passage evidence)."
        )
    mcq_tail = ""
    if rt == "mcq":
        mcq_tail = (
            "\n\nWhen you output Action: FINAL, end with: Answer: <LETTER> — <full option text> "
            "(and include [P#] citations for passage facts)."
        )
    header = (
        f"{sys_rag}\n{REACT_TOOL_PROTOCOL}{routing_block}{yesno_tail}{mcq_tail}\n\n"
        f"## Passage index (use READ_PASSAGE to load full text)\n{index_block}\n\n"
        f"## Question\n{question}\n\n"
        "Begin. Write Thought: then Action: on the following lines."
    )

    transcript = header
    final_answer: Optional[str] = None
    rep_pen, ngram = _t5_yesno_decode_controls(reader_model, question_type)

    for step in range(react_max_steps):
        prompt = transcript + "\n\nAssistant:"
        gen = reader_generate(
            tokenizer,
            reader_model,
            prompt,
            device,
            max_new_tokens=per_step,
            max_input_length=max_input_length,
            repetition_penalty=rep_pen,
            no_repeat_ngram_size=ngram,
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
        max_new_tokens=min(2048, max_new_tokens),
        max_input_length=max_input_length,
        repetition_penalty=rep_pen,
        no_repeat_ngram_size=ngram,
    )
    last_act = _parse_react_action(gen)
    if last_act and last_act[0] == "FINAL":
        return _extract_final_answer(gen).strip()
    return gen.strip()


def _t5_yesno_decode_controls(reader_model: torch.nn.Module, question_type: Optional[str]) -> Tuple[float, int]:
    """Reduce Flan-T5 loops on PubMedQA-style yes/no when passages are long."""
    if (question_type or "").strip().lower() != "yesno":
        return 1.0, 0
    if not getattr(reader_model.config, "is_encoder_decoder", False):
        return 1.0, 0
    return 1.12, 3


def prepare_reader_tokenizer(tokenizer) -> None:
    """Causal LMs often lack pad_token; HF generate expects pad_token_id."""
    if getattr(tokenizer, "pad_token_id", None) is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token


def format_prompt_for_reader(
    tokenizer,
    reader_model: torch.nn.Module,
    system: str,
    user: str,
    *,
    instruct_wrap: bool = True,
) -> str:
    """
    Flat string for encoder–decoder models; chat template or ``[INST]`` wrap for causal LMs.

    ``instruct_wrap=False`` is used for ReAct-style transcripts that append many turns
    to the same string (no closing ``[/INST]`` mid-loop).
    """
    enc_dec = getattr(reader_model.config, "is_encoder_decoder", False)
    if enc_dec or not instruct_wrap:
        return f"{system}\n\n{user}".strip()
    tpl = getattr(tokenizer, "chat_template", None)
    if tpl and hasattr(tokenizer, "apply_chat_template"):
        # Mistral/BioMistral templates often reject ``system`` + ``user`` ("roles must
        # alternate"); a single ``user`` turn with merged instructions works more often.
        messages_variants: List[List[Dict[str, str]]] = [
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            [{"role": "user", "content": f"{system}\n\n{user}".strip()}],
        ]
        for msgs in messages_variants:
            try:
                return tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                continue
    combined = f"{system}\n\n{user}".strip()
    return f"<s>[INST] {combined} [/INST]"


def load_hf_reader(model_name: str, device: str) -> Tuple[Any, torch.nn.Module]:
    """
    Load ``AutoTokenizer`` + reader weights.

    Uses ``AutoConfig.model_type`` to pick **Seq2Seq** (Flan-T5, BART, …) vs **CausalLM**
    (BioMistral, Llama, Mistral, …), so Mistral-based checkpoints do not hit a noisy
    failed ``AutoModelForSeq2SeqLM`` attempt. If ``model_type`` is missing, falls back to
    the legacy try-seq2seq-then-causal path.
    """
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prepare_reader_tokenizer(tokenizer)

    cfg = AutoConfig.from_pretrained(model_name)
    mt = (getattr(cfg, "model_type", None) or "").strip().lower()

    def _load_seq2seq() -> torch.nn.Module:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype)
        model.to(device)
        model.eval()
        return model

    def _load_causal() -> torch.nn.Module:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        model.to(device)
        model.eval()
        return model

    if mt in _SEQ2SEQ_MODEL_TYPES:
        logger.info(f"Loading reader as Seq2SeqLM (model_type={mt!r}): {model_name}")
        return tokenizer, _load_seq2seq()

    if mt:
        logger.info(f"Loading reader as CausalLM (model_type={mt!r}): {model_name}")
        return tokenizer, _load_causal()

    try:
        logger.info(f"Loading reader as Seq2SeqLM (unknown model_type): {model_name}")
        return tokenizer, _load_seq2seq()
    except Exception as e:
        logger.warning(f"Seq2SeqLM load failed ({type(e).__name__}: {e}); trying CausalLM")
    return tokenizer, _load_causal()


def build_llm_only_prompt_split(
    question: str,
    question_type: Optional[str] = None,
    *,
    reader_task: str = "open",
) -> Tuple[str, str]:
    """System and user blocks before model-specific formatting."""
    rt = (reader_task or "open").strip().lower()
    if rt == "mcq":
        return SYSTEM_PROMPT_LLM_ONLY_MCQ.strip(), f"{question}\nAnswer:"
    if (question_type or "").strip().lower() == "yesno":
        return SYSTEM_PROMPT_LLM_ONLY_YESNO.strip(), f"Question: {question}\nAnswer:"
    return SYSTEM_PROMPT_LLM_ONLY.strip(), f"Question: {question}\nAnswer:"


def build_llm_only_prompt(
    question: str,
    question_type: Optional[str] = None,
    *,
    reader_task: str = "open",
) -> str:
    """Prompt for reader-only QA (no retrieved passages)."""
    s, u = build_llm_only_prompt_split(question, question_type=question_type, reader_task=reader_task)
    return f"{s}\n\n{u}"


def build_rag_prompt_split(
    question: str,
    context: str,
    routing_note: Optional[str] = None,
    question_type: Optional[str] = None,
    *,
    reader_task: str = "open",
) -> Tuple[str, str]:
    routing = f"\n\n{routing_note}" if routing_note else ""
    rt = (reader_task or "open").strip().lower()
    if rt == "mcq":
        sys_block = SYSTEM_PROMPT_MCQ_RAG
    elif (question_type or "").strip().lower() == "yesno":
        sys_block = SYSTEM_PROMPT_YESNO_RAG
    else:
        sys_block = SYSTEM_PROMPT
    system = f"{sys_block}{routing}".strip()
    user = f"Passages:\n{context}\n\nQuestion: {question}\nAnswer:"
    return system, user


def build_rag_prompt(
    question: str,
    context: str,
    routing_note: Optional[str] = None,
    question_type: Optional[str] = None,
    *,
    reader_task: str = "open",
) -> str:
    """Concatenate system instructions, passages, and the user question (encoder–decoder layout)."""
    s, u = build_rag_prompt_split(
        question, context, routing_note=routing_note, question_type=question_type, reader_task=reader_task
    )
    return f"{s}\n\n{u}"


def reader_generate(
    tokenizer,
    reader_model,
    prompt: str,
    device: str,
    *,
    max_new_tokens: int,
    max_input_length: int = 2048,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> str:
    """
    Run the reader LM on ``prompt`` and return the generated answer text.

    For encoder–decoder models, returns the decoder output only. For causal LMs,
    returns tokens after the prompt. Inputs are placed on the model's parameter device
    (works with ``device_map`` layouts).
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    inp_device = next(reader_model.parameters()).device
    inputs = inputs.to(inp_device)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    gen_kw: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if pad_id is not None:
        gen_kw["pad_token_id"] = pad_id
    enc_dec = getattr(reader_model.config, "is_encoder_decoder", False)
    if not enc_dec and tokenizer.eos_token_id is not None:
        gen_kw["eos_token_id"] = tokenizer.eos_token_id
    if repetition_penalty > 1.0:
        gen_kw["repetition_penalty"] = float(repetition_penalty)
    if no_repeat_ngram_size > 0:
        gen_kw["no_repeat_ngram_size"] = int(no_repeat_ngram_size)

    with torch.no_grad():
        out = reader_model.generate(**inputs, **gen_kw)

    if enc_dec:
        return tokenizer.decode(out[0], skip_special_tokens=True).strip()

    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()


def run_reader_llm_only_block(
    *,
    question: str,
    tokenizer,
    reader_model,
    device: str,
    max_new_tokens: int,
    max_input_length: int = 2048,
    question_type: Optional[str] = None,
    reader_task: str = "open",
) -> str:
    """
    Reader with **no** retrieved passages (parametric / LLM-only baseline).

    Uses :func:`format_prompt_for_reader` then :func:`reader_generate` so causal
    models (BioMistral, etc.) get a proper instruct envelope.
    """
    sys_p, usr_p = build_llm_only_prompt_split(
        question, question_type=question_type, reader_task=reader_task
    )
    prompt = format_prompt_for_reader(tokenizer, reader_model, sys_p, usr_p, instruct_wrap=True)
    is_yesno = (question_type or "").strip().lower() == "yesno"
    enc_dec = getattr(reader_model.config, "is_encoder_decoder", False)
    tok_budget = max_new_tokens
    rep_pen, ngram = _t5_yesno_decode_controls(reader_model, question_type)
    if is_yesno and enc_dec:
        tok_budget = min(int(max_new_tokens), 128)
        rep_pen = max(rep_pen, 1.18)
    answer = reader_generate(
        tokenizer,
        reader_model,
        prompt,
        device,
        max_new_tokens=tok_budget,
        max_input_length=max_input_length,
        repetition_penalty=rep_pen,
        no_repeat_ngram_size=ngram,
    )
    return (answer or "").strip()


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
    reader_task: str = "open",
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
            reader_task=reader_task,
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
    sys_p, usr_p = build_rag_prompt_split(
        question,
        context,
        routing_note=routing_note,
        question_type=question_type,
        reader_task=reader_task,
    )
    prompt = format_prompt_for_reader(tokenizer, reader_model, sys_p, usr_p, instruct_wrap=True)
    rep_pen, ngram = _t5_yesno_decode_controls(reader_model, question_type)
    answer = reader_generate(
        tokenizer,
        reader_model,
        prompt,
        device,
        max_new_tokens=max_new_tokens,
        max_input_length=max_input_length,
        repetition_penalty=rep_pen,
        no_repeat_ngram_size=ngram,
    )
    if answer and not _is_weak_answer(answer):
        return answer.strip()

    # Fallback for occasional empty generations (e.g., immediate EOS):
    # shrink context and simplify prompt to force a concise answer.
    fallback_context = format_reader_context(
        passages_top_k,
        top_k=max(1, min(3, top_k_passages)),
        max_chars_per_passage=max(350, max_chars_per_passage // 2),
        include_signal_features=False,
    )
    is_yesno = (question_type or "").strip().lower() == "yesno"
    rt_fb = (reader_task or "open").strip().lower()
    if is_yesno:
        sys_fb = SYSTEM_PROMPT_YESNO_RAG.strip()
    elif rt_fb == "mcq":
        sys_fb = SYSTEM_PROMPT_MCQ_RAG.strip()
    else:
        sys_fb = (
            "You are a helpful medical assistant.\n"
            "Answer using only the provided passages. If insufficient evidence, say \"I don't know\".\n"
            "End each supported sentence with [P#] citations; final line: Sources: [P1],..."
        ).strip()
    usr_fb = (
        f"Passages:\n{fallback_context}\n\nQuestion: {question}\nAnswer:"
        if (is_yesno or rt_fb == "mcq")
        else (
            f"Passages:\n{fallback_context}\n\n"
            f"Question: {question}\n"
            "Write at least 3 informative sentences. Explain mechanism and trial evidence if present.\n"
            "Answer:"
        )
    )
    fallback_prompt = format_prompt_for_reader(
        tokenizer, reader_model, sys_fb, usr_fb.strip(), instruct_wrap=True
    )
    fallback_answer = reader_generate(
        tokenizer,
        reader_model,
        fallback_prompt,
        device,
        max_new_tokens=max(96, max_new_tokens),
        max_input_length=max_input_length,
        repetition_penalty=rep_pen,
        no_repeat_ngram_size=ngram,
    )
    clean = fallback_answer.strip() if fallback_answer else ""
    return clean if not _is_weak_answer(clean) else "I don't know"
