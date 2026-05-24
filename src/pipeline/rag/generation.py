"""HF tokenizer + model generation for RAG."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from loguru import logger

_SEQ2SEQ_TYPES = frozenset(
    {
        "t5", "mt5", "bart", "pegasus", "led", "longt5", "switch_transformers",
        "encoder_decoder",
    }
)


def prepare_tokenizer(tokenizer) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        tokenizer.clean_up_tokenization_spaces = False
    except Exception:
        pass


def load_reader(model_name: str, device: str) -> Tuple[Any, torch.nn.Module]:
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prepare_tokenizer(tokenizer)
    cfg = AutoConfig.from_pretrained(model_name)
    mt = (getattr(cfg, "model_type", None) or "").strip().lower()
    if mt in _SEQ2SEQ_TYPES:
        logger.info(f"RAG reader Seq2Seq ({mt}): {model_name}")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=dtype)
    else:
        logger.info(f"RAG reader CausalLM ({mt or 'auto'}): {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()
    gc = getattr(model, "generation_config", None)
    if gc is not None:
        try:
            gc.max_length = None
        except Exception:
            pass
    return tokenizer, model


def format_chat_prompt(
    tokenizer,
    reader_model: torch.nn.Module,
    system: str,
    user: str,
) -> str:
    enc_dec = getattr(reader_model.config, "is_encoder_decoder", False)
    if enc_dec:
        return f"{system}\n\n{user}".strip()
    tpl = getattr(tokenizer, "chat_template", None)
    if tpl and hasattr(tokenizer, "apply_chat_template"):
        for messages in (
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            [{"role": "user", "content": f"{system}\n\n{user}".strip()}],
        ):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                continue
    return f"<s>[INST] {system}\n\n{user} [/INST]"


def generate_text(
    tokenizer,
    reader_model: torch.nn.Module,
    prompt: str,
    *,
    max_new_tokens: int,
    max_input_length: int,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    device = next(reader_model.parameters()).device
    inputs = inputs.to(device)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    gen_kw: Dict[str, Any] = {"max_new_tokens": int(max_new_tokens), "do_sample": False}
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
