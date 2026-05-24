"""Cross-encoder / MonoBERT / MonoT5 relevance scoring for reranking baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List, Literal, Sequence

import torch
from loguru import logger
from sentence_transformers import CrossEncoder
from transformers import (
    AutoModelForSeq2SeqLM,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    T5Tokenizer,
)

CrossEncoderBackend = Literal["auto", "st", "monobert", "monot5"]

# Strong rerankers (pick one in configs/base.yaml cross_encoder_model).
CROSS_ENCODER_PRESETS: dict[str, str] = {
    "bge_v2_m3": "BAAI/bge-reranker-v2-m3",
    "msmarco_electra": "cross-encoder/ms-marco-electra-base",
    "msmarco_minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "monobert_large": "castorini/monobert-large-msmarco",
    "monot5_base": "castorini/monot5-base-msmarco-10k",
    "monot5_med": "castorini/monot5-base-med-msmarco",
}

# MonoT5 models use SentencePiece "▁false" / "▁true" (or locale variants).
_MONOT5_PREDICTION_TOKENS: dict[str, tuple[str, str]] = {
    "default": ("▁false", "▁true"),
    "unicamp-dl/mt5-base-en-msmarco": ("▁no", "▁yes"),
    "unicamp-dl/mt5-base-mmarco-v2": ("▁no", "▁yes"),
}


def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_cross_encoder_backend(model_name: str, backend: str = "auto") -> str:
    """Map model id + backend hint to st | monobert | monot5."""
    if backend not in ("auto", "st", "monobert", "monot5"):
        raise ValueError(f"Unknown cross-encoder backend: {backend!r}")
    if backend != "auto":
        return backend
    lower = model_name.lower()
    if "monot5" in lower or "duot5" in lower or "inranker" in lower:
        return "monot5"
    if "monobert" in lower:
        return "monobert"
    if any(x in lower for x in ("/mt5-", "mt5-base", "mt5-large", "mt5-3b", "ptt5")):
        return "monot5"
    return "st"


def _batched(items: Sequence[str], batch_size: int) -> Iterator[List[str]]:
    n = max(1, int(batch_size))
    for i in range(0, len(items), n):
        yield list(items[i : i + n])


def _load_monot5_tokenizer(model_name: str) -> T5Tokenizer:
    """Load T5/MonoT5 slow SentencePiece tokenizer (avoid fast/tiktoken conversion)."""
    try:
        import sentencepiece  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "MonoT5 requires sentencepiece. Install: pip install sentencepiece protobuf"
        ) from e
    try:
        import google.protobuf  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "MonoT5 requires protobuf. Install: pip install sentencepiece protobuf"
        ) from e
    return T5Tokenizer.from_pretrained(model_name, legacy=True)


def _load_monobert_tokenizer(model_name: str) -> BertTokenizer:
    """MonoBERT uses BERT WordPiece; avoid broken AutoTokenizer fast conversion."""
    return BertTokenizer.from_pretrained(model_name)


def _torch_at_least(version: str) -> bool:
    import re

    m = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    if not m:
        return False
    major, minor = int(m.group(1)), int(m.group(2))
    req_major, req_minor = (int(x) for x in version.split(".", 1))
    return (major, minor) >= (req_major, req_minor)


def _load_monobert_model(model_name: str, *, dtype: torch.dtype) -> BertForSequenceClassification:
    """Castorini MonoBERT: BERT config without model_type; checkpoint has 2-way classifier."""
    config = BertConfig.from_pretrained(model_name)
    config.num_labels = 2
    kwargs: dict = {"config": config, "dtype": dtype}
    try:
        return BertForSequenceClassification.from_pretrained(
            model_name,
            use_safetensors=True,
            **kwargs,
        )
    except (ValueError, OSError):
        pass
    if not _torch_at_least("2.6"):
        raise RuntimeError(
            f"MonoBERT weights require torch.load (pytorch_model.bin) but torch is "
            f"{torch.__version__}. Upgrade: pip install 'torch>=2.6.0' "
            "--index-url https://download.pytorch.org/whl/cu121"
        )
    return BertForSequenceClassification.from_pretrained(
        model_name,
        use_safetensors=False,
        **kwargs,
    )


def _monot5_prediction_tokens(model_name: str) -> tuple[str, str]:
    for prefix, tokens in _MONOT5_PREDICTION_TOKENS.items():
        if prefix != "default" and prefix in model_name:
            return tokens
    return _MONOT5_PREDICTION_TOKENS["default"]


class _PairScorerBackend(ABC):
    model_name: str
    device: str
    batch_size: int
    max_length: int

    @abstractmethod
    def score_pairs(
        self,
        queries: Sequence[str],
        passages: Sequence[str],
    ) -> List[float]:
        ...


class _SentenceTransformerBackend(_PairScorerBackend):
    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        max_length: int,
        batch_size: int,
        fp16: bool,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        model_kwargs: dict = {}
        if fp16 and device.startswith("cuda"):
            model_kwargs["torch_dtype"] = torch.float16
        self.model = CrossEncoder(
            model_name,
            max_length=self.max_length,
            device=device,
            model_kwargs=model_kwargs or None,
        )
        logger.info(
            f"Loaded ST CrossEncoder {model_name!r} on {device!r} "
            f"(fp16={bool(model_kwargs)})"
        )

    def score_pairs(
        self,
        queries: Sequence[str],
        passages: Sequence[str],
    ) -> List[float]:
        pairs = [(str(q), str(p)) for q, p in zip(queries, passages)]
        raw = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(s) for s in raw]


class _MonoBERTBackend(_PairScorerBackend):
    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        max_length: int,
        batch_size: int,
        fp16: bool,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.tokenizer = _load_monobert_tokenizer(model_name)
        dtype = torch.float16 if fp16 and device.startswith("cuda") else torch.float32
        self.model = _load_monobert_model(model_name, dtype=dtype)
        self.model.eval()
        self.model.to(device)
        logger.info(f"Loaded MonoBERT {model_name!r} on {device!r} (dtype={dtype})")

    @torch.inference_mode()
    def score_pairs(
        self,
        queries: Sequence[str],
        passages: Sequence[str],
    ) -> List[float]:
        scores: List[float] = []
        use_amp = self.device.startswith("cuda") and next(self.model.parameters()).dtype == torch.float16
        for q_batch, p_batch in zip(
            _batched(list(queries), self.batch_size),
            _batched(list(passages), self.batch_size),
        ):
            enc = self.tokenizer(
                q_batch,
                p_batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(**enc).logits
            else:
                logits = self.model(**enc).logits
            if logits.shape[-1] == 1:
                batch_scores = logits.squeeze(-1)
            elif logits.shape[-1] == 2:
                # Castorini MonoBERT: index 1 = relevant (MS MARCO fine-tuning convention).
                batch_scores = logits[:, 1]
            else:
                batch_scores = logits[:, 0]
            scores.extend(batch_scores.detach().float().cpu().tolist())
        return scores


class _MonoT5Backend(_PairScorerBackend):
    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        max_length: int,
        batch_size: int,
        fp16: bool,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.tokenizer = _load_monot5_tokenizer(model_name)
        dtype = torch.float16 if fp16 and device.startswith("cuda") else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            dtype=dtype,
            use_safetensors=True,
        )
        self.model.eval()
        self.model.to(device)
        dec_start = self.model.config.decoder_start_token_id
        if dec_start is None:
            dec_start = self.tokenizer.pad_token_id
        self.decoder_start_id = int(dec_start if dec_start is not None else 0)
        false_tok, true_tok = _monot5_prediction_tokens(model_name)
        vocab = self.tokenizer.get_vocab()
        if false_tok not in vocab or true_tok not in vocab:
            false_id = self.tokenizer.encode(false_tok, add_special_tokens=False)[0]
            true_id = self.tokenizer.encode(true_tok, add_special_tokens=False)[0]
        else:
            false_id = vocab[false_tok]
            true_id = vocab[true_tok]
        self.token_false_id = false_id
        self.token_true_id = true_id
        self.use_amp = fp16 and device.startswith("cuda")
        logger.info(
            f"Loaded MonoT5 {model_name!r} on {device!r} "
            f"(tokens {false_tok!r}/{true_tok!r}, fp16={self.use_amp})"
        )

    @staticmethod
    def _format_input(query: str, passage: str) -> str:
        return f"Query: {query} Document: {passage} relevant:"

    @torch.inference_mode()
    def score_pairs(
        self,
        queries: Sequence[str],
        passages: Sequence[str],
    ) -> List[float]:
        texts = [self._format_input(q, p) for q, p in zip(queries, passages)]
        scores: List[float] = []
        for batch in _batched(texts, self.batch_size):
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            batch_n = enc["input_ids"].shape[0]
            decoder_input_ids = torch.full(
                (batch_n, 1),
                self.decoder_start_id,
                dtype=torch.long,
                device=self.device,
            )
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(
                        **enc,
                        decoder_input_ids=decoder_input_ids,
                    ).logits
            else:
                logits = self.model(
                    **enc,
                    decoder_input_ids=decoder_input_ids,
                ).logits
            last_logits = logits[:, -1, :]
            pair = torch.stack(
                [
                    last_logits[:, self.token_false_id],
                    last_logits[:, self.token_true_id],
                ],
                dim=1,
            )
            rel = torch.log_softmax(pair, dim=1)[:, 1]
            scores.extend(rel.detach().float().cpu().tolist())
        return scores


def build_pair_scorer(
    model_name: str,
    *,
    backend: str = "auto",
    device: str = "auto",
    max_length: int = 512,
    batch_size: int = 64,
    fp16: bool = True,
) -> _PairScorerBackend:
    resolved_device = resolve_device(device)
    resolved_backend = resolve_cross_encoder_backend(model_name, backend)
    use_fp16 = bool(fp16) and resolved_device.startswith("cuda")
    if resolved_backend == "monobert":
        return _MonoBERTBackend(
            model_name,
            device=resolved_device,
            max_length=max_length,
            batch_size=batch_size,
            fp16=use_fp16,
        )
    if resolved_backend == "monot5":
        return _MonoT5Backend(
            model_name,
            device=resolved_device,
            max_length=max_length,
            batch_size=batch_size,
            fp16=use_fp16,
        )
    return _SentenceTransformerBackend(
        model_name,
        device=resolved_device,
        max_length=max_length,
        batch_size=batch_size,
        fp16=use_fp16,
    )


class CrossEncoderScorer:
    """Score (query, passage) pairs with a cross-encoder, MonoBERT, or MonoT5."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 64,
        backend: str = "auto",
        fp16: bool = True,
    ) -> None:
        self.model_name = model_name
        self.backend = resolve_cross_encoder_backend(model_name, backend)
        self.device = resolve_device(device)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self._scorer = build_pair_scorer(
            model_name,
            backend=backend,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            fp16=fp16,
        )

    def score_pairs(
        self,
        queries: Sequence[str],
        passages: Sequence[str],
    ) -> List[float]:
        if len(queries) != len(passages):
            raise ValueError(
                f"queries and passages length mismatch: {len(queries)} vs {len(passages)}"
            )
        if not queries:
            return []
        return self._scorer.score_pairs(queries, passages)
