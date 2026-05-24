"""Controlled end-to-end QA evaluation (RQ4).

RQ4 primary comparison (use ``--rq4``): **hybrid RRF** vs **GARDIAN rerank** on the same
candidate pool and fixed reader — not LLM-only (that is an optional PubMedQA control).

Retrieval modes:
  * ``--online-retrieval`` — live BM25+FAISS → (optional GARDIAN rerank) → RAG reader.
  * Default — precomputed rank JSONL from ``scripts/03_generate_rank_data.py``.

PubMedQA:
  * ``--pubmedqa-open-domain`` — RQ4 / retrieval paper (retrieve from this dataset's index).
  * ``--pubmedqa-gold-context`` — standard benchmark (labeled abstract only; supplementary).
"""

import argparse
import logging
import warnings
import json
import pathlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from copy import deepcopy
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from loguru import logger

# Third-party deprecation noise during local runs (HF/transformers/torch).
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
from omegaconf import OmegaConf
sys.path.insert(0, ".")

from src.common.question_types import assert_cfg_question_types
from src.common.rank_data_paths import normalize_retriever_name, resolve_rank_data_file
from src.evaluation.pubmedqa_rag import (
    build_gold_passage_lookup,
    is_pubmedqa_dataset,
    resolve_pubmedqa_rag_mode,
)
from src.evaluation.qa_eval import evaluate_qa_from_rank_records, evaluate_qa_live
from src.model.gardian import build_gardian_from_model_cfg, load_checkpoint_state
from src.pipeline.rank_dense_features import (
    FaissPassageEmbeddingLookup,
    MedCPTFeatureEncoder,
    uses_faiss_dense,
    uses_medcpt_dense,
)
from src.pipeline.rag_reader import (
    build_retriever_for_qa,
    load_hf_reader,
    resolve_faiss_use_gpu,
    resolve_retrieval_paths,
)
from src.retrieval.index_paths import (
    assert_dataset_indices_exist,
    resolve_retrieval_paths_for_qa,
)

RETRIEVER_CHOICES = (
    "hybrid_bm25_faiss",
    "hybrid_bm25_medcpt",
    "hybrid_spladepp_faiss",
    "hybrid_spladepp_medcpt",
)


def _git_revision() -> str:
    root = pathlib.Path(__file__).resolve().parents[1]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return ""


def _load_questions(path: str, dataset_name: str, max_questions: Optional[int]):
    if not pathlib.Path(path).exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rec["dataset"] = dataset_name
            if "answer" not in rec:
                rec["answer"] = rec.get("final_decision", rec.get("label", ""))
            rows.append(rec)
            if max_questions is not None and max_questions > 0 and len(rows) >= max_questions:
                break
    return rows


# Per-dataset passage JSONL (same sources as scripts/03_generate_rank_data.py corpora).
_DATASET_PASSAGE_CORPUS: Dict[str, pathlib.Path] = {
    "pubmedqa_labeled": pathlib.Path("data/corpus_pubmedqa_labeled.jsonl"),
    "pubmedqa_artificial": pathlib.Path("data/corpus_pubmedqa_artificial.jsonl"),
    "medmcqa": pathlib.Path("data/corpus_medmcqa.jsonl"),
}


def _scan_corpus_for_pids(
    corpus_path: Union[str, pathlib.Path],
    want: set,
) -> Dict[str, str]:
    """Scan one JSONL corpus for passage ids in ``want``; return id -> text."""
    corpus_path = pathlib.Path(corpus_path)
    out: Dict[str, str] = {}
    if not want or not corpus_path.is_file():
        return out
    with corpus_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = obj.get("id")
            if isinstance(pid, str) and pid in want and pid not in out:
                out[pid] = str(obj.get("text") or "")
                if len(out) >= len(want):
                    break
    return out


def _build_passage_text_lookup(
    rank_records: List[Dict[str, Any]],
    corpus_paths: Sequence[pathlib.Path],
) -> Dict[str, str]:
    """
    Map passage id -> text for rank rows that omit the ``text`` field (compact JSONL).

    Tries each path in order (typically per-dataset corpus then unified
    ``paths.corpus_jsonl``) so QA/RAG matches passages stored only in the unified pool.
    """
    need = {
        str(r["pid"])
        for r in rank_records
        if isinstance(r.get("pid"), str)
        and not (isinstance(r.get("text"), str) and r.get("text", "").strip())
    }
    if not need:
        return {}
    out: Dict[str, str] = {}
    remaining = set(need)
    for corpus_path in corpus_paths:
        if not remaining:
            break
        chunk = _scan_corpus_for_pids(corpus_path, remaining)
        out.update(chunk)
        remaining -= set(chunk.keys())
    return out


def _dataset_cache_status(
    block: Any,
    systems: List[str],
    n_questions: int,
) -> Tuple[bool, str]:
    """Whether a checkpoint block matches this run; human-readable reason if not."""
    if not isinstance(block, dict):
        return False, "not a dataset dict"
    pq = block.get("per_question")
    agg = block.get("aggregate")
    if not isinstance(pq, dict) or not isinstance(agg, dict):
        return False, "missing per_question or aggregate"
    for s in systems:
        rows = pq.get(s)
        if not isinstance(rows, list):
            return False, f"missing per_question[{s!r}]"
        if len(rows) != n_questions:
            return False, f"cached {len(rows)} questions, need {n_questions}"
        if s not in agg:
            return False, f"missing aggregate[{s!r}]"
        n_agg = agg[s].get("n_questions")
        if n_agg is not None and int(n_agg) != n_questions:
            return False, f"aggregate n_questions={n_agg}, need {n_questions}"
    return True, "ok"


def _dataset_block_complete_for_systems(
    block: Any,
    systems: List[str],
    n_questions: int,
) -> bool:
    """True if cached JSON has full per-system rows for this run configuration."""
    ok, _ = _dataset_cache_status(block, systems, n_questions)
    return ok


def _build_model(cfg, device: str, retriever: str):
    # Prefer canonical checkpoint path used for final QA (results/gardian.pt),
    # then fall back to retriever-specific training artifact.
    canonical = normalize_retriever_name(retriever)
    explicit_ckpt = getattr(cfg.qa, "gardian_checkpoint", None)
    ckpt_candidates: List[pathlib.Path] = []
    if explicit_ckpt:
        ckpt_candidates.append(pathlib.Path(str(explicit_ckpt)))
    ckpt_candidates.append(pathlib.Path(cfg.paths.results_dir) / "gardian.pt")
    ckpt_candidates.append(pathlib.Path(cfg.paths.results_dir) / f"gardian_best_{canonical}.pt")
    ckpt_path = next((p for p in ckpt_candidates if p.exists()), None)
    if ckpt_path is None:
        tried = ", ".join(str(p) for p in ckpt_candidates)
        raise FileNotFoundError(f"Checkpoint not found. Tried: {tried}")
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_model_cfg = ckpt.get("cfg", {}).get("model") if isinstance(ckpt.get("cfg"), dict) else None
    model = build_gardian_from_model_cfg(ckpt_model_cfg or cfg.model)
    load_checkpoint_state(model, ckpt["model_state"], strict=False)
    model.to(device)
    model.eval()
    logger.info(f"Loaded GARDIAN checkpoint: {ckpt_path}")
    return model


def parse_args():
    p = argparse.ArgumentParser(
        description="Controlled end-to-end QA evaluation",
        epilog=(
            "RQ4 paper run (1k PubMedQA + 1k MedMCQA, hybrid vs GARDIAN only):\n"
            "  CUDA_VISIBLE_DEVICES=0 python scripts/06_end_to_end_qa.py --online-retrieval --rq4 "
            "--pubmedqa-open-domain --gardian-adaptive-retrieval "
            "--datasets pubmedqa_labeled,medmcqa --max-questions 1000 "
            "--retriever hybrid_bm25_faiss --out results/qa_rq4_od_1k.json\n"
            "  (pubmedqa_labeled has 1000 questions; medmcqa capped at 1000 via --max-questions)\n"
            "PubMedQA standard setting (gold abstract, supplementary table):\n"
            "  ... --pubmedqa-gold-context --rq4 --datasets pubmedqa_labeled "
            "--out results/qa_rq4_pubmedqa_gold_1k.json\n"
            "Live retrieval: add --online-retrieval\n"
            "Smoke test: add --max-questions 25. Wall-clock smoke (~5-15 s/q with Llama-8B, 3 systems): "
            "--quick-qa (one-shot + tight caps; never combine with --reader-react). "
            "Otherwise ReAct is off in cfg by default; use --fast if YAML enables ReAct; "
            "--reader-react for paper Self-RAG runs.\n"
            "50-question retrieval ablation (4 hybrid families × sparse/dense/hybrid/gardian + LLM-only):\n"
            "  python scripts/06_end_to_end_qa.py --max-questions 50 --datasets pubmedqa_labeled "
            "--retrievers hybrid_bm25_faiss,hybrid_bm25_medcpt,hybrid_spladepp_faiss,hybrid_spladepp_medcpt "
            "--systems llm_only,bm25,dense,hybrid,gardian --out results/qa_compare_50q.json\n"
            "Matrix (4 readers x 4 retrievers): "
            "python scripts/06_end_to_end_qa.py "
            "--reader-models google/flan-t5-small,google/flan-t5-base,google/flan-t5-large,google/flan-t5-xl "
            "--retrievers hybrid_bm25_faiss,hybrid_bm25_medcpt,hybrid_spladepp_faiss,hybrid_spladepp_medcpt "
            "--systems llm_only,hybrid,gardian --out results/qa_matrix.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cfg", type=str, default="configs/base.yaml")
    p.add_argument(
        "--retriever",
        type=str,
        choices=[*list(RETRIEVER_CHOICES), "hybrid", "hybrid_neural"],
        default="hybrid_bm25_faiss",
        help=(
            "Rank JSONL family (must match checkpoints gardian_best_<retriever>.pt). "
            "Default hybrid_bm25_faiss is the lexical-anchor baseline."
        ),
    )
    p.add_argument(
        "--reader-models",
        type=str,
        default=None,
        help=(
            "Comma-separated HuggingFace reader model ids (e.g. four LLMs). "
            "If omitted, uses cfg.qa.reader_model once. With --retrievers, evaluates the full grid."
        ),
    )
    p.add_argument(
        "--load-in-4bit",
        action="store_true",
        help=(
            "Load causal reader with bitsandbytes 4-bit NF4 (device_map=auto). "
            "Use for large models (e.g. Qwen2.5-32B) on a single GPU."
        ),
    )
    p.add_argument(
        "--retrievers",
        type=str,
        default=None,
        help=(
            "Comma-separated retriever names (same choices as --retriever). "
            "Each loads rank_data_<r>_*.jsonl and results/gardian_best_<r>.pt for hybrid/gardian. "
            "If omitted, uses --retriever once."
        ),
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="pubmedqa_labeled,medmcqa",
        help=(
            "Comma-separated datasets to evaluate. Supported: "
            "pubmedqa_labeled,pubmedqa_artificial,medmcqa. "
            "Default keeps only paper table datasets for faster runs."
        ),
    )
    p.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help=(
            "Max questions per dataset JSONL. Default 0 = no cap (use every question). "
            "Set a positive integer for faster smoke tests."
        ),
    )
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None)
    p.add_argument(
        "--systems",
        type=str,
        default="llm_only,hybrid,gardian",
        help=(
            "Comma-separated: llm_only, bm25, dense, hybrid, doc2query, gardian. "
            "For RQ4 use --rq4 (locks protocol + sparse,dense,hybrid,gardian). Full ablation: "
            "llm_only,bm25,dense,hybrid,gardian."
        ),
    )
    p.add_argument(
        "--rq4",
        action="store_true",
        help=(
            "RQ4 mode: lock top-k=10, full union pool, RRF hybrid top-k; run "
            "sparse,dense,hybrid,gardian on the same pool and reader."
        ),
    )
    p.add_argument(
        "--query-emb-cache",
        type=str,
        default=None,
        help=(
            "Pickle qid -> query_emb (scripts/12_precompute_query_cache.py). "
            "Comma-separated paths merge pickles (e.g. train_all + labeled_eval) so eval "
            "qids need no on-the-fly encoding. If omitted: use "
            "data/query_emb_cache_<retriever>_train_all.pkl when present, else "
            "cfg.training.query_emb_cache_path when that file exists."
        ),
    )
    p.add_argument(
        "--strict-query-emb-cache",
        action="store_true",
        help=(
            "Fail if a qid is missing from the query_emb pickle (no on-the-fly encoding). "
            "Default: allow encoding missing qids with cfg.encoder when cache is incomplete."
        ),
    )
    p.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help=(
            "Path to an existing qa_results_*.json from this script. Datasets that already "
            "contain aggregate + per_question rows for every --systems entry (same row count "
            "as the current question load) are skipped; others are recomputed. If --out is "
            "omitted, results are written back to this path when the file exists."
        ),
    )
    p.add_argument(
        "--checkpoint-every-dataset",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Write --out after each dataset finishes (PubMedQA then MedMCQA). Enables resume "
            "without losing the first dataset if the run crashes. Default: on with "
            "--online-retrieval."
        ),
    )
    p.add_argument(
        "--reader-react",
        action="store_true",
        help="Force Self-RAG–inspired ReAct reader (overrides cfg.qa.reader_react).",
    )
    p.add_argument(
        "--no-reader-react",
        action="store_true",
        help="Disable ReAct reader for this run (one-shot RAG); overrides cfg.",
    )
    p.add_argument(
        "--reader-react-max-steps",
        type=int,
        default=None,
        help="Max Thought/Action turns for ReAct (default: cfg.qa.reader_react_max_steps).",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Speed preset: disable ReAct (one-shot RAG) unless --reader-react is also set "
            "(then ReAct stays on). Use for smoke benchmarks. Tip: add --bootstrap 200 for quicker CIs."
        ),
    )
    p.add_argument(
        "--quick-qa",
        action="store_true",
        help=(
            "Aggressive wall-clock preset for smoke runs: disables ReAct (incompatible with "
            "--reader-react), caps max_new_tokens, top_k_passages, context length, and passage "
            "snippets. Targets roughly single-digit–15 s/question on a fast GPU with Llama-8B "
            "and three systems; not comparable to full paper settings."
        ),
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Override decoder budget for all reader tasks "
            "(yesno, mcq, open; sets max_new_tokens_yesno/mcq/open)."
        ),
    )
    p.add_argument(
        "--reader-no-retry",
        action="store_true",
        help="Disable second reader generation on format/citation failure (faster on large LMs).",
    )
    p.add_argument(
        "--top-k-passages",
        type=int,
        default=None,
        help="Override cfg.qa.top_k_passages (default 10; align with retrieval Hit@10).",
    )
    p.add_argument(
        "--reader-max-input-length",
        type=int,
        default=None,
        help="Override cfg.qa.reader_max_input_length (tokenizer truncation).",
    )
    p.add_argument(
        "--max-chars-per-passage",
        type=int,
        default=None,
        help="Override max chars per passage in the reader context (default 600).",
    )
    p.add_argument(
        "--online-retrieval",
        action="store_true",
        help=(
            "Live retrieval from per-dataset BM25/FAISS (or MedCPT/SPLADE++) indices, then "
            "GARDIAN rerank and RAG. Default: benchmark corpus, not unified 3M (see --unified-indices). "
            "Without this flag, uses precomputed rank JSONL (faster bulk eval)."
        ),
    )
    p.add_argument(
        "--top-candidates",
        type=int,
        default=None,
        help=(
            "Candidate pool size before GARDIAN rerank (live mode only; default "
            "retrieval.candidate_pool_size in cfg)."
        ),
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Ignore existing --out checkpoint (do not resume). Use when changing "
            "--max-questions, --top-k-passages, or protocol flags."
        ),
    )
    p.add_argument(
        "--gardian-adaptive-retrieval",
        action="store_true",
        help=(
            "Enable GARDIAN controller for fusion (cfg.qa.gardian_adaptive_retrieval). "
            "Does NOT shrink the QA pool when qa.rq4_full_union_pool=true (default): "
            "pool stays 50+50 union; check logs for pool=full_union_rrf."
        ),
    )
    p.add_argument(
        "--no-gardian-adaptive-retrieval",
        action="store_true",
        help=(
            "Disable adaptive controller path (ablation; overrides "
            "cfg.qa.gardian_adaptive_retrieval=true)."
        ),
    )
    p.add_argument(
        "--faiss-cpu",
        action="store_true",
        help=(
            "Keep the FAISS index on CPU during live retrieval (recommended with an 8B reader "
            "on one GPU; avoids ~9GB VRAM for 3M vectors)."
        ),
    )
    p.add_argument(
        "--faiss-gpu",
        action="store_true",
        help="Force FAISS on GPU for live QA (overrides qa.faiss_use_gpu=false).",
    )
    p.add_argument(
        "--unified-indices",
        action="store_true",
        help=(
            "Live QA: use unified 3M BM25+FAISS indices for all datasets (ablation). "
            "Default: per-dataset indices (pubmedqa_labeled, medmcqa, …)."
        ),
    )
    p.add_argument(
        "--pubmedqa-gold-context",
        action="store_true",
        help="PubMedQA: RAG uses labeled abstract(s) only (standard benchmark; overrides cfg).",
    )
    p.add_argument(
        "--pubmedqa-open-domain",
        action="store_true",
        help=(
            "PubMedQA: RAG retrieves from this benchmark's per-dataset indices "
            "(open-domain; overrides cfg). Not the unified 3M pool unless --unified-indices."
        ),
    )
    p.add_argument(
        "--rag-yesno-compact",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "PubMedQA yes/no RAG prompt: compact (cfg default) vs full (requires [P#] citations). "
            "BioMistral readers auto-use full prompt unless --rag-yesno-compact is set."
        ),
    )
    args = p.parse_args()
    if getattr(args, "rq4", False):
        args.systems = "sparse,dense,hybrid,gardian"
    return args


def _checkpoint_every_dataset(args: argparse.Namespace) -> bool:
    flag = getattr(args, "checkpoint_every_dataset", None)
    if flag is not None:
        return bool(flag)
    return bool(getattr(args, "online_retrieval", False))


def _load_prior_datasets_from_json(path: pathlib.Path) -> Dict[str, Any]:
    """Read completed dataset blocks from a prior QA JSON (single-cell or matrix run[0])."""
    prior = json.loads(path.read_text(encoding="utf-8"))
    if "runs" in prior and isinstance(prior["runs"], list) and prior["runs"]:
        return dict(prior["runs"][0].get("datasets") or {})
    return dict(prior.get("datasets") or {})


def _atomic_write_json(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _metrics_legend_payload() -> Dict[str, Any]:
    return {
        "answer_accuracy": "Bootstrap mean and 95% CI of per-question 0/1 correctness.",
        "citation_precision": "Fraction of [P#] citations pointing to gold evidence (PubMedQA only).",
        "citation_recall": "Fraction of gold evidence passages cited (PubMedQA only).",
        "unsupported_claim_rate": "Fraction of [P#] citations not on gold evidence (PubMedQA only).",
        "supported_citation_rate": "1 - unsupported_claim_rate over cited [P#] markers (PubMedQA only).",
        "gold_evidence_in_context_rate": (
            "Fraction of gold_passage_ids present in reader top-k (PubMedQA diagnostic; "
            "explains citation/accuracy vs retrieval Hit@10)."
        ),
        "ci_array_format": "[mean, ci95_low, ci95_high]",
        "reader_top_k": "Passages fed to reader (cfg.qa.top_k_passages; default 10, aligned with retrieval Hit@10).",
        "rq4_primary_comparison": (
            "hybrid vs gardian: same live candidate pool, same reader; "
            "Hybrid = top-k by RRF; GARDIAN = top-k by gardian_score after rerank "
            "(rq4_align_retrieval_top_k disables hybrid_balanced_top_k). "
            "llm_only is not part of RQ4 (optional PubMedQA control only)."
        ),
        "systems": {
            "llm_only": "Optional control: same reader, no passages (not the RQ4 contrast).",
            "bm25": "RAG with sparse channel only (BM25 or SPLADE++ on that hybrid's rank pool).",
            "dense": "RAG with dense channel only (FAISS or MedCPT on that hybrid's rank pool).",
            "hybrid": "RQ4 baseline: top-k passages by hybrid_rrf_score (retrieval-aligned when rq4_align_retrieval_top_k).",
            "gardian": "RQ4 treatment: top-k passages by gardian_score after GARDIAN rerank.",
        },
        "pubmedqa_rag_mode": {
            "gold_context": "Reader sees labeled PubMedQA abstract(s) from corpus (standard benchmark).",
            "open_domain": "Retrieve from per-dataset BM25+FAISS (benchmark corpus); harder than gold_context.",
        },
        "retriever_families": {
            "hybrid_bm25_faiss": "sparse=BM25, dense=FAISS",
            "hybrid_bm25_medcpt": "sparse=BM25, dense=MedCPT",
            "hybrid_spladepp_faiss": "sparse=SPLADE++, dense=FAISS",
            "hybrid_spladepp_medcpt": "sparse=SPLADE++, dense=MedCPT",
        },
    }


def _faiss_use_gpu_for_live_qa(cfg: Any, args: argparse.Namespace, device: str) -> bool:
    if getattr(args, "faiss_cpu", False) and getattr(args, "faiss_gpu", False):
        raise ValueError("Use only one of --faiss-cpu and --faiss-gpu")
    override: Optional[bool] = None
    if getattr(args, "faiss_cpu", False):
        override = False
    elif getattr(args, "faiss_gpu", False):
        override = True
    return resolve_faiss_use_gpu(cfg, device=device, for_qa=True, override=override)


def _pubmedqa_rag_mode(cfg: Any, args: argparse.Namespace) -> Optional[str]:
    if getattr(args, "pubmedqa_gold_context", False):
        return "gold_context"
    if getattr(args, "pubmedqa_open_domain", False):
        return "open_domain"
    return None


def _gardian_adaptive_flag(cfg: Any, args: argparse.Namespace) -> bool:
    if getattr(args, "no_gardian_adaptive_retrieval", False):
        return False
    return bool(
        getattr(args, "gardian_adaptive_retrieval", False)
        or getattr(cfg.qa, "gardian_adaptive_retrieval", False)
    )


def _live_qa_pool_mode(cfg: Any, args: argparse.Namespace, gardian_in_systems: bool) -> str:
    """Pool label written to meta and echoed at startup (matches qa_eval.evaluate_qa_live)."""
    if _gardian_adaptive_flag(cfg, args) and gardian_in_systems and not bool(
        getattr(cfg.qa, "rq4_full_union_pool", False)
    ):
        return "adaptive_live"
    return "full_union_rrf"


def _apply_rq4_protocol_cfg(cfg: Any) -> None:
    """Lock RQ4 E2E settings so Hybrid vs GARDIAN matches rank-data / retrieval @10."""
    cfg.qa.top_k_passages = 10
    cfg.qa.yesno_top_k_passages = 10
    cfg.qa.mcq_top_k_passages = 10
    cfg.qa.hybrid_balanced_top_k = False
    cfg.qa.rq4_align_retrieval_top_k = True
    cfg.qa.rq4_full_union_pool = True


def _log_rq4_protocol(cfg: Any, args: argparse.Namespace, *, gardian_in_systems: bool) -> Dict[str, Any]:
    """Echo RQ4 knobs so logs prove the run matches retrieval tables."""
    pool = _live_qa_pool_mode(cfg, args, gardian_in_systems)
    payload = {
        "reader_top_k": int(getattr(cfg.qa, "top_k_passages", 10) or 10),
        "hybrid_balanced_top_k": bool(getattr(cfg.qa, "hybrid_balanced_top_k", False)),
        "rq4_align_retrieval_top_k": bool(getattr(cfg.qa, "rq4_align_retrieval_top_k", True)),
        "rq4_full_union_pool": bool(getattr(cfg.qa, "rq4_full_union_pool", False)),
        "gardian_adaptive_retrieval": _gardian_adaptive_flag(cfg, args),
        "pool_mode": pool,
        "candidate_pool_size": int(getattr(cfg.retrieval, "candidate_pool_size", 100) or 100),
    }
    logger.info(
        "RQ4 protocol: "
        f"top_k={payload['reader_top_k']} | hybrid_balanced={payload['hybrid_balanced_top_k']} | "
        f"rq4_align={payload['rq4_align_retrieval_top_k']} | "
        f"rq4_full_union_pool={payload['rq4_full_union_pool']} | "
        f"gardian_adaptive={payload['gardian_adaptive_retrieval']} | pool={pool}"
    )
    if pool != "full_union_rrf":
        logger.warning(
            "pool_mode is not full_union_rrf — E2E QA may not match offline retrieval "
            "(set qa.rq4_full_union_pool: true in configs/base.yaml for RQ4)."
        )
    if payload["reader_top_k"] != 10:
        logger.warning(
            f"reader_top_k={payload['reader_top_k']} (paper retrieval uses Hit@10 / NDCG@10)."
        )
    return payload


def _split_csv(s: Optional[str]) -> List[str]:
    if not s or not str(s).strip():
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _apply_reader_react_cli(cfg: Any, args: argparse.Namespace) -> None:
    """Apply ``--fast`` / ``--reader-react`` / ``--no-reader-react`` / max-steps onto ``cfg.qa``."""
    if getattr(args, "no_reader_react", False) and getattr(args, "reader_react", False):
        raise ValueError("Use only one of --reader-react and --no-reader-react.")
    if getattr(args, "reader_react", False):
        cfg.qa.reader_react = True
        if getattr(args, "fast", False):
            logger.info("--reader-react overrides --fast (ReAct enabled).")
    elif getattr(args, "no_reader_react", False) or getattr(args, "fast", False):
        cfg.qa.reader_react = False
        if getattr(args, "fast", False):
            logger.info("Fast mode (--fast): qa.reader_react=false (one-shot RAG).")
    if getattr(args, "reader_react_max_steps", None) is not None:
        cfg.qa.reader_react_max_steps = int(args.reader_react_max_steps)


def _apply_pubmedqa_rag_cli(cfg: Any, args: argparse.Namespace) -> None:
    mode = _pubmedqa_rag_mode(cfg, args)
    if mode is not None:
        cfg.qa.pubmedqa_rag_mode = mode
    compact_flag = getattr(args, "rag_yesno_compact", None)
    if compact_flag is True:
        cfg.qa.rag_yesno_compact = True
    elif compact_flag is False:
        cfg.qa.rag_yesno_compact = False
    readers = _split_csv(getattr(args, "reader_models", None)) or [str(cfg.qa.reader_model)]
    if compact_flag is not True and any("biomistral" in r.lower() for r in readers):
        cfg.qa.rag_yesno_compact = False
        logger.info(
            "BioMistral reader: using full PubMedQA RAG prompt (expects [P#] citations). "
            "Pass --rag-yesno-compact to force the short prompt."
        )


def _apply_qa_speed_overrides(cfg: Any, args: argparse.Namespace) -> None:
    """
    Optional wall-clock presets and explicit QA reader overrides.

    ``--quick-qa`` forces one-shot RAG and tight caps; it cannot be combined with
    ``--reader-react`` (multi-hop ReAct cannot meet ~10 s/question budgets with 8B×3 systems).
    """
    if getattr(args, "quick_qa", False):
        if getattr(args, "reader_react", False):
            raise ValueError(
                "--quick-qa cannot be combined with --reader-react. "
                "Drop --reader-react for wall-clock smoke runs, or omit --quick-qa for Self-RAG ReAct."
            )
        ds_names = _split_csv(getattr(args, "datasets", None)) or [
            "pubmedqa_labeled",
            "pubmedqa_artificial",
            "medmcqa",
        ]
        if any(d.startswith("pubmedqa") for d in ds_names):
            logger.warning(
                "--quick-qa caps passages/tokens and often inflates Answer: maybe on PubMedQA; "
                "RAG accuracy will look much worse than llm_only. Omit --quick-qa for real QA numbers."
            )
        cfg.qa.reader_react = False
        tok_cap = min(int(cfg.qa.get("max_new_tokens", 2048) or 2048), 384)
        cfg.qa.max_new_tokens = tok_cap
        cfg.qa.max_new_tokens_yesno = tok_cap
        cfg.qa.max_new_tokens_mcq = tok_cap
        cfg.qa.top_k_passages = min(int(cfg.qa.get("top_k_passages", 10) or 10), 4)
        cfg.qa.reader_max_input_length = min(int(cfg.qa.get("reader_max_input_length", 8192) or 8192), 3072)
        cfg.qa["max_chars_per_passage"] = min(int(cfg.qa.get("max_chars_per_passage", 600) or 600), 320)
        logger.warning(
            f"Quick-qa: ReAct off | max_new_tokens={int(cfg.qa.max_new_tokens)} "
            f"top_k_passages={int(cfg.qa.top_k_passages)} "
            f"reader_max_input_length={int(cfg.qa.reader_max_input_length)} "
            f"max_chars_per_passage={int(cfg.qa.get('max_chars_per_passage', 320))} "
            "(smoke timings only; do not use for final paper numbers)."
        )
    if getattr(args, "max_new_tokens", None) is not None:
        tok = int(args.max_new_tokens)
        cfg.qa.max_new_tokens = tok
        cfg.qa.max_new_tokens_yesno = tok
        cfg.qa.max_new_tokens_mcq = tok
        logger.info(
            f"Reader token cap: max_new_tokens_yesno/mcq/open={tok} "
            "(PubMedQA uses yesno budget, not cfg.qa.max_new_tokens alone)."
        )
    if getattr(args, "reader_no_retry", False):
        cfg.qa.reader_allow_retry = False
        logger.info("Reader retry disabled (--reader-no-retry).")
    if getattr(args, "top_k_passages", None) is not None:
        cfg.qa.top_k_passages = int(args.top_k_passages)
    if getattr(args, "reader_max_input_length", None) is not None:
        cfg.qa.reader_max_input_length = int(args.reader_max_input_length)
    if getattr(args, "max_chars_per_passage", None) is not None:
        cfg.qa["max_chars_per_passage"] = int(args.max_chars_per_passage)


def _require_kg_artifacts(cfg) -> None:
    if bool(getattr(cfg.model, "text_only", True)):
        return
    kg_p = pathlib.Path(cfg.paths.kg_graph)
    lex_p = pathlib.Path(cfg.paths.kg_lexical_idx)
    if kg_p.is_file() and lex_p.is_file():
        return
    raise FileNotFoundError(
        f"KG artifacts required for legacy KG-enabled GARDIAN: "
        f"graph={kg_p}, lexical={lex_p}"
    )


def _use_per_dataset_indices(
    cfg: Any,
    args: argparse.Namespace,
    dataset_name: str,
    pmqa_mode: str,
) -> bool:
    """Default: each benchmark uses its own corpus indices (not unified 3M)."""
    if getattr(args, "unified_indices", False):
        return False
    if bool(getattr(cfg.qa, "use_per_dataset_indices", True)) is False:
        return False
    if is_pubmedqa_dataset(dataset_name) and pmqa_mode == "gold_context":
        # Gold-context PubMedQA skips live retrieval; path unused for index load.
        return False
    # open_domain PubMedQA + MedMCQA: per-dataset indices (paper default).
    return True


def _load_live_qa_stack(
    cfg: Any,
    device: str,
    retriever: str,
    *,
    use_faiss_gpu: bool,
    dataset_name: Optional[str] = None,
    use_per_dataset_indices: bool = True,
) -> Dict[str, Any]:
    """Hybrid retriever + PubMedBERT controller encoder + dense-feature helpers."""
    from sentence_transformers import SentenceTransformer

    retriever = normalize_retriever_name(retriever)
    idx_paths = resolve_retrieval_paths_for_qa(
        cfg,
        dataset_name=dataset_name,
        use_per_dataset=use_per_dataset_indices,
    )
    if use_per_dataset_indices and dataset_name:
        assert_dataset_indices_exist(idx_paths, retriever=retriever)
    logger.info(
        f"Live retrieval ({idx_paths.get('index_scope', 'unified')} "
        f"key={idx_paths.get('dataset_index_key', 'unified')}): "
        f"sparse={idx_paths['bm25_index_pkl']} | "
        f"dense={idx_paths.get('faiss_index') or idx_paths.get('medcpt_dir')} | "
        f"faiss_use_gpu={use_faiss_gpu}"
    )
    faiss_lookup = None
    if uses_faiss_dense(retriever) and pathlib.Path(idx_paths["faiss_index"]).is_file():
        faiss_lookup = FaissPassageEmbeddingLookup(
            idx_paths["faiss_index"],
            idx_paths["faiss_meta"],
            use_faiss_gpu=use_faiss_gpu,
            faiss_gpu_id=int(getattr(cfg.retrieval, "faiss_gpu_id", 0)),
        )
    medcpt_encoder = None
    if uses_medcpt_dense(retriever):
        medcpt_encoder = MedCPTFeatureEncoder(
            article_encoder=str(cfg.retrieval.medcpt_article_encoder),
            query_encoder=str(cfg.retrieval.medcpt_query_encoder),
            device=device,
            batch_size=int(cfg.retrieval.medcpt_batch_size),
            max_length=int(cfg.retrieval.medcpt_max_length),
            fp16=(device == "cuda"),
        )
    hybrid = build_retriever_for_qa(
        cfg,
        retriever,
        device=device,
        use_faiss_gpu=use_faiss_gpu,
        dataset_name=dataset_name,
        use_per_dataset_indices=use_per_dataset_indices,
    )
    encoder = SentenceTransformer(cfg.encoder.model_name, device=device)
    return {
        "retriever": hybrid,
        "encoder": encoder,
        "faiss_lookup": faiss_lookup,
        "medcpt_encoder": medcpt_encoder,
        "retriever_name": retriever,
        "retrieval_paths": idx_paths,
    }


def _resolve_query_emb_cache_paths(args: argparse.Namespace, cfg, retriever: str) -> List[str]:
    if getattr(args, "query_emb_cache", None):
        out: List[str] = []
        for part in [x.strip() for x in args.query_emb_cache.split(",") if x.strip()]:
            p = pathlib.Path(part)
            if p.is_file():
                out.append(str(p))
            else:
                logger.warning(f"--query-emb-cache path not found (skipped): {p}")
        return out
    per_retriever = pathlib.Path(f"data/query_emb_cache_{retriever}_train_all.pkl")
    if per_retriever.is_file():
        return [str(per_retriever)]
    cfg_p = pathlib.Path(str(getattr(cfg.training, "query_emb_cache_path", "") or ""))
    if cfg_p.is_file():
        return [str(cfg_p)]
    return []


def _eval_all_datasets(
    args: argparse.Namespace,
    cfg: Any,
    device: str,
    systems: List[str],
    retriever: str,
    gardian_model: Optional[torch.nn.Module],
    tokenizer,
    reader: torch.nn.Module,
    query_emb_cache_paths: List[str],
    prior_datasets: Dict[str, Any],
    resume_ok: bool,
    *,
    on_dataset_complete: Optional[Any] = None,
    faiss_use_gpu: bool = False,
) -> Dict[str, Any]:
    """Run QA eval for every dataset job; returns ``datasets`` payload (name -> aggregate + per_question)."""
    all_jobs = {
        "pubmedqa_labeled": ("pubmedqa_labeled", "data/pubmedqa_labeled_eval.jsonl", "eval"),
        "pubmedqa_artificial": ("pubmedqa_artificial", "data/pubmedqa_artificial_test.jsonl", "test"),
        "medmcqa": ("medmcqa", "data/medmcqa_test.jsonl", "test"),
    }
    selected = _split_csv(getattr(args, "datasets", None)) or list(all_jobs.keys())
    unknown = [d for d in selected if d not in all_jobs]
    if unknown:
        raise ValueError(f"Unknown --datasets entries: {unknown}; allowed={sorted(all_jobs.keys())}")
    dataset_jobs = [all_jobs[d] for d in selected]
    needs_rank = any(
        s in systems for s in ("sparse", "dense", "hybrid", "doc2query", "gardian")
    )
    online = bool(getattr(args, "online_retrieval", False))
    datasets_payload: Dict[str, Any] = {}
    pmqa_mode_global = resolve_pubmedqa_rag_mode(cfg, _pubmedqa_rag_mode(cfg, args))

    for dataset_name, q_path, split in dataset_jobs:
        rank_path = resolve_rank_data_file(retriever, dataset_name, split)
        if needs_rank and not online and not pathlib.Path(rank_path).exists():
            logger.warning(f"Missing rank data: {rank_path}, skipping")
            continue
        max_q = args.max_questions if args.max_questions and args.max_questions > 0 else None
        questions = _load_questions(q_path, dataset_name, max_q)
        if not questions:
            logger.warning(f"No questions from {q_path}, skipping")
            continue
        n_q = len(questions)
        cached = prior_datasets.get(dataset_name) if resume_ok else None
        if cached:
            cache_ok, cache_detail = _dataset_cache_status(cached, systems, n_q)
            if cache_ok:
                datasets_payload[dataset_name] = deepcopy(cached)
                logger.info(
                    f"Resume: skipped {dataset_name} (cached {n_q} questions × {len(systems)} systems)"
                )
                continue
            logger.info(
                f"Resume: re-running {dataset_name} ({cache_detail})"
            )

        if online:
            pool_k = getattr(args, "top_candidates", None)
            pmqa_mode = pmqa_mode_global
            gold_lookup: Dict[str, str] = {}
            if pmqa_mode == "gold_context" and dataset_name.startswith("pubmedqa"):
                corpus_p = _DATASET_PASSAGE_CORPUS.get(dataset_name)
                if corpus_p and corpus_p.is_file():
                    gold_lookup = build_gold_passage_lookup(
                        questions, [str(corpus_p)], _scan_corpus_for_pids
                    )
                else:
                    logger.warning(
                        f"pubmedqa_rag_mode=gold_context but corpus missing: {corpus_p}"
                    )
            from sentence_transformers import SentenceTransformer

            # Controller query_emb for GARDIAN rerank (required even in PubMedQA gold_context).
            encoder = SentenceTransformer(cfg.encoder.model_name, device=device)
            live_stack = None
            if not (
                is_pubmedqa_dataset(dataset_name) and pmqa_mode == "gold_context"
            ):
                per_ds = _use_per_dataset_indices(cfg, args, dataset_name, pmqa_mode)
                live_stack = _load_live_qa_stack(
                    cfg,
                    device,
                    retriever,
                    use_faiss_gpu=faiss_use_gpu,
                    dataset_name=dataset_name if per_ds else None,
                    use_per_dataset_indices=per_ds,
                )
                encoder = live_stack["encoder"]
            agg, per_q = evaluate_qa_live(
                questions,
                systems=systems,
                cfg=cfg,
                device=device,
                retriever=live_stack["retriever"] if live_stack else None,
                gardian_model=gardian_model,
                tokenizer=tokenizer,
                reader_model=reader,
                encoder=encoder,
                retriever_name=retriever,
                faiss_lookup=live_stack.get("faiss_lookup") if live_stack else None,
                medcpt_encoder=live_stack.get("medcpt_encoder") if live_stack else None,
                bootstrap_samples=int(args.bootstrap),
                bootstrap_seed=int(args.seed),
                top_candidates=int(pool_k) if pool_k else None,
                gardian_adaptive_retrieval=_gardian_adaptive_flag(cfg, args),
                pubmedqa_rag_mode=pmqa_mode,
                gold_passage_lookup=gold_lookup,
            )
            if live_stack is not None:
                datasets_payload.setdefault("_retrieval_paths", {})[dataset_name] = (
                    live_stack.get("retrieval_paths")
                )
                del live_stack
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            rank_records: List[Dict[str, Any]] = []
            if pathlib.Path(rank_path).exists():
                with open(rank_path, "r", encoding="utf-8") as f:
                    rank_records = [json.loads(line) for line in f if line.strip()]
            elif needs_rank:
                logger.warning(f"Expected rank file missing: {rank_path}, skipping")
                continue
            ds_corpus = _DATASET_PASSAGE_CORPUS.get(dataset_name, pathlib.Path(""))
            unified = pathlib.Path(str(getattr(cfg.paths, "corpus_jsonl", "") or ""))
            corpus_chain: List[pathlib.Path] = []
            if isinstance(ds_corpus, pathlib.Path) and ds_corpus.is_file():
                corpus_chain.append(ds_corpus)
            if unified.is_file() and all(p.resolve() != unified.resolve() for p in corpus_chain):
                corpus_chain.append(unified)
            passage_lookup: Dict[str, str] = {}
            if corpus_chain:
                passage_lookup = _build_passage_text_lookup(rank_records, corpus_chain)
                if passage_lookup:
                    logger.info(
                        f"Loaded {len(passage_lookup):,} passage texts from "
                        f"{', '.join(str(p) for p in corpus_chain)} (rank rows without inline text)"
                    )
            agg, per_q = evaluate_qa_from_rank_records(
                questions,
                rank_records,
                systems=systems,
                gardian_model=gardian_model,
                tokenizer=tokenizer,
                reader_model=reader,
                cfg=cfg,
                device=device,
                bootstrap_samples=int(args.bootstrap),
                bootstrap_seed=int(args.seed),
                passage_text_by_pid=passage_lookup if passage_lookup else None,
                query_emb_cache_path=query_emb_cache_paths or None,
                allow_query_emb_encode_on_cache_miss=not bool(args.strict_query_emb_cache),
                gardian_adaptive_retrieval=_gardian_adaptive_flag(cfg, args),
            )
        agg_out = dict(agg) if isinstance(agg, dict) else agg
        if isinstance(agg_out, dict):
            agg_out.pop("_ci_format", None)
        datasets_payload[dataset_name] = {
            "aggregate": agg_out,
            "per_question": per_q,
            "metrics_note": (
                "answer_accuracy and citation_* arrays are bootstrap 95% CI: "
                "[mean, ci95_low, ci95_high] (see also *_ci objects). "
                "Citation metrics are PubMedQA-only (null for MedMCQA). "
                "llm_only, hybrid, and gardian share the same reader LLM; RAG systems add passages."
            ),
        }
        logger.info(
            f"Completed QA eval for {dataset_name} | retriever={retriever} | systems={list(agg_out.keys())}"
        )
        if on_dataset_complete is not None:
            on_dataset_complete(dict(datasets_payload), dataset_name)
    return datasets_payload


def _load_reader_model(cfg, device: str, *, load_in_4bit: bool = False):
    """Load reader tokenizer + weights (seq2seq or causal — see ``load_hf_reader``)."""
    return load_hf_reader(str(cfg.qa.reader_model), device, load_in_4bit=load_in_4bit)


def main():
    args = parse_args()
    cfg0 = OmegaConf.load(args.cfg)
    assert_cfg_question_types(cfg0.model.question_types)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = pathlib.Path(str(cfg0.paths.results_dir))

    from src.pipeline.rag.reader_types import normalize_system_name

    systems = [
        normalize_system_name(x.strip())
        for x in args.systems.split(",")
        if x.strip()
    ]
    allowed = {"llm_only", "sparse", "dense", "hybrid", "gardian", "doc2query", "bm25"}
    bad = [s for s in systems if s not in allowed]
    if bad:
        raise ValueError(
            f"Unknown --systems entries: {bad}; allowed={sorted(allowed)} "
            "(bm25 is an alias for sparse)"
        )

    readers = _split_csv(args.reader_models) or [str(cfg0.qa.reader_model)]
    retrievers = _split_csv(args.retrievers) or [args.retriever]
    retrievers = [normalize_retriever_name(r) for r in retrievers]
    for r in retrievers:
        if r not in RETRIEVER_CHOICES:
            raise ValueError(f"Unknown retriever {r!r}; allowed={list(RETRIEVER_CHOICES)}")

    matrix_mode = len(readers) > 1 or len(retrievers) > 1
    if getattr(args, "online_retrieval", False) and matrix_mode:
        logger.warning("Matrix mode with multiple retrievers: each cell loads its own live retriever.")
    if matrix_mode and getattr(args, "resume_from", None):
        logger.warning("Matrix mode (--reader-models and/or --retrievers): ignoring --resume-from.")

    checkpoint_every = _checkpoint_every_dataset(args)
    resume_path: Optional[pathlib.Path] = None
    prior_datasets: Dict[str, Any] = {}
    resume_ok = not matrix_mode
    if getattr(args, "fresh", False):
        logger.info("--fresh: ignoring any existing checkpoint in --out")
        prior_datasets = {}
        resume_ok = False
    elif resume_ok:
        load_paths: List[pathlib.Path] = []
        if getattr(args, "resume_from", None):
            load_paths.append(pathlib.Path(args.resume_from))
        if checkpoint_every and args.out:
            out_p = pathlib.Path(args.out)
            if out_p not in load_paths:
                load_paths.append(out_p)
        for lp in load_paths:
            if not lp.is_file():
                if lp == pathlib.Path(args.resume_from or ""):
                    logger.warning(f"--resume-from not found: {lp}; running all datasets fresh.")
                continue
            try:
                loaded = _load_prior_datasets_from_json(lp)
                if loaded:
                    prior_datasets.update(loaded)
                    resume_path = lp
                    logger.info(f"Resume: loaded {len(loaded)} dataset(s) from {lp}")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not read {lp}: {e}; skipping for resume.")
    if checkpoint_every and args.out:
        logger.info(f"Per-dataset checkpointing enabled -> {args.out}")

    runs: List[Dict[str, Any]] = []
    for reader_name, retriever in product(readers, retrievers):
        cfg = OmegaConf.load(args.cfg)
        assert_cfg_question_types(cfg.model.question_types)
        cfg.qa.reader_model = reader_name
        if getattr(args, "rq4", False):
            _apply_rq4_protocol_cfg(cfg)
        _apply_reader_react_cli(cfg, args)
        _apply_pubmedqa_rag_cli(cfg, args)
        _apply_qa_speed_overrides(cfg, args)
        rq4_meta = _log_rq4_protocol(cfg, args, gardian_in_systems="gardian" in systems)
        logger.info(
            f"=== QA matrix cell: reader={reader_name!r} | retriever={retriever!r} | "
            f"reader_react={bool(cfg.qa.get('reader_react', False))} "
            f"(max_steps={int(cfg.qa.get('reader_react_max_steps', 6) or 6)}) | "
            f"max_new_tokens={int(cfg.qa.get('max_new_tokens', 0) or 0)} "
            f"top_k_passages={int(cfg.qa.get('top_k_passages', 0) or 0)} "
            f"quick_qa={bool(getattr(args, 'quick_qa', False))} ==="
        )
        tokenizer, reader = _load_reader_model(
            cfg, device, load_in_4bit=bool(getattr(args, "load_in_4bit", False))
        )

        gardian_model: Optional[torch.nn.Module] = None
        if "gardian" in systems:
            try:
                gardian_model = _build_model(cfg, device, retriever)
            except FileNotFoundError as e:
                logger.error(f"Skipping cell reader={reader_name!r} retriever={retriever!r}: {e}")
                del reader
                del tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        query_emb_cache_paths = _resolve_query_emb_cache_paths(args, cfg, retriever)
        faiss_gpu = _faiss_use_gpu_for_live_qa(cfg, args, device) if getattr(
            args, "online_retrieval", False
        ) else False
        if (
            not getattr(args, "online_retrieval", False)
            and "gardian" in systems
            and not query_emb_cache_paths
        ):
            logger.warning(
                f"No query_emb pickle for retriever={retriever!r} (see --query-emb-cache or "
                f"data/query_emb_cache_{retriever}_train_all.pkl)."
            )

        checkpoint_cb = None
        if checkpoint_every and args.out and not matrix_mode:
            ckpt_path = pathlib.Path(args.out)
            faiss_gpu_flag = _faiss_use_gpu_for_live_qa(cfg, args, device)

            def checkpoint_cb(
                datasets_so_far: Dict[str, Any],
                last_dataset: str,
                *,
                _path: pathlib.Path = ckpt_path,
                _reader: str = reader_name,
                _retriever: str = retriever,
                _cache: List[str] = query_emb_cache_paths,
                _faiss_gpu: bool = faiss_gpu_flag,
            ) -> None:
                meta_ckpt: Dict[str, Any] = {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "script": "scripts/06_end_to_end_qa.py",
                    "checkpoint": True,
                    "last_completed_dataset": last_dataset,
                    "reader_model": _reader,
                    "retriever": _retriever,
                    "online_retrieval": bool(getattr(args, "online_retrieval", False)),
                    "faiss_use_gpu": _faiss_gpu,
                    "query_emb_cache": _cache,
                    "resumed_from": str(resume_path) if resume_path else None,
                    "rq4_protocol": rq4_meta,
                }
                if datasets_so_far.get("_retrieval_paths"):
                    meta_ckpt["retrieval_paths_by_dataset"] = datasets_so_far[
                        "_retrieval_paths"
                    ]
                _atomic_write_json(
                    _path,
                    {
                        "metrics_legend": _metrics_legend_payload(),
                        "meta": meta_ckpt,
                        "datasets": datasets_so_far,
                    },
                )
                logger.success(f"Checkpoint saved after {last_dataset} -> {_path}")

        ds = _eval_all_datasets(
            args,
            cfg,
            device,
            systems,
            retriever,
            gardian_model,
            tokenizer,
            reader,
            query_emb_cache_paths,
            prior_datasets,
            resume_ok=resume_ok,
            on_dataset_complete=checkpoint_cb,
            faiss_use_gpu=faiss_gpu,
        )
        run_meta: Dict[str, Any] = {
            "reader_model": reader_name,
            "retriever": retriever,
            "query_emb_cache": query_emb_cache_paths,
            "online_retrieval": bool(getattr(args, "online_retrieval", False)),
            "rq4_protocol": rq4_meta,
            "datasets": {k: v for k, v in ds.items() if not k.startswith("_")},
        }
        if ds.get("_retrieval_paths"):
            run_meta["retrieval_paths_by_dataset"] = ds["_retrieval_paths"]
        runs.append(run_meta)

        del reader
        del tokenizer
        if gardian_model is not None:
            del gardian_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not runs:
        raise RuntimeError(
            "No QA runs completed (every matrix cell failed — e.g. missing "
            "results/gardian_best_<retriever>.pt for all retrievers when 'gardian' is in --systems)."
        )

    payload: Dict[str, Any] = {
        "metrics_legend": _metrics_legend_payload(),
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/06_end_to_end_qa.py",
            "args": vars(args),
            "git_revision": _git_revision(),
            "platform": platform.platform(),
            "python_version": sys.version,
            "reader_models": readers,
            "retrievers": retrievers,
            "matrix_mode": matrix_mode,
            "reader_model": readers[0],
            "retriever": retrievers[0],
            "online_retrieval": bool(getattr(args, "online_retrieval", False)),
            "checkpoint_every_dataset": checkpoint_every,
            "resumed_from": str(resume_path) if resume_path else None,
            "rq4_mode": bool(getattr(args, "rq4", False)),
        },
    }
    if getattr(args, "online_retrieval", False) and not matrix_mode:
        payload["meta"]["faiss_use_gpu"] = _faiss_use_gpu_for_live_qa(cfg0, args, device)
    if getattr(args, "online_retrieval", False):
        payload["meta"]["retrieval_paths"] = resolve_retrieval_paths(cfg0)
        if getattr(args, "rq4", False):
            payload["meta"]["pipeline"] = (
                "RQ4: shared hybrid pool → hybrid (RRF top-k) vs GARDIAN (rerank top-k) → same reader"
            )
        else:
            payload["meta"]["pipeline"] = (
                "BM25+FAISS retrieve → GARDIAN rerank → RAG reader (live; matches gardian server)"
            )
    if matrix_mode:
        payload["runs"] = runs
    else:
        payload["datasets"] = runs[0]["datasets"]
        payload["meta"]["query_emb_cache"] = runs[0].get("query_emb_cache", [])

    if matrix_mode and not getattr(args, "query_emb_cache", None):
        payload["meta"]["query_emb_cache_note"] = (
            "Per-cell query_emb_cache is under each run when --query-emb-cache is omitted "
            "(defaults use data/query_emb_cache_<retriever>_train_all.pkl)."
        )

    if args.out:
        out_path = pathlib.Path(args.out)
    elif matrix_mode:
        out_path = results_dir / "qa_results_matrix.json"
    elif resume_path is not None:
        out_path = resume_path
    else:
        out_path = results_dir / f"qa_results_{retrievers[0]}.json"
    payload["meta"]["checkpoint"] = False
    _atomic_write_json(out_path, payload)
    logger.success(f"QA results -> {out_path} ({len(runs)} run(s))")


if __name__ == "__main__":
    main()