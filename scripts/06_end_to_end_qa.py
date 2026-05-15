"""Controlled end-to-end QA evaluation (RQ4).

Two retrieval modes:
  * ``--online-retrieval`` — live BM25+FAISS (unified indices) → GARDIAN rerank → RAG reader
    (same pipeline as ``scripts/11_run_gardian_server.py``).
  * Default — precomputed rank JSONL from ``scripts/03_generate_rank_data.py``.
"""

import argparse
import warnings
import json
import pathlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from copy import deepcopy
from itertools import product
from typing import Any, Dict, List, Optional, Sequence

import torch
from loguru import logger

# Third-party deprecation noise during local runs (HF/transformers/torch).
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*", category=FutureWarning)
from omegaconf import OmegaConf
sys.path.insert(0, ".")

from src.common.question_types import assert_cfg_question_types
from src.common.rank_data_paths import normalize_retriever_name, resolve_rank_data_file
from src.evaluation.qa_eval import evaluate_qa_from_rank_records, evaluate_qa_live
from src.kg.builder import load_kg
from src.kg.linker import EntityLinker
from src.model.gardian import build_gardian_from_model_cfg
from src.features.kg_feat import build_degree_lookup, build_node_set
from src.pipeline.online_feature_cache import OnlinePassageFeatureCache
from src.pipeline.rag_reader import (
    build_retriever_for_qa,
    load_hf_reader,
    resolve_retrieval_paths,
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
    corpus_path: pathlib.Path,
    want: set,
) -> Dict[str, str]:
    """Scan one JSONL corpus for passage ids in ``want``; return id -> text."""
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


def _dataset_block_complete_for_systems(
    block: Any,
    systems: List[str],
    n_questions: int,
) -> bool:
    """True if cached JSON has full per-system rows for this run configuration."""
    if not isinstance(block, dict):
        return False
    pq = block.get("per_question")
    agg = block.get("aggregate")
    if not isinstance(pq, dict) or not isinstance(agg, dict):
        return False
    for s in systems:
        rows = pq.get(s)
        if not isinstance(rows, list) or len(rows) != n_questions:
            return False
        if s not in agg:
            return False
    return True


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
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    logger.info(f"Loaded GARDIAN checkpoint: {ckpt_path}")
    return model


def parse_args():
    p = argparse.ArgumentParser(
        description="Controlled end-to-end QA evaluation",
        epilog=(
            "Full eval (all questions per dataset; default): "
            "python scripts/06_end_to_end_qa.py --retriever hybrid_bm25_faiss "
            "--systems llm_only,hybrid,gardian\n"
            "Live retrieval (matches gardian server): add --online-retrieval\n"
            "Smoke test: add --max-questions 25. Wall-clock smoke (~5-15 s/q with Llama-8B, 3 systems): "
            "--quick-qa (one-shot + tight caps; never combine with --reader-react). "
            "Otherwise ReAct is off in cfg by default; use --fast if YAML enables ReAct; "
            "--reader-react for paper Self-RAG runs.\n"
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
            "Comma-separated: llm_only, bm25, dense, hybrid, doc2query, gardian "
            "(hybrid uses sparse+dense RRF; defaults to three-way LLM-only vs RAG hybrid vs RAG+GARDIAN)."
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
        help="Override cfg.qa.max_new_tokens (decoder budget per generate).",
    )
    p.add_argument(
        "--top-k-passages",
        type=int,
        default=None,
        help="Override cfg.qa.top_k_passages (passages fed to the reader).",
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
            "Retrieve from unified BM25+FAISS indices at QA time (data/indices/bm25/unified, "
            "data/indices/faiss/unified), GARDIAN rerank, then RAG — matches the live server. "
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
    return p.parse_args()


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
        cfg.qa.max_new_tokens = min(int(cfg.qa.get("max_new_tokens", 2048) or 2048), 384)
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
        cfg.qa.max_new_tokens = int(args.max_new_tokens)
    if getattr(args, "top_k_passages", None) is not None:
        cfg.qa.top_k_passages = int(args.top_k_passages)
    if getattr(args, "reader_max_input_length", None) is not None:
        cfg.qa.reader_max_input_length = int(args.reader_max_input_length)
    if getattr(args, "max_chars_per_passage", None) is not None:
        cfg.qa["max_chars_per_passage"] = int(args.max_chars_per_passage)


def _require_kg_artifacts(cfg) -> None:
    kg_p = pathlib.Path(cfg.paths.kg_graph)
    lex_p = pathlib.Path(cfg.paths.kg_lexical_idx)
    if kg_p.is_file() and lex_p.is_file():
        return
    raise FileNotFoundError(
        f"KG artifacts required for --online-retrieval or gardian system: "
        f"graph={kg_p}, lexical={lex_p}"
    )


def _load_live_qa_stack(cfg: Any, device: str, retriever: str) -> Dict[str, Any]:
    """BM25+FAISS retriever, encoder, FAISS passage cache, and KG (startup once per cell)."""
    from sentence_transformers import SentenceTransformer

    _require_kg_artifacts(cfg)
    idx_paths = resolve_retrieval_paths(cfg)
    logger.info(
        f"Live retrieval: sparse={idx_paths['bm25_index_pkl']} | dense={idx_paths['faiss_index']}"
    )
    kg, lex = load_kg(cfg.paths.kg_graph, cfg.paths.kg_lexical_idx)
    linker = EntityLinker(lexical_index=lex, max_entities=int(cfg.kg.max_entities_per_text))
    hybrid = build_retriever_for_qa(cfg, retriever, device=device)
    encoder = SentenceTransformer(cfg.encoder.model_name, device=device)
    feature_cache = OnlinePassageFeatureCache(
        embedding_index_path=idx_paths["faiss_index"],
        embedding_meta_path=idx_paths["faiss_meta"],
        linker=linker,
        encoder=encoder,
    )
    return {
        "kg": kg,
        "linker": linker,
        "degree_lookup": build_degree_lookup(kg),
        "node_set": build_node_set(kg),
        "retriever": hybrid,
        "encoder": encoder,
        "feature_cache": feature_cache,
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
    live_stack: Optional[Dict[str, Any]] = None,
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
    needs_rank = any(s in systems for s in ("bm25", "dense", "hybrid", "doc2query", "gardian"))
    online = bool(getattr(args, "online_retrieval", False)) and live_stack is not None
    datasets_payload: Dict[str, Any] = {}

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
        if cached and _dataset_block_complete_for_systems(cached, systems, n_q):
            datasets_payload[dataset_name] = deepcopy(cached)
            logger.info(
                f"Resume: skipped {dataset_name} (cached {n_q} questions × {len(systems)} systems)"
            )
            continue

        if online:
            pool_k = getattr(args, "top_candidates", None)
            agg, per_q = evaluate_qa_live(
                questions,
                systems=systems,
                cfg=cfg,
                device=device,
                retriever=live_stack["retriever"],
                gardian_model=gardian_model,
                tokenizer=tokenizer,
                reader_model=reader,
                encoder=live_stack["encoder"],
                feature_cache=live_stack["feature_cache"],
                kg=live_stack["kg"],
                linker=live_stack["linker"],
                degree_lookup=live_stack["degree_lookup"],
                node_set=live_stack["node_set"],
                bootstrap_samples=int(args.bootstrap),
                bootstrap_seed=int(args.seed),
                top_candidates=int(pool_k) if pool_k else None,
            )
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
    return datasets_payload


def _load_reader_model(cfg, device: str):
    """Load reader tokenizer + weights (seq2seq or causal — see ``load_hf_reader``)."""
    return load_hf_reader(str(cfg.qa.reader_model), device)


def main():
    args = parse_args()
    cfg0 = OmegaConf.load(args.cfg)
    assert_cfg_question_types(cfg0.model.question_types)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = pathlib.Path(str(cfg0.paths.results_dir))

    systems = [x.strip() for x in args.systems.split(",") if x.strip()]
    allowed = {"llm_only", "bm25", "dense", "hybrid", "doc2query", "gardian"}
    bad = [s for s in systems if s not in allowed]
    if bad:
        raise ValueError(f"Unknown --systems entries: {bad}; allowed={sorted(allowed)}")

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
    if getattr(args, "online_retrieval", False) and any(
        s in systems for s in ("bm25", "dense", "hybrid", "doc2query", "gardian")
    ):
        _require_kg_artifacts(cfg0)

    resume_path: Optional[pathlib.Path] = None
    prior_datasets: Dict[str, Any] = {}
    resume_ok = not matrix_mode and bool(getattr(args, "resume_from", None))
    if getattr(args, "resume_from", None):
        rp = pathlib.Path(args.resume_from)
        if resume_ok and rp.is_file():
            resume_path = rp
            try:
                prior = json.loads(rp.read_text(encoding="utf-8"))
                prior_datasets = dict(prior.get("datasets") or {})
                logger.info(f"Resume: loaded {len(prior_datasets)} dataset(s) from {rp}")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not read --resume-from {rp}: {e}; running all datasets fresh.")
                prior_datasets = {}
        elif resume_ok:
            logger.warning(f"--resume-from not found: {rp}; running all datasets fresh.")

    runs: List[Dict[str, Any]] = []
    for reader_name, retriever in product(readers, retrievers):
        cfg = OmegaConf.load(args.cfg)
        assert_cfg_question_types(cfg.model.question_types)
        cfg.qa.reader_model = reader_name
        _apply_reader_react_cli(cfg, args)
        _apply_qa_speed_overrides(cfg, args)
        logger.info(
            f"=== QA matrix cell: reader={reader_name!r} | retriever={retriever!r} | "
            f"reader_react={bool(cfg.qa.get('reader_react', False))} "
            f"(max_steps={int(cfg.qa.get('reader_react_max_steps', 6) or 6)}) | "
            f"max_new_tokens={int(cfg.qa.get('max_new_tokens', 0) or 0)} "
            f"top_k_passages={int(cfg.qa.get('top_k_passages', 0) or 0)} "
            f"quick_qa={bool(getattr(args, 'quick_qa', False))} ==="
        )
        tokenizer, reader = _load_reader_model(cfg, device)

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

        live_stack = None
        if getattr(args, "online_retrieval", False):
            live_stack = _load_live_qa_stack(cfg, device, retriever)

        query_emb_cache_paths = _resolve_query_emb_cache_paths(args, cfg, retriever)
        if (
            not getattr(args, "online_retrieval", False)
            and "gardian" in systems
            and not query_emb_cache_paths
        ):
            logger.warning(
                f"No query_emb pickle for retriever={retriever!r} (see --query-emb-cache or "
                f"data/query_emb_cache_{retriever}_train_all.pkl)."
            )

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
            live_stack=live_stack,
        )
        run_meta: Dict[str, Any] = {
            "reader_model": reader_name,
            "retriever": retriever,
            "query_emb_cache": query_emb_cache_paths,
            "online_retrieval": bool(getattr(args, "online_retrieval", False)),
            "datasets": ds,
        }
        if live_stack is not None:
            run_meta["retrieval_paths"] = resolve_retrieval_paths(cfg)
        runs.append(run_meta)

        del reader
        del tokenizer
        if gardian_model is not None:
            del gardian_model
        if live_stack is not None:
            del live_stack
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not runs:
        raise RuntimeError(
            "No QA runs completed (every matrix cell failed — e.g. missing "
            "results/gardian_best_<retriever>.pt for all retrievers when 'gardian' is in --systems)."
        )

    payload: Dict[str, Any] = {
        "metrics_legend": {
            "answer_accuracy": "Bootstrap mean and 95% CI of per-question 0/1 correctness.",
            "citation_precision": "Fraction of [P#] citations pointing to gold evidence (PubMedQA only).",
            "citation_recall": "Fraction of gold evidence passages cited (PubMedQA only).",
            "unsupported_claim_rate": "Fraction of citations not supporting gold (PubMedQA only).",
            "ci_array_format": "[mean, ci95_low, ci95_high]",
            "systems": {
                "llm_only": "Same reader LLM, no retrieved passages.",
                "hybrid": "Same reader LLM + RRF-ranked passages from rank JSONL.",
                "gardian": "Same reader LLM + GARDIAN-reranked passages.",
            },
        },
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
            "resumed_from": str(resume_path) if resume_path else None,
        },
    }
    if getattr(args, "online_retrieval", False):
        payload["meta"]["retrieval_paths"] = resolve_retrieval_paths(cfg0)
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.success(f"QA results -> {out_path} ({len(runs)} run(s))")


if __name__ == "__main__":
    main()