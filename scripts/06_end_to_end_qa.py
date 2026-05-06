"""Controlled end-to-end QA evaluation (RQ4) over precomputed rank data."""

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
from typing import Any, Dict, List, Optional

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
from src.evaluation.qa_eval import evaluate_qa_from_rank_records
from src.model.gardian import GARDIAN
from src.pipeline.rag_reader import load_hf_reader

RETRIEVER_CHOICES = (
    "hybrid_bm25_faiss",
    "hybrid_doc2query_biobert",
    "hybrid_bm25_biobert",
    "hybrid_doc2query_faiss",
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


def _build_passage_text_lookup(
    rank_records: List[Dict[str, Any]],
    corpus_path: pathlib.Path,
) -> Dict[str, str]:
    """
    Map passage id -> text for rank rows that omit the ``text`` field (compact JSONL).
    Scans ``corpus_path`` once; stops early once all needed ids are found.
    """
    need = {
        str(r["pid"])
        for r in rank_records
        if isinstance(r.get("pid"), str)
        and not (isinstance(r.get("text"), str) and r.get("text", "").strip())
    }
    if not need or not corpus_path.is_file():
        return {}
    out: Dict[str, str] = {}
    with corpus_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = obj.get("id")
            if isinstance(pid, str) and pid in need and pid not in out:
                out[pid] = str(obj.get("text") or "")
                if len(out) >= len(need):
                    break
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
    model = GARDIAN(
        sparse_dim=int(cfg.model.sparse_feat_dim),
        dense_dim=int(cfg.model.dense_feat_dim),
        kg_dim=int(cfg.model.kg_feat_dim),
        branch_hidden=int(cfg.model.branch_hidden),
        controller_hidden=int(cfg.model.controller_hidden),
        query_feat_dim=int(cfg.model.query_feat_dim),
        n_qtypes=len(cfg.model.question_types),
        dropout=float(cfg.model.dropout),
    )
    # Prefer canonical checkpoint path used for final QA (results/gardian.pt),
    # then fall back to retriever-specific training artifact.
    explicit_ckpt = getattr(cfg.qa, "gardian_checkpoint", None)
    ckpt_candidates: List[pathlib.Path] = []
    if explicit_ckpt:
        ckpt_candidates.append(pathlib.Path(str(explicit_ckpt)))
    ckpt_candidates.append(pathlib.Path(cfg.paths.results_dir) / "gardian.pt")
    ckpt_candidates.append(pathlib.Path(cfg.paths.results_dir) / f"gardian_best_{retriever}.pt")
    ckpt_path = next((p for p in ckpt_candidates if p.exists()), None)
    if ckpt_path is None:
        tried = ", ".join(str(p) for p in ckpt_candidates)
        raise FileNotFoundError(f"Checkpoint not found. Tried: {tried}")
    ckpt = torch.load(ckpt_path, map_location=device)
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
            "python scripts/06_end_to_end_qa.py --retriever hybrid_bm25_biobert "
            "--systems llm_only,hybrid,gardian\n"
            "Smoke test (cap N questions): add --max-questions 50\n"
            "Matrix (4 readers x 4 retrievers): "
            "python scripts/06_end_to_end_qa.py "
            "--reader-models google/flan-t5-small,google/flan-t5-base,google/flan-t5-large,google/flan-t5-xl "
            "--retrievers hybrid_bm25_faiss,hybrid_doc2query_biobert,hybrid_bm25_biobert,hybrid_doc2query_faiss "
            "--systems llm_only,hybrid,gardian --out results/qa_matrix.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cfg", type=str, default="configs/base.yaml")
    p.add_argument(
        "--retriever",
        type=str,
        choices=[*list(RETRIEVER_CHOICES), "hybrid", "hybrid_neural"],
        default="hybrid_bm25_biobert",
        help=(
            "Rank JSONL family (must match checkpoints gardian_best_<retriever>.pt). "
            "Default hybrid_bm25_biobert matches RAG = BM25+BioBERT in the paper table."
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
            "(defaults to three-way LLM-only vs RAG hybrid vs RAG+GARDIAN)."
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
    return p.parse_args()


def _split_csv(s: Optional[str]) -> List[str]:
    if not s or not str(s).strip():
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


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
    datasets_payload: Dict[str, Any] = {}

    for dataset_name, q_path, split in dataset_jobs:
        rank_path = resolve_rank_data_file(retriever, dataset_name, split)
        if needs_rank and not pathlib.Path(rank_path).exists():
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

        rank_records: List[Dict[str, Any]] = []
        if pathlib.Path(rank_path).exists():
            with open(rank_path, "r", encoding="utf-8") as f:
                rank_records = [json.loads(line) for line in f if line.strip()]
        elif needs_rank:
            logger.warning(f"Expected rank file missing: {rank_path}, skipping")
            continue
        corpus_path = _DATASET_PASSAGE_CORPUS.get(dataset_name, pathlib.Path(""))
        passage_lookup: Dict[str, str] = {}
        if isinstance(corpus_path, pathlib.Path) and corpus_path.is_file():
            passage_lookup = _build_passage_text_lookup(rank_records, corpus_path)
            if passage_lookup:
                logger.info(
                    f"Loaded {len(passage_lookup):,} passage texts from {corpus_path} "
                    f"(for rank rows without inline text)"
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
        datasets_payload[dataset_name] = {
            "aggregate": agg,
            "per_question": per_q,
        }
        logger.info(
            f"Completed QA eval for {dataset_name} | retriever={retriever} | systems={list(agg.keys())}"
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
    if matrix_mode and getattr(args, "resume_from", None):
        logger.warning("Matrix mode (--reader-models and/or --retrievers): ignoring --resume-from.")

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

        logger.info(f"=== QA matrix cell: reader={reader_name!r} | retriever={retriever!r} ===")
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

        query_emb_cache_paths = _resolve_query_emb_cache_paths(args, cfg, retriever)
        if "gardian" in systems and not query_emb_cache_paths:
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
        )
        runs.append(
            {
                "reader_model": reader_name,
                "retriever": retriever,
                "query_emb_cache": query_emb_cache_paths,
                "datasets": ds,
            }
        )

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
            "resumed_from": str(resume_path) if resume_path else None,
        },
    }
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