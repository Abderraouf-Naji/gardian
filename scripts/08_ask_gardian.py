"""Ask GARDIAN on one user question and return answer + alpha traces."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Dict, List

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf


def _ensure_local_hf_cache() -> None:
    """
    Force Hugging Face caches into a project-local writable directory.
    This avoids permission errors under ~/.cache in shared environments.
    """
    cache_root = pathlib.Path(".cache/huggingface").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    hub = cache_root / "hub"
    transformers = cache_root / "transformers"
    hub.mkdir(parents=True, exist_ok=True)
    transformers.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers)


_ensure_local_hf_cache()

from sentence_transformers import SentenceTransformer

sys.path.insert(0, ".")

from src.common.question_types import assert_cfg_question_types, normalize_question_type, qtype_onehot
from src.common.rank_data_paths import normalize_retriever_name
from src.evaluation.qa_eval import _enrich_live_candidates_for_gardian
from src.model.gardian import GARDIAN, build_gardian_from_model_cfg, load_checkpoint_state
from src.pipeline.rank_dense_features import (
    FaissPassageEmbeddingLookup,
    MedCPTFeatureEncoder,
    uses_faiss_dense,
    uses_medcpt_dense,
)
from src.pipeline.gardian_adaptive import retrieve_adaptive_candidates_live
from src.pipeline.rag_reader import (
    build_retriever_for_qa,
    load_hf_reader,
    resolve_faiss_use_gpu,
    resolve_retrieval_paths,
    retrieve_hybrid_candidates,
    run_reader_rag_block,
)


def _require_kg_artifacts(cfg) -> None:
    """Fail fast with fix instructions if paths.kg_* were never built."""
    kg_p = pathlib.Path(cfg.paths.kg_graph)
    lex_p = pathlib.Path(cfg.paths.kg_lexical_idx)
    if kg_p.is_file() and lex_p.is_file():
        return
    raise FileNotFoundError(
        "KG artifacts are missing (config points to the new layout under data/kg/default/).\n"
        f"  graph (missing or not a file): {kg_p}\n"
        f"  lexical (missing or not a file): {lex_p}\n\n"
        "If you already built per-source KGs under data/kg/sources/, copy the largest into default:\n"
        "  python3 scripts/02_build_kg.py --bootstrap-default-from-extra\n"
        "List candidates first:\n"
        "  python3 scripts/02_build_kg.py --skim-extra-kg\n\n"
        "Otherwise create the default KG:\n"
        "  python3 scripts/02_build_kg.py\n"
        "Set UMLS_DIR, UMLS_MRCONSO, or UMLS_MRCONSO_ZIP for a real UMLS KG; "
        "if none are set, the script bootstraps from sources/variants when possible, else synthetic.\n\n"
        "If you still have legacy pickles directly under data/, run once:\n"
        "  python3 scripts/02_build_kg.py --organize-only\n"
        "then run the command above again so data/kg/default/umls_kg.pkl exists.\n"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live RAG+GARDIAN: retrieve → rerank → reader; prints α_sparse, α_dense and answer.",
        epilog=(
            "One-shot:\n"
            "  python scripts/08_ask_gardian.py --question 'Is aspirin safe in pregnancy?' --pretty\n"
            "Interactive (load models once):\n"
            "  python scripts/08_ask_gardian.py --interactive --pretty\n"
            "Fast client if server is running:\n"
            "  python scripts/11_run_gardian_server.py --port 8787\n"
            "  python scripts/11_ask_gardian_live.py --question '...' --pretty"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cfg", type=str, default="configs/base.yaml")
    p.add_argument("--question", type=str, default=None, help="User medical question.")
    p.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="REPL: ask many questions without reloading models (omit --question).",
    )
    p.add_argument("--question-type", type=str, default="other", help="Optional question type.")
    p.add_argument(
        "--retriever",
        type=str,
        default="hybrid_bm25_faiss",
        help="hybrid_bm25_faiss (default), hybrid, hybrid_neural, etc.",
    )
    p.add_argument(
        "--adaptive",
        action="store_true",
        help="Query-first retrieval (controller α,β budgets per channel) before GARDIAN rerank.",
    )
    p.add_argument("--top-candidates", type=int, default=400, help="Candidate pool size before rerank.")
    p.add_argument("--top-passages", type=int, default=None, help="Top passages to send to reader.")
    p.add_argument("--device", type=str, default=None, help="cuda or cpu (auto if omitted).")
    p.add_argument("--max-input-length", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--reader-react", action="store_true", help="Use agentic ReAct reader (or set qa.reader_react in cfg).")
    p.add_argument("--no-reader-react", action="store_true", help="Disable ReAct even if cfg enables it.")
    p.add_argument(
        "--no-reader",
        action="store_true",
        help="Return only GARDIAN scores/ranking; skip loading and running the LLM reader.",
    )
    p.add_argument("--reader-react-max-steps", type=int, default=None, help="Max Thought/Action turns (default from cfg).")
    p.add_argument(
        "--reader-react-tokens-per-step",
        type=int,
        default=None,
        help="Max new tokens per ReAct step (default from cfg or heuristic).",
    )
    p.add_argument(
        "--disable-online-feature-cache",
        action="store_true",
        help=(
            "Disable online passage-side caches and re-encode/link candidate passages "
            "for debugging. Production should leave this off."
        ),
    )
    p.add_argument("--pretty", action="store_true", help="Pretty-print output JSON.")
    return p.parse_args()


def infer_qtype(question: str, fallback: str) -> str:
    if fallback and fallback.lower() != "other":
        return normalize_question_type(fallback)
    q = question.lower()
    if any(t in q for t in ["diagnosis", "diagnose", "condition", "disease"]):
        return "diagnosis"
    if any(t in q for t in ["treatment", "therapy", "drug", "medication"]):
        return "treatment"
    if any(t in q for t in ["mechanism", "pathway", "cause", "etiology"]):
        return "mechanism"
    if any(t in q for t in ["contraindication", "interaction", "side effect", "adverse"]):
        return "contraindication"
    yesno_starts = ("is ", "are ", "can ", "does ", "do ", "did ", "should ", "could ", "would ", "will ")
    if q.strip().startswith(yesno_starts):
        return "yesno"
    return "factoid"


def load_gardian(cfg, retriever: str, device: str) -> GARDIAN:
    out_dir = pathlib.Path(cfg.paths.results_dir)
    canonical = normalize_retriever_name(retriever)
    ckpt_path = out_dir / f"gardian_best_{canonical}.pt"
    if not ckpt_path.exists():
        legacy = out_dir / "gardian_best.pt"
        if legacy.exists():
            logger.warning(
                f"Using legacy {legacy.name} ({ckpt_path.name} not found). "
                "Train per retriever for a matching checkpoint."
            )
            ckpt_path = legacy
        else:
            raise FileNotFoundError(f"Checkpoint not found: {out_dir / f'gardian_best_{retriever}.pt'}")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        # Backward compatibility with older torch versions without weights_only.
        ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_model_cfg = ckpt.get("cfg", {}).get("model") if isinstance(ckpt.get("cfg"), dict) else None
    model = build_gardian_from_model_cfg(ckpt_model_cfg or cfg.model)
    load_checkpoint_state(model, ckpt["model_state"], strict=False)
    model.to(device)
    model.eval()
    return model


def _print_human_summary(payload: Dict) -> None:
    """Readable α and answer (before optional JSON dump)."""
    print("\n" + "=" * 72)
    print(f"Question ({payload.get('question_type', 'other')}): {payload.get('question', '')}")
    print(
        f"α_sparse = {payload.get('sparse_alfa', 0):.4f}   "
        f"α_dense = {payload.get('dense_alfa', 0):.4f}   "
        f"retrieval = {payload.get('retrieval_mode', 'fixed_pool')}"
    )
    if payload.get("reader_skipped"):
        print("(Reader skipped — ranking only)")
    else:
        print("\n--- Answer ---\n")
        print(payload.get("answer", ""))
    print("\n--- Top passages (GARDIAN rank) ---")
    for row in payload.get("top_passages") or []:
        print(
            f"  [{row.get('rank')}] {row.get('pid')}  "
            f"gardian={row.get('gardian_score', 0):.4f}  "
            f"sparse_α·s={row.get('sparse_contribution', 0):.4f}  "
            f"dense_α·s={row.get('dense_contribution', 0):.4f}"
        )
        prev = (row.get("text_preview") or "").replace("\n", " ")
        if prev:
            print(f"      {prev[:200]}{'…' if len(prev) > 200 else ''}")
    print("=" * 72 + "\n")


class GardianAskRuntime:
    """Load retriever + GARDIAN + reader once; answer many questions."""

    def __init__(self, args: argparse.Namespace, cfg) -> None:
        self.args = args
        self.cfg = cfg
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_passages = int(args.top_passages or cfg.qa.top_k_passages)
        self.max_new_tokens = int(args.max_new_tokens or cfg.qa.max_new_tokens)
        self.retriever_name = normalize_retriever_name(args.retriever)
        self.adaptive = bool(
            args.adaptive or getattr(cfg.qa, "gardian_adaptive_retrieval", False)
        )
        self.pool_k = int(args.top_candidates or getattr(cfg.retrieval, "candidate_pool_size", 100))
        faiss_gpu = resolve_faiss_use_gpu(cfg, device=self.device, for_qa=True)

        logger.info("Loading retrieval + encoders ...")
        idx_paths = resolve_retrieval_paths(cfg)
        self.retriever = build_retriever_for_qa(
            cfg, self.retriever_name, device=self.device, use_faiss_gpu=faiss_gpu
        )
        self.encoder = SentenceTransformer(cfg.encoder.model_name, device=self.device)
        self.faiss_lookup = None
        if uses_faiss_dense(self.retriever_name) and pathlib.Path(idx_paths["faiss_index"]).is_file():
            self.faiss_lookup = FaissPassageEmbeddingLookup(
                idx_paths["faiss_index"], idx_paths["faiss_meta"]
            )
        self.medcpt_encoder = None
        if uses_medcpt_dense(self.retriever_name):
            self.medcpt_encoder = MedCPTFeatureEncoder(
                article_encoder=str(cfg.retrieval.medcpt_article_encoder),
                query_encoder=str(cfg.retrieval.medcpt_query_encoder),
                device=self.device,
                batch_size=int(cfg.retrieval.medcpt_batch_size),
                max_length=int(cfg.retrieval.medcpt_max_length),
                fp16=(self.device == "cuda"),
            )

        logger.info("Loading GARDIAN ...")
        self.gardian = load_gardian(cfg, retriever=args.retriever, device=self.device)
        self.tokenizer = None
        self.reader = None
        if not bool(args.no_reader):
            logger.info("Loading reader LLM ...")
            self.tokenizer, self.reader = load_hf_reader(cfg.qa.reader_model, self.device)
        logger.success(
            f"Ready | retriever={self.retriever_name} | adaptive={self.adaptive} | device={self.device}"
        )

    def ask(self, question: str, question_type: str = "other") -> Dict:
        question = (question or "").strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        if self.adaptive:
            qtype = infer_qtype(question, question_type)
            q_emb = self.encoder.encode(
                [question],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )[0].tolist()
            qtype_oh = qtype_onehot(normalize_question_type(qtype))
            candidates, ctrl_sparse, ctrl_dense = retrieve_adaptive_candidates_live(
                question,
                self.retriever,
                self.gardian,
                query_emb=q_emb,
                qtype_onehot=qtype_oh,
                cfg=self.cfg,
                device=self.device,
            )
            retrieval_mode = f"adaptive (controller α_sparse≈{ctrl_sparse:.3f}, α_dense≈{ctrl_dense:.3f} pre-retrieve)"
        else:
            candidates = retrieve_hybrid_candidates(
                question, self.retriever, top_k=self.pool_k
            )
            retrieval_mode = f"fixed_pool (top {self.pool_k} hybrid union)"
            qtype = infer_qtype(question, question_type)

        if not candidates:
            return {
                "question": question,
                "question_type": qtype,
                "retrieval_mode": retrieval_mode,
                "answer": "I don't know",
                "sparse_alfa": 0.0,
                "dense_alfa": 0.0,
                "top_passages": [],
            }

        qtype_oh, q_emb = _enrich_live_candidates_for_gardian(
            question=question,
            candidates=candidates,
            qtype=qtype,
            retriever_name=self.retriever_name,
            encoder=self.encoder,
            faiss_lookup=self.faiss_lookup,
            medcpt_encoder=self.medcpt_encoder,
        )
        ranked = self.gardian.rerank(
            candidates=candidates,
            query_features={"query_emb": q_emb, "qtype_onehot": qtype_oh},
            device=self.device,
        )
        top_for_reader = ranked[: self.top_passages]
        first = ranked[0]
        sparse_alfa = float(first["sparse_alfa"])
        dense_alfa = float(first["dense_alfa"])

        args = self.args
        if getattr(args, "no_reader_react", False):
            use_react = False
        elif getattr(args, "reader_react", False):
            use_react = True
        else:
            use_react = bool(self.cfg.qa.get("reader_react", False))
        react_max_steps = (
            int(args.reader_react_max_steps)
            if args.reader_react_max_steps is not None
            else int(self.cfg.qa.get("reader_react_max_steps", 6))
        )
        _rtp = self.cfg.qa.get("reader_react_tokens_per_step")
        react_tokens = (
            int(args.reader_react_tokens_per_step)
            if args.reader_react_tokens_per_step is not None
            else None
        )
        if react_tokens is None and _rtp is not None:
            react_tokens = int(_rtp)

        yesno_compact = bool(getattr(self.cfg.qa, "rag_yesno_compact", False))
        if bool(args.no_reader):
            answer = ""
        else:
            answer = run_reader_rag_block(
                question=question,
                passages_top_k=top_for_reader,
                tokenizer=self.tokenizer,
                reader_model=self.reader,
                device=self.device,
                top_k_passages=self.top_passages,
                max_new_tokens=self.max_new_tokens,
                max_input_length=int(args.max_input_length),
                question_type=qtype,
                reader_task="yesno" if qtype == "yesno" else "open",
                alpha_sparse=sparse_alfa,
                alpha_dense=dense_alfa,
                include_signal_features=True,
                use_react=use_react,
                react_max_steps=react_max_steps,
                react_tokens_per_step=react_tokens,
                yesno_compact=yesno_compact,
            )
            answer = (answer or "").strip() or "I don't know"

        return {
            "question": question,
            "question_type": qtype,
            "retriever": self.retriever_name,
            "retrieval_mode": retrieval_mode,
            "reader_skipped": bool(args.no_reader),
            "answer": answer,
            "fusion_formula": "score = alpha_sparse*sparse + alpha_dense*dense",
            "sparse_alfa": sparse_alfa,
            "dense_alfa": dense_alfa,
            "αsparse": sparse_alfa,
            "αdense": dense_alfa,
            "rag_how_used": (
                "Retrieve → GARDIAN rerank (fused sparse+dense branch scores) → "
                "top-k passages → reader LLM."
                + (" ReAct reader enabled." if use_react else "")
            ),
            "reader_react": use_react,
            "top_passages": [
                {
                    "rank": i + 1,
                    "pid": c.get("id"),
                    "gardian_score": float(c.get("gardian_score", 0.0)),
                    "bm25_score": float(c.get("bm25_score", 0.0)),
                    "dense_score": float(c.get("dense_score", 0.0)),
                    "sparse_branch_score": float(c.get("sparse_branch_score", 0.0)),
                    "dense_branch_score": float(c.get("dense_branch_score", 0.0)),
                    "sparse_contribution": float(c.get("sparse_contribution", 0.0)),
                    "dense_contribution": float(c.get("dense_contribution", 0.0)),
                    "text_preview": (c.get("text", "") or "")[:260],
                }
                for i, c in enumerate(top_for_reader)
            ],
        }


def _emit(payload: Dict, pretty: bool) -> None:
    if pretty:
        _print_human_summary(payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if not args.interactive and not (args.question and args.question.strip()):
        raise SystemExit("Provide --question or use --interactive")
    if args.interactive and args.question:
        logger.warning("Ignoring --question in --interactive mode")

    cfg = OmegaConf.load(args.cfg)
    assert_cfg_question_types(cfg.model.question_types)
    runtime = GardianAskRuntime(args, cfg)

    if args.interactive:
        print("GARDIAN live REPL — type a question (empty line or 'quit' to exit).\n")
        while True:
            try:
                q = input("Question> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q or q.lower() in {"quit", "exit", "q"}:
                break
            payload = runtime.ask(q, question_type=args.question_type)
            _emit(payload, bool(args.pretty))
        return

    payload = runtime.ask(args.question.strip(), question_type=args.question_type)
    _emit(payload, bool(args.pretty))


if __name__ == "__main__":
    main()
