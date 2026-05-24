"""Persistent GARDIAN server: load once, answer many questions quickly.

Pipeline per request:
  1. Hybrid retrieval (BM25 ``index.pkl`` + FAISS ``faiss.index`` via ``rag_reader``)
  2. GARDIAN rerank (α-weighted sparse + dense fusion)
  3. RAG reader on GARDIAN top-k passages

Run in background:
  .venv/bin/python scripts/11_run_gardian_server.py --port 8787

Ask from another terminal:
  .venv/bin/python scripts/11_ask_gardian_live.py --question "..."
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf


def _ensure_local_hf_cache() -> None:
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
from src.pipeline.rag_reader import (
    build_retriever_for_qa,
    load_hf_reader,
    resolve_retrieval_paths,
    retrieve_hybrid_candidates,
    run_reader_rag_block,
)


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


def _require_kg_artifacts(cfg) -> None:
    kg_p = pathlib.Path(cfg.paths.kg_graph)
    lex_p = pathlib.Path(cfg.paths.kg_lexical_idx)
    if kg_p.is_file() and lex_p.is_file():
        return
    raise FileNotFoundError(f"Missing KG artifacts: graph={kg_p}, lexical={lex_p}")


def load_gardian(cfg, retriever: str, device: str) -> GARDIAN:
    out_dir = pathlib.Path(cfg.paths.results_dir)
    canonical = normalize_retriever_name(retriever)
    ckpt_path = out_dir / f"gardian_best_{canonical}.pt"
    if not ckpt_path.exists():
        legacy = out_dir / "gardian_best.pt"
        if legacy.exists():
            ckpt_path = legacy
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_model_cfg = ckpt.get("cfg", {}).get("model") if isinstance(ckpt.get("cfg"), dict) else None
    model = build_gardian_from_model_cfg(ckpt_model_cfg or cfg.model)
    load_checkpoint_state(model, ckpt["model_state"], strict=False)
    model.to(device)
    model.eval()
    return model


class GardianRuntime:
    def __init__(
        self,
        cfg_path: str,
        retriever_name: str,
        device: str | None,
        top_candidates: int | None,
        top_passages: int | None,
        max_new_tokens: int | None,
        no_reader: bool = False,
    ):
        self.cfg = OmegaConf.load(cfg_path)
        assert_cfg_question_types(self.cfg.model.question_types)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.retriever_name = normalize_retriever_name(retriever_name)
        pool_default = int(getattr(self.cfg.retrieval, "candidate_pool_size", 100))
        self.top_candidates = int(top_candidates or pool_default)
        self.top_passages = int(top_passages or self.cfg.qa.top_k_passages)
        self.max_new_tokens = int(max_new_tokens or self.cfg.qa.max_new_tokens)
        self.no_reader = bool(no_reader)

        logger.info("Loading retriever + encoders (startup only)...")
        idx_paths = resolve_retrieval_paths(self.cfg)
        logger.info(
            f"Sparse index: {idx_paths['bm25_index_pkl']} | "
            f"Dense index: {idx_paths['faiss_index']}"
        )
        self.faiss_lookup = None
        if uses_faiss_dense(self.retriever_name) and pathlib.Path(idx_paths["faiss_index"]).is_file():
            self.faiss_lookup = FaissPassageEmbeddingLookup(
                idx_paths["faiss_index"], idx_paths["faiss_meta"]
            )
        self.medcpt_encoder = None
        if uses_medcpt_dense(self.retriever_name):
            self.medcpt_encoder = MedCPTFeatureEncoder(
                article_encoder=str(self.cfg.retrieval.medcpt_article_encoder),
                query_encoder=str(self.cfg.retrieval.medcpt_query_encoder),
                device=self.device,
                batch_size=int(self.cfg.retrieval.medcpt_batch_size),
                max_length=int(self.cfg.retrieval.medcpt_max_length),
                fp16=(self.device == "cuda"),
            )
        self.retriever = build_retriever_for_qa(self.cfg, self.retriever_name, device=self.device)
        self.encoder = SentenceTransformer(self.cfg.encoder.model_name, device=self.device)

        logger.info("Loading GARDIAN (startup only)...")
        self.gardian = load_gardian(self.cfg, self.retriever_name, self.device)
        self.tokenizer = None
        self.reader = None
        if not self.no_reader:
            logger.info("Loading reader (startup only)...")
            self.tokenizer, self.reader = load_hf_reader(self.cfg.qa.reader_model, self.device)
        logger.success("GARDIAN server is warm and ready.")

    def ask(
        self,
        question: str,
        question_type: str = "other",
        use_react: bool | None = None,
        no_reader: bool | None = None,
    ) -> Dict:
        question = (question or "").strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        qtype = infer_qtype(question, question_type)
        skip_reader = self.no_reader if no_reader is None else bool(no_reader)
        use_react = bool(self.cfg.qa.get("reader_react", False)) if use_react is None else bool(use_react)
        react_max_steps = int(self.cfg.qa.get("reader_react_max_steps", 6))
        react_tokens = self.cfg.qa.get("reader_react_tokens_per_step")
        react_tokens = int(react_tokens) if react_tokens is not None else None

        candidates = retrieve_hybrid_candidates(
            question,
            self.retriever,
            top_k=self.top_candidates,
        )
        if not candidates:
            return {
                "ok": True,
                "question": question,
                "question_type": qtype,
                "retriever": self.retriever_name,
                "answer": "I don't know",
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

        if skip_reader:
            answer = ""
        else:
            if self.tokenizer is None or self.reader is None:
                raise RuntimeError("Reader was not loaded at startup; restart without --no-reader.")
            answer = run_reader_rag_block(
                question=question,
                passages_top_k=top_for_reader,
                tokenizer=self.tokenizer,
                reader_model=self.reader,
                device=self.device,
                top_k_passages=self.top_passages,
                max_new_tokens=self.max_new_tokens,
                max_input_length=int(self.cfg.qa.get("reader_max_input_length", 2048) or 2048),
                question_type=qtype,
                alpha_sparse=sparse_alfa,
                alpha_dense=dense_alfa,
                include_signal_features=True,
                use_react=use_react,
                react_max_steps=react_max_steps,
                react_tokens_per_step=react_tokens,
            )
            answer = (answer or "").strip() or "I don't know"

        return {
            "ok": True,
            "question": question,
            "question_type": qtype,
            "retriever": self.retriever_name,
            "reader_skipped": skip_reader,
            "reader_react": use_react,
            "answer": answer,
            "fusion_formula": "score = alpha_sparse*sparse + alpha_dense*dense",
            "sparse_alfa": sparse_alfa,
            "dense_alfa": dense_alfa,
            "rag_how_used": (
                "BM25+FAISS candidates → GARDIAN rerank by fused branch scores → "
                "top passages sent to the reader LLM."
            ),
            "top_passages": [
                {
                    "rank": i + 1,
                    "pid": c.get("id"),
                    "gardian_score": float(c.get("gardian_score", 0.0)),
                    "bm25_score": float(c.get("bm25_score", 0.0)),
                    "dense_score": float(c.get("dense_score", 0.0)),
                    "sparse_contribution": float(c.get("sparse_contribution", 0.0)),
                    "dense_contribution": float(c.get("dense_contribution", 0.0)),
                    "text_preview": (c.get("text", "") or "")[:260],
                }
                for i, c in enumerate(top_for_reader)
            ],
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run persistent GARDIAN server.")
    p.add_argument("--cfg", type=str, default="configs/base.yaml")
    p.add_argument(
        "--retriever",
        type=str,
        choices=["hybrid", "hybrid_bm25_faiss", "hybrid_neural", "hybrid_spladepp_medcpt"],
        default="hybrid",
        help="hybrid → BM25+FAISS (unified indices); hybrid_neural → SPLADE++ + MedCPT.",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument(
        "--top-candidates",
        type=int,
        default=None,
        help="Candidate pool before GARDIAN rerank (default: retrieval.candidate_pool_size in cfg).",
    )
    p.add_argument("--top-passages", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument(
        "--no-reader",
        action="store_true",
        help="Serve only GARDIAN scores/ranking; do not load the LLM reader.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    state: Dict[str, object] = {"ready": False, "error": None, "runtime": None}

    def _load_runtime() -> None:
        try:
            rt = GardianRuntime(
                cfg_path=args.cfg,
                retriever_name=args.retriever,
                device=args.device,
                top_candidates=args.top_candidates,
                top_passages=args.top_passages,
                max_new_tokens=args.max_new_tokens,
                no_reader=bool(args.no_reader),
            )
            state["runtime"] = rt
            state["ready"] = True
            logger.success("Startup complete: server is ready to answer.")
        except Exception as e:
            logger.exception("Startup failed")
            state["error"] = str(e)
            state["ready"] = False

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, payload: Dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                if state["error"]:
                    self._send_json(500, {"ok": False, "status": "failed", "error": state["error"]})
                    return
                status = "ready" if state["ready"] else "loading"
                self._send_json(200, {"ok": True, "status": status})
                return
            self._send_json(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            if self.path != "/ask":
                self._send_json(404, {"ok": False, "error": "not found"})
                return
            try:
                if state["error"]:
                    self._send_json(500, {"ok": False, "error": state["error"]})
                    return
                if not state["ready"] or state["runtime"] is None:
                    self._send_json(503, {"ok": False, "error": "server still loading models; retry shortly"})
                    return
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n)
                data = json.loads(raw.decode("utf-8") if raw else "{}")
                question = data.get("question", "")
                qtype = data.get("question_type", "other")
                use_react = data.get("reader_react", None)
                no_reader = data.get("no_reader", data.get("score_only", None))
                runtime = state["runtime"]
                result = runtime.ask(
                    question=question,
                    question_type=qtype,
                    use_react=use_react,
                    no_reader=no_reader,
                )
                self._send_json(200, result)
            except Exception as e:
                logger.exception("Request failed")
                self._send_json(500, {"ok": False, "error": str(e)})

        def log_message(self, fmt, *args):
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.success(f"Listening on http://{args.host}:{args.port} (startup in progress)")
    threading.Thread(target=_load_runtime, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
