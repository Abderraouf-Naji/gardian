"""Persistent GARDIAN server: load once, answer many questions quickly.

Run in background:
  .venv/bin/python scripts/11_run_guardian_server.py --port 8787

Ask from another terminal:
  .venv/bin/python scripts/11_ask_guardian_live.py --question "..."
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
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
from src.features.dense_feat import batch_cosines, compute_dense_features
from src.features.kg_feat import build_degree_lookup, build_node_set, build_query_kg_cache, compute_kg_features
from src.features.sparse import compute_sparse_features
from src.kg.builder import load_kg
from src.kg.linker import EntityLinker
from src.model.gardian import GARDIAN
from src.pipeline.rag_reader import load_hf_reader, run_reader_rag_block
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.colbert import ColBERTRetriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridBm25FaissRetriever, HybridSpladev3ColbertRetriever
from src.retrieval.spladev3 import SpladeV3Retriever


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


def build_hybrid_retriever(cfg):
    bm25 = BM25Retriever(index_dir="data/indices/bm25/unified")
    dense = DenseRetriever(
        faiss_index_path="data/indices/faiss/unified/faiss.index",
        meta_path="data/indices/faiss/unified/faiss_meta.jsonl",
        encoder_name=cfg.encoder.model_name,
        batch_size=int(cfg.encoder.batch_size),
        max_length=int(cfg.encoder.max_length),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return HybridBm25FaissRetriever(
        bm25=bm25,
        dense=dense,
        top_k_bm25=int(cfg.retrieval.top_k_bm25),
        top_k_dense=int(cfg.retrieval.top_k_faiss),
    )


def build_hybrid_neural_retriever(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spladev3 = SpladeV3Retriever(
        index_path="data/indices/spladev3/unified",
        model_name=cfg.retrieval.get("spladev3_encoder", "naver/splade-v3-distilbert"),
        device=device,
        batch_size=int(cfg.retrieval.get("spladev3_batch_size", 256)),
        max_length=int(cfg.retrieval.get("spladev3_max_length", 256)),
    )
    colbert = ColBERTRetriever(
        index_path="data/indices/colbert/unified",
        model_name=cfg.retrieval.get("colbert_encoder", "BAAI/bge-large-en-v1.5"),
        device=device,
        batch_size=int(cfg.retrieval.get("colbert_batch_size", 256)),
        max_length=int(cfg.retrieval.get("colbert_max_length", 256)),
    )
    return HybridSpladev3ColbertRetriever(
        spladev3=spladev3,
        colbert=colbert,
        top_k_spladev3=int(cfg.retrieval.get("top_k_spladev3", 50)),
        top_k_colbert=int(cfg.retrieval.get("top_k_colbert", 50)),
    )


def load_gardian(cfg, retriever: str, device: str) -> GARDIAN:
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
    out_dir = pathlib.Path(cfg.paths.results_dir)
    ckpt_path = out_dir / f"gardian_best_{retriever}.pt"
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
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


class GuardianRuntime:
    def __init__(self, cfg_path: str, retriever_name: str, device: str | None, top_candidates: int | None, top_passages: int | None, max_new_tokens: int | None):
        self.cfg = OmegaConf.load(cfg_path)
        assert_cfg_question_types(self.cfg.model.question_types)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.retriever_name = retriever_name
        self.top_candidates = int(top_candidates or 400)
        self.top_passages = int(top_passages or self.cfg.qa.top_k_passages)
        self.max_new_tokens = int(max_new_tokens or self.cfg.qa.max_new_tokens)

        logger.info("Loading KG + linker (startup only)...")
        _require_kg_artifacts(self.cfg)
        self.kg, lex = load_kg(self.cfg.paths.kg_graph, self.cfg.paths.kg_lexical_idx)
        self.linker = EntityLinker(lexical_index=lex, max_entities=int(self.cfg.kg.max_entities_per_text))
        self.degree_lookup = build_degree_lookup(self.kg)
        self.node_set = build_node_set(self.kg)

        logger.info("Loading retriever + encoders (startup only)...")
        self.retriever = build_hybrid_retriever(self.cfg) if retriever_name == "hybrid" else build_hybrid_neural_retriever(self.cfg)
        self.encoder = SentenceTransformer(self.cfg.encoder.model_name, device=self.device)

        logger.info("Loading GARDIAN + reader (startup only)...")
        self.gardian = load_gardian(self.cfg, retriever_name, self.device)
        self.tokenizer, self.reader = load_hf_reader(self.cfg.qa.reader_model, self.device)
        logger.success("GARDIAN server is warm and ready.")

    @staticmethod
    def _reader_focus_terms(question: str) -> List[str]:
        stop = {
            "what", "which", "when", "where", "why", "how", "with", "without", "from", "into", "through",
            "about", "this", "that", "these", "those", "versus", "vs", "compared", "placebo", "adults",
            "patients", "patient", "established", "evidence", "trial", "trials", "pathway", "pathways",
            "reduce", "reduced", "reducing", "effect", "effects", "both", "type", "diabetes", "ascvd",
        }
        toks = re.findall(r"[a-z0-9][a-z0-9\-\+]{2,}", question.lower())
        uniq: List[str] = []
        for t in toks:
            if t in stop:
                continue
            if t not in uniq:
                uniq.append(t)
        # keep strongest medical anchors first
        priority = [t for t in uniq if any(x in t for x in ("sglt2", "empagliflozin", "dapagliflozin", "canagliflozin", "hospital", "failure", "cvot"))]
        rest = [t for t in uniq if t not in priority]
        return (priority + rest)[:10]

    def _select_reader_passages(self, question: str, ranked: List[Dict]) -> List[Dict]:
        terms = self._reader_focus_terms(question)
        if not terms:
            return ranked[: self.top_passages]
        rescored = []
        for c in ranked[: max(self.top_passages * 6, 40)]:
            text = (c.get("text", "") or "").lower()
            hits = sum(1 for t in terms if t in text)
            overlap = hits / max(1, len(terms))
            boosted = float(c.get("gardian_score", 0.0)) + 1.2 * overlap
            d = dict(c)
            d["reader_focus_overlap"] = overlap
            d["reader_focus_score"] = boosted
            rescored.append(d)
        rescored.sort(key=lambda x: float(x.get("reader_focus_score", -1e9)), reverse=True)
        return rescored[: self.top_passages]

    def ask(self, question: str, question_type: str = "other", use_react: bool | None = None) -> Dict:
        question = (question or "").strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        qtype = infer_qtype(question, question_type)
        use_react = bool(self.cfg.qa.get("reader_react", False)) if use_react is None else bool(use_react)
        react_max_steps = int(self.cfg.qa.get("reader_react_max_steps", 6))
        react_tokens = self.cfg.qa.get("reader_react_tokens_per_step")
        react_tokens = int(react_tokens) if react_tokens is not None else None

        candidates = self.retriever.retrieve(question)[: self.top_candidates]
        if not candidates:
            return {"question": question, "question_type": qtype, "answer": "I don't know", "top_passages": []}

        qtype_oh = qtype_onehot(qtype)
        q_emb = self.encoder.encode([question], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)[0]
        p_texts = [c.get("text", "") for c in candidates]
        p_embs = self.encoder.encode(p_texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        cosines = batch_cosines(q_emb, p_embs)
        cos_mean = float(np.mean(cosines))
        cos_std = float(np.std(cosines) + 1e-8)

        q_entities = self.linker.link(question)
        kg_coverage = 1.0 if q_entities else 0.0
        query_kg_cache = build_query_kg_cache(q_entities, self.kg, node_set=self.node_set)

        for i, cand in enumerate(candidates):
            p_entities = self.linker.link(cand.get("text", ""))
            cand["sparse_feats"] = compute_sparse_features(
                query=question,
                passage=cand.get("text", ""),
                bm25_score=float(cand.get("bm25_score", cand.get("spladev3_score", 0.0))),
                idf_table=None,
            ).tolist()
            cand["dense_feats"] = compute_dense_features(
                q_emb=q_emb,
                p_emb=p_embs[i],
                cosine_mean=cos_mean,
                cosine_std=cos_std,
            ).tolist()
            cand["kg_feats"] = compute_kg_features(
                q_entities=q_entities,
                p_entities=p_entities,
                G=self.kg,
                query_cache=query_kg_cache,
                degree_lookup=self.degree_lookup,
                node_set=self.node_set,
            ).tolist()

        ranked = self.gardian.rerank(
            candidates=candidates,
            query_features={"query_emb": q_emb.tolist(), "qtype_onehot": qtype_oh, "kg_coverage": kg_coverage},
            device=self.device,
        )
        top_for_reader = self._select_reader_passages(question, ranked)
        first = ranked[0]
        sparse_alfa = float(first["sparse_alfa"])
        dense_alfa = float(first["dense_alfa"])
        kg_alfa = float(first["kg_alfa"])

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
            alpha_kg=kg_alfa,
            include_signal_features=True,
            use_react=use_react,
            react_max_steps=react_max_steps,
            react_tokens_per_step=react_tokens,
        )
        answer = (answer or "").strip() or "I don't know"

        return {
            "question": question,
            "question_type": qtype,
            "answer": answer,
            "sparse_alfa": sparse_alfa,
            "dense_alfa": dense_alfa,
            "kg_alfa": kg_alfa,
            "top_passages": [
                {"rank": i + 1, "pid": c.get("id"), "text_preview": (c.get("text", "") or "")[:220]}
                for i, c in enumerate(top_for_reader)
            ],
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run persistent GARDIAN server.")
    p.add_argument("--cfg", type=str, default="configs/base.yaml")
    p.add_argument("--retriever", type=str, choices=["hybrid", "hybrid_neural"], default="hybrid")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--top-candidates", type=int, default=400)
    p.add_argument("--top-passages", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    state: Dict[str, object] = {"ready": False, "error": None, "runtime": None}

    def _load_runtime() -> None:
        try:
            rt = GuardianRuntime(
                cfg_path=args.cfg,
                retriever_name=args.retriever,
                device=args.device,
                top_candidates=args.top_candidates,
                top_passages=args.top_passages,
                max_new_tokens=args.max_new_tokens,
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
                runtime = state["runtime"]
                result = runtime.ask(question=question, question_type=qtype, use_react=use_react)
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
