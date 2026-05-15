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
from src.features.dense_feat import compute_dense_features_with_score
from src.features.kg_feat import build_degree_lookup, build_node_set, build_query_kg_cache, compute_kg_features
from src.features.sparse import compute_sparse_features
from src.kg.builder import load_kg
from src.kg.linker import EntityLinker
from src.model.gardian import GARDIAN, build_gardian_from_model_cfg
from src.pipeline.online_feature_cache import OnlinePassageFeatureCache
from src.pipeline.rag_reader import (
    build_retriever_for_qa,
    load_hf_reader,
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
    p = argparse.ArgumentParser(description="Ask GARDIAN and inspect alpha fusion.")
    p.add_argument("--cfg", type=str, default="configs/base.yaml")
    p.add_argument("--question", type=str, required=True, help="User medical question.")
    p.add_argument("--question-type", type=str, default="other", help="Optional question type.")
    p.add_argument("--retriever", type=str, choices=["hybrid", "hybrid_neural"], default="hybrid")
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
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)
    assert_cfg_question_types(cfg.model.question_types)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    top_passages = int(args.top_passages or cfg.qa.top_k_passages)
    max_new_tokens = int(args.max_new_tokens or cfg.qa.max_new_tokens)
    question = args.question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    logger.info("Loading KG + linker ...")
    _require_kg_artifacts(cfg)
    kg, lex = load_kg(cfg.paths.kg_graph, cfg.paths.kg_lexical_idx)
    linker = EntityLinker(lexical_index=lex, max_entities=int(cfg.kg.max_entities_per_text))
    degree_lookup = build_degree_lookup(kg)
    node_set = build_node_set(kg)

    logger.info("Loading retrieval + encoders ...")
    retriever = build_retriever_for_qa(cfg, args.retriever, device=device)
    encoder = SentenceTransformer(cfg.encoder.model_name, device=device)
    feature_cache = None
    if not bool(args.disable_online_feature_cache):
        idx_paths = resolve_retrieval_paths(cfg)
        feature_cache = OnlinePassageFeatureCache(
            embedding_index_path=idx_paths["faiss_index"],
            embedding_meta_path=idx_paths["faiss_meta"],
            linker=linker,
            encoder=encoder,
        )

    logger.info("Loading GARDIAN model ...")
    gardian = load_gardian(cfg, retriever=args.retriever, device=device)
    tokenizer = reader = None
    if not bool(args.no_reader):
        logger.info("Loading reader model ...")
        tokenizer, reader = load_hf_reader(cfg.qa.reader_model, device)

    pool_k = int(args.top_candidates or getattr(cfg.retrieval, "candidate_pool_size", 100))
    candidates = retrieve_hybrid_candidates(question, retriever, top_k=pool_k)
    if not candidates:
        raise RuntimeError("No retrieval candidates found.")

    qtype = infer_qtype(question, args.question_type)
    qtype_oh = qtype_onehot(qtype)
    q_emb = encoder.encode([question], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)[0]
    if feature_cache is not None:
        # Production path: passage embeddings are already in the FAISS index.
        # Reconstruct by passage id instead of re-encoding ~100 candidate texts
        # for every user question.
        p_embs = feature_cache.get_passage_embeddings(candidates)
    else:
        p_texts = [c.get("text", "") for c in candidates]
        p_embs = encoder.encode(p_texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    active_dense_scores = [
        float(c.get("medcpt_score", c.get("dense_score", c.get("score", 0.0))))
        for c in candidates
    ]
    dense_score_mean = float(np.mean(active_dense_scores))
    dense_score_std = float(np.std(active_dense_scores) + 1e-8)

    q_entities = linker.link(question)
    kg_coverage = 1.0 if q_entities else 0.0
    # Match the configured feature semantics. If exact distance mode is enabled,
    # the only graph traversal happens once per query, not once per candidate.
    query_kg_cache = build_query_kg_cache(
        q_entities,
        kg,
        node_set=node_set,
        compute_distances=bool(getattr(cfg.kg, "exact_distance_features", False)),
        max_path=int(getattr(cfg.kg, "max_path_length", 4)),
    )

    for i, cand in enumerate(candidates):
        if feature_cache is not None:
            p_entities = feature_cache.get_passage_entities(cand)
        else:
            p_entities = linker.link(cand.get("text", ""))
        cand["sparse_feats"] = compute_sparse_features(
            query=question,
            passage=cand.get("text", ""),
            bm25_score=float(cand.get("bm25_score", cand.get("spladepp_score", 0.0))),
            idf_table=None,
        ).tolist()
        cand["dense_feats"] = compute_dense_features_with_score(
            q_emb=q_emb,
            p_emb=p_embs[i],
            dense_score=active_dense_scores[i],
            score_mean=dense_score_mean,
            score_std=dense_score_std,
        ).tolist()
        cand["kg_feats"] = compute_kg_features(
            q_entities=q_entities,
            p_entities=p_entities,
            G=kg,
            max_path=int(getattr(cfg.kg, "max_path_length", 4)),
            query_cache=query_kg_cache,
            degree_lookup=degree_lookup,
            node_set=node_set,
        ).tolist()

    ranked = gardian.rerank(
        candidates=candidates,
        query_features={
            "query_emb": q_emb.tolist(),
            "qtype_onehot": qtype_oh,
            "kg_coverage": kg_coverage,
        },
        device=device,
    )
    top_for_reader = ranked[:top_passages]
    # Query-level alphas are identical across the candidate batch.
    first = ranked[0]
    sparse_alfa = float(first["sparse_alfa"])
    dense_alfa = float(first["dense_alfa"])
    kg_alfa = float(first["kg_alfa"])

    if getattr(args, "no_reader_react", False):
        use_react = False
    elif getattr(args, "reader_react", False):
        use_react = True
    else:
        use_react = bool(cfg.qa.get("reader_react", False))
    react_max_steps = (
        int(args.reader_react_max_steps)
        if args.reader_react_max_steps is not None
        else int(cfg.qa.get("reader_react_max_steps", 6))
    )
    _rtp = cfg.qa.get("reader_react_tokens_per_step")
    react_tokens = int(args.reader_react_tokens_per_step) if args.reader_react_tokens_per_step is not None else None
    if react_tokens is None and _rtp is not None:
        react_tokens = int(_rtp)

    if bool(args.no_reader):
        answer = ""
    else:
        answer = run_reader_rag_block(
            question=question,
            passages_top_k=top_for_reader,
            tokenizer=tokenizer,
            reader_model=reader,
            device=device,
            top_k_passages=top_passages,
            max_new_tokens=max_new_tokens,
            max_input_length=int(args.max_input_length),
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

    payload: Dict = {
        "question": question,
        "question_type": qtype,
        "reader_skipped": bool(args.no_reader),
        "answer": answer,
        "fusion_formula": "score = alpha_sparse*sparse + alpha_dense*dense + alpha_kg*kg",
        "sparse_alfa": sparse_alfa,
        "dense_alfa": dense_alfa,
        "kg_alfa": kg_alfa,
        "αsparse": sparse_alfa,
        "αdense": dense_alfa,
        "αkg": kg_alfa,
        "rag_how_used": (
            "Alphas weight branch scores for every candidate; candidates are sorted by fused score; "
            "top passages are sent to the reader LLM to generate the final answer."
            + (" ReAct: multi-turn READ_PASSAGE / LIST_SIGNALS / FINAL." if use_react else "")
        ),
        "reader_react": use_react,
        "reader_react_max_steps": react_max_steps,
        "reader_react_tokens_per_step": react_tokens,
        "top_passages": [
            {
                "rank": i + 1,
                "pid": c.get("id"),
                "gardian_score": float(c.get("gardian_score", 0.0)),
                "sparse_branch_score": float(c.get("sparse_branch_score", 0.0)),
                "dense_branch_score": float(c.get("dense_branch_score", 0.0)),
                "kg_branch_score": float(c.get("kg_branch_score", 0.0)),
                "sparse_contribution": float(c.get("sparse_contribution", 0.0)),
                "dense_contribution": float(c.get("dense_contribution", 0.0)),
                "kg_contribution": float(c.get("kg_contribution", 0.0)),
                "text_preview": (c.get("text", "") or "")[:260],
            }
            for i, c in enumerate(top_for_reader)
        ],
    }

    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
