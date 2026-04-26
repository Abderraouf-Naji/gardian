"""
Build BioBERT indices and precheck Doc2Query model availability.
"""

import argparse
import os
import pathlib
import sys

import torch
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, ".")

from src.retrieval.biobert import BioBERTRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.doc2query import Doc2QueryRetriever


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build neural indices (BioBERT + Doc2Query).")
    p.add_argument("--cfg", type=str, default="configs/base.yaml")
    p.add_argument("--only", type=str, choices=["all", "biobert", "doc2query"], default="all")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dataset", type=str, default="all")
    return p.parse_args()


def build_biobert_index(corpus_jsonl: str, output_dir: str, cfg, overwrite: bool = False):
    out = pathlib.Path(output_dir)
    idx = out / "biobert_index.pt"
    if idx.exists() and not overwrite:
        logger.info(f"BioBERT index exists at {out}, skipping")
        return
    out.mkdir(parents=True, exist_ok=True)
    biobert = BioBERTRetriever(
        index_path=str(out),
        checkpoint=cfg.retrieval.get("biobert_checkpoint", "dmis-lab/biobert-v1.1"),
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=int(cfg.retrieval.get("biobert_batch_size", 32)),
        max_length=int(cfg.retrieval.get("biobert_max_length", 512)),
    )
    biobert.build_index(corpus_jsonl, str(out), overwrite=overwrite)
    logger.success(f"BioBERT index built for {corpus_jsonl}")


def precheck_doc2query(cfg):
    bm25_dir = pathlib.Path("data/indices/bm25/unified")
    if not (bm25_dir / "index.pkl").exists():
        raise FileNotFoundError(f"BM25 index missing for Doc2Query: {bm25_dir}")
    bm25 = BM25Retriever(index_dir=str(bm25_dir))
    retriever = Doc2QueryRetriever(
        bm25=bm25,
        model_name=str(cfg.retrieval.get("doc2query_model", "doc2query/msmarco-t5-base-v1")),
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_expansions=int(cfg.retrieval.get("doc2query_num_expansions", 4)),
        max_new_tokens=int(cfg.retrieval.get("doc2query_max_new_tokens", 24)),
    )
    _ = retriever.retrieve("doc2query precheck", top_k=1)
    logger.success("Doc2Query precheck passed")


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)
    indices_root = pathlib.Path("data/indices")
    corpora = [
        ("pubmedqa_artificial", "data/corpus_pubmedqa_artificial.jsonl"),
        ("pubmedqa_labeled", "data/corpus_pubmedqa_labeled.jsonl"),
        ("medmcqa", "data/corpus_medmcqa.jsonl"),
        ("medrag_pubmed", "data/corpus_medrag_pubmed.jsonl"),
        ("unified", "data/indices/unified/corpus_unified.jsonl"),
    ]

    logger.info("=" * 72)
    logger.info("Building Neural Indices (BioBERT + Doc2Query)")
    logger.info("=" * 72)

    if args.only in ("all", "doc2query"):
        precheck_doc2query(cfg)

    if args.only in ("all", "biobert"):
        for name, corpus in corpora:
            if args.dataset != "all" and name != args.dataset:
                continue
            if not os.path.exists(corpus):
                logger.warning(f"Corpus missing: {corpus}")
                continue
            out = (indices_root / "biobert" / name).as_posix()
            build_biobert_index(corpus, out, cfg, overwrite=args.overwrite)

    logger.success("Done.")


if __name__ == "__main__":
    main()

"""
Build BioBERT indices and precheck Doc2Query model availability.
"""

import argparse
import os
import pathlib
import sys

import torch
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, ".")

from src.retrieval.biobert import BioBERTRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.doc2query import Doc2QueryRetriever


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build neural indices (BioBERT + Doc2Query).")
    p.add_argument("--cfg", type=str, default="configs/base.yaml")
    p.add_argument("--only", type=str, choices=["all", "biobert", "doc2query"], default="all")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dataset", type=str, default="all")
    return p.parse_args()


def build_biobert_index(corpus_jsonl: str, output_dir: str, cfg, overwrite: bool = False):
    out = pathlib.Path(output_dir)
    idx = out / "biobert_index.pt"
    if idx.exists() and not overwrite:
        logger.info(f"BioBERT index exists at {out}, skipping")
        return
    out.mkdir(parents=True, exist_ok=True)
    biobert = BioBERTRetriever(
        index_path=str(out),
        checkpoint=cfg.retrieval.get("biobert_checkpoint", "dmis-lab/biobert-v1.1"),
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=int(cfg.retrieval.get("biobert_batch_size", 32)),
        max_length=int(cfg.retrieval.get("biobert_max_length", 512)),
    )
    biobert.build_index(corpus_jsonl, str(out), overwrite=overwrite)
    logger.success(f"BioBERT index built for {corpus_jsonl}")


def precheck_doc2query(cfg):
    bm25_dir = pathlib.Path("data/indices/bm25/unified")
    if not (bm25_dir / "index.pkl").exists():
        raise FileNotFoundError(f"BM25 index missing for Doc2Query: {bm25_dir}")
    bm25 = BM25Retriever(index_dir=str(bm25_dir))
    retriever = Doc2QueryRetriever(
        bm25=bm25,
        model_name=str(cfg.retrieval.get("doc2query_model", "doc2query/msmarco-t5-base-v1")),
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_expansions=int(cfg.retrieval.get("doc2query_num_expansions", 4)),
        max_new_tokens=int(cfg.retrieval.get("doc2query_max_new_tokens", 24)),
    )
    _ = retriever.retrieve("doc2query precheck", top_k=1)
    logger.success("Doc2Query precheck passed")


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)
    indices_root = pathlib.Path("data/indices")
    corpora = [
        ("pubmedqa_artificial", "data/corpus_pubmedqa_artificial.jsonl"),
        ("pubmedqa_labeled", "data/corpus_pubmedqa_labeled.jsonl"),
        ("medmcqa", "data/corpus_medmcqa.jsonl"),
        ("medrag_pubmed", "data/corpus_medrag_pubmed.jsonl"),
        ("unified", "data/indices/unified/corpus_unified.jsonl"),
    ]

    logger.info("=" * 72)
    logger.info("Building Neural Indices (BioBERT + Doc2Query)")
    logger.info("=" * 72)

    if args.only in ("all", "doc2query"):
        precheck_doc2query(cfg)

    if args.only in ("all", "biobert"):
        for name, corpus in corpora:
            if args.dataset != "all" and name != args.dataset:
                continue
            if not os.path.exists(corpus):
                logger.warning(f"Corpus missing: {corpus}")
                continue
            out = (indices_root / "biobert" / name).as_posix()
            build_biobert_index(corpus, out, cfg, overwrite=args.overwrite)

    logger.success("Done.")


if __name__ == "__main__":
    main()