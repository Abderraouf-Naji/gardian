"""
Build SPLADEv3 and ColBERT indices.
"""

import argparse
import os
import pathlib
import sys

import torch
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, ".")

from src.retrieval.colbert import ColBERTRetriever
from src.retrieval.spladev3 import SpladeV3Retriever


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build neural indices (SPLADEv3 + ColBERT).")
    p.add_argument("--cfg", type=str, default="configs/base.yaml")
    p.add_argument("--only", type=str, choices=["all", "spladev3", "colbert"], default="all")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dataset", type=str, default="all")
    return p.parse_args()


def main() -> None:
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 72)
    logger.info("Building Neural Indices (SPLADEv3 + ColBERT)")
    logger.info("=" * 72)

    for name, corpus in corpora:
        if args.dataset != "all" and name != args.dataset:
            continue
        if not os.path.exists(corpus):
            logger.warning(f"Corpus missing: {corpus}")
            continue

        if args.only in ("all", "spladev3"):
            retriever = SpladeV3Retriever(
                index_path=(indices_root / "spladev3" / name).as_posix(),
                model_name=str(cfg.retrieval.get("spladev3_encoder", "naver/splade-v3-distilbert")),
                device=device,
                batch_size=int(cfg.retrieval.get("spladev3_batch_size", 256)),
                max_length=int(cfg.retrieval.get("spladev3_max_length", 256)),
            )
            retriever.build_index(
                corpus_jsonl=corpus,
                output_dir=(indices_root / "spladev3" / name).as_posix(),
                overwrite=bool(args.overwrite),
            )

        if args.only in ("all", "colbert"):
            retriever = ColBERTRetriever(
                index_path=(indices_root / "colbert" / name).as_posix(),
                model_name=str(cfg.retrieval.get("colbert_encoder", "colbert-ir/colbertv2.0")),
                device=device,
                batch_size=int(cfg.retrieval.get("colbert_batch_size", 256)),
                max_length=int(cfg.retrieval.get("colbert_max_length", 256)),
            )
            retriever.build_index(
                corpus_jsonl=corpus,
                output_dir=(indices_root / "colbert" / name).as_posix(),
                overwrite=bool(args.overwrite),
            )

    logger.success("Done.")


if __name__ == "__main__":
    main()
