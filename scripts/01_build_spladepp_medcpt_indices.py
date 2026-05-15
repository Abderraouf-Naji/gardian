"""
Build the two neural first-stage indices used by GARDIAN:

  * SPLADE++ — real sparse-MLM (vocabulary-space), stored as a CSR ``.pt``.
  * MedCPT   — NCBI biomedical dense (asymmetric Q/A encoders), stored as
               an ``.npy`` matrix + a ``.jsonl`` metadata file.

One folder per corpus is created under ``data/indices/{spladepp,medcpt}/<corpus>/``.

Examples:

    # Build everything for every corpus
    python scripts/01_build_spladepp_medcpt_indices.py --only all --dataset all

    # Just MedCPT for the medmcqa corpus
    python scripts/01_build_spladepp_medcpt_indices.py --only medcpt --dataset medmcqa

    # Force rebuild even if the index files already exist
    python scripts/01_build_spladepp_medcpt_indices.py --only spladepp --overwrite
"""

import argparse
import os
import pathlib
import sys

import torch
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, ".")

from src.retrieval.medcpt import MedCPTRetriever
from src.retrieval.spladepp import SpladePPRetriever


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build neural indices (real SPLADE++ + MedCPT).")
    p.add_argument("--cfg", type=str, default="configs/base.yaml")
    p.add_argument("--only", type=str, choices=["all", "spladepp", "medcpt"], default="all")
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
    logger.info("Building Neural Indices (real SPLADE++ + MedCPT)")
    logger.info("=" * 72)

    for name, corpus in corpora:
        if args.dataset != "all" and name != args.dataset:
            continue
        if not os.path.exists(corpus):
            logger.warning(f"Corpus missing: {corpus}")
            continue

        if args.only in ("all", "spladepp"):
            retriever = SpladePPRetriever(
                index_path=(indices_root / "spladepp" / name).as_posix(),
                model_name=str(
                    cfg.retrieval.get(
                        "spladepp_encoder",
                        "naver/splade-cocondenser-ensembledistil",
                    )
                ),
                device=device,
                batch_size=int(cfg.retrieval.get("spladepp_batch_size", 64)),
                max_length=int(cfg.retrieval.get("spladepp_max_length", 256)),
            )
            retriever.build_index(
                corpus_jsonl=corpus,
                output_dir=(indices_root / "spladepp" / name).as_posix(),
                overwrite=bool(args.overwrite),
            )

        if args.only in ("all", "medcpt"):
            retriever = MedCPTRetriever(
                index_path=(indices_root / "medcpt" / name).as_posix(),
                article_encoder=str(
                    cfg.retrieval.get(
                        "medcpt_article_encoder", "ncbi/MedCPT-Article-Encoder"
                    )
                ),
                query_encoder=str(
                    cfg.retrieval.get(
                        "medcpt_query_encoder", "ncbi/MedCPT-Query-Encoder"
                    )
                ),
                device=device,
                batch_size=int(cfg.retrieval.get("medcpt_batch_size", 256)),
                max_length=int(cfg.retrieval.get("medcpt_max_length", 512)),
            )
            retriever.build_index(
                corpus_jsonl=corpus,
                output_dir=(indices_root / "medcpt" / name).as_posix(),
                overwrite=bool(args.overwrite),
            )

    logger.success("Done.")


if __name__ == "__main__":
    main()
