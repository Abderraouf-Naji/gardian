"""GARDIAN RAG reader package (v2)."""

from src.pipeline.rag.metrics import compute_citation_metrics
from src.pipeline.rag.parser import extract_citations
from src.pipeline.rag.reader import RAGReader, run_rag_reader
from src.pipeline.rag.reader_types import (
    EVAL_SYSTEMS,
    ReaderConfig,
    ReaderTask,
    normalize_system_name,
)

__all__ = [
    "EVAL_SYSTEMS",
    "RAGReader",
    "ReaderConfig",
    "ReaderTask",
    "compute_citation_metrics",
    "extract_citations",
    "normalize_system_name",
    "run_rag_reader",
]
