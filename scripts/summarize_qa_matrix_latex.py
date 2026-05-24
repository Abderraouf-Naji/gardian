#!/usr/bin/env python3
"""Print LaTeX rows for the E2E QA ablation table from ``06`` matrix JSON output."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple

# Paper table order (SPLADE++ families first, then BM25).
RETRIEVER_ORDER = [
    ("hybrid_spladepp_faiss", "SPLADEpp{+}FAISS"),
    ("hybrid_spladepp_medcpt", "SPLADEpp{+}MedCPT"),
    ("hybrid_bm25_faiss", "BM25{+}FAISS"),
    ("hybrid_bm25_medcpt", "BM25{+}MedCPT"),
]

READER_ORDER = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "BioMistral/BioMistral-7B-DARE",
    "bio-mistral/BioMistral-7B",
]

FAMILY_SYSTEMS: Dict[str, List[Tuple[str, str]]] = {
    "hybrid_bm25_faiss": [
        ("sparse", "Sparse (BM25)"),
        ("dense", "Dense (FAISS)"),
        ("hybrid", "Hybrid RAG"),
        ("gardian", "RAG{+}GARDIAN"),
    ],
    "hybrid_bm25_medcpt": [
        ("sparse", "Sparse (BM25)"),
        ("dense", "Dense (MedCPT)"),
        ("hybrid", "Hybrid RAG"),
        ("gardian", "RAG{+}GARDIAN"),
    ],
    "hybrid_spladepp_faiss": [
        ("sparse", "Sparse (SPLADEpp)"),
        ("dense", "Dense (FAISS)"),
        ("hybrid", "Hybrid RAG"),
        ("gardian", "RAG{+}GARDIAN"),
    ],
    "hybrid_spladepp_medcpt": [
        ("sparse", "Sparse (SPLADEpp)"),
        ("dense", "Dense (MedCPT)"),
        ("hybrid", "Hybrid RAG"),
        ("gardian", "RAG{+}GARDIAN"),
    ],
}

READER_LABEL = {
    "meta-llama/Meta-Llama-3-8B-Instruct": "Llama-3-8B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "BioMistral/BioMistral-7B-DARE": "BioMistral-7B",
    "bio-mistral/BioMistral-7B": "BioMistral-7B",
}

PQA_METRICS = 4  # Acc, CitP, CitR, Unsupp
MCQ_METRICS = 1  # Acc only
METRICS_PER_READER = PQA_METRICS + MCQ_METRICS
PUBMEDQA_LATEX = "PubMedQA-L"
MEDMCQA_LATEX = "MedMCQA"
MIN_CITE_FRAC_DEFAULT = 0.05


def _extract_citations(text: str) -> List[str]:
    return re.findall(r"\[P(\d+)\]", text or "", re.I)


def _cite_fraction(block: Dict[str, Any], system: str) -> Optional[float]:
    pq = (block.get("per_question") or {}).get(system) or []
    if not pq:
        return None
    n = sum(1 for r in pq if _extract_citations(r.get("answer", "")))
    return n / len(pq)


def _agg_block(agg: Dict[str, Any], system: str) -> Dict[str, Any]:
    if system in agg:
        return agg.get(system) or {}
    if system == "sparse" and "bm25" in agg:
        return agg.get("bm25") or {}
    return {}


def _mean_metric(agg: Dict[str, Any], system: str, key: str) -> Optional[float]:
    block = _agg_block(agg, system)
    ci = block.get(f"{key}_ci") or {}
    if isinstance(ci, dict) and ci.get("mean") is not None:
        return float(ci["mean"])
    arr = block.get(key)
    if isinstance(arr, list) and arr:
        return float(arr[0])
    return None


def _fmt(v: Optional[float], *, digits: int = 3) -> str:
    if v is None:
        return "--"
    return f"{v:.{digits}f}"


def _collect_input_paths(raw: List[str]) -> List[pathlib.Path]:
    paths: List[pathlib.Path] = []
    for item in raw:
        p = pathlib.Path(item)
        if p.is_dir():
            paths.extend(sorted(p.glob("qa_*.json")))
        elif p.is_file():
            paths.append(p)
    return paths


def _load_matrix(paths: List[pathlib.Path]) -> Tuple[List[str], Dict[Tuple[str, str, str], Dict[str, Any]], Dict[Tuple[str, str, str], Dict[str, Any]]]:
    """Return readers, aggregate cells, full dataset blocks (for citation audit)."""
    readers: List[str] = []
    cells: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    blocks: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def ingest(reader: str, retriever: str, datasets: Dict[str, Any]) -> None:
        if reader and reader not in readers:
            readers.append(reader)
        for ds_name, block in datasets.items():
            cells[(reader, retriever, ds_name)] = block.get("aggregate") or {}
            blocks[(reader, retriever, ds_name)] = block

    if len(paths) == 1:
        obj = json.loads(paths[0].read_text(encoding="utf-8"))
        if "runs" in obj:
            for run in obj["runs"]:
                ingest(
                    str(run.get("reader_model", "")),
                    str(run.get("retriever", "")),
                    run.get("datasets") or {},
                )
        elif "datasets" in obj:
            meta = obj.get("meta") or {}
            ingest(
                str(meta.get("reader_model", "")),
                str(meta.get("retriever", "")),
                obj["datasets"],
            )
        else:
            raise ValueError(f"{paths[0]}: need 'runs' or 'datasets'")
    else:
        for path in paths:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if "runs" in obj and obj["runs"]:
                run = obj["runs"][0]
                ingest(
                    str(run.get("reader_model", "")),
                    str(run.get("retriever", "")),
                    run.get("datasets") or {},
                )
            elif "datasets" in obj:
                meta = obj.get("meta") or {}
                ingest(
                    str(meta.get("reader_model", "")),
                    str(meta.get("retriever", "")),
                    obj["datasets"],
                )
            else:
                raise ValueError(f"{path}: unrecognized QA JSON")

    ordered = [r for r in READER_ORDER if r in readers] + [r for r in readers if r not in READER_ORDER]
    return ordered, cells, blocks


def _reader_metric_cells(
    agg_pqa: Dict[str, Any],
    agg_mcq: Dict[str, Any],
    system: str,
    pqa_block: Optional[Dict[str, Any]],
    *,
    min_cite_frac: float,
) -> List[str]:
    """One reader block: PQA Acc, CitP, CitR, Unsupp, MCQ Acc."""
    cite_frac = _cite_fraction(pqa_block, system) if pqa_block else None
    cite_ok = cite_frac is None or cite_frac >= min_cite_frac
    if cite_ok:
        cit_cells = [
            _fmt(_mean_metric(agg_pqa, system, "citation_precision")),
            _fmt(_mean_metric(agg_pqa, system, "citation_recall")),
            _fmt(_mean_metric(agg_pqa, system, "unsupported_claim_rate")),
        ]
    else:
        cit_cells = ["--", "--", "--"]
    return [
        _fmt(_mean_metric(agg_pqa, system, "answer_accuracy")),
        *cit_cells,
        _fmt(_mean_metric(agg_mcq, system, "answer_accuracy")),
    ]


def print_tabular_header(readers: List[str]) -> None:
    n = len(readers)
    span = METRICS_PER_READER
    print("% --- paste below \\midrule ---")
    row1 = ["", ""] + [
        f"\\multicolumn{{{span}}}{{c}}{{\\textbf{{{READER_LABEL.get(r, r.split('/')[-1][:12])}}}}}"
        for r in readers
    ]
    print(" & ".join(row1) + " \\\\")
    cmid_reader = []
    base = 3
    for i in range(n):
        lo, hi = base + i * span, base + (i + 1) * span - 1
        cmid_reader.append(f"\\cmidrule(lr){{{lo}-{hi}}}")
    print(" ".join(cmid_reader))
    row2 = ["Hybrid family", "System"]
    for _ in readers:
        row2.append(f"\\multicolumn{{{PQA_METRICS}}}{{c}}{{{PUBMEDQA_LATEX}}}")
        row2.append(f"\\multicolumn{{{MCQ_METRICS}}}{{c}}{{{MEDMCQA_LATEX}}}")
    print(" & ".join(row2) + " \\\\")
    cmid_ds = []
    for i in range(n):
        pqa_lo = base + i * span
        pqa_hi = pqa_lo + PQA_METRICS - 1
        mcq_col = pqa_hi + 1
        cmid_ds.append(f"\\cmidrule(lr){{{pqa_lo}-{pqa_hi}}}")
        cmid_ds.append(f"\\cmidrule(lr){{{mcq_col}-{mcq_col}}}")
    print(" ".join(cmid_ds))
    row3 = ["", ""]
    for _ in readers:
        row3.extend(["Acc.", "Cit.P", "Cit.R", "Uns.$\\downarrow$", "Acc."])
    print(" & ".join(row3) + " \\\\")


def print_table_body(
    readers: List[str],
    cells: Dict[Tuple[str, str, str], Dict[str, Any]],
    blocks: Dict[Tuple[str, str, str], Dict[str, Any]],
    *,
    pubmedqa_ds: str = "pubmedqa_labeled",
    medmcqa_ds: str = "medmcqa",
    min_cite_frac: float = MIN_CITE_FRAC_DEFAULT,
) -> None:
    for ret_key, ret_label in RETRIEVER_ORDER:
        systems = FAMILY_SYSTEMS[ret_key]
        for si, (sys_key, sys_label) in enumerate(systems):
            row: List[str] = []
            if si == 0:
                row.append(f"\\multirow{{{len(systems)}}}{{*}}{{{ret_label}}}")
            else:
                row.append("")
            row.append(sys_label)
            for reader in readers:
                pqa = cells.get((reader, ret_key, pubmedqa_ds), {})
                mcq = cells.get((reader, ret_key, medmcqa_ds), {})
                pqa_block = blocks.get((reader, ret_key, pubmedqa_ds))
                row.extend(
                    _reader_metric_cells(
                        pqa, mcq, sys_key, pqa_block, min_cite_frac=min_cite_frac
                    )
                )
            print(" & ".join(row) + " \\\\")
        print("\\addlinespace[3pt]")


def main() -> None:
    p = argparse.ArgumentParser(
        description="LaTeX E2E ablation table from QA JSON (single matrix file or qa_*.json glob)."
    )
    p.add_argument(
        "json_paths",
        nargs="*",
        default=["results"],
        help="Matrix JSON, cell JSON(s), or directory (default: results/)",
    )
    p.add_argument("--pubmedqa-dataset", type=str, default="pubmedqa_labeled")
    p.add_argument("--medmcqa-dataset", type=str, default="medmcqa")
    p.add_argument("--header-only", action="store_true")
    p.add_argument("--body-only", action="store_true")
    p.add_argument(
        "--min-cite-frac",
        type=float,
        default=MIN_CITE_FRAC_DEFAULT,
        help="If fewer answers contain [P#], emit -- for Cit.P/R/Uns. (honest reporting).",
    )
    args = p.parse_args()
    paths = _collect_input_paths(args.json_paths)
    if not paths:
        raise SystemExit("No QA JSON files found.")
    readers, cells, blocks = _load_matrix(paths)
    if not readers:
        raise SystemExit("No readers in JSON.")
    if not args.body_only:
        print_tabular_header(readers)
    if not args.header_only:
        print_table_body(
            readers,
            cells,
            blocks,
            pubmedqa_ds=args.pubmedqa_dataset,
            medmcqa_ds=args.medmcqa_dataset,
            min_cite_frac=args.min_cite_frac,
        )
    print(
        f"% Built from {len(paths)} file(s); readers={', '.join(READER_LABEL.get(r, r) for r in readers)}"
    )


if __name__ == "__main__":
    main()
