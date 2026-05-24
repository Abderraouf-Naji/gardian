#!/usr/bin/env python3
"""Audit E2E QA result files for Table tab:e2e-ablation (completion + citation validity)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RETRIEVERS = [
    "hybrid_spladepp_faiss",
    "hybrid_spladepp_medcpt",
    "hybrid_bm25_faiss",
    "hybrid_bm25_medcpt",
]
READERS = [
    ("meta-llama/Meta-Llama-3-8B-Instruct", "Llama-3-8B"),
    ("BioMistral/BioMistral-7B-DARE", "BioMistral-7B"),
]
SYSTEMS = ["bm25", "dense", "hybrid", "gardian"]
DATASETS = ["pubmedqa_labeled", "medmcqa"]
N_EXPECTED = 1000
MIN_CITE_FRAC = 0.05


def _extract_citations(text: str) -> List[str]:
    return re.findall(r"\[P(\d+)\]", text or "", re.I)


def _mean_metric(agg: Dict[str, Any], system: str, key: str) -> Optional[float]:
    block = agg.get(system) or {}
    ci = block.get(f"{key}_ci") or {}
    if isinstance(ci, dict) and ci.get("mean") is not None:
        return float(ci["mean"])
    arr = block.get(key)
    if isinstance(arr, list) and arr:
        return float(arr[0])
    return None


def _load_cells(paths: List[Path]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    cells: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for p in paths:
        obj = json.loads(p.read_text(encoding="utf-8"))
        meta = obj.get("meta") or {}
        reader = str(meta.get("reader_model") or "")
        retriever = str(meta.get("retriever") or "")
        if "runs" in obj and obj["runs"]:
            reader = str(obj["runs"][0].get("reader_model") or reader)
            retriever = str(obj["runs"][0].get("retriever") or retriever)
            datasets = obj["runs"][0].get("datasets") or {}
        else:
            datasets = obj.get("datasets") or {}
        cells[(reader, retriever)] = {
            "path": str(p),
            "checkpoint": bool(meta.get("checkpoint")),
            "datasets": datasets,
        }
    return cells


def _retrieval_audit(block: Dict[str, Any], system: str = "gardian") -> Dict[str, Any]:
    """Check per-question retrieval provenance when stored by live QA."""
    pq = (block.get("per_question") or {}).get(system) or []
    if not pq:
        return {"n": 0}
    with_ret = [r for r in pq if isinstance(r.get("retrieval"), dict)]
    if not with_ret:
        return {"n": len(pq), "with_retrieval_field": 0}
    ks = [int(r["retrieval"].get("k_sparse", 0)) for r in with_ret]
    kd = [int(r["retrieval"].get("k_dense", 0)) for r in with_ret]
    pools = [int(r["retrieval"].get("pool_size", 0)) for r in with_ret]
    alphas = [float(r["retrieval"].get("alpha_sparse", 0)) for r in with_ret if "alpha_sparse" in r["retrieval"]]
    return {
        "n": len(pq),
        "with_retrieval_field": len(with_ret),
        "k_sparse_median": sorted(ks)[len(ks) // 2] if ks else None,
        "k_dense_median": sorted(kd)[len(kd) // 2] if kd else None,
        "pool_size_median": sorted(pools)[len(pools) // 2] if pools else None,
        "alpha_sparse_mean": (sum(alphas) / len(alphas)) if alphas else None,
        "budget_mode": with_ret[0]["retrieval"].get("adaptive_channel_budget"),
    }


def _cite_fraction(block: Dict[str, Any], system: str) -> Optional[float]:
    pq = (block.get("per_question") or {}).get(system) or []
    if not pq:
        return None
    n = sum(1 for r in pq if _extract_citations(r.get("answer", "")))
    return n / len(pq)


def _dataset_ok(block: Optional[Dict[str, Any]], system: str) -> bool:
    if not block:
        return False
    agg = block.get("aggregate") or {}
    sys_block = agg.get(system) or {}
    n = sys_block.get("n_questions")
    if n != N_EXPECTED:
        return False
    pq = (block.get("per_question") or {}).get(system) or []
    return len(pq) == N_EXPECTED


def main() -> None:
    p = argparse.ArgumentParser(description="Audit QA JSONs for the E2E ablation table.")
    p.add_argument(
        "inputs",
        nargs="*",
        default=["results"],
        help="Files or directories of qa_*.json (default: results/)",
    )
    p.add_argument("--min-cite-frac", type=float, default=MIN_CITE_FRAC)
    args = p.parse_args()

    paths: List[Path] = []
    for raw in args.inputs:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.glob("qa_*.json")))
        elif path.is_file():
            paths.append(path)

    cells = _load_cells(paths)
    print(f"Found {len(paths)} QA file(s), {len(cells)} unique (reader, retriever) cell(s)\n")

    missing: List[str] = []
    invalid_cite: List[str] = []

    for ret in RETRIEVERS:
        print(f"=== {ret} ===")
        for reader_id, reader_label in READERS:
            cell = cells.get((reader_id, ret))
            if cell is None:
                print(f"  {reader_label}: MISSING")
                missing.append(f"{ret} | {reader_label}")
                continue
            ck = " [checkpoint]" if cell["checkpoint"] else ""
            print(f"  {reader_label}: {Path(cell['path']).name}{ck}")
            for ds in DATASETS:
                block = cell["datasets"].get(ds)
                if not block:
                    print(f"    {ds}: MISSING")
                    missing.append(f"{ret} | {reader_label} | {ds}")
                    continue
                agg = block.get("aggregate") or {}
                ok = all(_dataset_ok(block, s) for s in SYSTEMS)
                tag = "OK" if ok else "INCOMPLETE"
                print(f"    {ds}: {tag} (n={[(agg.get(s) or {}).get('n_questions') for s in SYSTEMS]})")
                if ds == "pubmedqa_labeled":
                    ret_audit = _retrieval_audit(block, "gardian")
                    if ret_audit.get("with_retrieval_field", 0) > 0:
                        print(
                            f"      retrieval: k_sparse={ret_audit.get('k_sparse_median')} "
                            f"k_dense={ret_audit.get('k_dense_median')} "
                            f"pool≈{ret_audit.get('pool_size_median')} "
                            f"budget={ret_audit.get('budget_mode')!r} "
                            f"α_mean={ret_audit.get('alpha_sparse_mean'):.2f}"
                            if ret_audit.get("alpha_sparse_mean") is not None
                            else f"      retrieval: pool≈{ret_audit.get('pool_size_median')} "
                            f"budget={ret_audit.get('budget_mode')!r}"
                        )
                    for s in SYSTEMS:
                        cf = _cite_fraction(block, s)
                        if cf is not None and cf < args.min_cite_frac:
                            print(f"      WARN {s}: citation tags in {cf:.1%} of answers (<{args.min_cite_frac:.0%})")
                            invalid_cite.append(f"{ret} | {reader_label} | {s}")
        print()

    print("--- Missing runs (copy commands below) ---")
    for m in missing:
        print(f"  - {m}")

    if invalid_cite:
        print("\n--- Re-run needed (citation metrics not trustworthy) ---")
        for x in invalid_cite:
            print(f"  - {x}")

    print("\n--- Run template (one cell) ---")
    print(
        "CUDA_VISIBLE_DEVICES=0 python scripts/06_end_to_end_qa.py \\\n"
        "  --online-retrieval --pubmedqa-open-domain --gardian-adaptive-retrieval \\\n"
        "  --checkpoint-every-dataset \\\n"
        "  --datasets pubmedqa_labeled,medmcqa --max-questions 1000 \\\n"
        "  --systems bm25,dense,hybrid,gardian \\\n"
        "  --retriever <FAMILY> \\\n"
        "  --reader-models <READER> \\\n"
        "  --out results/qa_<FAMILY>_<READER_SLUG>.json"
    )


if __name__ == "__main__":
    main()
