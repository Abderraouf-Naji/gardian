"""
Measure whether the question-type one-hot carries usable signal.

Regenerates every number in ``docs/QUESTION_TYPE.md``:

  1. type distribution per dataset (is the one-hot constant?)
  2. whether the one-hot separates datasets (does it leak dataset identity?)
  3. declared categories that never occur

    python scripts/analyze_question_types.py
    python scripts/analyze_question_types.py --json results/question_type_analysis.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import pathlib
import sys
from typing import Any, Dict

os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, ".")

from omegaconf import OmegaConf  # noqa: E402

from src.common.question_types import normalize_question_type  # noqa: E402


def dataset_of(path: str) -> str:
    """Infer the benchmark from a rank-data filename."""
    for name in ("pubmedqa_labeled", "pubmedqa_artificial", "medmcqa"):
        if name in path:
            return name
    return "unknown"


def query_types(path: str, max_lines: int = 0) -> Dict[str, str]:
    """
    Map qid -> question type for one rank file (first row per query wins).

    ``max_lines`` caps the scan; 0 reads the whole file. Rank rows are
    query-contiguous, so a cap yields a prefix of the queries rather than a
    biased sample of each query.
    """
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = rec.get("qid")
            if qid is None or qid in out:
                continue
            out[qid] = normalize_question_type(rec.get("question_type"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rank-glob",
        default="data/hybrid_*/rank_data_*.jsonl",
        help="Rank JSONL files to scan.",
    )
    parser.add_argument("--json", default=None, help="Optional path to write the report.")
    parser.add_argument(
        "--max-lines-per-file",
        type=int,
        default=200_000,
        help=(
            "Cap lines read per rank file during the dataset-leakage scan "
            "(0 = read everything). The per-dataset distribution in section 1 "
            "always reads its evaluation file in full."
        ),
    )
    args = parser.parse_args()

    cfg = OmegaConf.load("configs/base.yaml")
    declared = list(cfg.evaluation.question_types)

    # Per evaluation split, using one canonical file per dataset.
    eval_files = {
        "pubmedqa_labeled": "data/hybrid_bm25_faiss/rank_data_hybrid_bm25_faiss_pubmedqa_labeled_eval.jsonl",
        "pubmedqa_artificial": "data/hybrid_bm25_faiss/rank_data_hybrid_bm25_faiss_pubmedqa_artificial_test.jsonl",
        "medmcqa": "data/hybrid_bm25_faiss/rank_data_hybrid_bm25_faiss_medmcqa_test.jsonl",
    }

    report: Dict[str, Any] = {"declared_categories": declared, "per_dataset": {}}

    print("=" * 72)
    print("1. TYPE DISTRIBUTION PER EVALUATION SET")
    print("=" * 72)
    for ds, path in eval_files.items():
        if not os.path.exists(path):
            print(f"  {ds}: MISSING {path}")
            continue
        types = query_types(path)
        counts = collections.Counter(types.values())
        constant = len(counts) <= 1
        report["per_dataset"][ds] = {
            "n_queries": len(types),
            "distinct_types": len(counts),
            "counts": dict(counts),
            "one_hot_is_constant": constant,
        }
        print(f"\n  {ds}: {len(types)} queries, {len(counts)} distinct type(s)")
        for t, n in counts.most_common():
            print(f"      {t:<18} {n:>6}  ({100 * n / len(types):5.1f}%)")
        print(f"      -> {'CONSTANT: zero information' if constant else 'informative'}")

    # Dataset leakage across every rank file.
    print("\n" + "=" * 72)
    print("2. DOES THE ONE-HOT ENCODE DATASET IDENTITY?")
    print("=" * 72)
    by_ds: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for path in sorted(glob.glob(args.rank_glob)):
        ds = dataset_of(path)
        if ds == "unknown":
            continue
        for t in query_types(path, args.max_lines_per_file).values():
            by_ds[ds][t] += 1

    leak = {}
    for ds in sorted(by_ds):
        total = sum(by_ds[ds].values())
        yesno = by_ds[ds].get("yesno", 0)
        share = 100.0 * yesno / total if total else 0.0
        leak[ds] = {"n": total, "yesno": yesno, "yesno_share_pct": round(share, 2)}
        print(f"  {ds:<22} yesno = {yesno:>7}/{total:<7} ({share:5.1f}%)")
    report["dataset_leakage"] = leak
    report["leakage_scan_max_lines_per_file"] = int(args.max_lines_per_file)

    pubmed = [d for d in leak if d.startswith("pubmedqa")]
    separates = (
        all(leak[d]["yesno_share_pct"] == 100.0 for d in pubmed)
        and leak.get("medmcqa", {}).get("yesno_share_pct", 0.0) == 0.0
    )
    report["one_hot_separates_datasets"] = bool(separates)
    if separates:
        print("\n  -> the 'yesno' bit separates PubMedQA from MedMCQA PERFECTLY.")
        print("     In the combined training file it acts as a dataset indicator,")
        print("     so the controller can adapt to the dataset, not only the query.")

    # Unused declared categories.
    print("\n" + "=" * 72)
    print("3. DECLARED CATEGORIES THAT NEVER OCCUR")
    print("=" * 72)
    overall: collections.Counter = collections.Counter()
    for counter in by_ds.values():
        overall.update(counter)
    unused = [t for t in declared if overall.get(t, 0) == 0]
    for t in declared:
        n = overall.get(t, 0)
        flag = "  <-- NEVER USED" if n == 0 else ""
        print(f"  {t:<18} {n:>7}{flag}")
    report["category_counts"] = dict(overall)
    report["unused_categories"] = unused
    if unused:
        print(
            f"\n  -> {len(unused)} declared categor{'y' if len(unused) == 1 else 'ies'} "
            f"never fire(s): {unused}. A conditioned controller carries that many "
            "permanently-zero input dimensions."
        )

    if args.json:
        out = pathlib.Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
