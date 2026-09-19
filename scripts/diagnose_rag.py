"""
Diagnose where end-to-end RAG accuracy is actually lost.

Answers four questions the paper currently asserts without evidence:

  A. Does the system beat the dataset's own majority-class baseline?
  B. When the gold evidence IS in context, how often is the answer still wrong?
     (separates reader failure from retrieval failure)
  C. What is the dominant error mode? (verdict confusion matrix)
  D. Is the ceiling set by first-stage retrieval or by ranking?
     (pool_recall@K' vs the oracle re-ranking bound -- reviewer 1's question)

    python scripts/diagnose_rag.py --qa-json results/qa_hybrid_bm25_faiss_Llama3-8B_n300.json
    python scripts/diagnose_rag.py --pool-only          # D only, no QA run needed
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import pathlib
import re
import statistics
import sys
from typing import Any, Dict, List

os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, ".")

from src.evaluation.metrics import ndcg_at_k  # noqa: E402

VERDICT_RE = re.compile(r"(?i)\banswer\s*:\s*(yes|no|maybe)\b")
CITE_RE = re.compile(r"\[P(\d+)\]")


def load_gold(eval_jsonl: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(eval_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["id"]] = rec
    return out


def majority_baseline(gold: Dict[str, Dict[str, Any]]) -> tuple[float, str, collections.Counter]:
    counts = collections.Counter(
        str(r.get("answer") or "").strip().lower() for r in gold.values()
    )
    total = sum(counts.values())
    label, n = counts.most_common(1)[0]
    return (n / total if total else 0.0), label, counts


def analyse_reader(
    qa_json: str, dataset: str, system: str, gold: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    qa = json.load(open(qa_json, encoding="utf-8"))
    recs = qa["datasets"][dataset]["per_question"][system]

    conf: collections.Counter = collections.Counter()
    n_cites: List[int] = []
    n_gold: List[int] = []
    ctx_hit = ctx_hit_wrong = 0
    n = 0

    for r in recs:
        src = gold.get(r["qid"])
        if src is None:
            continue
        n += 1
        g = str(src.get("answer") or "").strip().lower()
        m = VERDICT_RE.search(r.get("answer") or "")
        p = m.group(1).lower() if m else "unparseable"
        conf[(g, p)] += 1
        n_cites.append(len(set(CITE_RE.findall(r.get("answer") or ""))))
        n_gold.append(len(src.get("gold_passage_ids") or []))
        if r.get("gold_evidence_in_context_rate", 0.0) == 1.0:
            ctx_hit += 1
            if r.get("accuracy", 0.0) == 0.0:
                ctx_hit_wrong += 1

    labels = ["yes", "no", "maybe"]
    correct = sum(conf[(g, g)] for g in labels)
    hedge = sum(conf[(g, "maybe")] for g in ("yes", "no"))
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "confusion": {f"{g}->{p}": c for (g, p), c in sorted(conf.items())},
        "gold_in_context": ctx_hit,
        "gold_in_context_but_wrong": ctx_hit_wrong,
        "reader_failure_rate_given_evidence": ctx_hit_wrong / ctx_hit if ctx_hit else None,
        "hedged_to_maybe_on_decidable": hedge,
        "hedge_rate": hedge / n if n else 0.0,
        "mean_citations_emitted": statistics.fmean(n_cites) if n_cites else 0.0,
        "mean_gold_passages": statistics.fmean(n_gold) if n_gold else 0.0,
        "citation_coverage": (
            statistics.fmean(n_cites) / statistics.fmean(n_gold) if n_gold and statistics.fmean(n_gold) else None
        ),
        "_conf_counter": conf,
    }


def pool_and_oracle(rank_path: str, k: int = 10) -> Dict[str, float]:
    """pool_recall@K' and the nDCG@k of a perfect ranking of that same pool."""
    positives: Dict[str, List[str]] = collections.defaultdict(list)
    pool: Dict[str, List[str]] = collections.defaultdict(list)
    with open(rank_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pool[rec["qid"]].append(rec["pid"])
            if rec["label"] == 1:
                positives[rec["qid"]].append(rec["pid"])

    ndcgs, hits, sizes = [], [], []
    for qid, pids in pool.items():
        rel = positives.get(qid, [])
        hits.append(1.0 if rel else 0.0)
        sizes.append(len(pids))
        ranked = rel + [p for p in pids if p not in set(rel)]
        ndcgs.append(ndcg_at_k(ranked, rel, k))
    return {
        "pool_recall": statistics.fmean(hits) if hits else 0.0,
        "oracle_ndcg@10": statistics.fmean(ndcgs) if ndcgs else 0.0,
        "mean_pool_size": statistics.fmean(sizes) if sizes else 0.0,
        "n_queries": len(pool),
    }


def print_confusion(conf: collections.Counter, n: int) -> None:
    labels = ["yes", "no", "maybe"]
    cols = labels + ["unparseable"]
    header = "gold\\pred"
    print(f"    {header:<12}" + "".join(f"{c:>13}" for c in cols) + f"{'recall':>9}")
    for g in labels:
        tot = sum(conf[(g, p)] for p in cols)
        row = f"    {g:<12}" + "".join(f"{conf[(g, p)]:>13}" for p in cols)
        row += f"{(conf[(g, g)] / tot if tot else 0):>9.3f}"
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qa-json", default=None)
    ap.add_argument("--dataset", default="pubmedqa_labeled")
    ap.add_argument("--system", default="gardian")
    ap.add_argument("--eval-jsonl", default="data/pubmedqa_labeled_eval.jsonl")
    ap.add_argument("--rank-glob", default="data/hybrid_*/rank_data_*_{ds}_{split}.jsonl")
    ap.add_argument("--eval-json-glob", default="results/seeds/seed_42/evaluation_hybrid_*.json")
    ap.add_argument("--pool-only", action="store_true")
    ap.add_argument("--out", default="results/rag_diagnosis.json")
    args = ap.parse_args()

    report: Dict[str, Any] = {}

    if not args.pool_only and args.qa_json:
        gold = load_gold(args.eval_jsonl)
        base, base_label, dist = majority_baseline(gold)
        res = analyse_reader(args.qa_json, args.dataset, args.system, gold)

        print("=" * 92)
        print("A. IS THE SYSTEM BEATING THE DATASET'S OWN MAJORITY-CLASS BASELINE?")
        print("=" * 92)
        total = sum(dist.values())
        for lab, c in dist.most_common():
            print(f"    gold {lab!r:<8} {c:>5} ({100 * c / total:5.1f}%)")
        print(f"\n    majority-class baseline (always answer {base_label!r}): {base:.4f}")
        print(f"    system accuracy                                  : {res['accuracy']:.4f}")
        verdict = "ABOVE" if res["accuracy"] > base else "AT OR BELOW"
        print(f"    -> {verdict} baseline (delta {res['accuracy'] - base:+.4f})")

        print("\n" + "=" * 92)
        print("B. WHEN THE GOLD EVIDENCE IS IN CONTEXT, IS THE ANSWER RIGHT?")
        print("=" * 92)
        print(f"    gold evidence in context : {res['gold_in_context']}/{res['n']}")
        print(f"    ...yet answered WRONG    : {res['gold_in_context_but_wrong']}"
              f"  ({100 * (res['reader_failure_rate_given_evidence'] or 0):.1f}% of them)")
        print("    -> this share of the error is the READER, not retrieval.")

        print("\n" + "=" * 92)
        print("C. DOMINANT ERROR MODE")
        print("=" * 92)
        print_confusion(res["_conf_counter"], res["n"])
        print(f"\n    decidable questions (gold yes/no) answered 'maybe': "
              f"{res['hedged_to_maybe_on_decidable']} ({100 * res['hedge_rate']:.1f}% of all)")
        print(f"    citations emitted per answer : {res['mean_citations_emitted']:.2f}")
        print(f"    gold passages per question   : {res['mean_gold_passages']:.2f}")
        print(f"    -> citation coverage         : {(res['citation_coverage'] or 0):.2f}"
              "  (caps citation_recall)")
        res.pop("_conf_counter")
        report["reader"] = {
            "qa_json": args.qa_json,
            "dataset": args.dataset,
            "system": args.system,
            "majority_baseline": base,
            "majority_label": base_label,
            **res,
        }

    print("\n" + "=" * 92)
    print("D. IS THE CEILING FIRST-STAGE RETRIEVAL, OR RANKING?")
    print("=" * 92)

    gardian_ndcg: Dict[tuple, float] = {}
    for path in sorted(glob.glob(args.eval_json_glob)):
        if "_cross" in path:
            continue
        payload = json.load(open(path, encoding="utf-8"))
        for retriever, ds_block in payload.get("results", {}).items():
            for ds, systems in ds_block.items():
                if not ds.startswith("_") and isinstance(systems, dict):
                    val = (systems.get("gardian") or {}).get("ndcg@10")
                    if val is not None:
                        gardian_ndcg[(retriever, ds)] = float(val)

    print(f"    {'backend':<24}{'dataset':<20}{'pool_rec':>9}{'oracle':>8}"
          f"{'GARDIAN':>9}{'lost:rank':>11}{'lost:pool':>11}")
    print("    " + "-" * 88)
    rows = []
    for backend in ("hybrid_bm25_faiss", "hybrid_bm25_medcpt",
                    "hybrid_spladepp_faiss", "hybrid_spladepp_medcpt"):
        for ds, split in (("pubmedqa_labeled", "eval"),
                          ("pubmedqa_artificial", "test"),
                          ("medmcqa", "test")):
            path = f"data/{backend}/rank_data_{backend}_{ds}_{split}.jsonl"
            if not os.path.exists(path):
                continue
            st = pool_and_oracle(path)
            g = gardian_ndcg.get((backend, ds))
            row = {"backend": backend, "dataset": ds, **st, "gardian_ndcg@10": g}
            if g is not None:
                row["lost_to_ranking"] = st["oracle_ndcg@10"] - g
                row["lost_to_pool"] = 1.0 - st["pool_recall"]
                print(f"    {backend:<24}{ds:<20}{st['pool_recall']:>9.3f}"
                      f"{st['oracle_ndcg@10']:>8.3f}{g:>9.3f}"
                      f"{row['lost_to_ranking']:>11.3f}{row['lost_to_pool']:>11.3f}")
            rows.append(row)
    report["pool_vs_ranking"] = rows

    scored = [r for r in rows if r.get("lost_to_ranking") is not None]
    if scored:
        print("\n    lost:rank = nDCG@10 recoverable by better ranking of the SAME pool")
        print("    lost:pool = queries with no gold passage retrieved at all")
        by_ds: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in scored:
            by_ds[r["dataset"]].append(r)
        print()
        for ds, rs in by_ds.items():
            lr = statistics.fmean(r["lost_to_ranking"] for r in rs)
            lp = statistics.fmean(r["lost_to_pool"] for r in rs)
            dominant = "RANKING" if lr > lp else "FIRST-STAGE RETRIEVAL"
            ratio = lr / lp if lp > 1e-9 else float("inf")
            ratio_s = f"{ratio:.1f}x" if ratio != float("inf") else "all of it"
            print(f"    {ds:<22} mean lost:rank={lr:.3f}  lost:pool={lp:.3f}"
                  f"   -> {dominant} dominates ({ratio_s})")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
