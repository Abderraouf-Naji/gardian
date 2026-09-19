#!/usr/bin/env python3
"""Paired bootstrap significance for the end-to-end QA table.

Usage:
  .venv/bin/python scripts/e2e_qa_significance.py results/rq3_1k/*.json

Compares GARDIAN against the best available non-GARDIAN system on each metric
with a one-sided paired bootstrap, and marks † at p < 0.05.

Pairing is by ``qid``, over the questions where BOTH systems report the metric.
This matters: ``unsupported_claim_rate`` and ``citation_precision`` are None
when an answer cites nothing (an uncited answer must not score as perfectly
grounded), and the two systems abstain on different questions. Dropping the
Nones and zipping what is left -- the previous behaviour -- silently paired
question i of one system with a different question of the other, on exactly the
two metrics where abstention happens.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Candidate baselines, best-first only in the sense of "try each that exists".
BASELINE_SYSTEMS = ("hybrid", "sparse", "dense")


def _rows_by_qid(data: dict, dataset: str, system: str) -> Dict[str, dict]:
    block = data.get("datasets", {}).get(dataset, {})
    rows = (block.get("per_question") or {}).get(system) or []
    return {str(r.get("qid")): r for r in rows if r.get("qid") is not None}


def paired_values(
    data: dict, dataset: str, system_a: str, system_b: str, key: str
) -> Tuple[List[float], List[float]]:
    """Values for *key* on questions where both systems report it, qid-aligned."""
    a_rows = _rows_by_qid(data, dataset, system_a)
    b_rows = _rows_by_qid(data, dataset, system_b)
    xs: List[float] = []
    ys: List[float] = []
    for qid in a_rows:
        if qid not in b_rows:
            continue
        av = a_rows[qid].get(key)
        bv = b_rows[qid].get(key)
        if av is None or bv is None:
            continue
        xs.append(float(av))
        ys.append(float(bv))
    return xs, ys


def bootstrap_paired_greater(
    a: Sequence[float], b: Sequence[float], *, n_boot: int, seed: int
) -> float:
    """One-sided p for H1: mean(a) > mean(b), on paired observations."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.size == 0 or x.size != y.size:
        return 1.0
    d = x - y
    if float(d.mean()) <= 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return float((int((means <= 0.0).sum()) + 1) / (n_boot + 1))


def available_baselines(data: dict, dataset: str) -> List[str]:
    agg = (data.get("datasets", {}).get(dataset, {}).get("aggregate") or {})
    return [s for s in BASELINE_SYSTEMS if s in agg]


def compare(
    data: dict,
    dataset: str,
    key: str,
    *,
    lower_is_better: bool,
    n_boot: int,
    seed: int,
    alpha: float,
) -> Optional[dict]:
    """GARDIAN vs the strongest baseline present, paired on qid."""
    baselines = available_baselines(data, dataset)
    if not baselines:
        return None

    scored = []
    for sysname in baselines:
        g_vals, b_vals = paired_values(data, dataset, "gardian", sysname, key)
        if not g_vals:
            continue
        scored.append((sysname, float(np.mean(b_vals)), g_vals, b_vals))
    if not scored:
        return None

    # "Best" baseline is the hardest one to beat on this metric.
    best = (min if lower_is_better else max)(scored, key=lambda t: t[1])
    best_sys, best_mean, g_vals, b_vals = best
    g_mean = float(np.mean(g_vals))

    if lower_is_better:
        wins = g_mean < best_mean
        p = bootstrap_paired_greater(b_vals, g_vals, n_boot=n_boot, seed=seed) if wins else 1.0
    else:
        wins = g_mean > best_mean
        p = bootstrap_paired_greater(g_vals, b_vals, n_boot=n_boot, seed=seed) if wins else 1.0

    return {
        "gardian": g_mean,
        "best_system": best_sys,
        "best_mean": best_mean,
        "delta": g_mean - best_mean,
        "p": p,
        "significant": bool(wins and p < alpha),
        "n_paired": len(g_vals),
    }


# (dataset, per-question key, lower_is_better, printed label)
METRICS: Tuple[Tuple[str, str, bool, str], ...] = (
    ("pubmedqa_labeled", "accuracy", False, "PQA-L Acc"),
    ("pubmedqa_labeled", "gold_evidence_in_context_any", False, "PQA-L Hit@10"),
    ("pubmedqa_labeled", "gold_evidence_in_context_rate", False, "PQA-L Ctx"),
    ("pubmedqa_labeled", "citation_recall", False, "PQA-L Cit.R"),
    ("pubmedqa_labeled", "unsupported_claim_rate", True, "PQA-L Uns."),
    ("pubmedqa_artificial", "accuracy", False, "PQA-A Acc"),
    ("pubmedqa_artificial", "gold_evidence_in_context_any", False, "PQA-A Hit@10"),
    ("pubmedqa_artificial", "gold_evidence_in_context_rate", False, "PQA-A Ctx"),
    ("pubmedqa_artificial", "unsupported_claim_rate", True, "PQA-A Uns."),
    ("medmcqa", "accuracy", False, "MedMCQA Acc"),
    ("medmcqa", "gold_evidence_in_context_any", False, "MedMCQA Hit@10"),
    ("medmcqa", "gold_evidence_in_context_rate", False, "MedMCQA Ctx"),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_paths", nargs="+", type=Path)
    ap.add_argument("--bootstrap", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    for path in args.json_paths:
        if not path.exists():
            print(f"\n=== {path.name} === (missing)")
            continue
        data = json.loads(path.read_text())
        reader = (data.get("meta") or {}).get("reader_model", "?")
        print(f"\n=== {path.name}  reader={reader} ===")
        for dataset, key, lower, label in METRICS:
            if dataset not in data.get("datasets", {}):
                continue
            res = compare(
                data,
                dataset,
                key,
                lower_is_better=lower,
                n_boot=args.bootstrap,
                seed=args.seed,
                alpha=args.alpha,
            )
            if res is None:
                print(f"  {label:18s}: (not reported)")
                continue
            mark = "†" if res["significant"] else " "
            print(
                f"  {label:18s}: gardian={res['gardian']:.4f} "
                f"vs {res['best_system']}={res['best_mean']:.4f} "
                f"delta={res['delta']:+.4f} p={res['p']:.4f} {mark} "
                f"(n={res['n_paired']})"
            )


if __name__ == "__main__":
    main()
