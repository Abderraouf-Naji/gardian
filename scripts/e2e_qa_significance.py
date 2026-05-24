#!/usr/bin/env python3
"""Paired bootstrap significance: GARDIAN vs best baseline (sparse/dense/hybrid).

Usage:
  .venv/bin/python scripts/e2e_qa_significance.py results/qa_hybrid_bm25_faiss_Qwen32B_n300.json

Marks † when GARDIAN strictly beats the best non-GARDIAN system on bootstrap mean
(one-sided paired bootstrap, default 10k resamples, p < 0.05).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASELINES = ("sparse", "dense", "hybrid")


def _per_q(path: Path, dataset: str, system: str, key: str) -> list[float]:
    data = json.loads(path.read_text())
    rows = data["datasets"][dataset]["per_question"][system]
    return [float(r[key]) for r in rows if r.get(key) is not None]


def bootstrap_paired_greater(a: list[float], b: list[float], *, n_boot: int, seed: int) -> float:
    """One-sided p: H1 mean(a) > mean(b)."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    n = min(x.size, y.size)
    if n == 0:
        return 1.0
    d = x[:n] - y[:n]
    if float(d.mean()) <= 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    le = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if float(d[idx].mean()) <= 0.0:
            le += 1
    return float((le + 1) / (n_boot + 1))


def best_baseline_means(path: Path, dataset: str, key: str, *, lower_is_better: bool) -> tuple[str, float]:
    means = {s: float(np.mean(_per_q(path, dataset, s, key))) for s in BASELINES}
    best = min(means, key=means.get) if lower_is_better else max(means, key=means.get)
    return best, means[best]


def gardian_beats_best(
    path: Path,
    dataset: str,
    key: str,
    *,
    lower_is_better: bool,
    n_boot: int,
    seed: int,
) -> tuple[bool, float, str, float]:
    g = _per_q(path, dataset, "gardian", key)
    best_sys, best_mean = best_baseline_means(path, dataset, key, lower_is_better=lower_is_better)
    b = _per_q(path, dataset, best_sys, key)
    g_mean = float(np.mean(g))
    if lower_is_better:
        wins = g_mean < best_mean
        p = bootstrap_paired_greater(b, g, n_boot=n_boot, seed=seed) if wins else 1.0
    else:
        wins = g_mean > best_mean
        p = bootstrap_paired_greater(g, b, n_boot=n_boot, seed=seed) if wins else 1.0
    return wins and p < 0.05, p, best_sys, best_mean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("json_paths", nargs="+", type=Path)
    p.add_argument("--bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    metrics = [
        ("pubmedqa_labeled", "accuracy", False, "Acc"),
        ("pubmedqa_labeled", "citation_precision", False, "Cit.P"),
        ("pubmedqa_labeled", "citation_recall", False, "Cit.R"),
        ("pubmedqa_labeled", "unsupported_claim_rate", True, "Uns."),
        ("medmcqa", "accuracy", False, "MCQ Acc"),
    ]

    for path in args.json_paths:
        print(f"\n=== {path.name} ===")
        for dataset, key, lower, label in metrics:
            data = json.loads(path.read_text())
            if dataset not in data.get("datasets", {}):
                print(f"  {label}: (dataset missing)")
                continue
            sig, pval, best_sys, best_mean = gardian_beats_best(
                path,
                dataset,
                key,
                lower_is_better=lower,
                n_boot=args.bootstrap,
                seed=args.seed,
            )
            g_mean = float(np.mean(_per_q(path, dataset, "gardian", key)))
            mark = "†" if sig else ""
            print(
                f"  {label}: gardian={g_mean:.3f} best={best_sys}({best_mean:.3f}) "
                f"p={pval:.4f} {mark}"
            )


if __name__ == "__main__":
    main()
