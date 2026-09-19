#!/usr/bin/env python3
"""
RQ3 report: does improved evidence retrieval translate into better answers?

Reads the end-to-end QA JSONs of the n=1000 matrix and prints three blocks:

1. **Retrieval into the reader context**, once per back-end. Hit@10 (at least
   one gold passage in the reader's top-10) and Ctx (fraction of gold passages
   present) do not depend on the reader, so reporting them inside each reader's
   column -- as the submitted table did -- duplicates a number and reads as a
   copy-paste error.

2. **Answer accuracy**, per (back-end, reader), always beside the trivial
   baseline it has to beat. PubMedQA-Labeled is 55.2% "yes", so a constant
   "yes" scores .552; MedMCQA is 4-way, so random scores .250 and always-A
   scores .269. An accuracy below those is not a weak result, it is no result.

3. **Where the gain is lost.** Gold evidence present but the answer still
   wrong, and the share of decidable (gold yes/no) questions the reader hedges
   to "maybe". This is what turns "improvements are mixed" from a weakness into
   a diagnosis.

Usage:
  .venv/bin/python scripts/report_rq3.py results/rq3_1k/*.json
  .venv/bin/python scripts/report_rq3.py results/rq3_1k/*.json --latex
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e_qa_significance import bootstrap_paired_greater  # noqa: E402
from src.evaluation.qa_eval import extract_pubmedqa_label  # noqa: E402

DATASETS = ("pubmedqa_labeled", "pubmedqa_artificial", "medmcqa")
DATASET_LABEL = {
    "pubmedqa_labeled": "PQA-L",
    "pubmedqa_artificial": "PQA-A",
    "medmcqa": "MedMCQA",
}
GOLD_FILES = {
    "pubmedqa_labeled": ROOT / "data/pubmedqa_labeled_eval.jsonl",
    "pubmedqa_artificial": ROOT / "data/pubmedqa_artificial_test.jsonl",
    "medmcqa": ROOT / "data/medmcqa_test.jsonl",
}


def load_gold(dataset: str) -> Dict[str, str]:
    """qid -> gold answer label, for the hedging analysis."""
    path = GOLD_FILES.get(dataset)
    if path is None or not path.exists():
        return {}
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out[str(rec["id"])] = str(rec.get("answer", "")).strip().lower()
    return out


def rows_by_qid(data: dict, dataset: str, system: str) -> Dict[str, dict]:
    block = data.get("datasets", {}).get(dataset, {})
    rows = (block.get("per_question") or {}).get(system) or []
    return {str(r.get("qid")): r for r in rows if r.get("qid") is not None}


def paired(
    data: dict, dataset: str, key: str, a: str = "gardian", b: str = "hybrid"
) -> Tuple[List[float], List[float]]:
    ar, br = rows_by_qid(data, dataset, a), rows_by_qid(data, dataset, b)
    xs: List[float] = []
    ys: List[float] = []
    for qid, arow in ar.items():
        brow = br.get(qid)
        if brow is None:
            continue
        av, bv = arow.get(key), brow.get(key)
        if av is None or bv is None:
            continue
        xs.append(float(av))
        ys.append(float(bv))
    return xs, ys


def delta_with_p(
    data: dict,
    dataset: str,
    key: str,
    *,
    lower_is_better: bool = False,
    n_boot: int,
    seed: int,
) -> Optional[Dict[str, float]]:
    g, h = paired(data, dataset, key)
    if not g:
        return None
    gm, hm = float(np.mean(g)), float(np.mean(h))
    if lower_is_better:
        p = bootstrap_paired_greater(h, g, n_boot=n_boot, seed=seed) if gm < hm else 1.0
    else:
        p = bootstrap_paired_greater(g, h, n_boot=n_boot, seed=seed) if gm > hm else 1.0
    return {"gardian": gm, "hybrid": hm, "delta": gm - hm, "p": p, "n": len(g)}


def baselines_of(data: dict, dataset: str) -> Tuple[Optional[float], Optional[float], str]:
    """(majority accuracy, random accuracy, majority label) as stored in the run."""
    block = data.get("datasets", {}).get(dataset, {})
    b = block.get("baselines") or {}
    mc = b.get("majority_class") or {}
    rc = b.get("random_choice") or {}
    return mc.get("accuracy"), rc.get("accuracy"), str(mc.get("label") or "?")


def hedge_stats(data: dict, dataset: str, system: str, gold: Dict[str, str]) -> Optional[dict]:
    """Hedging to 'maybe' on questions whose gold answer is decidable (yes/no)."""
    if not gold or not dataset.startswith("pubmedqa"):
        return None
    rows = rows_by_qid(data, dataset, system)
    decidable = [r for qid, r in rows.items() if gold.get(qid) in ("yes", "no")]
    if not decidable:
        return None
    preds = [extract_pubmedqa_label(r.get("answer") or "") for r in decidable]
    return {
        "n_decidable": len(decidable),
        "hedged": sum(1 for p in preds if p == "maybe") / len(decidable),
        "no_verdict": sum(1 for p in preds if p is None) / len(decidable),
    }


def loss_decomposition(data: dict, dataset: str, system: str) -> Optional[dict]:
    """Gold evidence reaching the reader vs answers still wrong."""
    rows = list(rows_by_qid(data, dataset, system).values())
    if not rows:
        return None
    def has_gold(r: dict) -> bool:
        v = r.get("gold_evidence_in_context_any")
        if v is None:
            v = 1.0 if (r.get("gold_evidence_in_context_rate") or 0) > 0 else 0.0
        return bool(v)
    with_gold = [r for r in rows if has_gold(r)]
    without = [r for r in rows if not has_gold(r)]
    acc_in = float(np.mean([r["accuracy"] for r in with_gold])) if with_gold else None
    acc_out = float(np.mean([r["accuracy"] for r in without])) if without else None
    return {
        "n": len(rows),
        "gold_present": len(with_gold) / len(rows),
        "acc_given_gold": acc_in,
        "wrong_despite_gold": (1 - acc_in) if acc_in is not None else None,
        "acc_without_gold": acc_out,
        "evidence_lift": (acc_in - acc_out) if (acc_in is not None and acc_out is not None) else None,
    }


def pct(x: Optional[float], width: int = 6) -> str:
    return " " * (width - 1) + "-" if x is None else f"{x * 100:{width}.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_paths", nargs="+", type=pathlib.Path)
    ap.add_argument("--bootstrap", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    runs: List[Tuple[str, str, dict]] = []
    for path in args.json_paths:
        if not path.exists():
            print(f"(missing: {path})", file=sys.stderr)
            continue
        data = json.loads(path.read_text())
        meta = data.get("meta") or {}
        backend = str((meta.get("retrievers") or ["?"])[0])
        reader = str(meta.get("reader_model") or "?").split("/")[-1]
        runs.append((backend, reader, data))
    if not runs:
        print("no runs loaded", file=sys.stderr)
        sys.exit(1)

    gold_by_ds = {ds: load_gold(ds) for ds in DATASETS}
    star = lambda p: "†" if p is not None and p < args.alpha else " "  # noqa: E731

    # ---- 1. Retrieval into the reader context (reader-independent) ----------
    print("\n" + "=" * 78)
    print("1. EVIDENCE REACHING THE READER  (per back-end; does not depend on reader)")
    print("=" * 78)
    print(f"{'back-end':24s} {'dataset':9s} {'metric':8s} {'RRF':>7s} {'GARDIAN':>8s} {'delta':>7s} {'p':>8s}")
    seen_backends: set = set()
    for backend, reader, data in runs:
        if backend in seen_backends:
            continue
        seen_backends.add(backend)
        for ds in DATASETS:
            if ds not in data.get("datasets", {}):
                continue
            for key, label in (
                ("gold_evidence_in_context_any", "Hit@10"),
                ("gold_evidence_in_context_rate", "Ctx"),
            ):
                r = delta_with_p(data, ds, key, n_boot=args.bootstrap, seed=args.seed)
                if r is None:
                    continue
                print(
                    f"{backend:24s} {DATASET_LABEL[ds]:9s} {label:8s} "
                    f"{pct(r['hybrid'],7)} {pct(r['gardian'],8)} "
                    f"{r['delta']*100:+7.1f} {r['p']:8.4f} {star(r['p'])}"
                )

    # ---- 2. Answer accuracy against the trivial baseline --------------------
    print("\n" + "=" * 78)
    print("2. ANSWER ACCURACY  (beside the baseline it must beat)")
    print("=" * 78)
    print(f"{'back-end':24s} {'reader':14s} {'dataset':9s} {'RRF':>6s} {'GARD':>6s} {'delta':>7s} {'p':>7s} {'major':>6s} {'rand':>6s} {'G-maj':>7s}")
    for backend, reader, data in runs:
        for ds in DATASETS:
            if ds not in data.get("datasets", {}):
                continue
            r = delta_with_p(data, ds, "accuracy", n_boot=args.bootstrap, seed=args.seed)
            if r is None:
                continue
            maj, rand, _ = baselines_of(data, ds)
            gap = (r["gardian"] - maj) * 100 if maj is not None else None
            print(
                f"{backend:24s} {reader:14s} {DATASET_LABEL[ds]:9s} "
                f"{pct(r['hybrid'])} {pct(r['gardian'])} {r['delta']*100:+7.1f} "
                f"{r['p']:7.4f}{star(r['p'])}{pct(maj)} {pct(rand)} "
                f"{'      -' if gap is None else f'{gap:+7.1f}'}"
            )

    # ---- 3. Where the gain is lost -----------------------------------------
    print("\n" + "=" * 78)
    print("3. WHERE THE GAIN IS LOST")
    print("=" * 78)
    print(f"{'back-end':24s} {'reader':14s} {'dataset':9s} {'sys':8s} {'gold@ctx':>8s} {'acc|gold':>8s} {'wrong|gold':>10s} {'lift':>6s} {'hedge':>6s}")
    for backend, reader, data in runs:
        for ds in DATASETS:
            if ds not in data.get("datasets", {}):
                continue
            for system in ("hybrid", "gardian"):
                d = loss_decomposition(data, ds, system)
                if d is None:
                    continue
                h = hedge_stats(data, ds, system, gold_by_ds.get(ds, {}))
                print(
                    f"{backend:24s} {reader:14s} {DATASET_LABEL[ds]:9s} {system:8s} "
                    f"{pct(d['gold_present'],8)} {pct(d['acc_given_gold'],8)} "
                    f"{pct(d['wrong_despite_gold'],10)} {pct(d['evidence_lift'],6)} "
                    f"{pct(h['hedged'] if h else None,6)}"
                )
    print()


if __name__ == "__main__":
    main()
