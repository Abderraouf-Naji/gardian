#!/usr/bin/env python3
"""
Regenerate the MedMCQA gold-in-context flip table (paper Table 5 / tab:rq3-flip)
from the saved n=1000 QA JSON files. Also prints Hit@10 on that same slice.

Groups are defined from the 14B run of each back-end (gold-in-context is a
retrieval property; BM25 flags are identical across readers, SPLADE++ disagrees
on one question). Accuracy for 7B/32B is then measured on those same qids.

Usage:
  .venv/bin/python scripts/report_rq3_flip.py
  .venv/bin/python scripts/report_rq3_flip.py --latex paper/table_rq3_flip.tex
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QA_DIR = ROOT / "results" / "rq3_1k"
DATASET = "medmcqa"

BACKENDS = (
    ("hybrid_bm25_faiss", r"\makecell{BM25\\+FAISS}"),
    ("hybrid_spladepp_medcpt", r"\makecell{SPLADE\texttt{++}\\+MedCPT}"),
)
READERS = (
    ("Qwen7B", "7B"),
    ("Qwen14B", "14B"),
    ("Qwen32B", "32B"),
)
REF_READER = "Qwen7B"
GROUPS = (
    ("both_abs", "Absent both", lambda h, g: (not h) and (not g)),
    ("g_only", "GARDIAN only", lambda h, g: g and (not h)),
    ("h_only", "RRF only", lambda h, g: h and (not g)),
    ("both_pres", "Present both", lambda h, g: h and g),
)


def get_block(data: dict, ds: str = DATASET) -> Optional[dict]:
    block = (data.get("datasets") or {}).get(ds)
    if block and block.get("per_question"):
        return block
    top = data.get(ds)
    if isinstance(top, dict) and top.get("per_question"):
        return top
    return None


def by_qid(block: dict, system: str) -> Dict[str, dict]:
    return {str(r["qid"]): r for r in (block.get("per_question") or {}).get(system) or []}


def gold_flag(row: dict) -> bool:
    return float(row.get("gold_evidence_in_context_any") or 0.0) > 0.5


def load_run(backend: str, reader_tag: str) -> Optional[dict]:
    path = QA_DIR / f"qa_{backend}_{reader_tag}_n1000.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def hit_at_10(rows: Dict[str, dict]) -> float:
    return 100.0 * sum(gold_flag(r) for r in rows.values()) / len(rows)


def group_qids(
    hybrid: Dict[str, dict], gardian: Dict[str, dict]
) -> Dict[str, List[str]]:
    qids = [q for q in hybrid if q in gardian]
    out: Dict[str, List[str]] = {k: [] for k, _, _ in GROUPS}
    for q in qids:
        h, g = gold_flag(hybrid[q]), gold_flag(gardian[q])
        for key, _, pred in GROUPS:
            if pred(h, g):
                out[key].append(q)
                break
    return out


def acc_on(rows: Dict[str, dict], qids: List[str]) -> Optional[float]:
    if not qids:
        return None
    return 100.0 * sum(float(rows[q]["accuracy"]) > 0.5 for q in qids) / len(qids)


def collect() -> Tuple[List[dict], List[dict]]:
    """Return (hit_rows, flip_rows)."""
    hits: List[dict] = []
    flips: List[dict] = []
    empty = {"rrf": None, "gardian": None, "n": 0}
    for backend, tex_be in BACKENDS:
        ref = load_run(backend, REF_READER)
        ref_block = get_block(ref) if ref else None
        if not ref_block:
            continue
        h_ref = by_qid(ref_block, "hybrid")
        g_ref = by_qid(ref_block, "gardian")
        groups = group_qids(h_ref, g_ref)
        n_q = len(set(h_ref) & set(g_ref))
        hits.append(
            {
                "backend": backend,
                "n": n_q,
                "rrf": hit_at_10(h_ref),
                "gardian": hit_at_10(g_ref),
            }
        )
        accs: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
        for reader_tag, _ in READERS:
            data = load_run(backend, reader_tag)
            block = get_block(data) if data else None
            accs[reader_tag] = {}
            if not block:
                for key, _, _ in GROUPS:
                    accs[reader_tag][key] = dict(empty)
                continue
            h = by_qid(block, "hybrid")
            g = by_qid(block, "gardian")
            for key, _, _ in GROUPS:
                qids = [q for q in groups[key] if q in h and q in g]
                accs[reader_tag][key] = {
                    "rrf": acc_on(h, qids),
                    "gardian": acc_on(g, qids),
                    "n": len(qids),
                }
        for key, label, _ in GROUPS:
            flips.append(
                {
                    "backend": backend,
                    "tex_backend": tex_be,
                    "group": key,
                    "label": label,
                    "n": len(groups[key]),
                    "accs": {rt: accs[rt][key] for rt, _ in READERS},
                }
            )
    return hits, flips


def fmt_acc(x: Optional[float]) -> str:
    return "--" if x is None else f"{x:.1f}"


def print_text(hits: List[dict], flips: List[dict]) -> None:
    print("MedMCQA n=1000 slice  Hit@10  (gold_evidence_in_context_any)")
    print(f"{'back-end':28s} {'n':>5} {'RRF':>7} {'GARDIAN':>8} {'delta':>7}")
    for h in hits:
        print(
            f"{h['backend']:28s} {h['n']:5d} {h['rrf']:7.1f} {h['gardian']:8.1f} "
            f"{h['gardian']-h['rrf']:+7.1f}"
        )
    print()
    print(f"Flip Acc (%)  [groups from {REF_READER} gold flags]")
    hdr = f"{'back-end':24s} {'group':16s} {'n':>5}"
    for _, short in READERS:
        hdr += f"  {short+' RRF':>8} {short+' G':>8}"
    print(hdr)
    for row in flips:
        line = f"{row['backend']:24s} {row['label']:16s} {row['n']:5d}"
        for rt, _ in READERS:
            a = row["accs"][rt]
            line += f"  {fmt_acc(a['rrf']):>8} {fmt_acc(a['gardian']):>8}"
        print(line)


def latex_table(flips: List[dict]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{12pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\caption{MedMCQA Acc (\%) by whether gold is in the top-10 under RRF and GARDIAN ($n{=}1{,}000$).}",
        r"\label{tab:rq3-flip}",
        r"\begin{tabular}{ll r cc cc cc}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{7B} & \multicolumn{2}{c}{14B} & \multicolumn{2}{c}{32B} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"Back-end & Group & $n$ & RRF & GAR. & RRF & GAR. & RRF & GAR. \\",
        r"\midrule",
    ]
    by_be: Dict[str, List[dict]] = {}
    for row in flips:
        by_be.setdefault(row["backend"], []).append(row)
    first = True
    for backend, tex_be in BACKENDS:
        rows = by_be.get(backend) or []
        if not rows:
            continue
        if not first:
            lines.append(r"\addlinespace[2pt]")
        first = False
        n_groups = len(rows)
        for i, row in enumerate(rows):
            cells = []
            for rt, _ in READERS:
                a = row["accs"][rt]
                cells.append(fmt_acc(a["rrf"]))
                cells.append(fmt_acc(a["gardian"]))
            prefix = (
                f"\\multirow{{{n_groups}}}{{*}}{{{tex_be}}}\n & "
                if i == 0
                else " & "
            )
            lines.append(
                prefix
                + f"{row['label']:14s} & {row['n']:3d} & "
                + " & ".join(cells)
                + r" \\"
            )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--latex",
        nargs="?",
        const="-",
        default=None,
        help="Write LaTeX table to this path, or stdout if omitted after the flag.",
    )
    args = ap.parse_args()
    hits, flips = collect()
    print_text(hits, flips)
    if args.latex is not None:
        tex = latex_table(flips)
        if args.latex == "-":
            print("\n" + tex)
        else:
            path = pathlib.Path(args.latex)
            path.write_text(tex, encoding="utf-8")
            print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
