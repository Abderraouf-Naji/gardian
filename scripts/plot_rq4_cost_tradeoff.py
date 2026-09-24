#!/usr/bin/env python3
"""
RQ4 figure: the cost/effectiveness trade-off, as a Pareto plot.

The submitted version of this figure used grouped bars in two rows -- quality on
top, latency underneath -- across six panels. That layout has three problems for
a question about *cost*:

  * The trade-off is the point, but the reader has to join two separate rows by
    eye to see it. A system that is slightly worse and 50x faster looks the same
    as one that is slightly worse and 2x faster.
  * The latency row is near-identical in all six panels, because latency depends
    on pool size and model, not on which split the pool came from. Six copies of
    one fact.
  * RRF's bar is invisible: 0.03 ms on a log axis starting near 1 ms.

Here each system is one marker: x is per-query re-ranking latency (log), y is
nDCG@10, and marker area encodes model size. That is all three cost axes RQ4
asks about in one view, and the Pareto frontier -- the set of systems nothing
else beats on both axes at once -- can be drawn directly. GARDIAN's claim is
exactly that it sits on that frontier, so the figure should show the frontier.

Reads the same artifacts as scripts/report_rq4_cost.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]

# Palette keeps the submitted figure's colours for the systems it contained, so
# a reader comparing versions is not re-learning the mapping. The two new
# systems get a related blue (GARDIAN-Lite is the same model minus the
# controller) and a neutral gold (LambdaMART is not a neural re-ranker).
STYLE: Dict[str, Dict[str, Any]] = {
    "rrf": dict(label="RRF (no model)", color="#6E6E6E", marker="X", params=None),
    "cross_encoder_msmarco_minilm": dict(
        label="CE: MiniLM-L6", color="#E87722", marker="o", params=22.7e6
    ),
    "cross_encoder_bge_v2_m3": dict(
        label="CE: BGE-v2-M3", color="#009E73", marker="o", params=568e6
    ),
    "cross_encoder_monot5_med": dict(
        label="CE: MonoT5-med", color="#CC3311", marker="o", params=223e6
    ),
    "cross_encoder_monobert": dict(
        label="CE: MonoBERT", color="#7B68EE", marker="o", params=335e6
    ),
    "lambdamart": dict(label="LambdaMART", color="#B8860B", marker="^", params=None),
    "gardian_lite": dict(label="GARDIAN-Lite", color="#4C9BE8", marker="s", params=4.25e6),
    "gardian": dict(label="GARDIAN", color="#005BBB", marker="*", params=7.93e6),
}

PLOT_ORDER = list(STYLE.keys())

DATASETS = [
    ("pubmedqa_labeled", "PQA-L"),
    ("medmcqa", "MedMCQA"),
    ("pubmedqa_artificial", "PQA-A"),
]

BACKENDS = [
    ("hybrid_bm25_faiss", "BM25+FAISS"),
    ("hybrid_spladepp_medcpt", "SPLADE++ + MedCPT"),
]

# Marker area in points^2 for the smallest and largest model plotted.
AREA_MIN, AREA_MAX = 34.0, 620.0
PARAM_MIN, PARAM_MAX = 2e6, 600e6


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _merged(results_dir: Path, stem: str) -> Dict[str, Any]:
    """Merge a comparison file with its separate 1000-query PQA-A run."""
    main = _load(results_dir / f"{stem}.json")
    if not main:
        return {}
    out = dict(main.get("results") or {})
    extra = _load(results_dir / f"{stem}_pqa_a_q1000.json")
    if extra:
        out.update(extra.get("results") or {})
    return out


def _area_for_params(params: Optional[float]) -> float:
    """Marker area on a log scale, so 22M and 568M are visibly different."""
    if not params:
        return 70.0
    lo, hi = np.log10(PARAM_MIN), np.log10(PARAM_MAX)
    t = (np.log10(float(params)) - lo) / (hi - lo)
    t = float(np.clip(t, 0.0, 1.0))
    return AREA_MIN + t * (AREA_MAX - AREA_MIN)


def collect(results_dir: Path) -> Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]:
    """``{backend: {dataset: {system: (latency_ms, ndcg10)}}}``."""
    out: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {}

    for backend, _ in BACKENDS:
        full = _merged(results_dir, f"reranker_comparison_{backend}")
        lite = _merged(results_dir, f"reranker_comparison_{backend}_lite")
        ltr = _load(results_dir / f"ltr_baseline_{backend}.json") or {}
        ltr_block = ((ltr.get("results") or {}).get(backend) or {}).get("results") or {}
        ltr_sub = _load(results_dir / f"ltr_baseline_{backend}_pqa_a_q1000.json") or {}
        ltr_sub_block = (
            ((ltr_sub.get("results") or {}).get(backend) or {}).get("results") or {}
        )
        if "pubmedqa_artificial" in ltr_sub_block:
            ltr_block = dict(ltr_block)
            ltr_block["pubmedqa_artificial"] = ltr_sub_block["pubmedqa_artificial"]

        per_dataset: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for dataset, _ in DATASETS:
            points: Dict[str, Tuple[float, float]] = {}

            block = full.get(dataset) or {}
            for system in PLOT_ORDER:
                if system in ("gardian_lite", "lambdamart"):
                    continue
                entry = block.get(system) or {}
                ndcg = (entry.get("metrics") or {}).get("ndcg@10")
                lat = (entry.get("latency_ms") or {}).get("p50_ms")
                if ndcg is not None and lat:
                    points[system] = (float(lat), float(ndcg))

            lite_entry = (lite.get(dataset) or {}).get("gardian") or {}
            l_ndcg = (lite_entry.get("metrics") or {}).get("ndcg@10")
            l_lat = (lite_entry.get("latency_ms") or {}).get("p50_ms")
            if l_ndcg is not None and l_lat:
                points["gardian_lite"] = (float(l_lat), float(l_ndcg))

            t = ltr_block.get(dataset) or {}
            t_ndcg = (t.get("metrics") or {}).get("ndcg@10")
            t_lat = (t.get("latency_ms") or {}).get("p50_ms")
            if t_ndcg is not None and t_lat:
                points["lambdamart"] = (float(t_lat), float(t_ndcg))

            per_dataset[dataset] = points
        out[backend] = per_dataset
    return out


def pareto_front(points: Dict[str, Tuple[float, float]]) -> List[str]:
    """Systems no other system beats on BOTH lower latency and higher nDCG."""
    keys = list(points)
    front = []
    for k in keys:
        lat_k, q_k = points[k]
        dominated = any(
            (points[o][0] <= lat_k and points[o][1] >= q_k) and o != k
            and (points[o][0] < lat_k or points[o][1] > q_k)
            for o in keys
        )
        if not dominated:
            front.append(k)
    return sorted(front, key=lambda k: points[k][0])


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 8.0,
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def plot(data: Dict[str, Any], out_path: Path, *, contended: bool) -> None:
    apply_style()
    n_rows, n_cols = len(BACKENDS), len(DATASETS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(10.4, 5.5),
        facecolor="white",
        gridspec_kw={"hspace": 0.26, "wspace": 0.20},
    )

    for i, (backend, backend_label) in enumerate(BACKENDS):
        for j, (dataset, ds_label) in enumerate(DATASETS):
            ax = axes[i, j]
            ax.set_facecolor("white")
            points = data.get(backend, {}).get(dataset, {})

            if not points:
                ax.text(0.5, 0.5, "not yet measured", ha="center", va="center",
                        fontsize=8, color="#999999", transform=ax.transAxes)
                ax.set_xscale("log")
                continue

            front = pareto_front(points)
            if len(front) > 1:
                fx = [points[k][0] for k in front]
                fy = [points[k][1] * 100.0 for k in front]
                # Step line: moving right along x, the best achievable y so far.
                sx, sy = [], []
                for a in range(len(fx)):
                    if a:
                        sx.append(fx[a]); sy.append(fy[a - 1])
                    sx.append(fx[a]); sy.append(fy[a])
                ax.plot(sx, sy, color="#5a5a5a", lw=1.3, ls=(0, (5, 2.5)), zorder=2,
                        solid_capstyle="round")
                # Everything below/right of the frontier is dominated: some other
                # system is both faster and better. Shading it makes "on the
                # frontier" a visual fact rather than a claim in the caption.
                ax.fill_between(sx, min(sy) - 50, sy, color="#f2f2f2", zorder=1)

            # Cross-encoders of similar speed can land on top of each other;
            # a small multiplicative offset on a log axis separates them
            # without moving them meaningfully.
            spread: Dict[str, float] = {}
            by_lat = sorted(points, key=lambda k: points[k][0])
            for a in range(1, len(by_lat)):
                prev, cur = by_lat[a - 1], by_lat[a]
                r = points[cur][0] / max(points[prev][0], 1e-9)
                dy = abs(points[cur][1] - points[prev][1]) * 100.0
                if r < 1.9 and dy < 1.2:
                    spread[prev] = spread.get(prev, 1.0) * 0.80
                    spread[cur] = spread.get(cur, 1.0) * 1.25

            for system in PLOT_ORDER:
                if system not in points:
                    continue
                lat, ndcg = points[system]
                lat = lat * spread.get(system, 1.0)
                st = STYLE[system]
                is_ours = system in ("gardian", "gardian_lite")
                ax.scatter(
                    lat, ndcg * 100.0,
                    s=_area_for_params(st["params"]) * (1.35 if system == "gardian" else 1.0),
                    c=st["color"],
                    marker=st["marker"],
                    edgecolors="#111111" if is_ours else "white",
                    linewidths=0.9 if is_ours else 0.5,
                    alpha=0.95,
                    zorder=4 if is_ours else 3,
                )

            ax.set_xscale("log")
            ax.grid(True, which="major", ls="-", lw=0.4, color="#dddddd", zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=7.2)

            lats = [p[0] for p in points.values()]
            ax.set_xlim(min(lats) * 0.35, max(lats) * 3.2)
            ys = [p[1] * 100.0 for p in points.values()]
            pad = max((max(ys) - min(ys)) * 0.18, 1.2)
            ax.set_ylim(min(ys) - pad, max(ys) + pad)

            if i == 0:
                ax.set_title(ds_label, fontsize=9.5, fontweight="600", pad=5)
            if j == 0:
                ax.set_ylabel(f"{backend_label}\nnDCG@10 (%)", fontsize=8.6,
                              fontweight="600", linespacing=1.4)
            if i == n_rows - 1:
                ax.set_xlabel("re-ranking latency (ms/query, log)", fontsize=8.4)

            # "Better" corner, drawn once so the axes orientation is unambiguous.
            if i == 0 and j == 0:
                ax.annotate(
                    "better", xy=(0.055, 0.93), xytext=(0.30, 0.78),
                    textcoords="axes fraction", xycoords="axes fraction",
                    fontsize=7.4, style="italic", color="#444444",
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="#444444"),
                )

    handles = [
        Line2D([], [], marker=STYLE[s]["marker"], color="none",
               markerfacecolor=STYLE[s]["color"],
               markeredgecolor="#111111" if s in ("gardian", "gardian_lite") else "white",
               markeredgewidth=0.8, markersize=9 if s == "gardian" else 7.5,
               label=STYLE[s]["label"])
        for s in PLOT_ORDER
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.035),
        ncol=8, frameon=True, fancybox=False, edgecolor="#bbbbbb",
        facecolor="white", fontsize=8.0, handletextpad=0.35,
        columnspacing=1.0, borderpad=0.45,
    )

    # Marker area encodes model size, so it needs its own key.
    size_handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor="#999999",
               markeredgecolor="white",
               markersize=np.sqrt(_area_for_params(p)) * 0.82, label=lab)
        for p, lab in [(4.25e6, "4M"), (22.7e6, "23M"), (223e6, "223M"), (568e6, "568M")]
    ]
    fig.legend(
        handles=size_handles, loc="lower center", bbox_to_anchor=(0.5, -0.115),
        ncol=4, frameon=True, fancybox=False, edgecolor="#bbbbbb",
        facecolor="white", fontsize=7.6, title="marker area = model parameters",
        title_fontsize=7.8, handletextpad=1.1, columnspacing=2.4, borderpad=0.7,
        labelspacing=1.1, handleheight=2.2,
    )

    if contended:
        fig.text(
            0.5, -0.165,
            "Latency measured with other jobs resident on the GPU; "
            "absolute values are upper bounds, relative ordering is unaffected.",
            ha="center", fontsize=7.0, style="italic", color="#a03030",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=400, facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "results" / "figures" / "rq4_cost_tradeoff.png",
    )
    args = parser.parse_args()

    data = collect(args.results_dir)

    contended = False
    for backend, _ in BACKENDS:
        meta = (_load(args.results_dir / f"reranker_comparison_{backend}.json") or {}).get("meta", {})
        c = meta.get("latency_gpu_contention") or {}
        if c.get("available") and not c.get("exclusive", True):
            contended = True

    plot(data, args.out, contended=contended)
    n = sum(len(d) for b in data.values() for d in b.values())
    print(f"Wrote {args.out} and {args.out.with_suffix('.pdf')} ({n} points)")
    if contended:
        print("NOTE: latency was measured under GPU contention; figure is annotated.")


if __name__ == "__main__":
    main()
