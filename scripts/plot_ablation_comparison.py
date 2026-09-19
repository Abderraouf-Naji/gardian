"""CubeScope-style grouped bar figure for GARDIAN component ablations (tab:ablation replacement).

Six panels: 3 datasets × 2 hybrid back-ends. Each panel groups five ablation variants at
nDCG@10 and Recall@20 (with bootstrap 95% CI error bars when available).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]

VARIANTS = [
    ("no_sparse_signal", "NoSparse"),
    ("no_dense_signal", "NoDense"),
    ("fixed_alpha", "Fixed-$\\alpha$"),
    ("full", "GARDIAN"),
    ("oracle_branch", "Oracle-$\\alpha$"),
]

VARIANT_COLORS = {
    "no_sparse_signal": "#E87722",
    "no_dense_signal": "#CC3311",
    "fixed_alpha": "#6E6E6E",
    "full": "#005BBB",
    "oracle_branch": "#BBBBBB",
}

DATASETS = [
    ("pubmedqa_labeled", "PQA-L"),
    ("pubmedqa_artificial", "PQA-A"),
    ("medmcqa", "MedMCQA"),
]

BACKENDS = [
    ("BM25+FAISS", "hybrid_bm25_faiss", REPO_ROOT / "results" / "seeds" / "seed_42" / "ablation_paper.json"),
    ("BM25+MedCPT", "hybrid_bm25_medcpt", REPO_ROOT / "results" / "seeds" / "seed_42" / "ablation_paper.json"),
    ("SPLADE++\n+ FAISS", "hybrid_spladepp_faiss", REPO_ROOT / "results" / "seeds" / "seed_42" / "ablation_paper.json"),
    ("SPLADE++\n+ MedCPT", "hybrid_spladepp_medcpt", REPO_ROOT / "results" / "seeds" / "seed_42" / "ablation_paper.json"),
]

METRICS = [
    ("ndcg@10", "ndcg@10_bootstrap_ci95", "nDCG@10"),
    ("recall@20", "recall@20_bootstrap_ci95", "R@20"),
]


def _pct(x: float) -> float:
    return x * 100.0 if x <= 1.0 else x


def load_ablation(path: Path, retriever: str) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj["results"][retriever]


def variant_metrics(
    results: dict[str, Any], ds: str, variant: str
) -> tuple[list[float], list[tuple[float, float]]]:
    row = results[ds][variant]["gardian"]
    values: list[float] = []
    yerr: list[tuple[float, float]] = []
    for metric_key, ci_key, _ in METRICS:
        val = _pct(float(row[metric_key]))
        values.append(val)
        if ci_key in row and row[ci_key]:
            lo, hi = (_pct(float(x)) for x in row[ci_key])
            yerr.append((val - lo, hi - val))
        else:
            yerr.append((0.0, 0.0))
    return values, yerr


def _bar_style(variant: str) -> tuple[str, float]:
    if variant == "full":
        return "#1a1a1a", 1.1
    return "white", 0.5


def _draw_grouped_bars(
    ax: plt.Axes,
    x_centers: np.ndarray,
    values_per_variant: list[list[float]],
    yerr_per_variant: list[list[tuple[float, float]]],
    *,
    bar_width: float,
) -> None:
    n_variants = len(VARIANTS)
    for vi, (variant_key, _) in enumerate(VARIANTS):
        offsets = (vi - (n_variants - 1) / 2) * bar_width
        color = VARIANT_COLORS[variant_key]
        edge, lw = _bar_style(variant_key)
        yvals = values_per_variant[vi]
        yerr = np.array(yerr_per_variant[vi]).T
        ax.bar(
            x_centers + offsets,
            yvals,
            width=bar_width * 0.94,
            color=color,
            edgecolor=edge,
            linewidth=lw,
            yerr=yerr,
            capsize=1.5,
            error_kw={"elinewidth": 0.6, "ecolor": "#333333", "capthick": 0.6},
            zorder=3,
        )


def plot_ablation_cubescope(out_path: Path) -> None:
    panels: list[tuple[str, str, dict[str, Any]]] = []
    for backend_label, _retriever, path in BACKENDS:
        retriever = _retriever
        results = load_ablation(path, retriever)
        for ds_key, ds_short in DATASETS:
            panels.append((backend_label, ds_short, results))

    n_panels = len(panels)
    n_variants = len(VARIANTS)
    n_metrics = len(METRICS)

    fig_w = 2.05 * n_panels + 0.35
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(fig_w, 3.35),
        facecolor="white",
        gridspec_kw={"wspace": 0.28},
    )
    if n_panels == 1:
        axes = [axes]

    group_width = 0.72
    bar_width = group_width / n_variants
    x = np.arange(n_metrics)

    for col, (backend_label, ds_short, results) in enumerate(panels):
        ax = axes[col]
        ax.set_facecolor("white")
        ds_key = next(dk for dk, ds in DATASETS if ds == ds_short)

        values = []
        yerrs = []
        for variant_key, _ in VARIANTS:
            v, e = variant_metrics(results, ds_key, variant_key)
            values.append(v)
            yerrs.append(e)

        _draw_grouped_bars(ax, x, values, yerrs, bar_width=bar_width)

        ax.set_xticks(x)
        ax.set_xticklabels([m[2] for m in METRICS], fontsize=7.5)
        ax.set_title(
            f"#{col + 1} {backend_label.replace(chr(10), ' ')}\n· {ds_short}",
            fontsize=8.5,
            fontweight="600",
            pad=6,
            linespacing=0.95,
        )
        ax.yaxis.grid(True, linestyle="-", alpha=0.35, color="#cccccc", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=7.5)
        ymax = max(v for vs in values for v in vs)
        ymin = min(v for vs in values for v in vs)
        pad = max(3.0, (ymax - ymin) * 0.12)
        ax.set_ylim(max(0, ymin - pad * 0.3), ymax + pad)

    axes[0].set_ylabel("Score (%)", fontsize=9.5, fontweight="600")

    legend_handles = [
        Patch(
            facecolor=VARIANT_COLORS[vkey],
            edgecolor=_bar_style(vkey)[0],
            linewidth=_bar_style(vkey)[1],
        )
        for vkey, _ in VARIANTS
    ]
    fig.legend(
        handles=legend_handles,
        labels=[lab for _, lab in VARIANTS],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=5,
        frameon=True,
        fancybox=False,
        edgecolor="#bbbbbb",
        facecolor="white",
        fontsize=11,
        handlelength=0.65,
        handleheight=0.65,
        handletextpad=0.4,
        columnspacing=1.0,
        labelspacing=0.35,
        borderpad=0.45,
    )

    fig.subplots_adjust(left=0.07, right=0.99, top=0.80, bottom=0.14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor="white", pad_inches=0.10)
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white", pad_inches=0.10)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "figures" / "ablation_cubescope.png",
    )
    args = parser.parse_args()
    plot_ablation_cubescope(args.out)
    print(f"Wrote {args.out} and {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
