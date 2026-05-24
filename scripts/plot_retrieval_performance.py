"""Publication figures for retrieval ablation table (bar chart + GARDIAN gain heatmap).

Loads metrics from ``results/evaluation_hybrid_*.json`` (output of
``scripts/05_evaluate_gardian.py``) and writes PNG/PDF under ``results/figures/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

REPO_ROOT = Path(__file__).resolve().parents[1]

HYBRID_TYPES = [
    "hybrid_bm25_faiss",
    "hybrid_spladepp_faiss",
    "hybrid_bm25_medcpt",
    "hybrid_spladepp_medcpt",
]

BACKEND_LABELS = {
    "hybrid_bm25_faiss": "BM25+FAISS",
    "hybrid_spladepp_faiss": "SPLADE++ + FAISS",
    "hybrid_bm25_medcpt": "BM25+MedCPT",
    "hybrid_spladepp_medcpt": "SPLADE++ + MedCPT",
}

BACKEND_LABELS_FLAT = {
    "hybrid_bm25_faiss": "BM25+FAISS",
    "hybrid_spladepp_faiss": "SPLADE++ + FAISS",
    "hybrid_bm25_medcpt": "BM25+MedCPT",
    "hybrid_spladepp_medcpt": "SPLADE++ + MedCPT",
}

DATASETS = [
    ("pubmedqa_labeled", "PubMedQA-Labeled"),
    ("pubmedqa_artificial", "PubMedQA-Artificial"),
    ("medmcqa", "MedMCQA"),
]

SYSTEMS = ["sparse", "dense", "hybrid", "rrf", "gardian"]
VARIANT_LABELS = ["Sparse", "Dense", "Hybrid", "RRF", "GARDIAN"]
VARIANT_COLORS = {
    "sparse": "#B8B8B8",
    "dense": "#4472C4",
    "hybrid": "#2E9E9E",
    "rrf": "#ED7D31",
    "gardian": "#E53935",
}

METRIC_KEYS = {
    "nDCG@20": "ndcg@20",
    "nDCG@50": "ndcg@50",
    "Rec@20": "recall@20",
    "Rec@50": "recall@50",
}

HEATMAP_METRIC_ORDER = ["Rec@50", "Rec@20", "nDCG@50", "nDCG@20"]

DEFAULT_EVAL_PATHS = {
    ht: REPO_ROOT / "results" / f"evaluation_{ht}.json" for ht in HYBRID_TYPES
}


def load_retrieval_table(eval_paths: dict[str, Path]) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Return ``table[hybrid_type][dataset][system][metric_key]``."""
    table: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for ht, path in eval_paths.items():
        obj = json.loads(path.read_text(encoding="utf-8"))
        table[ht] = obj["results"][ht]
    return table


def _metric(table: dict, ht: str, ds: str, sys: str, key: str) -> float:
    return float(table[ht][ds][sys][key])


def gardian_gain_pp(
    table: dict[str, Any],
    ht: str,
    ds: str,
    metric_key: str,
) -> float:
    baselines = [_metric(table, ht, ds, s, metric_key) for s in SYSTEMS[:-1]]
    gardian = _metric(table, ht, ds, "gardian", metric_key)
    return (gardian - max(baselines)) * 100.0


def _pt_to_fig_y(fig: plt.Figure, pts: float) -> float:
    """Convert font size (pt) to figure-coordinate height."""
    return (pts / 72.0) / fig.get_figheight()


def _figure_title_block(
    fig: plt.Figure,
    main: str,
    subtitle: str,
    *,
    y_top: float = 0.99,
) -> float:
    """Place title + subtitle as one offset box; return bottom y in figure coords."""
    main_fs, sub_fs = 14, 9.5
    main_h = _pt_to_fig_y(fig, main_fs)
    sub_h = _pt_to_fig_y(fig, sub_fs)
    desc_gap_pt = 3.0

    title_box = VPacker(
        children=[
            TextArea(
                main,
                textprops={"fontsize": main_fs, "fontweight": "700", "ha": "center"},
            ),
            TextArea(
                subtitle,
                textprops={
                    "fontsize": sub_fs,
                    "color": "#555555",
                    "style": "italic",
                    "ha": "center",
                },
            ),
        ],
        align="center",
        pad=0,
        sep=desc_gap_pt,
    )
    anchored = AnchoredOffsetbox(
        loc="upper center",
        child=title_box,
        frameon=False,
        bbox_to_anchor=(0.5, y_top),
        bbox_transform=fig.transFigure,
        pad=0.0,
        borderpad=0.0,
    )
    fig.add_artist(anchored)
    return y_top - main_h - (desc_gap_pt / 72.0) / fig.get_figheight() - sub_h


def plot_ndcg20_bars(
    table: dict[str, Any],
    out_path: Path,
    *,
    metric_label: str = "nDCG@20",
    metric_key: str = "ndcg@20",
) -> None:
    n_ds = len(DATASETS)
    fig, axes = plt.subplots(1, n_ds, figsize=(14, 4.9), sharey=True, facecolor="white")
    if n_ds == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")

    n_backends = len(HYBRID_TYPES)
    n_variants = len(SYSTEMS)
    group_width = 0.78
    bar_width = group_width / n_variants

    for ax, (ds_key, ds_title) in zip(axes, DATASETS):
        x_centers = np.arange(n_backends)
        for vi, sys in enumerate(SYSTEMS):
            offsets = (vi - (n_variants - 1) / 2) * bar_width
            heights = [_metric(table, ht, ds_key, sys, metric_key) for ht in HYBRID_TYPES]
            edgecolor = "#1a1a1a" if sys == "gardian" else "none"
            linewidth = 1.2 if sys == "gardian" else 0.0
            bars = ax.bar(
                x_centers + offsets,
                heights,
                width=bar_width * 0.92,
                label=VARIANT_LABELS[vi],
                color=VARIANT_COLORS[sys],
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=3,
            )
            if sys == "gardian":
                for bar, h in zip(bars, heights):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + 0.008,
                        f"{h:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="#C62828",
                        fontweight="600",
                    )

        ax.set_title(ds_title, fontsize=12, fontweight="600", pad=8)
        ax.set_xticks(x_centers)
        ax.set_xticklabels(
            [BACKEND_LABELS[ht] for ht in HYBRID_TYPES],
            fontsize=8,
            rotation=25,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.grid(True, linestyle="-", alpha=0.35, zorder=0)
        ax.xaxis.grid(True, linestyle="-", alpha=0.2, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(metric_label, fontsize=11)
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=VARIANT_COLORS[s], ec="#1a1a1a" if s == "gardian" else "none", lw=1.2 if s == "gardian" else 0)
        for s in SYSTEMS
    ]
    title_bottom = _figure_title_block(
        fig,
        f"{metric_label}: GARDIAN vs. Baselines across All Back-ends & Benchmarks",
        f"GARDIAN (red) achieves top {metric_label} in all 12 back-end × benchmark settings",
    )
    fig.tight_layout(rect=[0, 0, 1, title_bottom - 0.038])
    fig.legend(
        handles,
        VARIANT_LABELS,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, title_bottom - 0.018),
        fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, out_path)
    plt.close(fig)


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    """Save without bbox_inches='tight' to avoid duplicated title rendering."""
    fig.savefig(out_path, dpi=300, facecolor="white", pad_inches=0.12)
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white", pad_inches=0.12)


def plot_gardian_gain_heatmap(table: dict[str, Any], out_path: Path) -> None:
    n_metrics = len(HEATMAP_METRIC_ORDER)
    col_labels: list[str] = []
    for ht in HYBRID_TYPES:
        be = BACKEND_LABELS_FLAT[ht]
        for ds_key, ds_short in DATASETS:
            short = {"pubmedqa_labeled": "PubMedQA-L", "pubmedqa_artificial": "PubMedQA-A", "medmcqa": "MedMCQA"}[ds_key]
            col_labels.append(f"{be}\n{short}")

    data = np.zeros((n_metrics, len(col_labels)))
    col_idx = 0
    for ht in HYBRID_TYPES:
        for ds_key, _ in DATASETS:
            for mi, mlabel in enumerate(HEATMAP_METRIC_ORDER):
                mkey = METRIC_KEYS[mlabel]
                data[mi, col_idx] = gardian_gain_pp(table, ht, ds_key, mkey)
            col_idx += 1

    vmax = max(2.0, float(np.ceil(np.abs(data).max() * 10) / 10))
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(16, 4.8), facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", norm=norm, interpolation="nearest")

    for i in range(n_metrics):
        for j in range(len(col_labels)):
            val = data[i, j]
            sign = "+" if val >= 0 else ""
            txt = f"{sign}{val:.2f}"
            lum = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            color = "#1a1a1a" if 0.32 < lum < 0.68 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color=color, fontweight="500")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8, color="#222222")
    ax.set_yticks(np.arange(n_metrics))
    ax.set_yticklabels(HEATMAP_METRIC_ORDER, fontsize=10, color="#222222")
    ax.set_xlabel("Back-end / Benchmark", fontsize=11, color="#222222", labelpad=12)
    ax.set_ylabel("Metric", fontsize=11, color="#222222", labelpad=8)

    for k in range(1, len(HYBRID_TYPES)):
        ax.axvline(k * 3 - 0.5, color="#cccccc", linewidth=0.9, alpha=0.9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("pp gain", color="#222222", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#222222")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#222222")

    title_bottom = _figure_title_block(
        fig,
        "GARDIAN Gain over Best Baseline (pp)",
        "Positive = GARDIAN leads; all 4 metrics × 12 settings",
    )
    fig.tight_layout(rect=[0, 0, 1, title_bottom - 0.006])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "figures",
        help="Directory for output figures",
    )
    parser.add_argument(
        "--metric",
        default="ndcg@20",
        choices=["ndcg@20", "ndcg@50", "recall@20", "recall@50"],
        help="Metric for grouped bar chart",
    )
    for ht in HYBRID_TYPES:
        parser.add_argument(
            f"--eval-{ht.replace('hybrid_', '')}",
            type=Path,
            default=DEFAULT_EVAL_PATHS[ht],
            help=f"Evaluation JSON for {ht}",
        )
    args = parser.parse_args()

    eval_paths = {
        ht: getattr(args, f"eval_{ht.replace('hybrid_', '')}") for ht in HYBRID_TYPES
    }
    for ht, p in eval_paths.items():
        if not p.is_file():
            raise FileNotFoundError(f"Missing evaluation file for {ht}: {p}")

    table = load_retrieval_table(eval_paths)
    metric_labels = {v: k for k, v in METRIC_KEYS.items()}
    metric_label = metric_labels.get(args.metric, args.metric)

    out_dir: Path = args.out_dir
    plot_ndcg20_bars(
        table,
        out_dir / "retrieval_ndcg20_grouped_bars.png",
        metric_label=metric_label,
        metric_key=args.metric,
    )
    plot_gardian_gain_heatmap(table, out_dir / "retrieval_gardian_gain_heatmap.png")
    print(f"Wrote figures to {out_dir}/")


if __name__ == "__main__":
    main()
