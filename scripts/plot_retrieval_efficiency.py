"""Publication figure: Hit@20, latency, and MRR (compact grid for two-column papers)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_STEM = "retrieval_hit_latency_mrr"

HYBRID_TYPES = [
    "hybrid_bm25_faiss",
    "hybrid_spladepp_faiss",
    "hybrid_bm25_medcpt",
    "hybrid_spladepp_medcpt",
]

BACKEND_XTICKS = ["BM25\n+FAISS", "SPL++\n+FAISS", "BM25\n+MedCPT", "SPL++\n+MedCPT"]

# Shorter than full \\textwidth, readable in a two-column paper (figure* at ~0.85–0.92\\textwidth)
DEFAULT_WIDTH_IN = 6.8
DEFAULT_HEIGHT_IN = 3.85
FIGURE_TITLE = "Retrieval Quality and Efficiency"

DATASETS = ["pubmedqa_labeled", "pubmedqa_artificial", "medmcqa"]
DATASET_LABELS = {
    "pubmedqa_labeled": "PubMedQA-L",
    "pubmedqa_artificial": "PubMedQA-A",
    "medmcqa": "MedMCQA",
}
PLOT_SYSTEMS = ["sparse", "dense", "hybrid", "gardian"]
SYSTEM_LABELS = {
    "sparse": "Sparse",
    "dense": "Dense",
    "hybrid": "Hybrid",
    "gardian": "GARDIAN",
}
SYSTEM_COLORS = {
    "sparse": "#B8B8B8",
    "dense": "#4472C4",
    "hybrid": "#2E9E9E",
    "gardian": "#E53935",
}


def _load_efficiency(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _system_metrics(systems: Dict[str, Any], sys: str) -> Dict[str, Any]:
    block = systems.get(sys, {})
    if isinstance(block, dict) and block:
        return block
    if sys == "sparse":
        return systems.get("bm25") or systems.get("spladepp") or {}
    if sys == "dense":
        return systems.get("faiss") or systems.get("medcpt") or {}
    return {}


def load_eval_metrics(
    eval_dir: Path,
) -> Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]]:
    """``metrics[hybrid_type][dataset][system] -> {hit@20, mrr}``."""
    out: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]] = {}
    for ht in HYBRID_TYPES:
        path = eval_dir / f"evaluation_{ht}.json"
        if not path.is_file():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        ds_root = obj.get("results", {}).get(ht, {})
        out[ht] = {}
        for ds in DATASETS:
            systems = ds_root.get(ds, {})
            if not isinstance(systems, dict):
                continue
            out[ht][ds] = {}
            for sys in PLOT_SYSTEMS:
                block = _system_metrics(systems, sys)
                out[ht][ds][sys] = {
                    "hit@20": float(block["hit@20"]) if "hit@20" in block else None,
                    "mrr": float(block["mrr"]) if "mrr" in block else None,
                }
    return out


def _draw_metric_panel(
    ax: plt.Axes,
    *,
    dataset: str,
    metric_key: str,
    eval_metrics: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]],
    latency: Optional[Dict[str, Any]] = None,
    ylim: Optional[tuple[float, float]] = None,
    show_xticklabels: bool = True,
) -> None:
    x = np.arange(len(HYBRID_TYPES))
    w = 0.19
    for si, sys in enumerate(PLOT_SYSTEMS):
        heights: list[float] = []
        for ht in HYBRID_TYPES:
            val: Optional[float] = None
            if metric_key == "mean_ms" and latency is not None:
                block = latency.get(ht, {}).get(dataset, {}).get(sys, {})
                if isinstance(block, dict) and block.get("mean_ms") is not None:
                    val = float(block["mean_ms"])
            else:
                cell = eval_metrics.get(ht, {}).get(dataset, {}).get(sys, {})
                if isinstance(cell, dict):
                    val = cell.get(metric_key)
            heights.append(float(val) if val is not None else np.nan)
        ax.bar(
            x + (si - 1.5) * w,
            heights,
            w * 0.9,
            label=SYSTEM_LABELS[sys],
            color=SYSTEM_COLORS[sys],
        )
    ax.set_xticks(x)
    if show_xticklabels:
        ax.set_xticklabels(BACKEND_XTICKS, fontsize=6, ha="center", linespacing=0.9)
    else:
        ax.set_xticklabels([])
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_retrieval_hit_latency_mrr(
    eval_metrics: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]],
    latency: Dict[str, Any],
    out_dir: Path,
    *,
    width_in: float = DEFAULT_WIDTH_IN,
    height_in: float = DEFAULT_HEIGHT_IN,
    title: str = FIGURE_TITLE,
) -> None:
    """3×3 grid: title + legend, then Hit@20 | latency | MRR × three datasets."""
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    panels = [
        ("hit@20", "Hit@20", (0.0, 1.05), None),
        ("mean_ms", "Latency (ms/q)", None, latency),
        ("mrr", "MRR", (0.0, 1.05), None),
    ]
    n_rows, n_cols = len(DATASETS), len(panels)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(width_in, height_in),
        facecolor="white",
        sharex=True,
        gridspec_kw={"wspace": 0.14, "hspace": 0.28},
    )

    handles: list[Any] = []
    labels_leg: list[str] = []
    for row_idx, ds in enumerate(DATASETS):
        for col_idx, (metric_key, col_title, ylim, lat_src) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            _draw_metric_panel(
                ax,
                dataset=ds,
                metric_key=metric_key,
                eval_metrics=eval_metrics,
                latency=lat_src,
                ylim=ylim,
                show_xticklabels=(row_idx == n_rows - 1),
            )
            if row_idx == 0:
                ax.set_title(col_title, fontweight="700", fontsize=8.5, pad=4)
            if col_idx == 0:
                ax.set_ylabel(
                    DATASET_LABELS.get(ds, ds),
                    fontsize=7.5,
                    fontweight="600",
                    labelpad=2,
                )
            if col_idx == 0 and row_idx == 0:
                handles, labels_leg = ax.get_legend_handles_labels()

    fig.subplots_adjust(left=0.11, right=0.99, top=0.845, bottom=0.14)
    fig.suptitle(title, fontsize=10.5, fontweight="700", y=0.995)
    fig.legend(
        handles,
        labels_leg,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.942),
        fontsize=7.5,
        borderaxespad=0.2,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_dir / f"{FIGURE_STEM}.{ext}"
        save_kw: dict[str, Any] = {"facecolor": "white", "pad_inches": 0.06}
        if ext == "png":
            save_kw["dpi"] = 300
        fig.savefig(path, **save_kw)
    plt.close(fig)
    print(f"Wrote {out_dir / FIGURE_STEM}.png ({width_in:.2f}×{height_in:.2f} in)")
    print(f"Wrote {out_dir / FIGURE_STEM}.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--efficiency-json",
        type=Path,
        default=REPO_ROOT / "results" / "retrieval_efficiency.json",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=REPO_ROOT / "results",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "figures",
    )
    parser.add_argument(
        "--width-in",
        type=float,
        default=DEFAULT_WIDTH_IN,
        help="Figure width in inches (default: compact for two-column papers).",
    )
    parser.add_argument(
        "--height-in",
        type=float,
        default=DEFAULT_HEIGHT_IN,
        help="Figure height in inches.",
    )
    parser.add_argument("--title", type=str, default=FIGURE_TITLE)
    args = parser.parse_args()

    if not args.efficiency_json.is_file():
        raise FileNotFoundError(
            f"Missing {args.efficiency_json}. Run scripts/benchmark_retrieval_efficiency.py first."
        )

    eff = _load_efficiency(args.efficiency_json)
    eval_metrics = load_eval_metrics(args.eval_dir)
    if not eval_metrics:
        raise FileNotFoundError(
            f"No evaluation_*.json under {args.eval_dir}. Run scripts/05_evaluate_gardian.py first."
        )

    missing = [
        f"{ht}/{ds}"
        for ht in HYBRID_TYPES
        for ds in DATASETS
        if eval_metrics.get(ht, {}).get(ds, {}).get("gardian", {}).get("hit@20") is None
    ]
    if missing:
        print(
            "Warning: GARDIAN Hit@20 missing for "
            + ", ".join(missing[:6])
            + ("..." if len(missing) > 6 else "")
            + ".\n"
            "Run once (not in parallel):\n"
            "  .venv/bin/python scripts/backfill_hit_metrics.py "
            "results/evaluation_hybrid_*.json --with-gardian --device cuda"
        )

    plot_retrieval_hit_latency_mrr(
        eval_metrics,
        eff.get("latency_ms", {}),
        args.out_dir,
        width_in=args.width_in,
        height_in=args.height_in,
        title=args.title,
    )


if __name__ == "__main__":
    main()
