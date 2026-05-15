"""High-quality static figures for GARDIAN training (loss + dev nDCG@10)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

HYBRID_TYPES = [
    "hybrid_bm25_faiss",
    "hybrid_bm25_medcpt",
    "hybrid_spladepp_faiss",
    "hybrid_spladepp_medcpt",
]

HYBRID_LABELS = {
    "hybrid_bm25_faiss": "BM25 + FAISS",
    "hybrid_bm25_medcpt": "BM25 + MedCPT",
    "hybrid_spladepp_faiss": "SPLADE++ + FAISS",
    "hybrid_spladepp_medcpt": "SPLADE++ + MedCPT",
}

HYBRID_COLORS = {
    "hybrid_bm25_faiss": "#0072B2",
    "hybrid_bm25_medcpt": "#D55E00",
    "hybrid_spladepp_faiss": "#009E73",
    "hybrid_spladepp_medcpt": "#CC79A7",
}

FACE = "#fbfbfb"
GRID = "#e6e6e6"
TEXT_MUTED = "#4a4a4a"


def _apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.titleweight": "600",
            "axes.edgecolor": "#2a2a2a",
            "axes.linewidth": 0.9,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "lines.linewidth": 2.35,
            "lines.markersize": 0,
            "lines.antialiased": True,
            "figure.dpi": 128,
            "savefig.dpi": 300,
            "savefig.facecolor": FACE,
            "figure.facecolor": FACE,
            "axes.facecolor": FACE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
        }
    )


def _load_epoch_logs(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _collect_training_data(results_dir: Path, *, skip_missing: bool = False) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for hybrid_type in HYBRID_TYPES:
        log_path = results_dir / "gardian_training" / hybrid_type / "epoch_logs.jsonl"
        if not log_path.exists():
            msg = f"Missing log file: {log_path}"
            if skip_missing:
                logger.warning(msg + " (skipped)")
                continue
            raise FileNotFoundError(msg)
        rows = _load_epoch_logs(log_path)
        epochs = [int(r["epoch"]) for r in rows]
        losses = [float(r["train_loss"]) for r in rows]
        eval_epochs = [int(r["epoch"]) for r in rows if bool(r.get("did_eval"))]
        eval_ndcg10 = [
            float(r["dev_ndcg@10"]) for r in rows if bool(r.get("did_eval")) and r.get("dev_ndcg@10") is not None
        ]
        out[hybrid_type] = {
            "epochs": epochs,
            "losses": losses,
            "eval_epochs": eval_epochs,
            "eval_ndcg10": eval_ndcg10,
        }
    if not out:
        raise FileNotFoundError(
            f"No epoch_logs.jsonl found under {results_dir / 'gardian_training'} "
            f"(tried {len(HYBRID_TYPES)} hybrid types)."
        )
    return out


def _ordered_keys(training_data: Dict[str, dict]) -> List[str]:
    return [k for k in HYBRID_TYPES if k in training_data]


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins="auto"))


def _plot_line_with_fill(
    ax: plt.Axes,
    x: List[int],
    y: List[float],
    *,
    color: str,
    baseline: float | None = None,
    fill: bool = True,
) -> None:
    """Line + optional translucent fill to baseline (min of y or given baseline)."""
    if not x:
        return
    ax.plot(x, y, color=color, lw=2.35, solid_capstyle="round", zorder=3)
    if not fill:
        return
    base = baseline if baseline is not None else min(y)
    ax.fill_between(x, y, base, color=color, alpha=0.14, linewidth=0, zorder=1)


def _plot_dev_ndcg10(training_data: Dict[str, dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.4), layout="constrained")
    handles: List[Line2D] = []
    for hybrid_type in _ordered_keys(training_data):
        values = training_data[hybrid_type]
        label = HYBRID_LABELS.get(hybrid_type, hybrid_type)
        c = HYBRID_COLORS.get(hybrid_type, "#333333")
        _plot_line_with_fill(ax, values["eval_epochs"], values["eval_ndcg10"], color=c, baseline=None)
        handles.append(Line2D([0], [0], color=c, lw=2.35, label=label))
        if values["eval_ndcg10"]:
            bi = int(np.argmax(values["eval_ndcg10"]))
            bx, by = values["eval_epochs"][bi], values["eval_ndcg10"][bi]
            ax.scatter([bx], [by], s=95, color=c, zorder=5, edgecolors="white", linewidths=0.9, marker="D")
    ax.set_title("Development nDCG@10 — all hybrid families", pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("nDCG@10  (↑ higher is better)")
    _style_axis(ax)
    ax.legend(handles=handles, loc="lower right", frameon=True, fancybox=False, edgecolor="#d0d0d0", framealpha=0.95)
    fig.savefig(output_path.with_suffix(".png"))
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_train_loss(training_data: Dict[str, dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.4), layout="constrained")
    handles: List[Line2D] = []
    for hybrid_type in _ordered_keys(training_data):
        values = training_data[hybrid_type]
        label = HYBRID_LABELS.get(hybrid_type, hybrid_type)
        c = HYBRID_COLORS.get(hybrid_type, "#333333")
        lo = min(values["losses"])
        _plot_line_with_fill(ax, values["epochs"], values["losses"], color=c, baseline=lo * 0.98)
        handles.append(Line2D([0], [0], color=c, lw=2.35, label=label))
    ax.set_title("Training loss — all hybrid families", pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss  (↓ lower is better)")
    _style_axis(ax)
    ax.legend(handles=handles, loc="upper right", frameon=True, fancybox=False, edgecolor="#d0d0d0", framealpha=0.95)
    fig.savefig(output_path.with_suffix(".png"))
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def _panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.add_patch(
        Rectangle(
            (0.01, 0.86),
            0.038,
            0.11,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="#bbbbbb",
            linewidth=0.6,
            zorder=10,
        )
    )
    ax.text(
        0.0295,
        0.915,
        letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="700",
        va="center",
        ha="center",
        zorder=11,
        color="#222",
    )


def _plot_matrix_rows_loss_ndcg(training_data: Dict[str, dict], output_path: Path) -> None:
    """
    One row per retriever: left = loss only, right = nDCG@10 only (no twin axes).
    Best nDCG marked; light fills for readability.
    """
    keys = _ordered_keys(training_data)
    n = len(keys)
    fig_h = max(7.8, 2.05 * n + 1.4)
    fig, axes = plt.subplots(n, 2, figsize=(12.4, fig_h), layout="constrained", squeeze=False, sharex=False)

    letters = "abcdefghijklmnop"
    li = 0
    for i, hybrid_type in enumerate(keys):
        v = training_data[hybrid_type]
        c = HYBRID_COLORS[hybrid_type]
        name = HYBRID_LABELS[hybrid_type]
        ax_l = axes[i, 0]
        ax_r = axes[i, 1]

        lo = min(v["losses"])
        _plot_line_with_fill(ax_l, v["epochs"], v["losses"], color=c, baseline=lo * 0.98)
        _plot_line_with_fill(ax_r, v["eval_epochs"], v["eval_ndcg10"], color=c, baseline=None)
        if v["eval_ndcg10"]:
            bi = int(np.argmax(v["eval_ndcg10"]))
            bx, by = v["eval_epochs"][bi], v["eval_ndcg10"][bi]
            ax_r.scatter([bx], [by], s=88, color=c, zorder=5, edgecolors="white", linewidths=0.85, marker="D")

        ax_l.set_ylabel("Loss", color=TEXT_MUTED)
        ax_r.set_ylabel("nDCG@10", color=TEXT_MUTED)
        ax_l.set_title(name, loc="left", fontsize=11.5, pad=6, color="#111")
        _style_axis(ax_l)
        _style_axis(ax_r)
        ax_l.tick_params(axis="y", labelcolor=TEXT_MUTED)
        ax_r.tick_params(axis="y", labelcolor=TEXT_MUTED)

        _panel_letter(ax_l, letters[li])
        li += 1
        _panel_letter(ax_r, letters[li])
        li += 1

        if i < n - 1:
            ax_l.set_xticklabels([])
            ax_r.set_xticklabels([])

    axes[-1, 0].set_xlabel("Epoch")
    axes[-1, 1].set_xlabel("Epoch")

    fig.suptitle(
        "GARDIAN training dynamics by retriever\n(left: training loss — right: development nDCG@10)",
        fontsize=13.5,
        fontweight="600",
        color="#111",
        y=1.01,
    )
    fig.savefig(output_path.with_suffix(".png"))
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_wide_two_column(training_data: Dict[str, dict], output_path: Path) -> None:
    """
    Landscape summary: nDCG@10 | training loss (lines only, no fills), compact hybrid key
    with colored names and readable metrics.
    """
    keys = _ordered_keys(training_data)
    fig = plt.figure(figsize=(15.0, 6.0), facecolor=FACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.08, 1.08, 0.58], wspace=0.3)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    ax_key = fig.add_subplot(gs[0, 2])
    ax_key.set_axis_off()
    ax_key.set_xlim(0, 1)
    ax_key.set_ylim(0, 1)

    y0, y_step = 0.96, 0.228
    for row, hybrid_type in enumerate(keys):
        v = training_data[hybrid_type]
        c = HYBRID_COLORS[hybrid_type]
        name = HYBRID_LABELS[hybrid_type]
        ex, ey = v["eval_epochs"], v["eval_ndcg10"]
        ep, ls = v["epochs"], v["losses"]
        best_ndcg = float(max(ey))
        best_i = int(np.argmax(ey))
        e_best, y_best = ex[best_i], ey[best_i]
        final_ndcg = float(ey[-1])
        final_loss = float(ls[-1])
        last_ep = int(ep[-1])
        last_i = len(ey) - 1

        _plot_line_with_fill(ax_l, ex, ey, color=c, baseline=None, fill=False)
        _plot_line_with_fill(ax_r, ep, ls, color=c, baseline=None, fill=False)

        ax_r.scatter([last_ep], [final_loss], s=58, color=c, zorder=6, edgecolors="white", linewidths=0.65)
        if best_i == last_i:
            ax_l.scatter([ex[-1]], [ey[-1]], s=68, color=c, zorder=7, marker="D", edgecolors="white", linewidths=0.7)
        else:
            ax_l.scatter([e_best], [y_best], s=76, color=c, zorder=7, marker="D", edgecolors="white", linewidths=0.75)
            ax_l.scatter([ex[-1]], [ey[-1]], s=52, color=c, zorder=6, marker="o", edgecolors="white", linewidths=0.65)

        y = y0 - row * y_step
        swatch_bottom = y - 0.021
        ax_key.add_patch(
            Rectangle(
                (0.03, swatch_bottom),
                0.032,
                0.014,
                transform=ax_key.transAxes,
                facecolor=c,
                edgecolor="#ffffff",
                linewidth=0.5,
                clip_on=False,
                zorder=5,
            )
        )
        ax_key.text(
            0.11,
            y,
            name,
            color=c,
            fontsize=11.5,
            fontweight="700",
            transform=ax_key.transAxes,
            va="top",
        )
        if abs(best_ndcg - final_ndcg) < 1e-6:
            ndcg_txt = f"nDCG@10  {best_ndcg:.4f}"
        else:
            ndcg_txt = f"nDCG@10  best {best_ndcg:.4f}  last {final_ndcg:.4f}"
        ax_key.text(
            0.11,
            y - 0.052,
            ndcg_txt,
            color="#222222",
            fontsize=10.2,
            transform=ax_key.transAxes,
            va="top",
            family="monospace",
        )
        ax_key.text(
            0.11,
            y - 0.095,
            f"loss     {final_loss:.4f}",
            color="#333333",
            fontsize=10.2,
            transform=ax_key.transAxes,
            va="top",
            family="monospace",
        )

    ax_l.set_title("Development nDCG@10", fontsize=12.5, pad=10, fontweight="600")
    ax_r.set_title("Training loss", fontsize=12.5, pad=10, fontweight="600")
    ax_l.set_xlabel("Epoch")
    ax_r.set_xlabel("Epoch")
    ax_l.set_ylabel("nDCG@10")
    ax_r.set_ylabel("Loss")
    _style_axis(ax_l)
    _style_axis(ax_r)
    ax_l.yaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
    ax_r.yaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
    ax_l.tick_params(axis="both", labelsize=10)
    ax_r.tick_params(axis="both", labelsize=10)
    ax_key.text(
        0.03,
        1.0,
        "Hybrid summary",
        fontsize=11,
        fontweight="600",
        color="#111",
        transform=ax_key.transAxes,
        va="top",
    )

    fig.suptitle("GARDIAN — training curves by hybrid family", fontsize=13, fontweight="600", color="#111", y=0.96)
    fig.text(
        0.5,
        0.905,
        "◆ best dev nDCG@10   ● last eval when it differs from best",
        ha="center",
        fontsize=9.2,
        color="#444444",
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.11, wspace=0.3)
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.22)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GARDIAN training figures: matrix (loss|nDCG per row), wide summary, single-metric plots."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--no-separate", action="store_true", help="Skip single-metric nDCG / loss PNG+PDF.")
    parser.add_argument("--no-matrix", action="store_true", help="Skip 4×2 loss|nDCG matrix figure.")
    parser.add_argument("--no-wide", action="store_true", help="Skip wide two-column overlaid figure.")
    args = parser.parse_args()

    if args.no_separate and args.no_matrix and args.no_wide:
        raise ValueError("Nothing to plot.")

    _apply_publication_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = _collect_training_data(args.results_dir, skip_missing=bool(args.skip_missing))

    ndcg_out = args.output_dir / "gardian_dev_ndcg10_hybrid_types.png"
    loss_out = args.output_dir / "gardian_train_loss_hybrid_types.png"
    matrix_out = args.output_dir / "gardian_training_matrix_loss_ndcg.png"
    wide_out = args.output_dir / "gardian_training_wide_summary.png"

    if not args.no_separate:
        _plot_dev_ndcg10(data, ndcg_out)
        _plot_train_loss(data, loss_out)
        print(f"Saved: {ndcg_out} (+ .pdf)")
        print(f"Saved: {loss_out} (+ .pdf)")
    if not args.no_matrix:
        _plot_matrix_rows_loss_ndcg(data, matrix_out)
        print(f"Saved: {matrix_out} (+ .pdf)")
    if not args.no_wide:
        _plot_wide_two_column(data, wide_out)
        print(f"Saved: {wide_out} (+ .pdf)")


if __name__ == "__main__":
    main()
