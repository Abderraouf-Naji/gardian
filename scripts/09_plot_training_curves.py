"""High-quality static figures for GARDIAN training (loss + dev nDCG@10 + LR).

Includes a creative dark-theme **Class A** hero panel (nDCG, loss, LR stacked).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
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

# Distinct linestyle + marker per family so curves stay separable in print/B&W.
HYBRID_LINESTYLES = {
    "hybrid_bm25_faiss": "-",
    "hybrid_bm25_medcpt": "--",
    "hybrid_spladepp_faiss": "-.",
    "hybrid_spladepp_medcpt": ":",
}

HYBRID_MARKERS = {
    "hybrid_bm25_faiss": "o",
    "hybrid_bm25_medcpt": "s",
    "hybrid_spladepp_faiss": "^",
    "hybrid_spladepp_medcpt": "D",
}

FACE = "#fbfbfb"
GRID = "#e6e6e6"
TEXT_MUTED = "#4a4a4a"

# IEEE / ACM single-column figure (~\\linewidth in two-column templates).
LATEX_COL_WIDTH_IN = 3.35
LATEX_COL_HEIGHT_IN = 2.45
FULL_WIDTH_SIZE = (10.5, 5.6)


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


def _apply_latex_column_style() -> None:
    """Compact typography for single-column LaTeX (use with LATEX_COL_* figsize)."""
    _apply_publication_style()
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "legend.fontsize": 6.2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "lines.linewidth": 1.75,
            "lines.markersize": 4.5,
        }
    )


def _triplet_figsize(*, latex_column: bool, width_in: float = LATEX_COL_WIDTH_IN) -> tuple[float, float]:
    if latex_column:
        return (width_in, LATEX_COL_HEIGHT_IN)
    return FULL_WIDTH_SIZE


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
        lrs = [float(r.get("lr", 0.0)) for r in rows]
        out[hybrid_type] = {
            "epochs": epochs,
            "losses": losses,
            "lrs": lrs,
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
    if not x:
        return
    ax.plot(x, y, color=color, lw=2.35, solid_capstyle="round", zorder=3)
    if not fill:
        return
    base = baseline if baseline is not None else min(y)
    ax.fill_between(x, y, base, color=color, alpha=0.14, linewidth=0, zorder=1)


def _hybrid_series_style(hybrid_type: str) -> dict:
    return {
        "color": HYBRID_COLORS.get(hybrid_type, "#333333"),
        "linestyle": HYBRID_LINESTYLES.get(hybrid_type, "-"),
        "marker": HYBRID_MARKERS.get(hybrid_type, "o"),
    }


def _plot_hybrid_curve(
    ax: plt.Axes,
    x: List[int],
    y: List[float],
    hybrid_type: str,
    *,
    mark_best: bool = False,
    fill: bool = False,
    fill_baseline: float | None = None,
    linewidth: float = 2.6,
    markersize: float = 7.5,
    best_marker_size: float = 130,
) -> Line2D:
    """One hybrid family: color + linestyle + per-epoch markers."""
    st = _hybrid_series_style(hybrid_type)
    if not x or not y:
        return Line2D([0], [0], color=st["color"], label=HYBRID_LABELS.get(hybrid_type, hybrid_type))

    ax.plot(
        x,
        y,
        color=st["color"],
        linestyle=st["linestyle"],
        linewidth=linewidth,
        marker=st["marker"],
        markersize=markersize,
        markerfacecolor=st["color"],
        markeredgecolor="white",
        markeredgewidth=0.65,
        solid_capstyle="round",
        zorder=3,
    )
    if fill:
        base = fill_baseline if fill_baseline is not None else min(y)
        ax.fill_between(x, y, base, color=st["color"], alpha=0.10, linewidth=0, zorder=1)

    if mark_best and y:
        bi = int(np.argmax(y))
        ax.scatter(
            [x[bi]],
            [y[bi]],
            s=best_marker_size,
            facecolors="none",
            edgecolors=st["color"],
            linewidths=1.6,
            marker=st["marker"],
            zorder=5,
        )

    return Line2D(
        [0],
        [0],
        color=st["color"],
        linestyle=st["linestyle"],
        marker=st["marker"],
        markersize=markersize * 0.9,
        markerfacecolor=st["color"],
        markeredgecolor="white",
        markeredgewidth=0.55,
        linewidth=linewidth,
        label=HYBRID_LABELS.get(hybrid_type, hybrid_type),
    )


def _save_figure(fig: plt.Figure, output_path: Path, *, latex_column: bool = False) -> None:
    pad = 0.03 if latex_column else 0.12
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", pad_inches=pad, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=pad)
    plt.close(fig)


def _triplet_curve_kw(*, latex_column: bool) -> dict:
    if not latex_column:
        return {"linewidth": 2.6, "markersize": 7.5, "best_marker_size": 130}
    return {"linewidth": 1.75, "markersize": 4.8, "best_marker_size": 72}


def _triplet_legend(ax: plt.Axes, handles: List[Line2D], *, latex_column: bool, loc: str) -> None:
    if latex_column:
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=2,
            frameon=True,
            fancybox=False,
            edgecolor="#d0d0d0",
            framealpha=0.98,
            columnspacing=0.9,
            handletextpad=0.45,
            borderpad=0.35,
        )
    else:
        ax.legend(
            handles=handles,
            loc=loc,
            frameon=True,
            fancybox=False,
            edgecolor="#d0d0d0",
            framealpha=0.98,
            ncol=1,
        )


def _plot_dev_ndcg10(
    training_data: Dict[str, dict],
    output_path: Path,
    *,
    latex_column: bool = True,
    width_in: float = LATEX_COL_WIDTH_IN,
) -> None:
    if latex_column:
        _apply_latex_column_style()
    fig, ax = plt.subplots(
        figsize=_triplet_figsize(latex_column=latex_column, width_in=width_in),
        layout="constrained",
    )
    ck = _triplet_curve_kw(latex_column=latex_column)
    handles: List[Line2D] = []
    for hybrid_type in _ordered_keys(training_data):
        values = training_data[hybrid_type]
        h = _plot_hybrid_curve(
            ax,
            values["eval_epochs"],
            values["eval_ndcg10"],
            hybrid_type,
            mark_best=True,
            **ck,
        )
        handles.append(h)
    title = "Dev nDCG@10" if latex_column else "Development nDCG@10 — four hybrid families"
    ax.set_title(title, pad=6 if latex_column else 12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("nDCG@10" if latex_column else "nDCG@10  (↑ higher is better)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
    _style_axis(ax)
    _triplet_legend(ax, handles, latex_column=latex_column, loc="lower right")
    _save_figure(fig, output_path, latex_column=latex_column)
    if latex_column:
        _apply_publication_style()


def _plot_train_loss(
    training_data: Dict[str, dict],
    output_path: Path,
    *,
    latex_column: bool = True,
    width_in: float = LATEX_COL_WIDTH_IN,
) -> None:
    if latex_column:
        _apply_latex_column_style()
    fig, ax = plt.subplots(
        figsize=_triplet_figsize(latex_column=latex_column, width_in=width_in),
        layout="constrained",
    )
    ck = _triplet_curve_kw(latex_column=latex_column)
    handles: List[Line2D] = []
    for hybrid_type in _ordered_keys(training_data):
        values = training_data[hybrid_type]
        lo = min(values["losses"]) if values["losses"] else 0.0
        h = _plot_hybrid_curve(
            ax,
            values["epochs"],
            values["losses"],
            hybrid_type,
            fill=True,
            fill_baseline=lo * 0.98,
            **ck,
        )
        handles.append(h)
    title = "Training loss" if latex_column else "Training loss — four hybrid families"
    ax.set_title(title, pad=6 if latex_column else 12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss" if latex_column else "Loss  (↓ lower is better)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
    _style_axis(ax)
    _triplet_legend(ax, handles, latex_column=latex_column, loc="upper right")
    _save_figure(fig, output_path, latex_column=latex_column)
    if latex_column:
        _apply_publication_style()


def _plot_learning_rate(
    training_data: Dict[str, dict],
    output_path: Path,
    *,
    latex_column: bool = True,
    width_in: float = LATEX_COL_WIDTH_IN,
) -> None:
    if latex_column:
        _apply_latex_column_style()
    fig, ax = plt.subplots(
        figsize=_triplet_figsize(latex_column=latex_column, width_in=width_in),
        layout="constrained",
    )
    ck = _triplet_curve_kw(latex_column=latex_column)
    handles: List[Line2D] = []
    for hybrid_type in _ordered_keys(training_data):
        values = training_data[hybrid_type]
        h = _plot_hybrid_curve(
            ax,
            values["epochs"],
            values["lrs"],
            hybrid_type,
            fill=True,
            fill_baseline=0.0,
            **ck,
        )
        handles.append(h)
    title = "Learning rate" if latex_column else "Learning-rate schedule — four hybrid families"
    ax.set_title(title, pad=6 if latex_column else 12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR" if latex_column else "Learning rate")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    _style_axis(ax)
    _triplet_legend(ax, handles, latex_column=latex_column, loc="upper right")
    _save_figure(fig, output_path, latex_column=latex_column)
    if latex_column:
        _apply_publication_style()


def _plot_triplet_separate(
    training_data: Dict[str, dict],
    output_dir: Path,
    *,
    latex_column: bool = True,
    width_in: float = LATEX_COL_WIDTH_IN,
) -> None:
    """Three standalone figures: nDCG@10, training loss, LR (all four hybrids)."""
    suffix = "_latexcol" if latex_column else ""
    kw = {"latex_column": latex_column, "width_in": width_in}
    _plot_dev_ndcg10(training_data, output_dir / f"gardian_ndcg10_four_hybrids{suffix}", **kw)
    _plot_train_loss(training_data, output_dir / f"gardian_loss_four_hybrids{suffix}", **kw)
    _plot_learning_rate(training_data, output_dir / f"gardian_lr_four_hybrids{suffix}", **kw)


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


def _apply_class_a_style() -> None:
    """Clean white background for Class A figures (matches publication style)."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": "#2a2a2a",
            "axes.linewidth": 0.9,
            "axes.labelcolor": "#222222",
            "axes.titleweight": "600",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#111111",
            "figure.facecolor": FACE,
            "axes.facecolor": FACE,
            "savefig.facecolor": FACE,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.alpha": 0.9,
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
        }
    )


def _gradient_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray, *, cmap_name: str, lw: float = 2.8) -> None:
    """Segment-colored line (epoch → metric) along the training trajectory."""
    if len(x) < 2:
        return
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(x.min(), x.max())
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=lw, capstyle="round", zorder=4)
    ax.add_collection(lc)
    ax.plot(x, y, color="#ffffff", lw=1.0, alpha=0.85, zorder=3)


def _class_a_badge(ax: plt.Axes, letter: str = "A") -> None:
    badge = FancyBboxPatch(
        (0.02, 0.90),
        0.055,
        0.085,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        transform=ax.transAxes,
        facecolor="#ffffff",
        edgecolor="#2a2a2a",
        linewidth=1.0,
        zorder=20,
    )
    ax.add_patch(badge)
    ax.text(
        0.0475,
        0.9425,
        letter,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="800",
        ha="center",
        va="center",
        color="#111111",
        zorder=21,
    )


def _plot_class_a_hero(
    values: dict,
    hybrid_type: str,
    output_path: Path,
) -> None:
    """
    Creative Class A panel: stacked nDCG@10, training loss, and LR schedule
    with gradient trajectories, warmup band, and best-epoch spine.
    """
    _apply_class_a_style()
    name = HYBRID_LABELS.get(hybrid_type, hybrid_type)
    accent = HYBRID_COLORS.get(hybrid_type, "#5eead4")

    ep = np.asarray(values["epochs"], dtype=float)
    loss = np.asarray(values["losses"], dtype=float)
    lr = np.asarray(values["lrs"], dtype=float)
    ex = np.asarray(values["eval_epochs"], dtype=float)
    ndcg = np.asarray(values["eval_ndcg10"], dtype=float)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11.2, 8.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0, 0.72], "hspace": 0.08},
    )
    fig.patch.set_facecolor(FACE)

    best_i = int(np.argmax(ndcg)) if len(ndcg) else 0
    best_ep = float(ex[best_i]) if len(ex) else 1.0
    best_ndcg = float(ndcg[best_i]) if len(ndcg) else 0.0

    # Warmup: LR strictly increasing from epoch 1 (config warmup_epochs=3).
    warmup_end = 3
    if len(lr) >= 2:
        for i in range(1, min(len(lr), 6)):
            if lr[i] <= lr[i - 1]:
                warmup_end = i
                break
        else:
            warmup_end = min(3, len(ep))

    for ax in axes:
        ax.set_facecolor(FACE)
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#2a2a2a")

    ax_n, ax_l, ax_lr = axes
    best_mark = "#c45c00"
    lr_color = "#009E73"

    # --- nDCG@10 ---
    if len(ex):
        ax_n.fill_between(ex, ndcg, ndcg.min() * 0.995, color=accent, alpha=0.18, zorder=1)
        _gradient_line(ax_n, ex, ndcg, cmap_name="viridis", lw=3.0)
        ax_n.scatter(
            [best_ep],
            [best_ndcg],
            s=200,
            c=best_mark,
            marker="*",
            zorder=8,
            edgecolors="white",
            linewidths=1.0,
            label=f"best {best_ndcg:.4f} @ ep {int(best_ep)}",
        )
        ax_n.axvline(best_ep, color=best_mark, ls="--", lw=1.0, alpha=0.55, zorder=2)
    ax_n.set_ylabel("nDCG@10", fontweight="600", color="#111")
    ax_n.set_title(f"GARDIAN training dynamics — {name}", loc="left", pad=14, fontsize=13, color="#111")
    ax_n.yaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
    _class_a_badge(ax_n, "A")

    # --- Loss ---
    lo = float(loss.min()) if len(loss) else 0.0
    ax_l.fill_between(ep, loss, lo * 0.97, color="#D55E00", alpha=0.16, zorder=1)
    _gradient_line(ax_l, ep, loss, cmap_name="Blues", lw=2.9)
    ax_l.scatter([ep[-1]], [loss[-1]], s=70, c="#D55E00", zorder=6, edgecolors="white", linewidths=0.9)
    ax_l.set_ylabel("Train loss", fontweight="600", color="#111")
    ax_l.yaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
    ax_l.axvline(best_ep, color=best_mark, ls="--", lw=1.0, alpha=0.4, zorder=2)

    # --- LR ---
    ax_lr.fill_between(ep, lr, 0.0, color=lr_color, alpha=0.2, zorder=1)
    ax_lr.plot(ep, lr, color=lr_color, lw=2.2, zorder=4, solid_capstyle="round")
    if warmup_end >= ep[0]:
        ax_lr.axvspan(ep[0] - 0.45, warmup_end + 0.45, color="#E69F00", alpha=0.12, zorder=0)
        ax_lr.text(
            warmup_end / 2 + 0.15,
            float(lr.max()) * 0.92,
            "warmup",
            ha="center",
            fontsize=8.5,
            color="#996600",
            style="italic",
        )
    ax_lr.set_ylabel("Learning rate", fontweight="600", color="#111")
    ax_lr.set_xlabel("Epoch", fontweight="600", color="#111")
    ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax_lr.axvline(best_ep, color=best_mark, ls="--", lw=1.0, alpha=0.4, zorder=2)

    fig.text(
        0.985,
        0.52,
        f"◆ best dev\n   epoch {int(best_ep)}",
        ha="right",
        va="center",
        fontsize=9.2,
        color=best_mark,
        linespacing=1.35,
    )

    ax_n.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#d0d0d0",
        labelcolor="#111",
    )

    if len(ex) == len(lr) and len(ex) >= 3:
        inset = ax_n.inset_axes([0.72, 0.12, 0.24, 0.38])
        inset.set_facecolor("#ffffff")
        sc = inset.scatter(
            lr,
            ndcg,
            c=ex,
            cmap="viridis",
            s=42,
            edgecolors="#666666",
            linewidths=0.45,
            zorder=3,
        )
        inset.set_xlabel("LR", fontsize=7.5, color=TEXT_MUTED)
        inset.set_ylabel("nDCG", fontsize=7.5, color=TEXT_MUTED)
        inset.tick_params(labelsize=6.5, colors=TEXT_MUTED)
        for sp in inset.spines.values():
            sp.set_color("#cccccc")
        inset.set_title("LR ↔ nDCG", fontsize=7.5, color=TEXT_MUTED, pad=3)
        cb = fig.colorbar(sc, ax=inset, fraction=0.08, pad=0.02)
        cb.set_label("epoch", fontsize=6.5, color=TEXT_MUTED)
        cb.ax.tick_params(labelsize=6, colors=TEXT_MUTED)
        cb.outline.set_edgecolor("#d0d0d0")

    fig.subplots_adjust(left=0.09, right=0.96, top=0.93, bottom=0.08, hspace=0.06)
    stem = output_path.stem
    if hybrid_type not in stem:
        out = output_path.parent / f"{stem}_{hybrid_type}"
    else:
        out = output_path
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.18)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    _apply_publication_style()


def _plot_class_a_orbit_grid(training_data: Dict[str, dict], output_path: Path) -> None:
    """
    Class A multi-hybrid: polar 'orbit' per family — angle = epoch, radius = normalized
    blend of nDCG (↑), inverted loss (↓), and LR.
    """
    _apply_class_a_style()
    keys = _ordered_keys(training_data)
    n = len(keys)
    cols = min(2, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(6.2 * cols, 5.8 * rows),
        subplot_kw={"projection": "polar"},
        facecolor=FACE,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, hybrid_type in zip(axes_flat, keys):
        v = training_data[hybrid_type]
        ep = np.asarray(v["epochs"], dtype=float)
        loss = np.asarray(v["losses"], dtype=float)
        lr = np.asarray(v["lrs"], dtype=float)
        ndcg_full = np.full(len(ep), np.nan)
        for e, n in zip(v["eval_epochs"], v["eval_ndcg10"]):
            idx = int(e) - 1
            if 0 <= idx < len(ndcg_full):
                ndcg_full[idx] = n
        valid = ~np.isnan(ndcg_full)
        if valid.sum() < 2:
            continue
        ndcg_interp = np.interp(ep, ep[valid], ndcg_full[valid])

        def _norm(arr: np.ndarray, invert: bool = False) -> np.ndarray:
            lo, hi = float(arr.min()), float(arr.max())
            if hi - lo < 1e-12:
                z = np.ones_like(arr) * 0.5
            else:
                z = (arr - lo) / (hi - lo)
            return 1.0 - z if invert else z

        r = 0.42 + 0.38 * (
            0.45 * _norm(ndcg_interp)
            + 0.35 * _norm(loss, invert=True)
            + 0.20 * _norm(lr)
        )
        theta = 2 * np.pi * (ep - ep.min()) / max(ep.max() - ep.min(), 1.0)
        c = HYBRID_COLORS[hybrid_type]
        ax.set_facecolor(FACE)
        ax.plot(theta, r, color=c, lw=2.4, zorder=3)
        ax.fill(theta, r, color=c, alpha=0.14, zorder=1)
        bi = int(np.argmax(ndcg_interp))
        ax.scatter([theta[bi]], [r[bi]], s=120, c="#c45c00", marker="*", zorder=5, edgecolors="white")
        ax.set_title(HYBRID_LABELS[hybrid_type], color="#111", pad=18, fontsize=11)
        ax.set_yticklabels([])
        ax.grid(color=GRID, alpha=0.85)
        ax.tick_params(colors=TEXT_MUTED, labelsize=8)

    for ax in axes_flat[len(keys) :]:
        ax.set_visible(False)

    if keys:
        _class_a_badge(axes_flat[0], "A")

    fig.suptitle(
        "Training orbit — epoch sweeps a composite of nDCG, loss, and LR",
        fontsize=12.5,
        fontweight="600",
        color="#111",
        y=1.02,
    )
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.2)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    _apply_publication_style()


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
    parser.add_argument(
        "--triplet-only",
        action="store_true",
        help="Only the three separate plots: nDCG@10, loss, LR (four hybrid families each).",
    )
    parser.add_argument(
        "--full-width",
        action="store_true",
        help="Wide figures (~10.5 in). Default is single-column LaTeX (3.35×2.45 in).",
    )
    parser.add_argument(
        "--latex-width-in",
        type=float,
        default=LATEX_COL_WIDTH_IN,
        help=f"Figure width in inches for single-column output (default {LATEX_COL_WIDTH_IN}).",
    )
    parser.add_argument("--no-separate", action="store_true", help="Skip single-metric nDCG / loss / LR PNG+PDF.")
    parser.add_argument("--no-matrix", action="store_true", help="Skip 4×2 loss|nDCG matrix figure.")
    parser.add_argument("--no-wide", action="store_true", help="Skip wide two-column overlaid figure.")
    parser.add_argument(
        "--class-a",
        action="store_true",
        help="Generate creative Class A figures (stacked nDCG/loss/LR hero + polar orbit grid).",
    )
    parser.add_argument(
        "--hybrid",
        type=str,
        default="hybrid_bm25_faiss",
        help="Hybrid family for the Class A hero panel (default: hybrid_bm25_faiss).",
    )
    args = parser.parse_args()
    latex_column = not bool(args.full_width)
    col_width_in = float(args.latex_width_in)

    if args.triplet_only:
        args.no_matrix = True
        args.no_wide = True
        args.class_a = False

    if args.no_separate and args.no_matrix and args.no_wide and not args.class_a:
        raise ValueError("Nothing to plot.")

    _apply_publication_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = _collect_training_data(args.results_dir, skip_missing=bool(args.skip_missing))

    triplet_suffix = "_latexcol" if latex_column else ""
    ndcg_out = args.output_dir / f"gardian_ndcg10_four_hybrids{triplet_suffix}"
    loss_out = args.output_dir / f"gardian_loss_four_hybrids{triplet_suffix}"
    lr_out = args.output_dir / f"gardian_lr_four_hybrids{triplet_suffix}"
    matrix_out = args.output_dir / "gardian_training_matrix_loss_ndcg.png"
    wide_out = args.output_dir / "gardian_training_wide_summary.png"

    if not args.no_separate:
        _plot_triplet_separate(
            data, args.output_dir, latex_column=latex_column, width_in=col_width_in
        )
        w = f"{col_width_in:.2f}×{LATEX_COL_HEIGHT_IN:.2f} in" if latex_column else "full width"
        print(f"Saved triplet figures ({w}):")
        print(f"Saved: {ndcg_out} (+ .png/.pdf)")
        print(f"Saved: {loss_out} (+ .png/.pdf)")
        print(f"Saved: {lr_out} (+ .png/.pdf)")
    if not args.no_matrix:
        _plot_matrix_rows_loss_ndcg(data, matrix_out)
        print(f"Saved: {matrix_out} (+ .pdf)")
    if not args.no_wide:
        _plot_wide_two_column(data, wide_out)
        print(f"Saved: {wide_out} (+ .pdf)")

    if args.class_a:
        hero_key = args.hybrid.strip()
        if hero_key not in data:
            available = ", ".join(data.keys())
            raise SystemExit(
                f"Hybrid {hero_key!r} not in loaded logs. Available: {available}. "
                "Use --skip-missing or train that family first."
            )
        class_a_hero = args.output_dir / "gardian_class_a_hero.png"
        class_a_orbit = args.output_dir / "gardian_class_a_orbit_grid.png"
        _plot_class_a_hero(data[hero_key], hero_key, class_a_hero)
        _plot_class_a_orbit_grid(data, class_a_orbit)
        print(f"Saved: {class_a_hero.parent / (class_a_hero.stem + '_' + hero_key)} (+ .png/.pdf)")
        print(f"Saved: {class_a_orbit} (+ .pdf)")


if __name__ == "__main__":
    main()
