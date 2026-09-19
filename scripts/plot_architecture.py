"""
Figure 1: the GARDIAN architecture, drawn from the code rather than by hand.

Every dimension, width and constant in the rendered figure is read from
``configs/base.yaml`` and ``src/features/schema.py``, so the figure cannot drift
away from the implementation the way a hand-drawn diagram does.

Style follows scientific-manuscript convention rather than presentation
software: vector output, a serif face matching LNCS body text, thin black rules,
a restrained grey palette with a single accent for the learned components, and
no icons or clip art.

    python scripts/plot_architecture.py
    python scripts/plot_architecture.py --out results/figures/fig1_architecture
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, ".")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.features.schema import DENSE_FEAT_DIM, SPARSE_FEAT_DIM  # noqa: E402

# Restrained palette: grey for fixed/frozen components, one accent for the
# trained ones, so a reader can see at a glance what is learned.
INK = "#1a1a1a"
FROZEN = "#f2f2f2"     # frozen / off-the-shelf components
LEARNED = "#dce6f1"    # trained components
POOL = "#ffffff"
RULE = 0.8


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        "font.size": 7.2,
        "text.color": INK,
        "axes.edgecolor": INK,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # embed TrueType so the PDF is editable/searchable
        "ps.fonttype": 42,
    })


def box(ax, x, y, w, h, text, *, fc=POOL, fs=7.2, weight="normal", ls="solid"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=RULE, edgecolor=INK, facecolor=fc, linestyle=ls, zorder=2,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=INK, zorder=3, linespacing=1.35)


def arrow(ax, p, q, *, style="-|>", rad=0.0, lw=RULE):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle=style, mutation_scale=7, linewidth=lw,
        color=INK, connectionstyle=f"arc3,rad={rad}",
        shrinkA=1.5, shrinkB=1.5, zorder=4,
    ))


def elbow(ax, points, *, lw=RULE):
    """Orthogonal (right-angled) connector, so long links never cross a box."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs[:-1] + [xs[-1]], ys[:-1] + [ys[-1]],
            color=INK, linewidth=lw, solid_joinstyle="miter", zorder=4)
    arrow(ax, points[-2], points[-1], lw=lw)


def stage(ax, x, y, label):
    ax.text(x, y, label, ha="left", va="center", fontsize=7.6,
            fontweight="bold", color=INK)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", default="configs/base.yaml")
    ap.add_argument("--out", default="results/figures/fig1_architecture")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.cfg)
    k_channel = int(cfg.retrieval.top_k_bm25)
    k_pool = int(cfg.retrieval.candidate_pool_size)
    hidden = int(cfg.model.branch_hidden)
    mid = max(hidden // 2, 4)
    qdim = int(cfg.model.query_feat_dim)
    reader_k = int(cfg.qa.top_k_passages)

    _style()
    # Wide and short: the pipeline is a left-to-right flow with one vertical
    # split inside stage 3. Generous whitespace keeps it legible at \textwidth.
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ══ Stage 1-2: query and first-stage retrieval ═══════════════════════════
    stage(ax, 0.00, 0.965, "(1)  Query")
    box(ax, 0.00, 0.855, 0.175, 0.085, "query $q$", fs=7.4)

    stage(ax, 0.225, 0.965, "(2)  First-stage retrieval   (frozen)")
    box(ax, 0.225, 0.890, 0.215, 0.055,
        f"BM25 / SPLADE++   top-{k_channel}", fc=FROZEN, fs=6.6)
    box(ax, 0.225, 0.820, 0.215, 0.055,
        f"FAISS / MedCPT   top-{k_channel}", fc=FROZEN, fs=6.6)
    box(ax, 0.480, 0.820, 0.150, 0.125,
        f"union +\ndeduplicate\n\npool  $K'={k_pool}$", fs=6.8)

    arrow(ax, (0.175, 0.905), (0.225, 0.917))
    arrow(ax, (0.175, 0.888), (0.225, 0.848))
    arrow(ax, (0.440, 0.917), (0.480, 0.900))
    arrow(ax, (0.440, 0.848), (0.480, 0.865))

    # ══ Stage 3: GARDIAN ═════════════════════════════════════════════════════
    stage(ax, 0.00, 0.745, "(3)  GARDIAN re-ranker   (trained)")
    ax.add_patch(FancyBboxPatch(
        (0.00, 0.215), 0.660, 0.505,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        linewidth=RULE, edgecolor="#8c8c8c", facecolor="none",
        linestyle=(0, (3, 2)), zorder=1))

    # --- upper lane: per-candidate scoring -----------------------------------
    ax.text(0.015, 0.688, "per candidate $p_i$", fontsize=6.6,
            style="italic", color="#555555")
    box(ax, 0.015, 0.575, 0.165, 0.095,
        f"sparse features\n$\\in\\mathbb{{R}}^{{{SPARSE_FEAT_DIM}}}$", fs=7.0)
    box(ax, 0.015, 0.455, 0.165, 0.095,
        f"dense features\n$\\in\\mathbb{{R}}^{{{DENSE_FEAT_DIM}}}$", fs=7.0)
    box(ax, 0.215, 0.575, 0.185, 0.095,
        f"sparse branch MLP\n${SPARSE_FEAT_DIM}\\!\\to\\!{hidden}\\!\\to\\!{mid}\\!\\to\\!1$",
        fc=LEARNED, fs=6.8)
    box(ax, 0.215, 0.455, 0.185, 0.095,
        f"dense branch MLP\n${DENSE_FEAT_DIM}\\!\\to\\!{hidden}\\!\\to\\!{mid}\\!\\to\\!1$",
        fc=LEARNED, fs=6.8)
    arrow(ax, (0.180, 0.622), (0.215, 0.622))
    arrow(ax, (0.180, 0.502), (0.215, 0.502))
    ax.text(0.408, 0.622, "$s^{\\mathrm{sparse}}_i$", fontsize=7.2, va="center")
    ax.text(0.408, 0.502, "$s^{\\mathrm{dense}}_i$", fontsize=7.2, va="center")

    # --- lower lane: query-only controller ------------------------------------
    ax.add_patch(FancyBboxPatch(
        (0.010, 0.240), 0.475, 0.170,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        linewidth=0, facecolor="#f7f9fc", zorder=0))
    ax.text(0.015, 0.388, "per query   (no candidate is involved)",
            fontsize=6.6, style="italic", color="#555555")
    box(ax, 0.015, 0.265, 0.165, 0.095,
        f"PubMedBERT\n(mean-pooled)\n$h_q\\in\\mathbb{{R}}^{{{qdim}}}$", fc=FROZEN, fs=6.0)
    box(ax, 0.215, 0.265, 0.185, 0.095,
        f"controller MLP\n${qdim}\\!\\to\\!{hidden}\\!\\to\\!{mid}\\!\\to\\!2$\n+ softmax",
        fc=LEARNED, fs=6.8)
    arrow(ax, (0.180, 0.312), (0.215, 0.312))
    ax.text(0.408, 0.325, "$(\\alpha_s,\\alpha_d)$", fontsize=7.2, va="center")
    ax.text(0.408, 0.288, "$\\alpha_s\\!+\\!\\alpha_d\\!=\\!1$", fontsize=6.2,
            va="center", color="#555555")

    # --- fusion --------------------------------------------------------------
    box(ax, 0.500, 0.505, 0.155, 0.125,
        "$r_i=\\alpha_s\\, s^{\\mathrm{sparse}}_i$\n"
        "$\\quad\\;+\\,\\alpha_d\\, s^{\\mathrm{dense}}_i$", fs=7.2)
    box(ax, 0.500, 0.270, 0.155, 0.100,
        f"sort by $r_i$\n$\\rightarrow$ top-{reader_k}", fs=7.0)
    elbow(ax, [(0.455, 0.622), (0.478, 0.622), (0.478, 0.600), (0.500, 0.600)])
    elbow(ax, [(0.455, 0.502), (0.478, 0.502), (0.478, 0.530), (0.500, 0.530)])
    elbow(ax, [(0.455, 0.312), (0.478, 0.312), (0.478, 0.565), (0.500, 0.565)])
    arrow(ax, (0.5775, 0.503), (0.5775, 0.372))
    elbow(ax, [(0.5625, 0.818), (0.5625, 0.760), (0.150, 0.760), (0.150, 0.672)])
    ax.text(0.330, 0.770, "$K'$ candidates", fontsize=6.4, color="#555555")

    # ══ Stage 4: reader ══════════════════════════════════════════════════════
    stage(ax, 0.700, 0.745, "(4)  Reader   (frozen)")
    box(ax, 0.700, 0.575, 0.290, 0.095,
        f"top-{reader_k} re-ranked\npassages", fs=7.0)
    box(ax, 0.700, 0.430, 0.290, 0.095,
        "reader LLM\nnot fine-tuned", fc=FROZEN, fs=7.0)
    box(ax, 0.700, 0.265, 0.290, 0.115,
        "answer with inline\ncitations [P1] [P2] …", fs=7.0)
    elbow(ax, [(0.655, 0.320), (0.678, 0.320), (0.678, 0.622), (0.700, 0.622)])
    arrow(ax, (0.845, 0.572), (0.845, 0.527))
    arrow(ax, (0.845, 0.427), (0.845, 0.382))

    # ══ legend / caption strip ═══════════════════════════════════════════════
    ax.add_patch(FancyBboxPatch((0.00, 0.020), 0.990, 0.165,
                                boxstyle="round,pad=0.005,rounding_size=0.008",
                                linewidth=RULE, edgecolor="#8c8c8c",
                                facecolor="#fafafa", zorder=1))
    ax.add_patch(FancyBboxPatch((0.015, 0.138), 0.022, 0.030,
                                boxstyle="square,pad=0",
                                linewidth=RULE, edgecolor=INK, facecolor=LEARNED))
    ax.text(0.046, 0.153, "trained: two branch MLPs + controller, pairwise ranking loss",
            fontsize=6.6, va="center")
    ax.add_patch(FancyBboxPatch((0.015, 0.093), 0.022, 0.030,
                                boxstyle="square,pad=0",
                                linewidth=RULE, edgecolor=INK, facecolor=FROZEN))
    ax.text(0.046, 0.108, "frozen: both retrievers, the query encoder, the reader",
            fontsize=6.6, va="center")
    ax.text(0.015, 0.055,
            "The controller sees the query embedding only. Fusion weights therefore vary "
            "across queries and are constant\nacross a query's candidates \u2014 this is what "
            "makes the re-ranker query-adaptive rather than a tuned global weighting.",
            fontsize=6.6, va="center", style="italic", linespacing=1.4)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=400)
    plt.close(fig)
    print(f"wrote {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")
    print(f"  read from {args.cfg}: sparse=R^{SPARSE_FEAT_DIM} dense=R^{DENSE_FEAT_DIM} "
          f"h_q=R^{qdim} hidden={hidden} K'={k_pool} top-k={reader_k}")


if __name__ == "__main__":
    main()
