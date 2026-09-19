"""
Shared drawing primitives for the paper's architecture figures.

Conventions follow current ML/IR practice (cf. LNCS system diagrams):

  * sharp rectangle  -- a processing module
  * rounded pill     -- a data or embedding object
  * snowflake        -- frozen / pretrained component
  * flame            -- trained component
  * dashed group box -- a stage, labelled (a) / (b) / (c) inside its top-left

Both icons are drawn as vector paths rather than emoji, so the resulting PDF is
font-independent and reproduces correctly in greyscale print.

Figures import these helpers so the two diagrams cannot drift apart stylistically.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

INK = "#111111"
COLD = "#5b9bd5"     # snowflake / frozen
HOT = "#ed7d31"      # flame / trained
ACCENT = "#c00000"   # emphasised data path
SOFT = "#7f7f7f"
RULE = 0.7


def apply_style(base: float = 6.0) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": base,
        "mathtext.fontset": "dejavuserif",
        "text.color": INK,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.015,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def canvas(w: float, h: float):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def rect(ax, x, y, w, h, text, *, fs=5.8, fc="white", ec=INK, lw=RULE, weight="normal"):
    """Sharp rectangle: a processing module."""
    ax.add_patch(Rectangle((x, y), w, h, linewidth=lw, edgecolor=ec,
                           facecolor=fc, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, zorder=5, linespacing=1.30)


def pill(ax, x, y, w, h, text, *, fs=5.8, fc="white", rounding=0.011):
    """
    Rounded pill: a data or embedding object.

    ``rounding`` is in data units. These figures use a wide canvas (aspect ~5:1),
    so a rounding radius derived from the box height would render as a distorted
    oval; a small fixed radius keeps the corners visually round in both axes.
    """
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.0015,rounding_size={rounding:.4f}",
                                linewidth=RULE, edgecolor=INK, facecolor=fc, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=5, linespacing=1.30)


def group(ax, x, y, w, h, tag, title, *, fs_title=6.0, fs_tag=6.0):
    """Stage container: label (a)/(b)/(c) inside the top-left, title centred."""
    ax.add_patch(Rectangle((x, y), w, h, linewidth=RULE, edgecolor=INK,
                           facecolor="none", zorder=2))
    ax.text(x + 0.005, y + h - 0.055, tag, ha="left", va="center",
            fontsize=fs_tag, fontweight="bold", zorder=6)
    ax.text(x + w / 2, y + h - 0.055, title, ha="center", va="center",
            fontsize=fs_title, zorder=6)


def arrow(ax, p, q, *, color=INK, lw=RULE, rad=0.0, ms=5.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms,
                                 linewidth=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0.8, shrinkB=0.8, zorder=4))


def elbow(ax, pts, *, color=INK, lw=RULE, ms=5.0):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color,
            linewidth=lw, solid_joinstyle="miter", zorder=4)
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ms=ms)


def snowflake(ax, x, y, r=0.0075, aspect=2.4):
    """Frozen marker: six spokes with tick marks."""
    import numpy as np

    for k in range(3):
        a = np.pi * k / 3.0
        dx, dy = r * np.cos(a), r * np.sin(a) * aspect
        ax.plot([x - dx, x + dx], [y - dy, y + dy], color=COLD,
                linewidth=0.75, solid_capstyle="round", zorder=7)
        for s in (-1, 1):
            tx, ty = x + s * dx * 0.60, y + s * dy * 0.60
            ax.plot([tx, tx + r * 0.40 * np.cos(a + 1.05)],
                    [ty, ty + r * 0.40 * aspect * np.sin(a + 1.05)],
                    color=COLD, linewidth=0.6, zorder=7)


def flame(ax, x, y, r=0.0105, aspect=2.4):
    """Trained marker: a solid teardrop, legible at small print sizes."""
    pts = [
        (x, y + 1.25 * r * aspect / 2.4 * 1.0),
        (x + 0.44 * r, y + 0.30 * r * aspect / 2.4),
        (x + 0.60 * r, y - 0.40 * r * aspect / 2.4),
        (x + 0.30 * r, y - 1.05 * r * aspect / 2.4),
        (x, y - 1.20 * r * aspect / 2.4),
        (x - 0.30 * r, y - 1.05 * r * aspect / 2.4),
        (x - 0.60 * r, y - 0.40 * r * aspect / 2.4),
        (x - 0.40 * r, y + 0.28 * r * aspect / 2.4),
    ]
    pts = [(px, y + (py - y) * aspect / 1.0) for px, py in pts]
    ax.add_patch(Polygon(pts, closed=True, facecolor=HOT, edgecolor=HOT,
                         linewidth=0.4, zorder=7))


def legend_row(ax, y, items, *, fs=5.6, x0=0.0, gap=0.145):
    """Compact legend strip: [(kind, label), ...] with kind in {frozen, trained}."""
    x = x0
    for kind, label in items:
        if kind == "frozen":
            snowflake(ax, x + 0.006, y)
        elif kind == "trained":
            flame(ax, x + 0.006, y)
        elif kind == "module":
            ax.add_patch(Rectangle((x, y - 0.018), 0.020, 0.036, linewidth=RULE,
                                   edgecolor=INK, facecolor="white"))
        elif kind == "data":
            ax.add_patch(FancyBboxPatch((x, y - 0.018), 0.020, 0.036,
                                        boxstyle="round,pad=0.001,rounding_size=0.008",
                                        linewidth=RULE, edgecolor=INK, facecolor="white"))
        ax.text(x + 0.028, y, label, fontsize=fs, va="center")
        x += gap + 0.012 * len(label) * 0.5
