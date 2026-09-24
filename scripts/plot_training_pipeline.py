"""
Figure: GARDIAN's offline training pipeline.

All dimensions, layer shapes, the margin and the negative-sampling ratio are read
from ``configs/base.yaml`` and ``src/features/schema.py`` at render time, so the
figure cannot drift from the implementation.

    python scripts/plot_training_pipeline.py
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from omegaconf import OmegaConf  # noqa: E402

from figstyle import (  # noqa: E402
    ACCENT, INK, RULE, apply_style, arrow, canvas, elbow, flame,
    group, legend_row, pill, rect, snowflake,
)
from src.features.schema import DENSE_FEAT_DIM, SPARSE_FEAT_DIM  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", default="configs/base.yaml")
    ap.add_argument("--out", default="results/figures/fig_training")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.cfg)
    t = cfg.training
    H = int(cfg.model.branch_hidden)
    M = max(H // 2, 4)  # second hidden layer, as in src/model/gardian.py
    QD = int(cfg.model.query_feat_dim)
    KP = int(cfg.retrieval.candidate_pool_size)
    loss_name = str(getattr(t, "loss", "lambdarank"))
    group_size = int(getattr(t, "listwise_group_size", 64))
    ndcg_k = int(getattr(t, "listwise_ndcg_k", 10))
    n_neg, hard_n = int(t.num_negatives), int(t.hard_negative_top_n)
    hard_pct = int(round(float(t.hard_negative_fraction) * 100))

    apply_style(6.0)
    fig, ax = canvas(9.6, 2.05)

    Y = 0.30      # baseline of the main band
    HB = 0.36     # height of a standard component box

    # ── input ────────────────────────────────────────────────────────────────
    pill(ax, 0.000, Y + 0.09, 0.085, 0.19, "queries $\\{q_j\\}$", fs=6.0)

    # ── (a) frozen retrieval backbone ────────────────────────────────────────
    gx, gw = 0.105, 0.205
    group(ax, gx, Y - 0.055, gw, HB + 0.19, "(a)", "Frozen retrieval backbone")
    rect(ax, gx + 0.012, Y + 0.135, 0.085, 0.115, "BM25 /\nSPLADE++", fs=5.6)
    snowflake(ax, gx + 0.090, Y + 0.232)
    rect(ax, gx + 0.012, Y - 0.010, 0.085, 0.115, "FAISS /\nMedCPT", fs=5.6)
    snowflake(ax, gx + 0.090, Y + 0.087)
    pill(ax, gx + 0.112, Y + 0.055, 0.082, 0.150,
         f"pool $P(q)$\n$K'\\!=\\!{KP}$", fs=5.6)
    arrow(ax, (0.085, Y + 0.185), (gx + 0.012, Y + 0.192), rad=-0.12)
    arrow(ax, (0.085, Y + 0.185), (gx + 0.012, Y + 0.048), rad=0.12)
    arrow(ax, (gx + 0.097, Y + 0.192), (gx + 0.112, Y + 0.150))
    arrow(ax, (gx + 0.097, Y + 0.048), (gx + 0.112, Y + 0.110))

    # ── negative sampling → training pools ───────────────────────────────────
    # A listwise objective consumes one whole pool per query, not isolated
    # pairs: every positive is kept and the pool is filled with sampled
    # negatives, so the candidate distribution matches the pairwise arm exactly.
    rect(ax, 0.322, Y + 0.040, 0.100, 0.180,
         f"all positives +\nnegatives from $P(q)$\n"
         f"fill to {group_size}", fs=5.2)
    pill(ax, 0.428, Y + 0.065, 0.070, 0.130,
         f"pool of {group_size}\n$(p^{{+}}\\!,p^{{-}}_{{1..{group_size - 1}}})$", fs=5.4)
    arrow(ax, (gx + 0.194, Y + 0.130), (0.322, Y + 0.130))
    arrow(ax, (0.422, Y + 0.130), (0.428, Y + 0.130))

    # ── (b) trainable GARDIAN modules ────────────────────────────────────────
    bx, bw = 0.513, 0.278
    group(ax, bx, Y - 0.055, bw, HB + 0.19, "(b)", "Trainable GARDIAN modules")
    rect(ax, bx + 0.012, Y + 0.135, 0.072, 0.115,
         f"sparse\n$\\mathbb{{R}}^{{{SPARSE_FEAT_DIM}}}$", fs=5.6)
    rect(ax, bx + 0.012, Y - 0.010, 0.072, 0.115,
         f"dense\n$\\mathbb{{R}}^{{{DENSE_FEAT_DIM}}}$", fs=5.6)
    rect(ax, bx + 0.098, Y + 0.135, 0.082, 0.115,
         f"$f_{{sp}}$\n${SPARSE_FEAT_DIM}\\!\\to\\!{H}\\!\\to\\!{M}\\!\\to\\!1$", fs=4.6)
    flame(ax, bx + 0.172, Y + 0.232)
    rect(ax, bx + 0.098, Y - 0.010, 0.082, 0.115,
         f"$f_{{de}}$\n${DENSE_FEAT_DIM}\\!\\to\\!{H}\\!\\to\\!{M}\\!\\to\\!1$", fs=4.6)
    flame(ax, bx + 0.172, Y + 0.087)
    arrow(ax, (bx + 0.084, Y + 0.192), (bx + 0.098, Y + 0.192))
    arrow(ax, (bx + 0.084, Y + 0.048), (bx + 0.098, Y + 0.048))
    arrow(ax, (0.498, Y + 0.130), (bx + 0.012, Y + 0.192))
    arrow(ax, (0.498, Y + 0.130), (bx + 0.012, Y + 0.048))

    # controller: query only -- drawn on its own row, in the accent colour
    rect(ax, bx + 0.098, Y - 0.155, 0.082, 0.115,
         f"controller\n${QD}\\!\\to\\!{H}\\!\\to\\!{M}\\!\\to\\!2$", fs=4.6)
    flame(ax, bx + 0.172, Y - 0.058)
    pill(ax, bx + 0.012, Y - 0.155, 0.072, 0.115,
         f"$h_q\\!\\in\\!\\mathbb{{R}}^{{{QD}}}$\nmean-pooled", fs=5.2)
    snowflake(ax, bx + 0.076, Y - 0.058)
    arrow(ax, (bx + 0.084, Y - 0.098), (bx + 0.098, Y - 0.098), color=ACCENT)
    elbow(ax, [(0.043, Y + 0.088), (0.043, Y - 0.098), (bx + 0.012, Y - 0.098)],
          color=ACCENT)
    ax.text(0.245, Y - 0.128, "query embedding only (no candidate)",
            fontsize=5.2, color=ACCENT, style="italic")

    rect(ax, bx + 0.196, Y - 0.010, 0.070, 0.260,
         "$r_i=$\n$\\alpha_s\\hat{s}_{sp}$\n$+\\,\\alpha_d\\hat{s}_{de}$", fs=5.8)
    arrow(ax, (bx + 0.180, Y + 0.192), (bx + 0.196, Y + 0.170))
    arrow(ax, (bx + 0.180, Y + 0.048), (bx + 0.196, Y + 0.085))
    elbow(ax, [(bx + 0.180, Y - 0.098), (bx + 0.231, Y - 0.098),
               (bx + 0.231, Y - 0.010)], color=ACCENT)
    ax.text(bx + 0.238, Y - 0.098, "$(\\alpha_s,\\alpha_d)$", fontsize=5.4,
            color=ACCENT, va="center", ha="left")

    # ── (c) training objective ───────────────────────────────────────────────
    # Listwise: the whole pool is scored at once and each pair is weighted by
    # the nDCG@k change of swapping it, so the figure shows a ranked list with
    # the cutoff rather than an isolated (positive, negative) pair.
    cx, cw = 0.806, 0.194
    group(ax, cx, Y - 0.055, cw, HB + 0.19, "(c)", "Listwise objective")

    mp = __import__("matplotlib")
    rows = 6
    rh, rw = 0.030, 0.086
    top = Y + 0.246
    gold = 2                                        # position of the relevant one
    for j in range(rows):
        yj = top - j * (rh + 0.006)
        ax.add_patch(mp.patches.Rectangle(
            (cx + 0.026, yj - rh), rw, rh, linewidth=RULE, edgecolor=INK,
            facecolor="#dce6f1" if j == gold else "#ededed", zorder=4))
    ax.text(cx + 0.020, top - gold * (rh + 0.006) - rh / 2, "$p^{+}$",
            fontsize=5.4, ha="right", va="center")
    ax.text(cx + 0.020, top - rh / 2, "$r_1$", fontsize=5.2, ha="right", va="center")

    # the nDCG cutoff, which is what makes the weighting listwise
    ycut = top - 3 * (rh + 0.006) - 0.003
    ax.plot([cx + 0.020, cx + 0.026 + rw + 0.008], [ycut, ycut],
            color=ACCENT, linewidth=0.7, linestyle=(0, (2.2, 1.4)), zorder=5)
    ax.text(cx + 0.026 + rw + 0.012, ycut, f"$k\\!=\\!{ndcg_k}$", fontsize=5.2,
            color=ACCENT, va="center", ha="left")

    ax.annotate("", xy=(cx + 0.026 + rw + 0.004, top - gold * (rh + 0.006) - rh / 2),
                xytext=(cx + 0.026 + rw + 0.004, top - rh / 2),
                arrowprops=dict(arrowstyle="<->", linewidth=0.6, color=INK))
    ax.text(cx + 0.026 + rw + 0.010, top - 0.030, "$|\\Delta\\mathrm{nDCG}|$",
            fontsize=5.0, va="center")

    ax.text(cx + 0.008, Y - 0.022,
            "$\\mathcal{L}=\\sum_{i \\succ j}"
            "|\\Delta\\mathrm{nDCG}_{ij}|\\,"
            "\\mathrm{softplus}(-\\sigma(r_i\\!-\\!r_j))$",
            fontsize=4.7, va="center")
    arrow(ax, (bx + 0.266, Y + 0.120), (cx, Y + 0.120))

    # ── legend ───────────────────────────────────────────────────────────────
    legend_row(ax, 0.045, [("frozen", "frozen"), ("trained", "trained"),
                           ("module", "module"), ("data", "data object")], fs=5.4)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=420)
    print(f"wrote {out.with_suffix('.pdf')} / .png")


if __name__ == "__main__":
    main()
