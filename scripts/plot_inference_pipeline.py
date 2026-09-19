"""
Figure: GARDIAN at inference time.

Companion to ``scripts/plot_training_pipeline.py``; both import
``scripts/figstyle.py`` so the two diagrams share one visual language. Every
dimension and constant is read from ``configs/base.yaml`` at render time.

The figure makes the cost claim precise: a single controller forward pass per
query, two small MLP passes per candidate, and no query-passage re-encoding --
which is what distinguishes the re-ranker from a cross-encoder.

    python scripts/plot_inference_pipeline.py
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
    ACCENT, apply_style, arrow, canvas, elbow, flame, group, legend_row,
    pill, rect, snowflake,
)
from src.features.schema import DENSE_FEAT_DIM, SPARSE_FEAT_DIM  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", default="configs/base.yaml")
    ap.add_argument("--out", default="results/figures/fig_inference")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.cfg)
    H = int(cfg.model.branch_hidden)
    QD = int(cfg.model.query_feat_dim)
    KP = int(cfg.retrieval.candidate_pool_size)
    KC = int(cfg.retrieval.top_k_bm25)
    RK = int(cfg.qa.top_k_passages)

    apply_style(6.0)
    fig, ax = canvas(9.6, 2.05)

    Y = 0.30

    # ── input ────────────────────────────────────────────────────────────────
    pill(ax, 0.000, Y + 0.09, 0.076, 0.19, "query $q$", fs=6.0)

    # ── (a) frozen first-stage retrieval ─────────────────────────────────────
    gx, gw = 0.096, 0.196
    group(ax, gx, Y - 0.055, gw, 0.55, "(a)", "First-stage retrieval")
    rect(ax, gx + 0.012, Y + 0.135, 0.082, 0.115, "BM25 /\nSPLADE++", fs=5.6)
    snowflake(ax, gx + 0.087, Y + 0.232)
    rect(ax, gx + 0.012, Y - 0.010, 0.082, 0.115, "FAISS /\nMedCPT", fs=5.6)
    snowflake(ax, gx + 0.087, Y + 0.087)
    pill(ax, gx + 0.106, Y + 0.055, 0.078, 0.150,
         f"pool $P(q)$\n$K'\\!=\\!{KP}$", fs=5.6)
    ax.text(gx + 0.145, Y + 0.020, f"top-{KC} each", fontsize=5.0,
            color="#666666", ha="center")
    arrow(ax, (0.076, Y + 0.185), (gx + 0.012, Y + 0.192), rad=-0.12)
    arrow(ax, (0.076, Y + 0.185), (gx + 0.012, Y + 0.048), rad=0.12)
    arrow(ax, (gx + 0.094, Y + 0.192), (gx + 0.106, Y + 0.150))
    arrow(ax, (gx + 0.094, Y + 0.048), (gx + 0.106, Y + 0.110))

    # ── (b) GARDIAN re-ranking ───────────────────────────────────────────────
    bx, bw = 0.312, 0.330
    group(ax, bx, Y - 0.055, bw, 0.55, "(b)", "GARDIAN re-ranking")
    rect(ax, bx + 0.012, Y + 0.135, 0.070, 0.115,
         f"sparse\n$\\mathbb{{R}}^{{{SPARSE_FEAT_DIM}}}$", fs=5.6)
    rect(ax, bx + 0.012, Y - 0.010, 0.070, 0.115,
         f"dense\n$\\mathbb{{R}}^{{{DENSE_FEAT_DIM}}}$", fs=5.6)
    rect(ax, bx + 0.096, Y + 0.135, 0.078, 0.115,
         f"$f_{{sp}}$\n{SPARSE_FEAT_DIM}$\\to${H}$\\to$1", fs=5.4)
    flame(ax, bx + 0.166, Y + 0.232)
    rect(ax, bx + 0.096, Y - 0.010, 0.078, 0.115,
         f"$f_{{de}}$\n{DENSE_FEAT_DIM}$\\to${H}$\\to$1", fs=5.4)
    flame(ax, bx + 0.166, Y + 0.087)
    arrow(ax, (bx + 0.082, Y + 0.192), (bx + 0.096, Y + 0.192))
    arrow(ax, (bx + 0.082, Y + 0.048), (bx + 0.096, Y + 0.048))
    arrow(ax, (gx + 0.184, Y + 0.130), (bx + 0.012, Y + 0.192), rad=-0.10)
    arrow(ax, (gx + 0.184, Y + 0.130), (bx + 0.012, Y + 0.048), rad=0.10)
    ax.text(bx + 0.045, Y + 0.278, "$\\times K'$ candidates", fontsize=5.0,
            color="#666666", ha="center")

    # controller: one pass per query
    pill(ax, bx + 0.012, Y - 0.155, 0.070, 0.115,
         f"$h_q\\!\\in\\!\\mathbb{{R}}^{{{QD}}}$\nmean-pooled", fs=5.2)
    snowflake(ax, bx + 0.074, Y - 0.058)
    rect(ax, bx + 0.096, Y - 0.155, 0.078, 0.115,
         f"controller\n{QD}$\\to${H}$\\to$2", fs=5.4)
    flame(ax, bx + 0.166, Y - 0.058)
    arrow(ax, (bx + 0.082, Y - 0.098), (bx + 0.096, Y - 0.098), color=ACCENT)
    elbow(ax, [(0.038, Y + 0.088), (0.038, Y - 0.098), (bx + 0.012, Y - 0.098)],
          color=ACCENT)
    ax.text(0.175, Y - 0.128, "one controller pass per query",
            fontsize=5.0, color=ACCENT, style="italic")

    rect(ax, bx + 0.190, Y - 0.010, 0.062, 0.260,
         "$r_i=$\n$\\alpha_s\\hat{s}_{sp}$\n$+\\,\\alpha_d\\hat{s}_{de}$", fs=5.6)
    arrow(ax, (bx + 0.174, Y + 0.192), (bx + 0.190, Y + 0.170))
    arrow(ax, (bx + 0.174, Y + 0.048), (bx + 0.190, Y + 0.085))
    elbow(ax, [(bx + 0.174, Y - 0.098), (bx + 0.221, Y - 0.098),
               (bx + 0.221, Y - 0.010)], color=ACCENT)
    pill(ax, bx + 0.264, Y + 0.055, 0.056, 0.150,
         f"sort\n$\\to$ top-{RK}", fs=5.6)
    arrow(ax, (bx + 0.252, Y + 0.120), (bx + 0.264, Y + 0.130))

    # ── (c) reader ───────────────────────────────────────────────────────────
    cx, cw = 0.662, 0.338
    group(ax, cx, Y - 0.055, cw, 0.55, "(c)", "Reader")
    rect(ax, cx + 0.014, Y + 0.055, 0.100, 0.150,
         f"top-{RK} passages\nin rank order", fs=5.6)
    rect(ax, cx + 0.130, Y + 0.055, 0.098, 0.150,
         "reader LLM\nnot fine-tuned", fs=5.6)
    snowflake(ax, cx + 0.220, Y + 0.187)
    pill(ax, cx + 0.244, Y + 0.055, 0.082, 0.150,
         "answer +\ncitations [P#]", fs=5.6)
    arrow(ax, (bx + 0.320, Y + 0.130), (cx + 0.014, Y + 0.130))
    arrow(ax, (cx + 0.114, Y + 0.130), (cx + 0.130, Y + 0.130))
    arrow(ax, (cx + 0.228, Y + 0.130), (cx + 0.244, Y + 0.130))
    ax.text(cx + 0.014, Y - 0.098,
            "no query-passage re-encoding: the branch heads score\n"
            "cached first-stage features (two small MLP passes / candidate)",
            fontsize=5.0, color="#555555", style="italic", va="center",
            linespacing=1.3)

    legend_row(ax, 0.045, [("frozen", "frozen"), ("trained", "trained"),
                           ("module", "module"), ("data", "data object")], fs=5.4)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=420)
    print(f"wrote {out.with_suffix('.pdf')} / .png")


if __name__ == "__main__":
    main()
