"""CubeScope-style grouped bar figure for re-ranker comparison (RQ3 table replacement).

Six panels × two rows:
  Row 1 — retrieval quality (nDCG@10, MRR, Recall@10)
  Row 2 — median query latency (p50 ms/q, log scale)
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

SYSTEMS = [
    "rrf",
    "cross_encoder_msmarco_minilm",
    "cross_encoder_bge_v2_m3",
    "cross_encoder_monot5_med",
    "cross_encoder_monobert",
    "gardian",
]

SYSTEM_LABELS = [
    "RRF",
    "CE: MiniLM",
    "CE: BGE-v2-M3",
    "CE: MonoT5-med",
    "CE: MonoBERT",
    "GARDIAN",
]

LEGEND_LABELS = [
    "RRF",
    "MiniLM",
    "BGE-v2-M3",
    "MonoT5-med",
    "MonoBERT",
    "GARDIAN",
]

# High-contrast, colorblind-friendly palette (GARDIAN = dark blue, RRF = neutral gray)
SYSTEM_COLORS = {
    "rrf": "#6E6E6E",
    "cross_encoder_msmarco_minilm": "#E87722",
    "cross_encoder_bge_v2_m3": "#009E73",
    "cross_encoder_monot5_med": "#CC3311",
    "cross_encoder_monobert": "#7B68EE",
    "gardian": "#005BBB",
}

DATASETS = [
    ("pubmedqa_labeled", "PQA-L"),
    ("medmcqa", "MedMCQA"),
    ("pubmedqa_artificial", "PQA-A"),
]

BACKENDS = [
    (
        "BM25+FAISS",
        REPO_ROOT / "results" / "reranker_comparison_hybrid_bm25_faiss.json",
        REPO_ROOT / "results" / "reranker_comparison_hybrid_bm25_faiss_pqa_a_q1000.json",
    ),
    (
        "SPLADE++\n+ MedCPT",
        REPO_ROOT / "results" / "reranker_comparison_hybrid_spladepp_medcpt.json",
        REPO_ROOT / "results" / "reranker_comparison_hybrid_spladepp_medcpt_pqa_a_q1000.json",
    ),
]

SCORE_METRICS = [
    ("ndcg@10", "nDCG@10"),
    ("mrr", "MRR"),
    ("recall@10", "R@10"),
]


def _pct(x: float) -> float:
    return x * 100.0 if x <= 1.0 else x


def load_backend(path_main: Path, path_pqa_a: Path) -> dict[str, Any]:
    main = json.loads(path_main.read_text(encoding="utf-8"))
    pqa_a = json.loads(path_pqa_a.read_text(encoding="utf-8"))
    merged = dict(main["results"])
    merged["pubmedqa_artificial"] = pqa_a["results"]["pubmedqa_artificial"]
    return merged


def score_values(results: dict[str, Any], ds: str, system: str) -> list[float]:
    row = results[ds][system]["metrics"]
    return [_pct(float(row[key])) for key, _ in SCORE_METRICS]


def latency_ms(results: dict[str, Any], ds: str, system: str) -> float:
    lat = results[ds][system]["latency_ms"]
    return float(lat.get("p50_ms", lat.get("mean_ms", 0.0)))


def _bar_style(system: str) -> tuple[str, float]:
    if system == "gardian":
        return "#1a1a1a", 1.1
    return "white", 0.5


def _draw_grouped_bars(
    ax: plt.Axes,
    x_centers: np.ndarray,
    heights_per_system: list[list[float]],
    *,
    bar_width: float,
) -> None:
    n_systems = len(SYSTEMS)
    for si, sys in enumerate(SYSTEMS):
        offsets = (si - (n_systems - 1) / 2) * bar_width
        color = SYSTEM_COLORS[sys]
        edge, lw = _bar_style(sys)
        ax.bar(
            x_centers + offsets,
            heights_per_system[si],
            width=bar_width * 0.94,
            color=color,
            edgecolor=edge,
            linewidth=lw,
            zorder=3,
            label=SYSTEM_LABELS[si] if si == 0 else None,
        )


def plot_reranker_cubescope(out_path: Path) -> None:
    panels: list[tuple[str, str, dict[str, Any]]] = []
    for backend_label, main_path, pqa_path in BACKENDS:
        results = load_backend(main_path, pqa_path)
        for ds_key, ds_short in DATASETS:
            panels.append((backend_label, ds_short, results))

    n_panels = len(panels)
    n_systems = len(SYSTEMS)
    n_score_metrics = len(SCORE_METRICS)

    fig_w = 2.05 * n_panels + 0.35
    fig, axes = plt.subplots(
        2,
        n_panels,
        figsize=(fig_w, 5.0),
        facecolor="white",
        gridspec_kw={"hspace": 0.12, "wspace": 0.28},
    )

    group_width = 0.72
    bar_width = group_width / n_systems
    x_score = np.arange(n_score_metrics)
    x_lat = np.array([0.0])

    for col, (backend_label, ds_short, results) in enumerate(panels):
        ds_key = next(dk for dk, ds in DATASETS if ds == ds_short)
        title = f"#{col + 1} {backend_label.replace(chr(10), ' ')}\n· {ds_short}"

        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        ax_top.set_facecolor("white")
        ax_bot.set_facecolor("white")

        heights = [score_values(results, ds_key, sys) for sys in SYSTEMS]
        _draw_grouped_bars(ax_top, x_score, heights, bar_width=bar_width)

        ax_top.set_xticks(x_score)
        ax_top.set_xticklabels([m[1] for m in SCORE_METRICS], fontsize=7.5)
        ax_top.set_title(title, fontsize=8.5, fontweight="600", pad=6, linespacing=0.95)
        ax_top.yaxis.grid(True, linestyle="-", alpha=0.35, color="#cccccc", zorder=0)
        ax_top.set_axisbelow(True)
        ax_top.tick_params(axis="y", labelsize=7.5)
        ymax = max(v for hs in heights for v in hs)
        ax_top.set_ylim(0, max(ymax * 1.08, 10))

        lat_heights = [[latency_ms(results, ds_key, sys)] for sys in SYSTEMS]
        _draw_grouped_bars(ax_bot, x_lat, lat_heights, bar_width=bar_width)

        ax_bot.set_xticks(x_lat)
        ax_bot.set_xticklabels(["ms/q"], fontsize=7.5)
        ax_bot.set_yscale("log")
        ax_bot.yaxis.grid(True, linestyle="-", alpha=0.35, color="#cccccc", zorder=0)
        ax_bot.set_axisbelow(True)
        ax_bot.tick_params(axis="y", labelsize=7.5)
        lat_vals = [latency_ms(results, ds_key, sys) for sys in SYSTEMS]
        ax_bot.set_ylim(max(min(lat_vals) * 0.5, 1e-4), max(lat_vals) * 2.5)

    axes[0, 0].set_ylabel("Score (%)", fontsize=9.5, fontweight="600")
    axes[1, 0].set_ylabel("Latency (ms/q, log)", fontsize=9.5, fontweight="600")

    legend_handles = [
        Patch(
            facecolor=SYSTEM_COLORS[sys],
            edgecolor=_bar_style(sys)[0],
            linewidth=_bar_style(sys)[1],
        )
        for sys in SYSTEMS
    ]
    fig.legend(
        handles=legend_handles,
        labels=LEGEND_LABELS,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=6,
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

    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor="white", pad_inches=0.10)
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white", pad_inches=0.10)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "figures" / "reranker_comparison_cubescope.png",
    )
    args = parser.parse_args()
    plot_reranker_cubescope(args.out)
    print(f"Wrote {args.out} and {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
