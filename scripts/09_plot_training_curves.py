"""Plot GARDIAN training curves for all hybrid retriever types."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


HYBRID_TYPES = [
    "hybrid",
    "hybrid_neural",
    "hybrid_bm25_biobert",
    "hybrid_doc2query_faiss",
]

HYBRID_LABELS = {
    "hybrid": "BM25+FAISS",
    "hybrid_neural": "Doc2Query+BioBERT",
    "hybrid_bm25_biobert": "BM25+BioBERT",
    "hybrid_doc2query_faiss": "Doc2Query+FAISS",
}


def _apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "axes.titleweight": "bold",
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "lines.linewidth": 2.6,
            "lines.markersize": 6,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
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


def _collect_training_data(results_dir: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for hybrid_type in HYBRID_TYPES:
        log_path = results_dir / "gardian_training" / hybrid_type / "epoch_logs.jsonl"
        if not log_path.exists():
            raise FileNotFoundError(f"Missing log file: {log_path}")
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
    return out


def _plot_dev_ndcg10(training_data: Dict[str, dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7))
    best_points = []
    for hybrid_type, values in training_data.items():
        line_label = HYBRID_LABELS.get(hybrid_type, hybrid_type)
        ax.plot(
            values["eval_epochs"],
            values["eval_ndcg10"],
            marker="o",
            label=line_label,
        )
        if values["eval_ndcg10"]:
            best_idx = max(
                range(len(values["eval_ndcg10"])),
                key=lambda i: values["eval_ndcg10"][i],
            )
            bx = values["eval_epochs"][best_idx]
            by = values["eval_ndcg10"][best_idx]
            ax.scatter([bx], [by], marker="*", s=180, zorder=5)
            best_points.append((bx, by))
    # Stagger value labels for close best points to avoid overlap.
    best_points_sorted = sorted(best_points, key=lambda p: (p[0], p[1]))
    y_offsets = [0, 10, -10, 18, -18, 26, -26]
    last_x = None
    cluster_index = 0
    for bx, by in best_points_sorted:
        if last_x is None or abs(bx - last_x) > 0.2:
            cluster_index = 0
        else:
            cluster_index += 1
        yoff = y_offsets[cluster_index % len(y_offsets)]
        ax.annotate(
            f"{by:.4f}",
            (bx, by),
            textcoords="offset points",
            xytext=(8, yoff),
            ha="left",
            va="center",
        )
        last_x = bx
    ax.set_title("GARDIAN Dev nDCG@10 Across Hybrid Retrieval Modes")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dev nDCG@10 (higher is better)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.22, linestyle="--")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"))
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_train_loss(training_data: Dict[str, dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7))
    for hybrid_type, values in training_data.items():
        line_label = HYBRID_LABELS.get(hybrid_type, hybrid_type)
        ax.plot(
            values["epochs"],
            values["losses"],
            marker="o",
            label=line_label,
        )
    ax.set_title("GARDIAN Train Loss Across Hybrid Retrieval Modes")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss (lower is better)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.22, linestyle="--")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"))
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GARDIAN training curves across hybrid types.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root results directory that contains gardian_training/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/plots"),
        help="Directory to save generated plots.",
    )
    args = parser.parse_args()

    _apply_publication_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_data = _collect_training_data(args.results_dir)

    ndcg_out = args.output_dir / "gardian_dev_ndcg10_hybrid_types.png"
    loss_out = args.output_dir / "gardian_train_loss_hybrid_types.png"
    _plot_dev_ndcg10(training_data, ndcg_out)
    _plot_train_loss(training_data, loss_out)

    print(f"Saved: {ndcg_out} and {ndcg_out.with_suffix('.pdf')}")
    print(f"Saved: {loss_out} and {loss_out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
