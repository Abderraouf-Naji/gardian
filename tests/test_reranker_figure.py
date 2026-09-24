"""
The RQ4 figure must consume exactly the schema script 14 writes.

These two files drifted apart once already: scripts/14_compare_rerankers.py
emitted a flat ``{system: metrics}`` mapping with no timings at all, while
scripts/plot_reranker_comparison.py read ``[system]["metrics"]`` and
``[system]["latency_ms"]["p50_ms"]``. The mismatch was invisible because the
comparison JSONs were never committed, so the figure could not be rebuilt to
expose it. This test rebuilds the figure from a synthetic payload written in
script 14's schema, so any future divergence fails here instead of silently
producing an unplottable result file hours into a GPU run.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load_plot_module():
    spec = importlib.util.spec_from_file_location(
        "plot_reranker", REPO / "scripts" / "plot_reranker_comparison.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _payload(systems, datasets) -> dict:
    """A comparison payload in exactly the shape script 14 writes."""
    return {
        "meta": {"retriever": "hybrid_bm25_faiss", "latency_measured": True},
        "results": {
            ds: {
                sysname: {
                    "metrics": {
                        "ndcg@10": 0.70 + 0.01 * i,
                        "mrr": 0.80 + 0.01 * i,
                        "recall@10": 0.60 + 0.01 * i,
                    },
                    "latency_ms": {"p50_ms": 1.0 + 10.0 * i, "mean_ms": 1.0 + 10.0 * i},
                }
                for i, sysname in enumerate(systems)
            }
            for ds in datasets
        },
    }


@pytest.fixture
def plot_mod(tmp_path, monkeypatch):
    mod = _load_plot_module()
    systems = mod.SYSTEMS
    main_datasets = ["pubmedqa_labeled", "medmcqa"]

    backends = []
    for name in ("hybrid_bm25_faiss", "hybrid_spladepp_medcpt"):
        main_path = tmp_path / f"reranker_comparison_{name}.json"
        pqa_path = tmp_path / f"reranker_comparison_{name}_pqa_a_q1000.json"
        main_path.write_text(json.dumps(_payload(systems, main_datasets)))
        pqa_path.write_text(json.dumps(_payload(systems, ["pubmedqa_artificial"])))
        backends.append((name, main_path, pqa_path))

    monkeypatch.setattr(mod, "BACKENDS", backends)
    return mod


def test_figure_renders_from_script14_schema(plot_mod, tmp_path):
    out = tmp_path / "fig.png"
    plot_mod.plot_reranker_cubescope(out)
    assert out.is_file() and out.stat().st_size > 0
    assert out.with_suffix(".pdf").is_file()


def test_plot_reads_nested_metrics_block(plot_mod):
    results = plot_mod.load_backend(*plot_mod.BACKENDS[0][1:])
    # All three splits must survive the merge of the main and PQA-A files.
    assert set(results) == {"pubmedqa_labeled", "medmcqa", "pubmedqa_artificial"}
    values = plot_mod.score_values(results, "medmcqa", "gardian")
    assert len(values) == len(plot_mod.SCORE_METRICS)
    assert all(v > 0 for v in values)


def test_plot_reads_latency_block(plot_mod):
    results = plot_mod.load_backend(*plot_mod.BACKENDS[0][1:])
    lat = plot_mod.latency_ms(results, "pubmedqa_labeled", "gardian")
    assert lat > 0


def test_every_plotted_system_is_produced_by_script14():
    """A system in the figure that script 14 never emits leaves a hole."""
    spec = importlib.util.spec_from_file_location(
        "cmp14", REPO / "scripts" / "14_compare_rerankers.py"
    )
    cmp14 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmp14)

    produced = set(cmp14._systems_reported(cmp14.DEFAULT_CE_VARIANTS))
    plotted = set(_load_plot_module().SYSTEMS)
    assert plotted <= produced, f"figure expects systems 14 never writes: {plotted - produced}"
