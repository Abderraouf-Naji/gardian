"""Tests for the multi-seed protocol: seeding, run layout, aggregation."""

from __future__ import annotations

import json
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from src.common.repro import set_global_seed, set_seed
from src.common.seeds import (
    DEFAULT_SEEDS,
    SeedArtifactExistsError,
    discover_seeds,
    guard_seed_artifact,
    parse_seeds,
    seed_dir,
    seed_path,
    write_seed_json,
)


# ── set_seed ─────────────────────────────────────────────────────────────────


def test_default_seeds_match_protocol():
    assert DEFAULT_SEEDS == (13, 21, 42, 87, 100)


def test_set_seed_makes_all_rngs_reproducible():
    set_seed(13)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    set_seed(13)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert first == second


def test_different_seeds_give_different_draws():
    set_seed(13)
    a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    set_seed(87)
    b = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert a != b


def test_set_seed_forces_cudnn_determinism_by_default():
    set_seed(42)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_set_seed_exports_pythonhashseed(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    set_seed(21)
    import os

    assert os.environ["PYTHONHASHSEED"] == "21"


def test_pythonhashseed_is_assigned_not_setdefault(monkeypatch):
    """A stale value from an outer process must not survive set_seed()."""
    monkeypatch.setenv("PYTHONHASHSEED", "999")
    set_seed(21)
    import os

    assert os.environ["PYTHONHASHSEED"] == "21"


def test_set_global_seed_alias_returns_seed():
    assert set_global_seed(100) == 100


# ── parse_seeds ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        (["13", "21", "42"], [13, 21, 42]),
        (["13,21,42"], [13, 21, 42]),
        (["42", "13"], [13, 42]),          # sorted
        (["42", "42", "13"], [13, 42]),    # de-duplicated
        (["13, 21"], [13, 21]),            # comma + space
    ],
)
def test_parse_seeds(raw, expected):
    assert parse_seeds(raw) == expected


def test_parse_seeds_rejects_non_integer():
    with pytest.raises(ValueError, match="must be integers"):
        parse_seeds(["13", "abc"])


def test_parse_seeds_rejects_empty():
    with pytest.raises(ValueError, match="No seeds"):
        parse_seeds([])


# ── run layout ───────────────────────────────────────────────────────────────


def test_seed_path_layout(tmp_path):
    p = seed_path(tmp_path, 13, "gardian_best_hybrid_bm25_faiss.pt")
    assert p == tmp_path / "seeds" / "seed_13" / "gardian_best_hybrid_bm25_faiss.pt"


def test_seed_dir_layout(tmp_path):
    assert seed_dir(tmp_path, 87) == tmp_path / "seeds" / "seed_87"


def test_guard_refuses_to_overwrite_existing_artifact(tmp_path):
    target = seed_path(tmp_path, 42, "evaluation.json")
    write_seed_json(target, {"ndcg@10": 0.7})
    with pytest.raises(SeedArtifactExistsError, match="Refusing to overwrite"):
        guard_seed_artifact(target)


def test_guard_allows_explicit_overwrite(tmp_path):
    target = seed_path(tmp_path, 42, "evaluation.json")
    write_seed_json(target, {"ndcg@10": 0.7})
    write_seed_json(target, {"ndcg@10": 0.9}, overwrite=True)
    assert json.loads(target.read_text())["ndcg@10"] == 0.9


def test_guard_creates_parent_directories(tmp_path):
    target = seed_path(tmp_path, 13, "gardian_training", "hybrid_bm25_faiss", "logs.json")
    guard_seed_artifact(target)
    assert target.parent.is_dir()


def test_discover_seeds(tmp_path):
    for seed in (100, 13, 42):
        write_seed_json(seed_path(tmp_path, seed, "x.json"), {})
    (tmp_path / "seeds" / "not_a_seed").mkdir()
    assert discover_seeds(tmp_path) == [13, 42, 100]


def test_discover_seeds_on_missing_tree(tmp_path):
    assert discover_seeds(tmp_path / "nope") == []


# ── aggregation ──────────────────────────────────────────────────────────────


def _write_eval(root, seed, ndcg):
    write_seed_json(
        seed_path(root, seed, "evaluation_hybrid_bm25_faiss.json"),
        {
            "meta": {"seed": seed},
            "results": {
                "hybrid_bm25_faiss": {
                    "medmcqa": {
                        "gardian": {"ndcg@10": ndcg, "mrr": 0.5},
                        "_meta": {"ignored": True},
                    }
                }
            },
        },
    )


def test_summarize_mean_and_sample_std():
    from scripts.aggregate_seeds import summarize

    out = summarize([0.1, 0.2, 0.3], [13, 21, 42])
    assert out["mean"] == pytest.approx(0.2)
    # sample std (n-1), not population std (which would be ~0.08165)
    assert out["std"] == pytest.approx(0.1)
    assert out["n"] == 3


def test_summarize_single_seed_has_no_std():
    from scripts.aggregate_seeds import summarize

    out = summarize([0.42], [42])
    assert out["mean"] == pytest.approx(0.42)
    assert out["std"] is None, "std over one seed is undefined, not 0.0"


def test_collect_retrieval_averages_across_seeds(tmp_path):
    from scripts.aggregate_seeds import collect_retrieval

    _write_eval(tmp_path, 13, 0.60)
    _write_eval(tmp_path, 21, 0.70)
    _write_eval(tmp_path, 42, 0.80)

    agg = collect_retrieval(tmp_path, [13, 21, 42])
    cell = agg["hybrid_bm25_faiss"]["medmcqa"]["gardian"]["ndcg@10"]
    assert cell["mean"] == pytest.approx(0.70)
    assert cell["std"] == pytest.approx(0.1)
    assert cell["seeds"] == [13, 21, 42]
    assert cell["values"] == [0.60, 0.70, 0.80]


def test_collect_retrieval_skips_meta_blocks(tmp_path):
    from scripts.aggregate_seeds import collect_retrieval

    _write_eval(tmp_path, 13, 0.6)
    agg = collect_retrieval(tmp_path, [13])
    assert "_meta" not in agg["hybrid_bm25_faiss"]["medmcqa"]


def test_report_incomplete_flags_missing_seeds(tmp_path):
    from scripts.aggregate_seeds import collect_retrieval, report_incomplete

    _write_eval(tmp_path, 13, 0.6)
    _write_eval(tmp_path, 21, 0.7)
    agg = collect_retrieval(tmp_path, [13, 21, 42])
    warnings = report_incomplete(agg, expected=3)
    assert warnings and "2/3 seeds" in warnings[0]


def test_aggregation_never_mutates_per_seed_artifacts(tmp_path):
    from scripts.aggregate_seeds import collect_retrieval

    _write_eval(tmp_path, 13, 0.6)
    src = seed_path(tmp_path, 13, "evaluation_hybrid_bm25_faiss.json")
    before = src.read_bytes()
    collect_retrieval(tmp_path, [13])
    assert src.read_bytes() == before


# ── CLI wiring ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "script", ["scripts/04_train_gardian.py", "scripts/05_evaluate_gardian.py"]
)
def test_scripts_expose_seeds_flag(script):
    out = subprocess.run(
        [sys.executable, script, "--help"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert out.returncode == 0, out.stderr
    assert "--seeds" in out.stdout
    assert "13 21 42 87 100" in " ".join(out.stdout.split())


# ---------------------------------------------------------------------------
# Within-pool normalisation (src/features/pool_norm.py)
# ---------------------------------------------------------------------------


from src.features.pool_norm import (
    minmax_normalise,
    pool_dispersion,
    rank_normalise,
    top_score_gap,
    zscore_normalise,
)
from src.features.schema import (
    DENSE_FEAT_DIM,
    SPARSE_FEAT_DIM,
    assert_feature_dims,
)


def test_minmax_hand_computed():
    # (10-1)/(10-1)=1, (5-1)/9=0.444..., (1-1)/9=0
    got = minmax_normalise(np.array([10.0, 5.0, 1.0]))
    assert got[0] == pytest.approx(1.0)
    assert got[1] == pytest.approx(4 / 9)
    assert got[2] == pytest.approx(0.0)


def test_zscore_has_zero_mean_unit_sd():
    out = zscore_normalise(np.array([1.0, 2.0, 3.0, 4.0]))
    assert out.mean() == pytest.approx(0.0, abs=1e-12)
    assert out.std() == pytest.approx(1.0)


def test_rank_is_one_for_best_and_zero_for_worst():
    out = rank_normalise(np.array([3.0, 9.0, 1.0]))
    assert out[1] == pytest.approx(1.0)   # 9.0 is best
    assert out[2] == pytest.approx(0.0)   # 1.0 is worst
    assert out[0] == pytest.approx(0.5)


def test_rank_is_scale_free():
    """The point of the rank feature: immune to the channel's units."""
    a = rank_normalise(np.array([1.0, 2.0, 3.0]))
    b = rank_normalise(np.array([1000.0, 2000.0, 3000.0]))
    assert np.allclose(a, b)


def test_constant_pool_is_neutral_not_nan():
    c = np.array([5.0, 5.0, 5.0])
    for fn in (minmax_normalise, zscore_normalise):
        out = fn(c)
        assert np.all(np.isfinite(out)) and np.allclose(out, 0.0)
    assert top_score_gap(c) == 0.0


def test_top_score_gap_hand_computed():
    # (10-6)/(10-2) = 0.5
    assert top_score_gap(np.array([10.0, 6.0, 2.0])) == pytest.approx(0.5)


def test_pool_dispersion_matches_std():
    v = np.array([1.0, 2.0, 3.0])
    assert pool_dispersion(v) == pytest.approx(float(v.std()))


def test_single_candidate_pool_does_not_crash():
    one = np.array([7.0])
    assert rank_normalise(one)[0] == pytest.approx(1.0)
    assert top_score_gap(one) == 0.0
    assert np.all(np.isfinite(minmax_normalise(one)))


def test_schema_widths_match_config():
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("configs/base.yaml")
    assert int(cfg.model.sparse_feat_dim) == SPARSE_FEAT_DIM
    assert int(cfg.model.dense_feat_dim) == DENSE_FEAT_DIM
    assert_feature_dims(cfg.model.sparse_feat_dim, cfg.model.dense_feat_dim)


def test_legacy_feature_widths_are_rejected_with_guidance():
    with pytest.raises(ValueError, match="Regenerate rank data"):
        assert_feature_dims(3, 4)


# ---------------------------------------------------------------------------
# Rank-data overwrite guard (scripts/03_generate_rank_data.py)
# ---------------------------------------------------------------------------


def _load_script03():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_gen_rank_data", "scripts/03_generate_rank_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


def _write_rank_file(path, n_queries):
    import json as _json

    with open(path, "w", encoding="utf-8") as f:
        for q in range(n_queries):
            for c in range(3):
                f.write(_json.dumps({"qid": f"q{q}", "pid": f"p{c}"}) + "\n")


def test_guard_blocks_smoke_test_from_destroying_a_full_generation(tmp_path):
    """A --max-queries run must not silently replace a completed rank file."""
    mod = _load_script03()
    target = tmp_path / "rank.jsonl"
    _write_rank_file(target, 500)
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        mod._guard_existing_rank_file(str(target), 5)


def test_guard_allows_explicit_override(tmp_path):
    mod = _load_script03()
    target = tmp_path / "rank.jsonl"
    _write_rank_file(target, 500)
    mod._guard_existing_rank_file(str(target), 5, allow_shrink=True)


def test_guard_allows_full_regeneration(tmp_path):
    mod = _load_script03()
    target = tmp_path / "rank.jsonl"
    _write_rank_file(target, 500)
    mod._guard_existing_rank_file(str(target), 500)


def test_guard_allows_a_new_path(tmp_path):
    mod = _load_script03()
    mod._guard_existing_rank_file(str(tmp_path / "fresh.jsonl"), 5)


def test_guard_ignores_tiny_existing_files(tmp_path):
    """A 10-query scratch file is not worth protecting; no false positives."""
    mod = _load_script03()
    target = tmp_path / "rank.jsonl"
    _write_rank_file(target, 10)
    mod._guard_existing_rank_file(str(target), 3)
