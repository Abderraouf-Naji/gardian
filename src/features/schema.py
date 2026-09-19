"""
Exact layout of the feature vectors written into rank JSONL.

Reviewers asked what the features are; this module is the single machine-checkable
answer. ``scripts/03_generate_rank_data.py`` writes these vectors and
``configs/base.yaml`` declares their widths -- :func:`assert_feature_dims` fails
loudly if the three ever disagree.

Sparse branch (8 dims)
    0  sparse_score          raw BM25 or SPLADE++ score; the missing marker 0.0 is
                             replaced by the channel's lowest observed score
    1  idf_overlap           sum of IDF over content tokens shared by query and passage
    2  jaccard               |q ∩ p| / |q ∪ p| over content-word sets
    3  sparse_minmax         min-max over SCORED candidates; unscored pinned to 0
    4  sparse_z              z-score over SCORED candidates; unscored pinned to 0
    5  sparse_rank           1 - rank/(n-1) by sparse score, 1.0 = best in pool
    6  sparse_dispersion     sd of the channel's scored values (query-level constant)
    7  retrieved_by_sparse   1.0 if this channel actually retrieved the candidate

Dense branch (8 dims)
    0  dense_score           raw FAISS cosine or MedCPT dot product, floor-imputed
    1  mean_abs_diff         mean |q_emb - p_emb| (0.0 in score-only mode)
    2  max_abs_diff          max  |q_emb - p_emb| (0.0 in score-only mode)
    3  dense_z               z-score over SCORED candidates; unscored pinned to 0
    4  dense_minmax          min-max over SCORED candidates; unscored pinned to 0
    5  dense_rank            1 - rank/(n-1) by dense score, 1.0 = best in pool
    6  dense_dispersion      sd of the channel's scored values (query-level constant)
    7  retrieved_by_dense    1.0 if this channel actually retrieved the candidate

Why dimension 7 exists
----------------------
A union pool takes top-k from each channel, so 43.4% of candidates carry no
score from one of them. Recording that as 0.0 tells the model a falsehood: the
smallest genuinely measured dense similarity in the data is 0.347, so a stored
0.0 means "never scored", not "scored zero". The indicator lets the model
distinguish "this channel ranked it last" from "this channel never saw it",
and dims 0/3/4 no longer let the marker corrupt the channel's statistics.

Whether the indicator earns its place was measured rather than assumed, since
floor-imputing the score already places unretrieved candidates below every
genuine one. The answer is that it neither helps nor hurts: dropping both
indicators moves LambdaMART by -0.0010 nDCG@10 on MedMCQA and +0.0005 on
PubMedQA-artificial, and costs the neural model 0.0008-0.0010. All of that is
inside seed noise, so both are kept and the model consumes all 8+8.
``model.dropped_features`` can exclude any of them without regenerating data;
cutting hard is not free, as an 8-feature model loses 0.0095 / 0.0231.

Also written per record, outside the branch vectors:
    pool_stats  {sparse_dispersion, dense_dispersion, sparse_top_gap,
                 dense_top_gap, pool_size}  -- query-level, identical across the
                query's candidates.
"""

from __future__ import annotations

from typing import Any, Sequence

SPARSE_FEATURE_NAMES: tuple[str, ...] = (
    "sparse_score",
    "idf_overlap",
    "jaccard",
    "sparse_minmax",
    "sparse_z",
    "sparse_rank",
    "sparse_dispersion",
    "retrieved_by_sparse",
)

DENSE_FEATURE_NAMES: tuple[str, ...] = (
    "dense_score",
    "mean_abs_diff",
    "max_abs_diff",
    "dense_z",
    "dense_minmax",
    "dense_rank",
    "dense_dispersion",
    "retrieved_by_dense",
)

POOL_STAT_NAMES: tuple[str, ...] = (
    "sparse_dispersion",
    "dense_dispersion",
    "sparse_top_gap",
    "dense_top_gap",
    "pool_size",
    "sparse_coverage",
    "dense_coverage",
)

SPARSE_FEAT_DIM = len(SPARSE_FEATURE_NAMES)
DENSE_FEAT_DIM = len(DENSE_FEATURE_NAMES)

# Widths before within-pool normalisation was added, used to give a clear error
# when rank data predates the change.
# (3, 4) is the CoopIS submission; (7, 7) is the first normalisation revision,
# before the missing-score marker was handled.
LEGACY_FEATURE_WIDTHS = ((3, 4), (7, 7))
LEGACY_SPARSE_FEAT_DIM = 3
LEGACY_DENSE_FEAT_DIM = 4


# Features excluded from the model input by default: none. Every stored feature
# is fed to the model unless ``model.dropped_features`` says otherwise.
#
# Dropping the two retrieval indicators was measured and is a wash, not a win:
# LambdaMART moves -0.0010 nDCG@10 on MedMCQA and +0.0005 on
# PubMedQA-artificial, and the neural model is 0.0008-0.0010 *worse* without
# them (mean of 2 seeds x {lambdarank, listnet} at lr 1e-3). All four numbers
# are inside seed noise, so there is no evidence for removing them. Cutting
# harder is not a wash -- the best 8 features by tree gain lose 0.0095 / 0.0231.
# See docs/OBJECTIVE.md.
DEFAULT_DROPPED_FEATURES: tuple[str, ...] = ()


def active_feature_indices(
    dropped: Sequence[str] | None = None,
) -> tuple[list[int], list[int]]:
    """
    Column indices of the sparse/dense features kept as model input.

    Rank JSONL always stores the full 8+8 schema; dropping a feature is a
    column selection made when tensors are built, never a change to the data.
    That means a feature set can be changed and re-trained without re-running
    ``scripts/03_generate_rank_data.py``.
    """
    drop = set(DEFAULT_DROPPED_FEATURES if dropped is None else dropped)
    unknown = drop - set(SPARSE_FEATURE_NAMES) - set(DENSE_FEATURE_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown feature name(s) in model.dropped_features: {sorted(unknown)}. "
            f"Valid sparse: {list(SPARSE_FEATURE_NAMES)}; "
            f"valid dense: {list(DENSE_FEATURE_NAMES)}."
        )
    sparse = [i for i, n in enumerate(SPARSE_FEATURE_NAMES) if n not in drop]
    dense = [i for i, n in enumerate(DENSE_FEATURE_NAMES) if n not in drop]
    if not sparse or not dense:
        raise ValueError(
            "model.dropped_features would empty a whole branch; each branch "
            "needs at least one feature."
        )
    return sparse, dense


def active_feature_dims(dropped: Sequence[str] | None = None) -> tuple[int, int]:
    """Model input widths after ``dropped`` features are removed."""
    sparse, dense = active_feature_indices(dropped)
    return len(sparse), len(dense)


def active_feature_names(
    dropped: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Names of the retained features, in model-input order."""
    si, di = active_feature_indices(dropped)
    return ([SPARSE_FEATURE_NAMES[i] for i in si],
            [DENSE_FEATURE_NAMES[i] for i in di])


def select_active(
    sparse_feats: Sequence[float],
    dense_feats: Sequence[float],
    dropped: Sequence[str] | None = None,
) -> tuple[list[float], list[float]]:
    """
    Slice one candidate's stored 8+8 vectors down to the model input.

    Every place that builds model inputs must go through the same selection.
    Scoring a model on misaligned columns does not raise -- it produces a
    plausible ranking that means nothing -- so this is the single definition.
    Vectors already at the active width are passed through, which keeps
    checkpoints trained on all 16 features working.
    """
    si, di = active_feature_indices(dropped)
    sparse = (list(sparse_feats) if len(sparse_feats) == len(si)
              else [sparse_feats[i] for i in si])
    dense = (list(dense_feats) if len(dense_feats) == len(di)
             else [dense_feats[i] for i in di])
    return sparse, dense


def assert_feature_dims(sparse_dim: int, dense_dim: int) -> None:
    """
    Fail loudly when config widths disagree with this schema.

    These are the widths of the vectors *stored in rank JSONL*, which stay 8+8
    regardless of how many features the model consumes -- see
    :func:`active_feature_indices`.
    """
    if (int(sparse_dim), int(dense_dim)) in LEGACY_FEATURE_WIDTHS:
        raise ValueError(
            f"Rank data / config use superseded feature widths "
            f"({sparse_dim}, {dense_dim}). Regenerate rank data with "
            "scripts/03_generate_rank_data.py and set model.sparse_feat_dim="
            f"{SPARSE_FEAT_DIM}, model.dense_feat_dim={DENSE_FEAT_DIM}."
        )
    if int(sparse_dim) != SPARSE_FEAT_DIM or int(dense_dim) != DENSE_FEAT_DIM:
        raise ValueError(
            f"Feature width mismatch: config says sparse={sparse_dim} dense={dense_dim}, "
            f"schema says sparse={SPARSE_FEAT_DIM} dense={DENSE_FEAT_DIM} "
            "(src/features/schema.py)."
        )


def describe(sparse_feats: Sequence[float], dense_feats: Sequence[float]) -> dict[str, Any]:
    """Name every value of one candidate's feature vectors, for debugging."""
    return {
        **{n: float(v) for n, v in zip(SPARSE_FEATURE_NAMES, sparse_feats)},
        **{n: float(v) for n, v in zip(DENSE_FEATURE_NAMES, dense_feats)},
    }
