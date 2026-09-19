"""
Fusion and learning-to-rank baselines that GARDIAN is measured against.

Reviewer 2's objection to the CoopIS submission was that without a globally
tuned weighting there is no evidence the gain comes from query adaptivity rather
than from simply learning better fusion weights. These are that evidence. Every
baseline consumes the SAME candidate pool, the SAME features and the SAME
splits as GARDIAN, so differences are attributable to the method alone.

Implemented here (src/baselines/fusion.py):

    global_alpha     one (alpha_s, alpha_d) grid-searched on dev, applied at test
                     -- the "Tuned-Static" baseline reviewer 2 asked for
    group_alpha      one alpha per question type, grid-searched on dev; tests
                     whether a learned controller is only a type detector
    oracle_alpha     per-query best alpha, chosen post hoc, plus the tie/invariance
                     diagnostics that say how much of that bound is reachable
    oracle_rerank    perfect ranking of the existing pool (ceiling from first-stage)
    pool_recall      fraction of queries whose gold passage is in the pool at all
    rrf_scores       reciprocal rank fusion
    sum_raw_scores   the submitted paper's UNNORMALISED sum, kept for continuity

Not yet implemented (tracked, needed for the full revision):

    static_ltr       GARDIAN's branches with a single query-independent learned
                     alpha -- isolates "learned branch heads" from "query-adaptive
                     weighting". Requires the trainer, so it lives with the model
                     rather than here.
    lambdamart       LightGBM ranker over the same features.

Read ``global_alpha`` first: on the CoopIS rank data it beat the submitted
GARDIAN on every back-end, which is what motivated the within-pool
normalisation in ``src/features/pool_norm.py``.
"""

from src.baselines.fusion import (
    ALPHA_GRID,
    global_alpha_fit,
    group_alpha_fit,
    group_alpha_scores,
    global_alpha_scores,
    oracle_alpha_per_query,
    oracle_rerank_ndcg,
    rrf_scores,
)

__all__ = [
    "ALPHA_GRID",
    "global_alpha_fit",
    "group_alpha_fit",
    "group_alpha_scores",
    "global_alpha_scores",
    "oracle_alpha_per_query",
    "oracle_rerank_ndcg",
    "rrf_scores",
]
