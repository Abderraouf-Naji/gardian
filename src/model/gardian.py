"""
GARDIAN: query-adaptive sparse-dense fusion for hybrid biomedical retrieval.

Architecture
------------
Two independent scoring branches and one controller, all three-layer MLPs with
LayerNorm. The branches and the controller share the *architecture*, not any
weights -- each is a separate ``nn.Module`` with its own parameters. (The paper
phrase "share a three-layer MLP" was ambiguous; the code is the ground truth
and the weights are not tied.)

    s_sparse = SparseBranch(f_sparse)          # f_sparse in R^3
    s_dense  = DenseBranch(f_dense)            # f_dense  in R^4
    (a_s, a_d) = softmax(Controller(h_q))      # per-query weights, a_s + a_d = 1
    score    = a_s * s_sparse + a_d * s_dense

The controller sees only query-level information, so the fusion weights are
constant across one query's candidates and vary between queries -- that is what
makes the re-ranker query-adaptive rather than a globally weighted fusion.

The controller's inputs are selected by ``controller_inputs``. Historically it
took the query embedding and nothing else, which made alpha a function of query
semantics alone -- fitted on PubMedQA/MedMCQA embeddings and extrapolated onto a
new collection. Measured consequence on TREC-COVID: alpha shifts systematically
downward out-of-domain on all four back-ends, and per-query adaptivity *loses*
to a single dev-fitted alpha on three of the four. ``"pool_feats"`` adds the
scale-free pool evidence of :mod:`src.features.pool_features` -- channel
agreement, per-channel peakedness, coverage -- which is what optimal alpha
actually depends on and which no query embedding can express.

``normalize_branches`` addresses the companion failure: with unbounded branch
outputs, ``alpha * s_sparse + (1 - alpha) * s_dense`` is not a mixing weight,
because the effective mix depends on each branch's learned output scale. Those
scales drift out-of-domain (sd ratio doubles on BM25+FAISS and SPLADE+++FAISS,
the two back-ends with the worst adaptivity), so an alpha calibrated in-domain
is arithmetically invalid on a new collection. Standardising both branches
within the pool before mixing makes alpha mean the same thing everywhere.

An earlier version
concatenated a one-hot question type; it was removed because the one-hot is
constant on both PubMedQA splits and separates PubMedQA from MedMCQA perfectly
in the combined training set, so it acted as a dataset indicator rather than a
query signal (measurements in ``docs/QUESTION_TYPE.md``). Question type remains
an *analysis* dimension -- see ``src/evaluation/qtype_breakdown.py`` -- and is
never a model input.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features.pool_features import POOL_FEATURE_DIM
from src.features.schema import (
    DEFAULT_DROPPED_FEATURES,
    active_feature_dims,
    select_active,
)

# Ablation identifiers accepted by ``GARDIAN.forward(..., ablation=...)``.
#   uniform_alpha     -- replace the controller with a fixed 0.5/0.5 split
#   fixed_alpha       -- replace it with a TUNED global alpha (see below)
#   no_sparse_signal  -- route all weight to the dense branch
#   no_dense_signal   -- route all weight to the sparse branch
#
# ``fixed_alpha`` is the ablation that isolates the paper's actual claim.
# ``uniform_alpha`` is not a fair fixed-weight control: 0.5/0.5 is an arbitrary
# point, so beating it conflates "adaptivity helps" with "0.5 is a bad
# constant". ``fixed_alpha`` uses the best *single* alpha for the whole dev
# split, applied to every test query, over the same learned branch outputs.
# GARDIAN minus that number is the value of per-query adaptation and nothing
# else. Measured on the CoopIS checkpoint (MedMCQA, 4,269 queries) it was
# +0.0007, against an Oracle-alpha headroom of +0.1208 over the same branches.
ABLATIONS = (
    "uniform_alpha",
    "fixed_alpha",
    "no_sparse_signal",
    "no_dense_signal",
)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a key from either a dict-like or attribute-style config object."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class BranchMLP(nn.Module):
    """
    Three-layer scoring head: ``in_dim -> hidden -> hidden//2 -> 1``.

    LayerNorm after each hidden linear stabilises training given that the raw
    input features live on very different scales (BM25 scores in the tens,
    cosine similarities in [0, 1]).
    """

    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        mid = max(int(hidden) // 2, 4)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, mid),
            nn.LayerNorm(mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


CONTROLLER_INPUTS = ("query_emb", "pool_feats")


class ControllerMLP(nn.Module):
    """
    Maps the query representation to simplex fusion weights ``(a_sparse, a_dense)``.

    Same three-layer shape as :class:`BranchMLP`, with its own parameters. The
    softmax makes the two weights sum to 1, so the controller allocates a fixed
    budget between the lexical and semantic channels rather than rescaling the
    combined score.

    ``inputs`` selects what the controller may condition on -- any non-empty
    subset of :data:`CONTROLLER_INPUTS`, concatenated in that fixed order:

    ``"query_emb"``
        The query embedding (768-d). Query semantics only.
    ``"pool_feats"``
        Scale-free retrieval evidence for this query's pool
        (:mod:`src.features.pool_features`): how much the two channels agree,
        how peaked each one is, how much of the pool each covered.

    The subset is an ablation axis: ``pool_feats`` alone, ``query_emb`` alone,
    and both, over identical branch outputs.

    ``forward`` accepts both tensors and consumes only the selected ones, so a
    caller can pass whatever it has without knowing the configuration -- the
    eval and training paths always pass both. An input that is selected but
    absent raises; an input that is present but unselected is ignored, which is
    why ``controller.inputs`` (recorded in every checkpoint's ``cfg``) is the
    only reliable statement of what a trained controller actually saw.
    """

    def __init__(
        self,
        query_feat_dim: int,
        hidden: int,
        dropout: float = 0.1,
        *,
        pool_feat_dim: int = 0,
        inputs: Sequence[str] = ("query_emb",),
    ):
        super().__init__()
        self.query_feat_dim = int(query_feat_dim)
        self.pool_feat_dim = int(pool_feat_dim)
        self.inputs = tuple(inputs)

        unknown = [i for i in self.inputs if i not in CONTROLLER_INPUTS]
        if unknown:
            raise ValueError(
                f"Unknown controller_inputs {unknown}; expected a subset of {CONTROLLER_INPUTS}"
            )
        if not self.inputs:
            raise ValueError("controller_inputs must be non-empty")
        if "pool_feats" in self.inputs and self.pool_feat_dim <= 0:
            raise ValueError(
                "controller_inputs includes 'pool_feats' but pool_feat_dim is 0"
            )

        self.in_dim = (
            (self.query_feat_dim if "query_emb" in self.inputs else 0)
            + (self.pool_feat_dim if "pool_feats" in self.inputs else 0)
        )
        mid = max(int(hidden) // 2, 4)
        # Pool features and embedding dims live on different scales, so a
        # LayerNorm on the concatenated input keeps one block from dominating
        # the first linear layer purely by magnitude. With a single input block
        # there is nothing to balance, and adding a LayerNorm there would
        # change the query-embedding-only model rather than reproduce it -- so
        # that case stays an identity, and checkpoints trained before this
        # parameter existed load unchanged.
        self.in_norm = (
            nn.LayerNorm(self.in_dim) if len(self.inputs) > 1 else nn.Identity()
        )
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, mid),
            nn.LayerNorm(mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 2),
        )

    def forward(
        self,
        query_emb: Optional[torch.Tensor] = None,
        pool_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        for name in CONTROLLER_INPUTS:
            if name not in self.inputs:
                continue
            x = query_emb if name == "query_emb" else pool_feats
            if x is None:
                raise ValueError(
                    f"controller_inputs includes {name!r} but it was not passed to forward()"
                )
            parts.append(x)
        h = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        return F.softmax(self.net(self.in_norm(h)), dim=-1)


class GARDIAN(nn.Module):
    """
    Query-adaptive sparse-dense re-ranker.

    Parameters
    ----------
    sparse_dim, dense_dim:
        Feature widths of the two branches (see ``src/features/``).
    branch_hidden, controller_hidden:
        Hidden widths of the branch heads and the controller.
    query_feat_dim:
        Dimensionality of the query embedding driving the controller
        (768 for the mean-pooled PubMedBERT sentence embedding).
    dropout:
        Dropout probability inside every MLP.
    pool_feat_dim:
        Width of the scale-free pool-evidence vector
        (:data:`src.features.pool_features.POOL_FEATURE_DIM`).
    controller_inputs:
        What the controller conditions on; see :class:`ControllerMLP`.
    normalize_branches:
        Standardise each branch's scores within the candidate pool before
        mixing, so ``alpha`` is a genuine mixing weight rather than a weight on
        two arbitrarily-scaled quantities. Requires pool-grouped input
        (``(B, N, F)`` with a mask); ignored for flat pairwise batches, which
        have no pool to normalise over.
    """

    def __init__(
        self,
        sparse_dim: int = 3,
        dense_dim: int = 4,
        branch_hidden: int = 128,
        controller_hidden: int = 128,
        query_feat_dim: int = 768,
        dropout: float = 0.1,
        use_controller: bool = True,
        pool_feat_dim: int = POOL_FEATURE_DIM,
        controller_inputs: Sequence[str] = ("query_emb",),
        normalize_branches: bool = False,
    ):
        super().__init__()
        self.sparse_dim = int(sparse_dim)
        self.dense_dim = int(dense_dim)
        self.query_feat_dim = int(query_feat_dim)
        self.use_controller = bool(use_controller)
        self.pool_feat_dim = int(pool_feat_dim)
        self.controller_inputs = tuple(controller_inputs)
        self.normalize_branches = bool(normalize_branches)

        self.sparse_head = BranchMLP(self.sparse_dim, branch_hidden, dropout)
        self.dense_head = BranchMLP(self.dense_dim, branch_hidden, dropout)
        # Without the controller the fusion weight is a constant, and a constant
        # is absorbable into each branch's final Linear layer -- so every fixed
        # alpha describes the same model class and there is nothing to tune.
        # The scores are simply summed, and no query embedding is needed at
        # inference, which removes the encoder pass entirely.
        self.controller = (
            ControllerMLP(
                self.query_feat_dim,
                controller_hidden,
                dropout,
                pool_feat_dim=self.pool_feat_dim,
                inputs=self.controller_inputs,
            )
            if self.use_controller else None
        )

    def fusion_weights(
        self,
        query_emb: torch.Tensor,
        *,
        pool_feats: Optional[torch.Tensor] = None,
        ablation: Optional[str] = None,
        fixed_alpha: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Per-query ``(a_sparse, a_dense)``, shape ``(batch, 2)``.

        ``ablation="uniform_alpha"`` forces 0.5/0.5. ``ablation="fixed_alpha"``
        forces ``(fixed_alpha, 1 - fixed_alpha)`` for every query, which with a
        dev-tuned value is the honest fixed-weight control for the adaptive
        claim -- see the note on :data:`ABLATIONS`.
        """
        if ablation == "uniform_alpha" or self.controller is None:
            return torch.full(
                (query_emb.shape[0], 2),
                0.5,
                device=query_emb.device,
                dtype=query_emb.dtype,
            )
        if ablation == "fixed_alpha":
            if fixed_alpha is None:
                raise ValueError(
                    "ablation='fixed_alpha' needs fixed_alpha=<value tuned on dev>. "
                    "Passing an untuned constant would make this ablation a "
                    "strawman rather than a control."
                )
            a = float(fixed_alpha)
            out = torch.empty(
                (query_emb.shape[0], 2), device=query_emb.device, dtype=query_emb.dtype
            )
            out[:, 0] = a
            out[:, 1] = 1.0 - a
            return out
        return self.controller(query_emb, pool_feats)

    @staticmethod
    def _standardize_in_pool(
        x: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Zero-mean, unit-sd within each pool along dim 1 of an ``(B, N)`` tensor.

        This is what makes ``alpha`` a mixing weight: after it, the two branches
        contribute on a common scale, so ``alpha = 0.6`` means the same thing on
        a new collection as it did in training. Masked (padding) entries are
        excluded from the statistics and left at zero. A pool with fewer than
        two live entries is returned unchanged -- there is no spread to divide by
        and rescaling a single point is meaningless.
        """
        if mask is None:
            m = torch.ones_like(x)
        else:
            m = mask.to(dtype=x.dtype)
        n = m.sum(dim=1, keepdim=True)
        safe = n.clamp(min=1.0)
        mean = (x * m).sum(dim=1, keepdim=True) / safe
        var = (((x - mean) ** 2) * m).sum(dim=1, keepdim=True) / safe
        out = (x - mean) / var.clamp(min=1e-12).sqrt()
        # Leave degenerate pools alone, and keep padding at exactly zero.
        out = torch.where(n > 1.0, out, x)
        return out * m if mask is not None else out

    def forward(
        self,
        sparse_feats: torch.Tensor,
        dense_feats: torch.Tensor,
        query_emb: torch.Tensor,
        *,
        pool_feats: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        ablation: Optional[str] = None,
        fixed_alpha: Optional[float] = None,
        return_breakdown: bool = False,
    ):
        """
        Score a batch of (query, candidate) pairs, or a batch of pools.

        Feature tensors are either flat ``(B, F)`` -- one row per candidate, the
        pairwise path -- or pool-grouped ``(B, N, F)`` with ``mask`` ``(B, N)``,
        the listwise path. ``normalize_branches`` needs the grouped form, since
        standardising a branch requires the pool it is standardised over.

        Returns ``(scores, weights)``, or ``(scores, weights, breakdown)`` when
        ``return_breakdown`` is set. ``weights`` is the controller output after
        any ablation has been applied, so logged weights always match the
        weights that produced the scores.
        """
        if ablation is not None and ablation not in ABLATIONS:
            raise ValueError(f"Unknown ablation {ablation!r}; expected one of {ABLATIONS}")

        grouped = sparse_feats.dim() == 3

        s_sparse = self.sparse_head(sparse_feats)
        s_dense = self.dense_head(dense_feats)

        if self.normalize_branches:
            if not grouped:
                raise ValueError(
                    "normalize_branches=True requires pool-grouped input "
                    "(sparse_feats of shape (B, N, F) with a mask); a flat batch "
                    "spans several queries, so normalising over it would mix pools."
                )
            s_sparse = self._standardize_in_pool(s_sparse, mask)
            s_dense = self._standardize_in_pool(s_dense, mask)

        weights = self.fusion_weights(
            query_emb, pool_feats=pool_feats, ablation=ablation, fixed_alpha=fixed_alpha
        )

        if ablation == "no_sparse_signal":
            beta = torch.ones_like(weights[:, 1])
            weights = torch.stack([torch.zeros_like(beta), beta], dim=1)
        elif ablation == "no_dense_signal":
            alpha = torch.ones_like(weights[:, 0])
            weights = torch.stack([alpha, torch.zeros_like(alpha)], dim=1)

        alpha, beta = weights[:, 0], weights[:, 1]
        if grouped:
            # One weight pair per query, broadcast across that query's pool.
            alpha = alpha.unsqueeze(1)
            beta = beta.unsqueeze(1)
        scores = alpha * s_sparse + beta * s_dense

        if return_breakdown:
            breakdown = {
                "s_sparse": s_sparse,
                "s_dense": s_dense,
                "sparse_contrib": alpha * s_sparse,
                "dense_contrib": beta * s_dense,
            }
            return scores, weights, breakdown
        return scores, weights

    @torch.no_grad()
    def controller_weights(
        self,
        query_emb: torch.Tensor,
        *,
        pool_feats: Optional[torch.Tensor] = None,
        ablation: Optional[str] = None,
        fixed_alpha: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Fusion weights from the query alone, before any retrieval happens.

        ``query_emb`` may be shape ``(D,)`` or ``(batch, D)``.
        """
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        return self.fusion_weights(
            query_emb, pool_feats=pool_feats, ablation=ablation, fixed_alpha=fixed_alpha
        )

    @torch.no_grad()
    def rerank(
        self,
        candidates: List[dict],
        query_features: dict,
        device: str = "cpu",
    ) -> List[dict]:
        """
        Score and sort one query's candidate pool, annotating each candidate.

        Each returned candidate gains its fused score, the controller weights
        used, and the per-branch contributions, so a ranking can be explained
        without re-running the model.
        """
        self.eval()
        self.to(device)
        batch = collate_candidates(
            candidates, query_features, device,
            dropped_features=query_features.get("dropped_features"),
        )
        scores, weights, breakdown = self(
            sparse_feats=batch["sparse_feats"],
            dense_feats=batch["dense_feats"],
            query_emb=batch["query_emb"],
            ablation=query_features.get("ablation"),
            return_breakdown=True,
        )

        scores_np = scores.cpu().numpy()
        sparse_np = breakdown["s_sparse"].cpu().numpy()
        dense_np = breakdown["s_dense"].cpu().numpy()
        sparse_contrib_np = breakdown["sparse_contrib"].cpu().numpy()
        dense_contrib_np = breakdown["dense_contrib"].cpu().numpy()
        weights_np = weights.cpu().numpy()

        for i, cand in enumerate(candidates):
            cand["gardian_score"] = float(scores_np[i])
            cand["ctrl_weights"] = [float(w) for w in weights_np[i]]
            cand["alpha_sparse"] = float(weights_np[i][0])
            cand["alpha_dense"] = float(weights_np[i][1])
            cand["sparse_branch_score"] = float(sparse_np[i])
            cand["dense_branch_score"] = float(dense_np[i])
            cand["sparse_contribution"] = float(sparse_contrib_np[i])
            cand["dense_contribution"] = float(dense_contrib_np[i])
            cand["fusion_formula"] = (
                "score = alpha_sparse * sparse_branch_score "
                "+ alpha_dense * dense_branch_score"
            )
        return sorted(candidates, key=lambda c: c["gardian_score"], reverse=True)


def dropped_features_from_cfg(model_cfg: Any) -> List[str]:
    """
    Feature names excluded from the model input, from the ``model`` config block.

    An unset key means "use the schema default" (the two retrieval indicators);
    an explicit empty list means "keep every stored feature".
    """
    dropped = _cfg_get(model_cfg, "dropped_features", None)
    if dropped is None:
        return list(DEFAULT_DROPPED_FEATURES)
    return [str(name) for name in dropped]


def build_gardian_from_model_cfg(model_cfg: Any) -> GARDIAN:
    """
    Construct a :class:`GARDIAN` from the ``model`` block of the YAML config.

    Branch widths come from the *active* feature set, not from
    ``sparse_feat_dim``/``dense_feat_dim`` -- those describe the vectors stored
    in rank JSONL, which stay 8+8 whatever the model consumes.
    """
    sparse_dim, dense_dim = active_feature_dims(dropped_features_from_cfg(model_cfg))
    return GARDIAN(
        sparse_dim=sparse_dim,
        dense_dim=dense_dim,
        use_controller=bool(_cfg_get(model_cfg, "use_controller", True)),
        branch_hidden=int(_cfg_get(model_cfg, "branch_hidden", 128)),
        controller_hidden=int(_cfg_get(model_cfg, "controller_hidden", 128)),
        query_feat_dim=int(_cfg_get(model_cfg, "query_feat_dim", 768)),
        dropout=float(_cfg_get(model_cfg, "dropout", 0.1)),
        controller_inputs=tuple(
            _cfg_get(model_cfg, "controller_inputs", ("query_emb",)) or ("query_emb",)
        ),
        normalize_branches=bool(_cfg_get(model_cfg, "normalize_branches", False)),
    )


def collate_candidates(
    candidates: List[dict],
    query_features: dict,
    device: str,
    dropped_features: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Stack one query's candidates into model inputs, broadcasting the query embedding.

    Candidates carry the full stored 8+8 feature vectors; the model consumes
    the active subset, so the same column selection used in training is applied
    here. ``dropped_features`` defaults to the schema default and should match
    what the checkpoint was trained with.
    """
    n = len(candidates)

    def to_tensor(values) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32, device=device)

    selected = [
        select_active(c["sparse_feats"], c["dense_feats"], dropped_features)
        for c in candidates
    ]
    return {
        "sparse_feats": to_tensor([sf for sf, _ in selected]),
        "dense_feats": to_tensor([df for _, df in selected]),
        "query_emb": to_tensor([query_features["query_emb"]] * n),
    }


def load_checkpoint_state(
    model: GARDIAN,
    state_dict: Dict[str, torch.Tensor],
    *,
    strict: bool = True,
):
    """
    Load weights, tolerating checkpoints from the earlier question-type model.

    Controllers trained before the one-hot was removed took
    ``[query_emb || qtype_onehot]``, so their first layer is wider than the
    current one. The query-embedding columns lead, so the trailing one-hot
    columns are dropped and the retained columns stay intact and in order.
    A narrower checkpoint is rejected rather than zero-padded, because
    inventing weights would silently produce a different model.
    """
    filtered: Dict[str, torch.Tensor] = {}
    # A GARDIAN-Lite model (use_controller=False) has no controller at all, so
    # there is no first layer to widen or narrow and nothing to reconcile.
    expected_in = (
        int(model.controller.net[0].weight.shape[1])
        if model.controller is not None
        else None
    )

    for key, value in state_dict.items():
        if key.startswith("controller.") and model.controller is None:
            # Loading a controller-trained checkpoint into a Lite model would
            # otherwise fail on unexpected keys under strict=True.
            continue
        if key == "controller.net.0.weight" and expected_in is not None and value.ndim == 2:
            got_in = int(value.shape[1])
            if got_in > expected_in:
                value = value[:, :expected_in].contiguous()
            elif got_in < expected_in:
                raise ValueError(
                    f"Checkpoint controller has {got_in} input features but this "
                    f"model needs {expected_in}. The checkpoint does not match "
                    "model.query_feat_dim in configs/base.yaml."
                )
        filtered[key] = value

    return model.load_state_dict(filtered, strict=strict)
