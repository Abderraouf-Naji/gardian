"""
Ranking objectives for GARDIAN.

Why this module exists
----------------------
The submitted model was trained with a pairwise margin loss,
``softplus(gamma - (r_pos - r_neg))``. That objective saturates: once a
positive clears a negative by the margin, the pair contributes almost no
gradient. Measured on 34,152 real training pairs from the epoch-1 checkpoint,
the median value of the gating term ``sigmoid(gamma - (r_pos - r_neg))`` is
0.0092, i.e. the median pair supplies under 1% of the gradient available at the
margin boundary. Training therefore stops improving the *ordering* long before
nDCG@10 stops improving.

The listwise objectives below score a whole candidate pool at once and weight
each comparison by how much swapping it would actually move nDCG, so they keep
optimising the top of the ranking. On the same features and the same combined
training set this is worth +4.9 nDCG@10 on PubMedQA-artificial and +1.4 on
MedMCQA over the pairwise loss (see ``docs/OBJECTIVE.md``).

Conventions
-----------
All listwise losses take ``scores``/``labels``/``mask`` of shape ``(B, N)``
where ``B`` is queries per batch and ``N`` the padded pool size. ``mask`` is
1.0 for real candidates and 0.0 for padding. Labels are non-negative gains
(binary 0/1 here; graded labels work unchanged). Queries with no positive
contribute zero loss rather than NaN.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn.functional as F

NEG_INF = -1.0e9


# --------------------------------------------------------------------------
# pairwise (the submitted objective, kept for the loss ablation)
# --------------------------------------------------------------------------
def pairwise_softplus_margin(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """``softplus(margin - (pos - neg))`` averaged over pairs."""
    return F.softplus(float(margin) - (pos_scores - neg_scores)).mean()


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def _gains(labels: torch.Tensor) -> torch.Tensor:
    """Exponential gain ``2^label - 1`` (identity on binary labels)."""
    return torch.pow(2.0, labels) - 1.0


def _ideal_dcg(labels: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """IDCG@k per query, shape ``(B, 1)``."""
    g = _gains(labels) * mask
    top = torch.sort(g, dim=1, descending=True).values[:, :k]
    disc = 1.0 / torch.log2(
        torch.arange(2, top.shape[1] + 2, device=labels.device, dtype=top.dtype)
    )
    return (top * disc).sum(dim=1, keepdim=True)


def _valid_queries(labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``(B, 1)`` float flag: query has at least one positive and 2+ candidates."""
    has_pos = ((labels * mask) > 0).any(dim=1, keepdim=True)
    has_two = mask.sum(dim=1, keepdim=True) >= 2
    return (has_pos & has_two).to(labels.dtype)


# --------------------------------------------------------------------------
# LambdaRank
# --------------------------------------------------------------------------
def lambdarank_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    sigma: float = 1.0,
    k: int = 10,
) -> torch.Tensor:
    """
    Pairwise logistic loss weighted by the nDCG@k change of swapping the pair.

    This is the objective behind LambdaMART. Unlike the margin loss it does not
    saturate on easy pairs and it does not spend capacity on pairs deep in the
    tail: ``|delta nDCG|`` is near zero for two candidates that are both far
    below the cutoff, and large for a pair straddling it.

    Ranks are computed from the current scores under ``no_grad`` -- the
    LambdaRank weight is treated as a constant, which is what makes the
    gradient well defined despite nDCG being a step function.
    """
    scores = scores.float()
    labels = labels.float()
    mask = mask.float()

    valid = _valid_queries(labels, mask)
    if float(valid.sum()) == 0.0:
        return scores.sum() * 0.0

    with torch.no_grad():
        masked = scores.masked_fill(mask == 0, NEG_INF)
        order = torch.argsort(masked, dim=1, descending=True)
        rank = torch.empty_like(order)
        ar = torch.arange(scores.shape[1], device=scores.device).expand_as(order)
        rank.scatter_(1, order, ar)
        disc = 1.0 / torch.log2(rank.float() + 2.0)          # 1 / log2(1 + rank_1based)

        g = _gains(labels) * mask
        delta_g = (g.unsqueeze(2) - g.unsqueeze(1)).abs()     # (B, N, N)
        delta_d = (disc.unsqueeze(2) - disc.unsqueeze(1)).abs()
        idcg = _ideal_dcg(labels, mask, k).clamp_min(1e-9).unsqueeze(2)
        weight = delta_g * delta_d / idcg

        pair_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)) * (
            labels.unsqueeze(2) > labels.unsqueeze(1)
        ).float()
        weight = weight * pair_mask

    diff = scores.unsqueeze(2) - scores.unsqueeze(1)          # s_i - s_j
    per_pair = F.softplus(-float(sigma) * diff) * weight
    per_query = per_pair.sum(dim=(1, 2)) * valid.squeeze(1)
    return per_query.sum() / valid.sum().clamp_min(1.0)


# --------------------------------------------------------------------------
# ApproxNDCG
# --------------------------------------------------------------------------
def approxndcg_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float = 1.0,
    k: int = 10,
) -> torch.Tensor:
    """
    ``1 - nDCG@k`` with the rank indicator replaced by a sigmoid.

    ``rank_i ~ 1 + sum_j sigmoid((s_j - s_i) / T)`` is differentiable end to
    end, so unlike LambdaRank the metric itself -- not a constant weight -- is
    what supplies the gradient. Smaller ``temperature`` is a tighter
    approximation but a harsher loss surface.
    """
    scores = scores.float()
    labels = labels.float()
    mask = mask.float()

    valid = _valid_queries(labels, mask)
    if float(valid.sum()) == 0.0:
        return scores.sum() * 0.0

    diff = scores.unsqueeze(1) - scores.unsqueeze(2)          # s_j - s_i at [b, i, j]
    pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
    eye = torch.eye(scores.shape[1], device=scores.device).unsqueeze(0)
    soft_rank = 1.0 + (
        torch.sigmoid(diff / float(temperature)) * pair_mask * (1.0 - eye)
    ).sum(dim=2)

    g = _gains(labels) * mask
    dcg = (g / torch.log2(soft_rank + 1.0)).sum(dim=1, keepdim=True)
    idcg = _ideal_dcg(labels, mask, k).clamp_min(1e-9)
    per_query = (1.0 - dcg / idcg) * valid
    return per_query.sum() / valid.sum().clamp_min(1.0)


# --------------------------------------------------------------------------
# ListNet
# --------------------------------------------------------------------------
def listnet_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    **_: object,
) -> torch.Tensor:
    """Cross-entropy between the score softmax and the gain distribution."""
    scores = scores.float()
    labels = labels.float()
    mask = mask.float()

    valid = _valid_queries(labels, mask)
    if float(valid.sum()) == 0.0:
        return scores.sum() * 0.0

    g = _gains(labels) * mask
    target = g / g.sum(dim=1, keepdim=True).clamp_min(1e-9)
    logp = torch.log_softmax(scores.masked_fill(mask == 0, NEG_INF), dim=1)
    per_query = -(target * logp * mask).sum(dim=1, keepdim=True) * valid
    return per_query.sum() / valid.sum().clamp_min(1.0)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
LISTWISE_LOSSES: Dict[str, Callable[..., torch.Tensor]] = {
    "lambdarank": lambdarank_loss,
    "approxndcg": approxndcg_loss,
    "listnet": listnet_loss,
}

PAIRWISE_LOSSES: Dict[str, Callable[..., torch.Tensor]] = {
    "pairwise_softplus_margin": pairwise_softplus_margin,
}

ALL_LOSSES = {**PAIRWISE_LOSSES, **LISTWISE_LOSSES}


def is_listwise(name: str) -> bool:
    return str(name) in LISTWISE_LOSSES


def build_loss(name: str) -> Callable[..., torch.Tensor]:
    """Look up a loss by its ``training.loss`` config value."""
    key = str(name)
    if key not in ALL_LOSSES:
        raise ValueError(
            f"Unknown training.loss={name!r}. Known: {sorted(ALL_LOSSES)}"
        )
    return ALL_LOSSES[key]
