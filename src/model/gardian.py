"""
GARDIAN text-only re-ranking model.

Architecture:
  - Two feature branches (Sparse, Dense): 3-layer MLP with LayerNorm → scalar score
  - Query-adaptive controller: query embedding + question type → softmax (α, β)
  - Fusion: score = α·sparse + β·dense

No knowledge-graph branch (text-only RAG re-ranking per RQ1–RQ4).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class BranchMLP(nn.Module):
    """3-layer MLP scoring head (in → hidden → mid → 1) with LayerNorm."""

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


class ControllerMLP(nn.Module):
    """
    Maps [query_emb ‖ question_type_onehot] → (α, β) via softmax.

    Two simplex weights for sparse and dense branches only.
    """

    def __init__(
        self,
        query_feat_dim: int,
        n_qtypes: int,
        hidden: int,
        dropout: float = 0.1,
        n_branches: int = 2,
    ):
        super().__init__()
        self.n_branches = int(n_branches)
        in_dim = query_feat_dim + n_qtypes
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
            nn.Linear(mid, self.n_branches),
        )

    def forward(
        self,
        query_emb: torch.Tensor,
        qtype_onehot: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([query_emb, qtype_onehot], dim=-1)
        return F.softmax(self.net(x), dim=-1)


class GARDIAN(nn.Module):
    """
    Text-only GARDIAN re-ranker (sparse + dense fusion).

    After sorting candidates, the reader block lives in ``src.pipeline.rag_reader``.
    """

    def __init__(
        self,
        sparse_dim: int = 3,
        dense_dim: int = 4,
        branch_hidden: int = 128,
        controller_hidden: int = 128,
        query_feat_dim: int = 768,
        n_qtypes: int = 6,
        dropout: float = 0.1,
        text_only: bool = True,
        kg_dim: int = 0,
    ):
        super().__init__()
        self.text_only = bool(text_only)
        self.sparse_head = BranchMLP(sparse_dim, branch_hidden, dropout)
        self.dense_head = BranchMLP(dense_dim, branch_hidden, dropout)
        self.kg_head = None
        if not self.text_only and int(kg_dim) > 0:
            self.kg_head = BranchMLP(int(kg_dim), branch_hidden, dropout)
        self.controller = ControllerMLP(
            query_feat_dim,
            n_qtypes,
            controller_hidden,
            dropout,
            n_branches=2 if self.text_only else 3,
        )

    def forward(
        self,
        sparse_feats: torch.Tensor,
        dense_feats: torch.Tensor,
        kg_feats: Optional[torch.Tensor] = None,
        query_emb: Optional[torch.Tensor] = None,
        qtype_onehot: Optional[torch.Tensor] = None,
        kg_coverage: Optional[torch.Tensor] = None,
        ablation: Optional[str] = None,
        return_breakdown: bool = False,
    ):
        _ = kg_feats, kg_coverage
        if query_emb is None or qtype_onehot is None:
            raise ValueError("query_emb and qtype_onehot are required")

        s_sparse = self.sparse_head(sparse_feats)
        s_dense = self.dense_head(dense_feats)

        qtype_in = qtype_onehot
        if ablation == "no_qtype":
            qtype_in = torch.zeros_like(qtype_onehot)

        if ablation == "uniform_alpha":
            b = query_emb.shape[0]
            weights = torch.full(
                (b, 2),
                0.5,
                device=query_emb.device,
                dtype=query_emb.dtype,
            )
        else:
            weights = self.controller(query_emb, qtype_in)

        if ablation == "no_sparse_signal":
            beta = weights[:, 1] / (weights[:, 1] + 1e-8)
            scores = beta * s_dense
            weights = torch.stack([torch.zeros_like(beta), beta], dim=1)
        elif ablation == "no_dense_signal":
            alpha = weights[:, 0] / (weights[:, 0] + 1e-8)
            scores = alpha * s_sparse
            weights = torch.stack([alpha, torch.zeros_like(alpha)], dim=1)
        else:
            alpha, beta = weights[:, 0], weights[:, 1]
            scores = alpha * s_sparse + beta * s_dense

        if return_breakdown:
            alpha, beta = weights[:, 0], weights[:, 1]
            return scores, weights, {
                "s_sparse": s_sparse,
                "s_dense": s_dense,
                "sparse_contrib": alpha * s_sparse,
                "dense_contrib": beta * s_dense,
            }
        return scores, weights

    @torch.no_grad()
    def controller_weights(
        self,
        query_emb: torch.Tensor,
        qtype_onehot: torch.Tensor,
        *,
        ablation: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Query-only controller: (α_sparse, α_dense) before retrieval.

        ``query_emb`` and ``qtype_onehot`` may be shape ``(D,)`` or ``(1, D)``.
        """
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        if qtype_onehot.dim() == 1:
            qtype_onehot = qtype_onehot.unsqueeze(0)
        qtype_in = qtype_onehot
        if ablation == "no_qtype":
            qtype_in = torch.zeros_like(qtype_onehot)
        if ablation == "uniform_alpha":
            b = query_emb.shape[0]
            return torch.full(
                (b, 2),
                0.5,
                device=query_emb.device,
                dtype=query_emb.dtype,
            )
        return self.controller(query_emb, qtype_in)

    @torch.no_grad()
    def rerank(
        self,
        candidates: list,
        query_features: dict,
        device: str = "cpu",
    ) -> list:
        self.eval()
        self.to(device)
        batch = collate_candidates(candidates, query_features, device)
        ablation = query_features.get("ablation")
        out = self(
            sparse_feats=batch["sparse_feats"],
            dense_feats=batch["dense_feats"],
            query_emb=batch["query_emb"],
            qtype_onehot=batch["qtype_onehot"],
            ablation=ablation,
            return_breakdown=True,
        )
        scores, weights, breakdown = out
        scores_np = scores.cpu().numpy()
        sparse_np = breakdown["s_sparse"].detach().cpu().numpy()
        dense_np = breakdown["s_dense"].detach().cpu().numpy()
        sparse_contrib_np = breakdown["sparse_contrib"].detach().cpu().numpy()
        dense_contrib_np = breakdown["dense_contrib"].detach().cpu().numpy()

        for i, cand in enumerate(candidates):
            cand["gardian_score"] = float(scores_np[i])
            cand["ctrl_weights"] = weights[i].cpu().tolist()
            cand["sparse_alfa"] = float(cand["ctrl_weights"][0])
            cand["dense_alfa"] = float(cand["ctrl_weights"][1])
            cand["sparse_branch_score"] = float(sparse_np[i])
            cand["dense_branch_score"] = float(dense_np[i])
            cand["sparse_contribution"] = float(sparse_contrib_np[i])
            cand["dense_contribution"] = float(dense_contrib_np[i])
            cand["fusion_formula"] = (
                "score = sparse_alfa*sparse_branch_score + dense_alfa*dense_branch_score"
            )
        return sorted(candidates, key=lambda x: x["gardian_score"], reverse=True)


def build_gardian_from_model_cfg(model_cfg: Any) -> GARDIAN:
    question_types = _cfg_get(model_cfg, "question_types", [])
    text_only = bool(_cfg_get(model_cfg, "text_only", True))
    return GARDIAN(
        sparse_dim=int(_cfg_get(model_cfg, "sparse_feat_dim", 3)),
        dense_dim=int(_cfg_get(model_cfg, "dense_feat_dim", 4)),
        branch_hidden=int(_cfg_get(model_cfg, "branch_hidden", 128)),
        controller_hidden=int(_cfg_get(model_cfg, "controller_hidden", 128)),
        query_feat_dim=int(_cfg_get(model_cfg, "query_feat_dim", 768)),
        n_qtypes=len(question_types),
        dropout=float(_cfg_get(model_cfg, "dropout", 0.1)),
        text_only=text_only,
        kg_dim=int(_cfg_get(model_cfg, "kg_feat_dim", 0)),
    )


def collate_candidates(
    candidates: List[dict],
    query_features: dict,
    device: str,
) -> Dict[str, torch.Tensor]:
    import numpy as np

    to_t = lambda arr: torch.tensor(arr, dtype=torch.float32, device=device)
    n = len(candidates)
    sparse = to_t([c["sparse_feats"] for c in candidates])
    dense = to_t([c["dense_feats"] for c in candidates])
    q_emb = to_t([query_features["query_emb"]] * n)
    qtype = to_t([query_features["qtype_onehot"]] * n)
    return {
        "sparse_feats": sparse,
        "dense_feats": dense,
        "query_emb": q_emb,
        "qtype_onehot": qtype,
    }


def load_checkpoint_state(
    model: GARDIAN,
    state_dict: Dict[str, torch.Tensor],
    *,
    strict: bool = False,
):
    """
    Load weights; drop legacy KG branch / 3-way controller keys for text-only models.
    """
    filtered = {}
    for k, v in state_dict.items():
        if model.text_only and k.startswith("kg_head."):
            continue
        if (
            model.text_only
            and int(getattr(model.controller, "n_branches", 2)) == 2
            and k.startswith("controller.net.")
            and k.endswith(".weight")
            and v.ndim >= 2
            and int(v.shape[0]) == 3
        ):
            continue
        if (
            model.text_only
            and int(getattr(model.controller, "n_branches", 2)) == 2
            and k.startswith("controller.net.")
            and k.endswith(".bias")
            and v.ndim == 1
            and int(v.shape[0]) == 3
        ):
            continue
        filtered[k] = v
    return model.load_state_dict(filtered, strict=strict)


# Backward-compatible alias
_collate_candidates = collate_candidates
