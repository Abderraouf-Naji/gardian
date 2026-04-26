"""
GARDIAN Re-ranking Model

Architecture (Section 3 of the paper):
  - Three feature branches (Sparse, Dense, KG), each a 2-layer MLP → scalar score
  - Query-adaptive controller: encodes query + auxiliary signals → softmax weights (α, β, γ)
  - Fusion: final_score = α·sparse + β·dense + γ·kg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BranchMLP(nn.Module):
    """2-layer MLP scoring head for one feature branch → scalar."""

    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)          # (batch,)


class ControllerMLP(nn.Module):
    """
    2-layer MLP that maps query representation → (α, β, γ) weights via softmax.

    Input: [query_emb ‖ question_type_onehot ‖ kg_coverage_scalar]
    Output: 3-dim simplex weights for [sparse, dense, kg] branches
    """

    def __init__(self, query_feat_dim: int, n_qtypes: int, hidden: int,
                 dropout: float = 0.1):
        super().__init__()
        in_dim = query_feat_dim + n_qtypes + 1   # +1 for kg_coverage scalar
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),               # logits for 3 branches
        )

    def forward(self, query_emb: torch.Tensor,
                qtype_onehot: torch.Tensor,
                kg_coverage: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 3) softmax-normalised weights."""
        x      = torch.cat([query_emb, qtype_onehot, kg_coverage.unsqueeze(-1)], dim=-1)
        logits = self.net(x)
        return F.softmax(logits, dim=-1)         # (batch, 3)


class GARDIAN(nn.Module):
    """
    Full GARDIAN re-ranking model.

    After sorting candidates by fused score, the **reader (RAG) last block**
    is implemented in ``src.pipeline.rag_reader`` (query + top-k passages →
    medical LLM).

    Parameters
    ----------
    sparse_dim      : dimensionality of sparse feature vector  (paper: 3)
    dense_dim       : dimensionality of dense feature vector   (paper: 4)
    kg_dim          : dimensionality of KG feature vector      (paper: 6)
    branch_hidden   : hidden units in each branch MLP          (paper: 128-256)
    controller_hidden: hidden units in controller MLP          (paper: 128-256)
    query_feat_dim  : query embedding size (from encoder)
    n_qtypes        : number of question-type categories
    dropout         : dropout probability
    """

    def __init__(self,
                 sparse_dim: int      = 3,
                 dense_dim: int       = 4,
                 kg_dim: int          = 6,
                 branch_hidden: int   = 128,
                 controller_hidden: int = 128,
                 query_feat_dim: int  = 768,
                 n_qtypes: int        = 6,
                 dropout: float       = 0.1):
        super().__init__()

        # Three branch scoring heads
        self.sparse_head = BranchMLP(sparse_dim, branch_hidden, dropout)
        self.dense_head  = BranchMLP(dense_dim,  branch_hidden, dropout)
        self.kg_head     = BranchMLP(kg_dim,     branch_hidden, dropout)

        # Query-adaptive controller
        self.controller  = ControllerMLP(query_feat_dim, n_qtypes,
                                         controller_hidden, dropout)

    def forward(
        self,
        sparse_feats: torch.Tensor,
        dense_feats: torch.Tensor,
        kg_feats: torch.Tensor,
        query_emb: torch.Tensor,
        qtype_onehot: torch.Tensor,
        kg_coverage: torch.Tensor,
        ablation: Optional[str] = None,
        return_breakdown: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        ablation
            Paper ablations (default ``None`` = full model):

            - ``uniform_alpha`` — fix (α,β,γ)=(⅓,⅓,⅓).
            - ``no_qtype`` — zero question-type one-hot before the controller.
            - ``no_kg_coverage`` — zero KG coverage scalar before the controller.
            - ``no_kg_signal`` — controller unchanged, but fusion drops the KG
              branch; sparse/dense masses are renormalized to sum to 1.

        Returns
        -------
        scores, weights
            Fused scores and (batch,3) weight matrix (last column 0 for
            ``no_kg_signal``).
        """
        s_sparse = self.sparse_head(sparse_feats)
        s_dense = self.dense_head(dense_feats)
        s_kg = self.kg_head(kg_feats)

        qtype_in = qtype_onehot
        kg_cov_in = kg_coverage
        if ablation == "no_qtype":
            qtype_in = torch.zeros_like(qtype_onehot)
        elif ablation == "no_kg_coverage":
            kg_cov_in = torch.zeros_like(kg_coverage)

        if ablation == "uniform_alpha":
            b = query_emb.shape[0]
            weights = torch.full(
                (b, 3),
                1.0 / 3.0,
                device=query_emb.device,
                dtype=query_emb.dtype,
            )
        else:
            weights = self.controller(query_emb, qtype_in, kg_cov_in)

        if ablation == "no_kg_signal":
            w = weights
            denom = w[:, 0] + w[:, 1] + 1e-8
            a = w[:, 0] / denom
            b = w[:, 1] / denom
            scores = a * s_sparse + b * s_dense
            weights = torch.stack([a, b, torch.zeros_like(a)], dim=1)
            if return_breakdown:
                sparse_contrib = a * s_sparse
                dense_contrib = b * s_dense
                kg_contrib = torch.zeros_like(sparse_contrib)
                return scores, weights, {
                    "s_sparse": s_sparse,
                    "s_dense": s_dense,
                    "s_kg": s_kg,
                    "sparse_contrib": sparse_contrib,
                    "dense_contrib": dense_contrib,
                    "kg_contrib": kg_contrib,
                }
            return scores, weights

        alpha, beta, gamma = weights[:, 0], weights[:, 1], weights[:, 2]
        scores = alpha * s_sparse + beta * s_dense + gamma * s_kg
        if return_breakdown:
            return scores, weights, {
                "s_sparse": s_sparse,
                "s_dense": s_dense,
                "s_kg": s_kg,
                "sparse_contrib": alpha * s_sparse,
                "dense_contrib": beta * s_dense,
                "kg_contrib": gamma * s_kg,
            }
        return scores, weights

    @torch.no_grad()
    def rerank(self, candidates: list, query_features: dict,
               device: str = "cpu") -> list:
        """
        High-level reranking call.
        candidates : list of dicts with precomputed feature tensors
        Returns candidates sorted by descending score.
        """
        self.eval()
        self.to(device)

        batch = _collate_candidates(candidates, query_features, device)
        ablation = query_features.get("ablation")
        scores, weights, breakdown = self(
            sparse_feats=batch["sparse_feats"],
            dense_feats=batch["dense_feats"],
            kg_feats=batch["kg_feats"],
            query_emb=batch["query_emb"],
            qtype_onehot=batch["qtype_onehot"],
            kg_coverage=batch["kg_coverage"],
            ablation=ablation,
            return_breakdown=True,
        )
        scores_np = scores.cpu().numpy()
        sparse_np = breakdown["s_sparse"].detach().cpu().numpy()
        dense_np = breakdown["s_dense"].detach().cpu().numpy()
        kg_np = breakdown["s_kg"].detach().cpu().numpy()
        sparse_contrib_np = breakdown["sparse_contrib"].detach().cpu().numpy()
        dense_contrib_np = breakdown["dense_contrib"].detach().cpu().numpy()
        kg_contrib_np = breakdown["kg_contrib"].detach().cpu().numpy()

        for i, cand in enumerate(candidates):
            cand["gardian_score"]   = float(scores_np[i])
            cand["ctrl_weights"]    = weights[i].cpu().tolist()
            cand["sparse_alfa"] = float(cand["ctrl_weights"][0])
            cand["dense_alfa"] = float(cand["ctrl_weights"][1])
            cand["kg_alfa"] = float(cand["ctrl_weights"][2])
            cand["sparse_branch_score"] = float(sparse_np[i])
            cand["dense_branch_score"] = float(dense_np[i])
            cand["kg_branch_score"] = float(kg_np[i])
            cand["sparse_contribution"] = float(sparse_contrib_np[i])
            cand["dense_contribution"] = float(dense_contrib_np[i])
            cand["kg_contribution"] = float(kg_contrib_np[i])
            cand["fusion_formula"] = (
                "score = sparse_alfa*sparse_branch_score + "
                "dense_alfa*dense_branch_score + kg_alfa*kg_branch_score"
            )

        return sorted(candidates, key=lambda x: x["gardian_score"], reverse=True)


def _collate_candidates(candidates, query_features, device):
    import torch, numpy as np
    to_t = lambda arr: torch.tensor(arr, dtype=torch.float32).to(device)

    sparse = to_t([c["sparse_feats"]  for c in candidates])
    dense  = to_t([c["dense_feats"]   for c in candidates])
    kg     = to_t([c["kg_feats"]      for c in candidates])
    q_emb  = to_t([query_features["query_emb"]] * len(candidates))
    qtype  = to_t([query_features["qtype_onehot"]] * len(candidates))
    kg_cov = to_t([query_features["kg_coverage"]] * len(candidates))

    return dict(sparse_feats=sparse, dense_feats=dense, kg_feats=kg,
                query_emb=q_emb, qtype_onehot=qtype, kg_coverage=kg_cov)