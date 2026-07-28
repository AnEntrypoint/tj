"""Contrastive training loss for positive (Claude Code) vs negative (public) data.

Implements an InfoNCE-style contrastive objective that pulls embeddings of
Claude Code (positive) agent sessions closer together while pushing embeddings
of public HF (negative) code apart. The loss operates on the pooled hidden
state from the hybrid stack, jointly trained with the standard LM loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss for positive vs negative sample separation.

    Positive samples are drawn from Claude Code agent sessions.
    Negative samples are drawn from public coding datasets.

    The loss encourages the model to produce similar embeddings for
    positive samples (same source) and dissimilar embeddings for
    negative samples (different source), creating a learned signal
    about what constitutes good agent behavior vs generic code.

    Args:
        temperature: Softmax temperature (lower = sharper separation).
        margin: Minimum margin between positive and negative similarity.
    """

    def __init__(self, temperature: float = 0.07, margin: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        pos_embeddings: torch.Tensor,  # (N, D) positive samples
        neg_embeddings: torch.Tensor,  # (M, D) negative samples
    ) -> torch.Tensor:
        """Compute InfoNCE contrastive loss.

        Args:
            pos_embeddings: Pooled hidden states from positive (Claude Code) data.
            neg_embeddings: Pooled hidden states from negative (public HF) data.

        Returns:
            Scalar contrastive loss.
        """
        if pos_embeddings.numel() == 0 or neg_embeddings.numel() == 0:
            return torch.tensor(0.0, device=pos_embeddings.device)

        # Normalize embeddings to unit sphere
        pos_norm = F.normalize(pos_embeddings, dim=-1)
        neg_norm = F.normalize(neg_embeddings, dim=-1)

        N = pos_norm.shape[0]
        M = neg_norm.shape[0]

        # Concatenate: [N positives, M negatives]
        all_emb = torch.cat([pos_norm, neg_norm], dim=0)  # (N+M, D)

        # Similarity matrix: all_emb @ all_emb^T
        sim = torch.matmul(all_emb, all_emb.T) / self.temperature  # (N+M, N+M)

        # Labels: for each positive, its own index is positive (0 to N-1)
        # For negatives, there are no positive labels (they're all negative)
        # We only compute loss over the positive samples
        labels = torch.arange(N, device=sim.device)

        # Mask out self-similarity
        mask = torch.eye(N + M, device=sim.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float("-inf"))

        # Only compute loss for the first N rows (positive samples)
        logits = sim[:N]  # (N, N+M)

        # InfoNCE: -log(exp(sim_pos) / sum(exp(sim_all)))
        loss = F.cross_entropy(logits, labels, reduction="mean")

        # Optionally add margin penalty for negative samples being too similar
        if self.margin > 0:
            # Penalize when negative samples are similar to positives
            pos_neg_sim = torch.matmul(pos_norm, neg_norm.T)  # (N, M)
            margin_loss = F.relu(pos_neg_sim - self.margin).mean()
            loss = loss + 0.1 * margin_loss

        return loss


class TripletLoss(nn.Module):
    """Triplet margin loss for positive/negative separation.

    For each positive pair (anchor, positive) and negative sample:
        loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)

    Simpler and more interpretable than InfoNCE, but requires careful
    triplet mining.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,   # (N, D)
        positive: torch.Tensor,  # (N, D)
        negative: torch.Tensor,  # (M, D)
    ) -> torch.Tensor:
        if anchor.numel() == 0 or positive.numel() == 0 or negative.numel() == 0:
            return torch.tensor(0.0, device=anchor.device)

        anchor_norm = F.normalize(anchor, dim=-1)
        positive_norm = F.normalize(positive, dim=-1)
        negative_norm = F.normalize(negative, dim=-1)

        # Pairwise distances
        d_pos = 1.0 - (anchor_norm * positive_norm).sum(dim=-1)  # (N,)
        d_neg = 1.0 - torch.matmul(anchor_norm, negative_norm.T)  # (N, M)

        # Hardest negative: the one closest to the anchor
        d_neg_hard = d_neg.min(dim=-1).values  # (N,)

        loss = F.relu(d_pos - d_neg_hard + self.margin).mean()
        return loss


def contrastive_loss_from_pooled(
    pos_pooled: torch.Tensor,  # (N_pos, D)
    neg_pooled: torch.Tensor,  # (N_neg, D)
    temperature: float = 0.07,
    loss_type: str = "infonce",
) -> torch.Tensor:
    """Convenience function: compute contrastive loss from pooled embeddings.

    Args:
        pos_pooled: Pooled hidden states from positive samples.
        neg_pooled: Pooled hidden states from negative samples.
        temperature: Temperature for InfoNCE.
        loss_type: "infonce" or "triplet".

    Returns:
        Scalar contrastive loss.
    """
    if loss_type == "triplet":
        # Use first half of positives as anchor, second half as positive
        n = pos_pooled.shape[0] // 2
        if n < 2:
            return torch.tensor(0.0, device=pos_pooled.device)
        anchor = pos_pooled[:n]
        positive = pos_pooled[n : 2 * n]
        criterion = TripletLoss(margin=0.5)
        return criterion(anchor, positive, neg_pooled)
    else:
        criterion = ContrastiveLoss(temperature=temperature)
        return criterion(pos_pooled, neg_pooled)